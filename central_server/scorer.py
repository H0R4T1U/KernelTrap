"""
Wraps the trained Isolation Forest model for scoring incoming syscall events.

Loads the model from the same directory structure produced by beth_iforest.py:
  model_dir/
    iforest.joblib
    scaler.joblib
    meta.json          (thresholds per entity + global fallback + freq tables)

Severity levels:
  0 = benign
  1 = low anomaly   (score < low_threshold)
  2 = high anomaly  (score < high_threshold)

Feature engineering pipeline KEPT IN SYNC with
masina_invata/isolation_forest/beth_iforest.py — every change to FEATURES,
extractor functions, or the lexicons MUST be mirrored here.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# FEATURES — must match beth_iforest.py exactly (same order).
# ============================================================================
FEATURES: List[str] = [
    # numerice
    "argsNum",
    "returnValue",
    # categorial frequency-encoded
    "eventId_freq",
    "mountNamespace_freq",
    # derivate din `args`
    "args_total_length",
    "has_sensitive_path",
    "has_canary_path",
    "has_path_traversal",
    "has_shell_metachar",
    "has_url_or_ip",
    # derivate din `processName`
    "processName_freq",
    "is_shell",
    "is_interpreter",
    "is_basic_recon",
    "is_recon_tool",
    "is_network_tool",
    "is_privileged_helper",
    # rolling per-(host, uid, pid) pe ultimele 5s
    "syscall_rate_5s",
    "unique_syscalls_5s",
    "failed_call_ratio_5s",
    "inter_arrival_ms",
    # threadId
    "is_thread_event",
]


# ============================================================================
# Lexicons & regexes — same constants as beth_iforest.py.
# ============================================================================
_SHELLS = frozenset({"bash", "sh", "dash", "zsh", "ksh", "ash", "csh", "tcsh", "fish"})
_INTERPRETERS = frozenset({"python", "python3", "perl", "ruby", "node", "lua", "php"})
_BASIC_RECON = frozenset({"id", "whoami", "w", "uname", "hostname", "ss", "netstat", "lsof", "ps", "who"})
_RECON_TOOLS = frozenset({"nmap", "masscan", "nikto", "gobuster", "dirb", "whatweb", "sqlmap", "hydra", "hashcat", "john"})
_NETWORK_TOOLS = frozenset({"nc", "ncat", "socat", "curl", "wget", "ssh", "scp", "rsync", "ftp", "telnet", "dig", "host", "nslookup"})
_PRIV_HELPERS = frozenset({"sudo", "su", "doas", "pkexec", "polkit", "sudoedit"})

_SENSITIVE_PATH_RE = re.compile(r"/etc/(shadow|passwd|sudoers|gshadow)|/root/|/\.ssh/|/\.aws/|\.bash_history|/\.docker/")
_CANARY_PATH_RE    = re.compile(r"\.aws/credentials|\.ssh/id_rsa|\.env\b|notes\.txt|\.jenkins_token")
_TRAVERSAL_RE      = re.compile(r"\.\./")
_SHELL_META_RE     = re.compile(r"[|;`]|\$\(|&&")
_URL_IP_RE         = re.compile(r"https?://|\b\d{1,3}(\.\d{1,3}){3}\b")


def _args_blob(args_value: Any) -> str:
    """Concatenate the .value of every dict in BETH's args field.

    Tolerant: accepts list-of-dicts directly OR a JSON string of the same.
    """
    if not args_value or args_value == "[]":
        return ""
    if isinstance(args_value, list):
        parsed = args_value
    else:
        try:
            parsed = json.loads(args_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
    if not isinstance(parsed, list):
        return ""
    parts: List[str] = []
    for a in parsed:
        if isinstance(a, dict):
            v = a.get("value", "")
            if v is not None:
                parts.append(str(v))
    return " ".join(parts)


def _extract_args_features(args_value: Any) -> Tuple[float, float, float, float, float, float]:
    blob = _args_blob(args_value)
    return (
        float(len(blob)),
        float(bool(_SENSITIVE_PATH_RE.search(blob))),
        float(bool(_CANARY_PATH_RE.search(blob))),
        float(bool(_TRAVERSAL_RE.search(blob))),
        float(bool(_SHELL_META_RE.search(blob))),
        float(bool(_URL_IP_RE.search(blob))),
    )


def _extract_processname_features(name: str, freq_table: Dict[str, float]) -> Tuple[float, ...]:
    n = (name or "").lower()
    return (
        float(freq_table.get(n, 0.0)),
        float(n in _SHELLS),
        float(n in _INTERPRETERS),
        float(n in _BASIC_RECON),
        float(n in _RECON_TOOLS),
        float(n in _NETWORK_TOOLS),
        float(n in _PRIV_HELPERS),
    )


# ============================================================================
# Rolling-feature cache: per-(host, uid, pid), last 60s of (timestamp, eventId, returnValue).
# Used to compute the same `syscall_rate_5s`, `unique_syscalls_5s`,
# `failed_call_ratio_5s`, `inter_arrival_ms` that beth_iforest.py computes
# at training time via groupby+rolling on a chunk.
# ============================================================================
class _RollingCache:
    """LRU per-process. Max ``max_keys`` distinct (host, uid, pid) tuples;
    oldest evicted when capacity hits."""

    __slots__ = ("_window", "_max", "_cache")

    def __init__(self, window_s: float = 60.0, max_keys: int = 50_000):
        self._window = window_s
        self._max = max_keys
        # key -> deque of (timestamp, eventId, returnValue)
        self._cache: "OrderedDict[Tuple[str, int, int], Deque[Tuple[float, int, float]]]" = OrderedDict()

    def update_and_features(
        self, host: str, uid: int, pid: int, ts: float, event_id: int, ret_val: float,
    ) -> Tuple[float, float, float, float]:
        key = (host, uid, pid)
        dq = self._cache.get(key)
        if dq is None:
            if len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            dq = deque()
            self._cache[key] = dq
        else:
            self._cache.move_to_end(key)

        # Evict events older than the window (60s).
        while dq and ts - dq[0][0] > self._window:
            dq.popleft()

        prev_ts = dq[-1][0] if dq else ts
        dq.append((ts, event_id, ret_val))

        # Rolling 5s features.
        cutoff_5s = ts - 5.0
        recent_eids = []
        recent_failed = 0
        recent_total = 0
        for t, eid, rv in dq:
            if t >= cutoff_5s:
                recent_eids.append(eid)
                recent_total += 1
                if rv < 0:
                    recent_failed += 1

        syscall_rate_5s      = float(recent_total)
        unique_syscalls_5s   = float(len(set(recent_eids))) if recent_eids else 0.0
        failed_call_ratio_5s = float(recent_failed / recent_total) if recent_total else 0.0
        inter_arrival_ms     = float((ts - prev_ts) * 1000.0)
        return syscall_rate_5s, unique_syscalls_5s, failed_call_ratio_5s, inter_arrival_ms


# ============================================================================
# Scorer
# ============================================================================
class Scorer:
    def __init__(self, model_dir: str):
        import joblib

        self._model_dir = Path(model_dir)
        self._scaler = joblib.load(self._model_dir / "scaler.joblib")
        self._model = joblib.load(self._model_dir / "iforest.joblib")

        meta_path = self._model_dir / "meta.json"
        self._meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)

        # Global fallback thresholds (used when per-entity thresholds are absent).
        gt = self._meta.get("global_thresholds", {})
        self._global_low = float(gt.get("low", -0.02))
        self._global_high = float(gt.get("high", -0.10))

        # Per-entity thresholds keyed by str(entity).
        self._entity_thresholds: Dict[str, Dict[str, float]] = self._meta.get("thresholds", {})

        self._host_col: Optional[str] = self._meta.get("host_col")

        # Frequency-encoding tables (keys are str(int) for eventId/mountNamespace,
        # lower-case strings for processName).
        self._eid_freq: Dict[str, float]    = self._meta.get("eventId_freq_table", {})
        self._mns_freq: Dict[str, float]    = self._meta.get("mountNamespace_freq_table", {})
        self._name_freq: Dict[str, float]   = self._meta.get("processName_freq_table", {})

        # Sanity: warn (don't crash) if the model was trained with the legacy
        # 7-feature vector — old joblib + new scorer = wrong feature alignment.
        meta_feats = self._meta.get("features")
        if meta_feats and list(meta_feats) != FEATURES:
            import sys
            print(
                f"[Scorer] WARNING: model meta.json features ({len(meta_feats)}) "
                f"differ from scorer FEATURES ({len(FEATURES)}). Re-train with "
                f"the current beth_iforest.py.",
                file=sys.stderr,
            )

        # Rolling cache for temporal features.
        self._rolling = _RollingCache(window_s=60.0, max_keys=50_000)

    def _entity_key(self, event: Dict[str, Any], hostname: str) -> str:
        if self._host_col:
            return f"{hostname}:{event.get('userId', 0)}"
        return str(event.get("userId", 0))

    def _thresholds_for(self, entity_key: str) -> Tuple[float, float]:
        t = self._entity_thresholds.get(entity_key, {})
        low = float(t.get("low", self._global_low))
        high = float(t.get("high", self._global_high))
        return low, high

    def _build_feature_row(self, event: Dict[str, Any], hostname: str) -> List[float]:
        """Build the canonical FEATURES vector for a single event.

        Mirrors build_features_for_chunk() in beth_iforest.py one row at a time.
        """
        # Numerics
        args_num    = float(event.get("argsNum", 0) or 0)
        return_val  = float(event.get("returnValue", 0) or 0)

        # Frequency-encoded categoricals
        eid_int     = int(event.get("eventId", 0) or 0)
        mns_int     = int(event.get("mountNamespace", 0) or 0)
        eid_freq    = float(self._eid_freq.get(str(eid_int), 0.0))
        mns_freq    = float(self._mns_freq.get(str(mns_int), 0.0))

        # Args-derived
        args_len, has_sens, has_canary, has_trav, has_meta, has_url = _extract_args_features(event.get("args"))

        # ProcessName-derived
        pn_freq, is_shell, is_interp, is_basic, is_recon, is_net, is_priv = _extract_processname_features(
            event.get("processName", ""), self._name_freq,
        )

        # Rolling-temporal
        ts          = float(event.get("timestamp", 0.0) or 0.0)
        uid_int     = int(event.get("userId", 0) or 0)
        pid_int     = int(event.get("processId", 0) or 0)
        rate_5s, uniq_5s, fail_5s, inter_ms = self._rolling.update_and_features(
            hostname, uid_int, pid_int, ts, eid_int, return_val,
        )

        # Thread
        tid_int     = int(event.get("threadId", pid_int) or pid_int)
        is_thread   = float(tid_int != pid_int)

        return [
            args_num, return_val,
            eid_freq, mns_freq,
            args_len, has_sens, has_canary, has_trav, has_meta, has_url,
            pn_freq, is_shell, is_interp, is_basic, is_recon, is_net, is_priv,
            rate_5s, uniq_5s, fail_5s, inter_ms,
            is_thread,
        ]

    def score_batch(self, events: List[Dict[str, Any]], hostname: str) -> List[Dict[str, Any]]:
        """Score a batch of event dicts. One result per input event."""
        if not events:
            return []

        rows = np.array([self._build_feature_row(e, hostname) for e in events], dtype=np.float64)

        scaled = self._scaler.transform(rows)
        raw_scores: np.ndarray = self._model.decision_function(scaled)

        results = []
        for i, event in enumerate(events):
            raw = float(raw_scores[i])
            entity_key = self._entity_key(event, hostname)
            low_thresh, high_thresh = self._thresholds_for(entity_key)

            if raw < high_thresh:
                severity = 2
            elif raw < low_thresh:
                severity = 1
            else:
                severity = 0

            results.append({
                "raw_score": raw,
                "severity": severity,
                "is_anomaly": severity > 0,
                "userId": event.get("userId", 0),
                "processName": event.get("processName", ""),
                "eventName": event.get("eventName", ""),
                "hostname": hostname,
            })

        return results
