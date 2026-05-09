#!/usr/bin/env python3
"""
BETH Isolation Forest — per-user (userId) percentile + two-tier severity version:

- Train IsolationForest on BENIGN-only events (sus==0 & evil==0) from training CSV
- Compute per-ENTITY percentile thresholds on VALIDATION scores
- Two-tier severity per entity:
    severity = 0 (benign)
    severity = 1 (low)  -> score < low_threshold(entity)
    severity = 2 (high) -> score < high_threshold(entity)

Entity can be:
- user (default):            entity = userId
- host_user (optional):      entity = f"{host}:{userId}" if host_col provided

Supports:
- large CSVs via chunked reads + sampling for training (reservoir sampling)
- evaluation on test CSV (if labels exist)
- scoring "outside logs" CSV (labels optional)

Expected feature columns (BETH kernel-process features):
processId, parentProcessId, userId, mountNamespace, eventId, argsNum, returnValue

Expected label columns (if evaluating/training):
sus, evil

Examples:

TRAIN (per-user):
  python beth_iforest_user2tier.py train \
    --train-csv labelled_training_data.csv \
    --val-csv labelled_validation_data.csv \
    --out-dir beth_iforest_model_user2tier \
    --low-percentile 2.0 \
    --high-percentile 0.2 \
    --n-estimators 600 \
    --n-jobs 32

TEST:
  python beth_iforest_user2tier.py test \
    --model-dir beth_iforest_model_user2tier \
    --test-csv labelled_testing_data.csv

PREDICT (outside logs):
  python beth_iforest_user2tier.py predict \
    --model-dir beth_iforest_model_user2tier \
    --csv my_new_logs.csv \
    --out-csv my_new_logs_scored.csv \
    --attach-features

Optional combined identity (host+user) if you have a host column:
  python ... train --host-col host --group-mode host_user
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================================
# FEATURES
#
# 22 features extracted from BETH-style syscall events. Replaces the 7-feature
# vector used by previous versions (processId / parentProcessId / userId /
# mountNamespace / eventId / argsNum / returnValue).
#
# Reasoning per group:
#  - argsNum, returnValue: real numeric measurements, kept as-is.
#  - eventId_freq, mountNamespace_freq: were treated as ordered numerics
#    before, now frequency-encoded (log(1+count) on benign training set).
#  - args_*: derived from BETH's `args` JSON field — was previously thrown away.
#  - processName_*: derived from BETH's `processName` field — was thrown away;
#    recovers signal from the H4 (LOLBAS / recon binary) heuristic that was
#    removed from window_v3.py.
#  - rolling_*: derived from `timestamp` field — captures temporal context
#    per-process which was previously absent (every event scored independently).
#  - is_thread_event: derived from `threadId` — distinguishes pthread_create
#    children from process forks.
#
# IMPORTANT: this list is the canonical order of the StandardScaler / model
# input vector. central_server/scorer.py MUST match it exactly.
# ============================================================================
FEATURES: List[str] = [
    # numerice OK (rămân)
    "argsNum",
    "returnValue",
    # categorial fix (frequency-encoded la training)
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
    # derivat din `threadId`
    "is_thread_event",
]

LABEL_COLS = ["sus", "evil"]

# Raw BETH columns we need to READ from CSV in order to derive FEATURES above.
# Superset of FEATURES because most of the new features come from raw fields
# that don't survive into the trained vector unchanged.
RAW_INPUT_COLS: List[str] = [
    "timestamp",
    "processId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "threadId",
    "processName",
    "hostName",
    "eventId",
    "argsNum",
    "returnValue",
    "args",
]

# Numeric raw columns — coerced via pd.to_numeric and required not-NaN.
# (processName, hostName, args are strings; missing → empty string is OK.)
RAW_NUMERIC_COLS: List[str] = [
    "timestamp",
    "processId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "threadId",
    "eventId",
    "argsNum",
    "returnValue",
]

# ============================================================================
# Feature lexicons & regexes
#
# Deterministic constants. Listed explicitly so it's easy to audit what the
# model considers "shell-ish" or "recon-ish" without digging into a separate
# feature module. Keep in sync with central_server/scorer.py.
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


# -----------------------------
# Helpers
# -----------------------------

def ensure_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns: {missing}. Found: {list(df.columns)}")


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def drop_bad_rows(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.dropna(subset=cols)


# ============================================================================
# Feature extractors (KEEP IN SYNC with central_server/scorer.py)
# ============================================================================

def _args_blob(args_value: Any) -> str:
    """Concatenate the .value of every dict in BETH's args JSON list.

    BETH stores args as: '[{"name":"pathname","type":"const char*","value":"/etc/shadow"},...]'
    We only care about the values (paths, IPs, command lines).
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
    parts = []
    for a in parsed:
        if isinstance(a, dict):
            v = a.get("value", "")
            if v is not None:
                parts.append(str(v))
    return " ".join(parts)


def extract_args_features_vec(args_series: pd.Series) -> pd.DataFrame:
    """Vectorized args features. Empty / unparseable args → all zeros."""
    blobs = args_series.fillna("").map(_args_blob)
    return pd.DataFrame({
        "args_total_length":  blobs.str.len().astype(np.float64),
        "has_sensitive_path": blobs.str.contains(_SENSITIVE_PATH_RE, regex=True, na=False).astype(np.float64),
        "has_canary_path":    blobs.str.contains(_CANARY_PATH_RE,    regex=True, na=False).astype(np.float64),
        "has_path_traversal": blobs.str.contains(_TRAVERSAL_RE,      regex=True, na=False).astype(np.float64),
        "has_shell_metachar": blobs.str.contains(_SHELL_META_RE,     regex=True, na=False).astype(np.float64),
        "has_url_or_ip":      blobs.str.contains(_URL_IP_RE,         regex=True, na=False).astype(np.float64),
    }, index=args_series.index)


def extract_processname_features_vec(name_series: pd.Series, freq_table: Dict[str, float]) -> pd.DataFrame:
    """Vectorized processName features. Names lower-cased before lookup."""
    names = name_series.fillna("").astype(str).str.lower()
    return pd.DataFrame({
        "processName_freq":     names.map(freq_table).fillna(0.0).astype(np.float64),
        "is_shell":             names.isin(_SHELLS).astype(np.float64),
        "is_interpreter":       names.isin(_INTERPRETERS).astype(np.float64),
        "is_basic_recon":       names.isin(_BASIC_RECON).astype(np.float64),
        "is_recon_tool":        names.isin(_RECON_TOOLS).astype(np.float64),
        "is_network_tool":      names.isin(_NETWORK_TOOLS).astype(np.float64),
        "is_privileged_helper": names.isin(_PRIV_HELPERS).astype(np.float64),
    }, index=name_series.index)


def compute_rolling_features_chunk(chunk: pd.DataFrame, window_s: float = 5.0) -> pd.DataFrame:
    """Per-chunk rolling features grouped by (hostName, userId, processId).

    Approximation: rolling window does NOT extend across chunk boundaries.
    For chunked training (chunksize=500k events) this is acceptable because a
    process typically completes its activity well within a single chunk; the
    inaccuracy at chunk borders affects a tiny fraction of rows.
    """
    n_total = len(chunk)
    if n_total == 0:
        return pd.DataFrame({
            "syscall_rate_5s":      np.zeros(0, dtype=np.float64),
            "unique_syscalls_5s":   np.zeros(0, dtype=np.float64),
            "failed_call_ratio_5s": np.zeros(0, dtype=np.float64),
            "inter_arrival_ms":     np.zeros(0, dtype=np.float64),
        }, index=chunk.index)

    # Default to safe values; overwritten in the group loop.
    rate_5s          = np.ones(n_total, dtype=np.float64)
    unique_5s        = np.ones(n_total, dtype=np.float64)
    failed_ratio_5s  = np.zeros(n_total, dtype=np.float64)
    inter_arrival_ms = np.zeros(n_total, dtype=np.float64)

    if not all(c in chunk.columns for c in ("timestamp", "hostName", "userId", "processId", "eventId", "returnValue")):
        return pd.DataFrame({
            "syscall_rate_5s":      rate_5s,
            "unique_syscalls_5s":   unique_5s,
            "failed_call_ratio_5s": failed_ratio_5s,
            "inter_arrival_ms":     inter_arrival_ms,
        }, index=chunk.index)

    # Sort once by group + timestamp; remember original positions to write back.
    sorted_idx_pos = np.argsort(
        chunk[["hostName", "userId", "processId", "timestamp"]].astype(str).agg("|".join, axis=1).values,
        kind="stable",
    )
    sorted_chunk = chunk.iloc[sorted_idx_pos]

    ts_arr   = sorted_chunk["timestamp"].to_numpy(dtype=np.float64)
    eid_arr  = sorted_chunk["eventId"].to_numpy()
    rv_arr   = sorted_chunk["returnValue"].to_numpy(dtype=np.float64)
    host_arr = sorted_chunk["hostName"].astype(str).to_numpy()
    uid_arr  = sorted_chunk["userId"].astype(str).to_numpy()
    pid_arr  = sorted_chunk["processId"].astype(str).to_numpy()

    # Two-pointer per group on the sorted view.
    n = len(sorted_chunk)
    out_rate    = np.empty(n, dtype=np.float64)
    out_unique  = np.empty(n, dtype=np.float64)
    out_failed  = np.empty(n, dtype=np.float64)
    out_inter   = np.empty(n, dtype=np.float64)

    left = 0
    group_start = 0
    for right in range(n):
        same_group = (
            right > 0
            and host_arr[right] == host_arr[right - 1]
            and uid_arr[right]  == uid_arr[right - 1]
            and pid_arr[right]  == pid_arr[right - 1]
        )
        if not same_group:
            left = right
            group_start = right
            out_inter[right] = 0.0
        else:
            out_inter[right] = (ts_arr[right] - ts_arr[right - 1]) * 1000.0
        # Advance left until window fits within 5s
        cutoff = ts_arr[right] - window_s
        if left < group_start:
            left = group_start
        while left < right and ts_arr[left] < cutoff:
            left += 1
        win_eid = eid_arr[left:right + 1]
        win_rv  = rv_arr[left:right + 1]
        size = win_eid.size
        out_rate[right]   = float(size)
        out_unique[right] = float(np.unique(win_eid).size)
        out_failed[right] = float((win_rv < 0).sum() / size) if size else 0.0

    # Write back to original positions.
    rate_5s[sorted_idx_pos]          = out_rate
    unique_5s[sorted_idx_pos]        = out_unique
    failed_ratio_5s[sorted_idx_pos]  = out_failed
    inter_arrival_ms[sorted_idx_pos] = out_inter

    return pd.DataFrame({
        "syscall_rate_5s":      rate_5s,
        "unique_syscalls_5s":   unique_5s,
        "failed_call_ratio_5s": failed_ratio_5s,
        "inter_arrival_ms":     inter_arrival_ms,
    }, index=chunk.index)


def build_features_for_chunk(chunk: pd.DataFrame, freq_tables: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Apply the full feature pipeline to a raw BETH chunk.

    Returns a DataFrame with EXACTLY the columns of FEATURES, in canonical
    order, ready to be fed to StandardScaler.partial_fit / transform.
    """
    eid_table  = freq_tables.get("eventId_freq_table", {})
    mns_table  = freq_tables.get("mountNamespace_freq_table", {})
    name_table = freq_tables.get("processName_freq_table", {})

    eventId_freq = (
        chunk["eventId"].astype("Int64").astype(str).map(eid_table).fillna(0.0).astype(np.float64)
        if "eventId" in chunk.columns else pd.Series(0.0, index=chunk.index)
    )
    mountNs_freq = (
        chunk["mountNamespace"].astype("Int64").astype(str).map(mns_table).fillna(0.0).astype(np.float64)
        if "mountNamespace" in chunk.columns else pd.Series(0.0, index=chunk.index)
    )

    if "args" in chunk.columns:
        args_feats = extract_args_features_vec(chunk["args"])
    else:
        args_feats = pd.DataFrame(0.0, index=chunk.index, columns=[
            "args_total_length", "has_sensitive_path", "has_canary_path",
            "has_path_traversal", "has_shell_metachar", "has_url_or_ip",
        ])

    if "processName" in chunk.columns:
        name_feats = extract_processname_features_vec(chunk["processName"], name_table)
    else:
        name_feats = pd.DataFrame(0.0, index=chunk.index, columns=[
            "processName_freq", "is_shell", "is_interpreter",
            "is_basic_recon", "is_recon_tool", "is_network_tool", "is_privileged_helper",
        ])

    rolling_feats = compute_rolling_features_chunk(chunk)

    if "threadId" in chunk.columns and "processId" in chunk.columns:
        is_thread = (chunk["threadId"].fillna(0).astype(np.int64) !=
                     chunk["processId"].fillna(0).astype(np.int64)).astype(np.float64)
    else:
        is_thread = pd.Series(0.0, index=chunk.index)

    out = pd.DataFrame({
        "argsNum":              chunk["argsNum"].fillna(0).astype(np.float64) if "argsNum" in chunk.columns else 0.0,
        "returnValue":          chunk["returnValue"].fillna(0).astype(np.float64) if "returnValue" in chunk.columns else 0.0,
        "eventId_freq":         eventId_freq.values,
        "mountNamespace_freq":  mountNs_freq.values,
        "args_total_length":    args_feats["args_total_length"].values,
        "has_sensitive_path":   args_feats["has_sensitive_path"].values,
        "has_canary_path":      args_feats["has_canary_path"].values,
        "has_path_traversal":   args_feats["has_path_traversal"].values,
        "has_shell_metachar":   args_feats["has_shell_metachar"].values,
        "has_url_or_ip":        args_feats["has_url_or_ip"].values,
        "processName_freq":     name_feats["processName_freq"].values,
        "is_shell":             name_feats["is_shell"].values,
        "is_interpreter":       name_feats["is_interpreter"].values,
        "is_basic_recon":       name_feats["is_basic_recon"].values,
        "is_recon_tool":        name_feats["is_recon_tool"].values,
        "is_network_tool":      name_feats["is_network_tool"].values,
        "is_privileged_helper": name_feats["is_privileged_helper"].values,
        "syscall_rate_5s":      rolling_feats["syscall_rate_5s"].values,
        "unique_syscalls_5s":   rolling_feats["unique_syscalls_5s"].values,
        "failed_call_ratio_5s": rolling_feats["failed_call_ratio_5s"].values,
        "inter_arrival_ms":     rolling_feats["inter_arrival_ms"].values,
        "is_thread_event":      is_thread.values,
    }, index=chunk.index)

    # Safety: guarantee the column order
    return out[FEATURES]


def build_frequency_tables(train_csv: str, chunksize: int) -> Dict[str, Dict[str, float]]:
    """Pre-pass over training CSV to build log-frequency tables for the three
    categorical fields we frequency-encode (eventId, mountNamespace,
    processName). Computed on BENIGN rows only (sus==0 & evil==0).

    Output values are log(1 + count) — handles wide dynamic ranges and gives
    unseen categories a score of 0 (lowest log-frequency).
    """
    eid_counter:  Counter = Counter()
    mns_counter:  Counter = Counter()
    name_counter: Counter = Counter()

    usecols = ["eventId", "mountNamespace", "processName", "sus", "evil"]
    for chunk in pd.read_csv(train_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue
        eid_counter.update(pd.to_numeric(benign["eventId"], errors="coerce").dropna().astype(int))
        mns_counter.update(pd.to_numeric(benign["mountNamespace"], errors="coerce").dropna().astype(int))
        name_counter.update(benign["processName"].fillna("").astype(str).str.lower())

    return {
        "eventId_freq_table":         {str(k): float(np.log1p(v)) for k, v in eid_counter.items()},
        "mountNamespace_freq_table":  {str(k): float(np.log1p(v)) for k, v in mns_counter.items()},
        "processName_freq_table":     {str(k): float(np.log1p(v)) for k, v in name_counter.items()},
    }


def _read_raw_chunks(csv_path: str, chunksize: int, extra_cols: Optional[List[str]] = None) -> Iterable[pd.DataFrame]:
    """Stream CSV reading the union of RAW_INPUT_COLS + LABEL_COLS + extras.

    Tolerant: columns missing from the CSV are silently skipped (older BETH
    dumps may not have `args` or `stackAddresses`).
    """
    extra_cols = list(extra_cols or [])
    wanted = list(dict.fromkeys(RAW_INPUT_COLS + LABEL_COLS + extra_cols))
    # Inspect header once to drop missing columns
    header = pd.read_csv(csv_path, nrows=0)
    available = [c for c in wanted if c in header.columns]
    missing = [c for c in wanted if c not in header.columns]
    if missing:
        # Non-fatal — features depending on them just default to 0.
        print(f"[warn] CSV {csv_path} is missing columns: {missing}")
    for chunk in pd.read_csv(csv_path, usecols=available, chunksize=chunksize, low_memory=False):
        chunk = coerce_numeric(chunk, [c for c in RAW_NUMERIC_COLS if c in chunk.columns])
        # Required for FEATURES; if eventId or mountNamespace is NaN we can't
        # compute the frequency-encoded features, so drop those rows.
        for required in ("eventId", "mountNamespace"):
            if required in chunk.columns:
                chunk = chunk.dropna(subset=[required])
        yield chunk


def read_csv_stream(
    csv_path: str,
    usecols: List[str],
    chunksize: int,
    dtype_map: Optional[Dict[str, Any]] = None,
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=dtype_map,
        low_memory=False,
    ):
        yield chunk


def reservoir_add(
    reservoir: List[np.ndarray],
    item: np.ndarray,
    seen: int,
    k: int,
    rng: np.random.Generator,
) -> None:
    """Classic reservoir sampling to keep a uniform sample of size k from a stream."""
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.integers(0, seen)
    if j < k:
        reservoir[j] = item


def safe_percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def make_entity_keys(
    df: pd.DataFrame,
    user_col: str,
    host_col: Optional[str],
    group_mode: str,
) -> np.ndarray:
    """
    Returns an array of entity keys per row (dtype object -> strings).
    group_mode:
      - "user": entity = userId
      - "host_user": entity = "{host}:{userId}" (requires host_col)
    """
    if user_col not in df.columns:
        raise ValueError(f"Missing user_col '{user_col}' in CSV. Available: {list(df.columns)}")

    u = df[user_col].astype("string").fillna("UNKNOWN_USER").to_numpy()

    if group_mode == "user":
        return u.astype(object)

    if group_mode == "host_user":
        if not host_col or host_col not in df.columns:
            raise ValueError("group_mode=host_user requires --host-col and that column must exist in CSV.")
        h = df[host_col].astype("string").fillna("UNKNOWN_HOST").to_numpy()
        return np.char.add(np.char.add(h, ":"), u).astype(object)

    raise ValueError("group_mode must be one of: user, host_user")


# -----------------------------
# Model bundle
# -----------------------------

@dataclass
class ModelBundle:
    scaler: StandardScaler
    model: IsolationForest
    features: List[str]

    user_col: str
    host_col: Optional[str]
    group_mode: str

    low_percentile: float
    high_percentile: float

    thresholds: Dict[str, Dict[str, float]]   # entity -> {low, high, n_scores}
    global_thresholds: Dict[str, float]       # fallback {low, high, n_scores}

    # Frequency tables for the categorical features (eventId, mountNamespace,
    # processName). Computed once on the benign training set and frozen here.
    freq_tables: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        dump(self.scaler, os.path.join(out_dir, "scaler.joblib"))
        dump(self.model, os.path.join(out_dir, "iforest.joblib"))
        meta = {
            "features": self.features,
            "user_col": self.user_col,
            "host_col": self.host_col,
            "group_mode": self.group_mode,
            "low_percentile": float(self.low_percentile),
            "high_percentile": float(self.high_percentile),
            "thresholds": self.thresholds,
            "global_thresholds": self.global_thresholds,
            "eventId_freq_table":        self.freq_tables.get("eventId_freq_table", {}),
            "mountNamespace_freq_table": self.freq_tables.get("mountNamespace_freq_table", {}),
            "processName_freq_table":    self.freq_tables.get("processName_freq_table", {}),
            "notes": (
                "normality_score = IsolationForest.decision_function (higher is more normal). "
                "Severity tiers are per-entity percentiles on validation scores. "
                "Entity is userId or host:userId depending on group_mode. "
                "Categorical features (eventId, mountNamespace, processName) are "
                "frequency-encoded with log(1+count) computed on benign training rows."
            ),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(model_dir: str) -> "ModelBundle":
        scaler = load(os.path.join(model_dir, "scaler.joblib"))
        model = load(os.path.join(model_dir, "iforest.joblib"))
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        freq_tables = {
            "eventId_freq_table":        dict(meta.get("eventId_freq_table", {})),
            "mountNamespace_freq_table": dict(meta.get("mountNamespace_freq_table", {})),
            "processName_freq_table":    dict(meta.get("processName_freq_table", {})),
        }
        return ModelBundle(
            scaler=scaler,
            model=model,
            features=list(meta["features"]),
            user_col=str(meta["user_col"]),
            host_col=meta.get("host_col"),
            group_mode=str(meta.get("group_mode", "user")),
            low_percentile=float(meta["low_percentile"]),
            high_percentile=float(meta["high_percentile"]),
            thresholds=dict(meta["thresholds"]),
            global_thresholds=dict(meta["global_thresholds"]),
            freq_tables=freq_tables,
        )


# -----------------------------
# Streaming: scaler fit + reservoir sample for training
# -----------------------------

def fit_scaler_on_benign(
    train_csv: str,
    chunksize: int,
    freq_tables: Dict[str, Dict[str, float]],
) -> StandardScaler:
    """Streaming partial_fit on the FEATURES vector (post feature-engineering)."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    benign_count = 0

    for chunk in _read_raw_chunks(train_csv, chunksize=chunksize):
        if "sus" not in chunk.columns or "evil" not in chunk.columns:
            raise ValueError(f"{train_csv}: missing label columns sus/evil")
        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue
        feats = build_features_for_chunk(benign, freq_tables)
        Xb = feats.to_numpy(dtype=np.float64, copy=False)
        scaler.partial_fit(Xb)
        benign_count += Xb.shape[0]

    if benign_count == 0:
        raise ValueError("No benign rows found in training set (sus==0 & evil==0).")
    return scaler


def sample_benign_for_training(
    train_csv: str,
    scaler: StandardScaler,
    chunksize: int,
    train_sample: int,
    random_state: int,
    freq_tables: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Reservoir-sample post-scaler benign feature vectors for the iForest fit."""
    rng = np.random.default_rng(random_state)
    reservoir: List[np.ndarray] = []
    seen = 0

    for chunk in _read_raw_chunks(train_csv, chunksize=chunksize):
        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue
        feats = build_features_for_chunk(benign, freq_tables)
        Xb = feats.to_numpy(dtype=np.float64, copy=False)
        Xb_scaled = scaler.transform(Xb)
        for i in range(Xb_scaled.shape[0]):
            seen += 1
            reservoir_add(reservoir, Xb_scaled[i].copy(), seen=seen, k=train_sample, rng=rng)

    if not reservoir:
        raise ValueError("Sampling produced 0 benign rows. Check data / labels.")
    return np.vstack(reservoir)


def train_iforest(
    X_benign_scaled: np.ndarray,
    n_estimators: int,
    contamination: float,
    max_samples: str,
    n_jobs: int,
    random_state: int,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_benign_scaled)
    return model


# -----------------------------
# Validation: per-entity two-tier thresholds
# -----------------------------

def compute_thresholds_from_validation(
    val_csv: str,
    scaler: StandardScaler,
    model: IsolationForest,
    chunksize: int,
    user_col: str,
    host_col: Optional[str],
    group_mode: str,
    low_percentile: float,
    high_percentile: float,
    max_scores_per_entity: int,
    random_state: int,
    freq_tables: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
    """Per-entity (user OR host:user) percentile thresholds on validation scores.

    Reads raw BETH columns, builds the FEATURES vector, scores it, then keeps
    a reservoir of scores per entity for percentile estimation.
    """
    rng = np.random.default_rng(random_state)

    entity_scores: Dict[str, List[float]] = {}
    entity_seen: Dict[str, int] = {}

    global_scores: List[float] = []
    global_seen = 0

    evil_all: List[np.ndarray] = []
    scores_all_for_auc: List[np.ndarray] = []
    evil_count = 0

    extras: List[str] = []
    if user_col not in RAW_INPUT_COLS:
        extras.append(user_col)
    if host_col and host_col not in RAW_INPUT_COLS:
        extras.append(host_col)

    for chunk in _read_raw_chunks(val_csv, chunksize=chunksize, extra_cols=extras):
        if user_col not in chunk.columns:
            raise ValueError(f"{val_csv}: missing user column '{user_col}'")
        keys = make_entity_keys(chunk, user_col=user_col, host_col=host_col, group_mode=group_mode)

        feats = build_features_for_chunk(chunk, freq_tables)
        X = feats.to_numpy(dtype=np.float64, copy=False)
        Xs = scaler.transform(X)
        scores = model.decision_function(Xs)

        if "evil" in chunk.columns:
            evil = chunk["evil"].to_numpy(dtype=np.int8, copy=False)
            evil_count += int(evil.sum())
            evil_all.append(evil)
            scores_all_for_auc.append(scores)

        for i in range(len(scores)):
            ent = str(keys[i])
            s = float(scores[i])

            if ent not in entity_scores:
                entity_scores[ent] = []
                entity_seen[ent] = 0

            entity_seen[ent] += 1
            if len(entity_scores[ent]) < max_scores_per_entity:
                entity_scores[ent].append(s)
            else:
                j = rng.integers(0, entity_seen[ent])
                if j < max_scores_per_entity:
                    entity_scores[ent][j] = s

            global_seen += 1
            if len(global_scores) < max_scores_per_entity:
                global_scores.append(s)
            else:
                jg = rng.integers(0, global_seen)
                if jg < max_scores_per_entity:
                    global_scores[jg] = s

    thresholds: Dict[str, Dict[str, float]] = {}
    for ent, lst in entity_scores.items():
        arr = np.array(lst, dtype=np.float64)
        thresholds[ent] = {
            "low": safe_percentile(arr, low_percentile),
            "high": safe_percentile(arr, high_percentile),
            "n_scores": int(arr.size),
        }

    garr = np.array(global_scores, dtype=np.float64)
    global_thresholds = {
        "low": safe_percentile(garr, low_percentile),
        "high": safe_percentile(garr, high_percentile),
        "n_scores": int(garr.size),
    }

    report: Dict[str, Any] = {
        "validation_csv": val_csv,
        "entities": len(thresholds),
        "group_mode": group_mode,
        "user_col": user_col,
        "host_col": host_col,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "global_thresholds": global_thresholds,
        "evil_count_in_validation": evil_count,
    }

    if evil_count > 0 and scores_all_for_auc:
        evil_vec = np.concatenate(evil_all)
        scores_vec = np.concatenate(scores_all_for_auc)
        report["roc_auc_evil"] = float(roc_auc_score(evil_vec, -scores_vec))
        report["avg_precision_evil"] = float(average_precision_score(evil_vec, -scores_vec))
    else:
        report["roc_auc_evil"] = None
        report["avg_precision_evil"] = None

    return thresholds, global_thresholds, report


# -----------------------------
# Severity + Evaluation
# -----------------------------

def severity_from_score(score: float, low_thr: float, high_thr: float) -> int:
    if np.isnan(low_thr) or np.isnan(high_thr):
        return 0
    if score < high_thr:
        return 2
    if score < low_thr:
        return 1
    return 0


@dataclass
class RunningConfusion:
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.tp += int(((y_true == 1) & (y_pred == 1)).sum())
        self.tn += int(((y_true == 0) & (y_pred == 0)).sum())
        self.fp += int(((y_true == 0) & (y_pred == 1)).sum())
        self.fn += int(((y_true == 1) & (y_pred == 0)).sum())

    def metrics(self) -> Dict[str, float]:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_on_test_stream(
    test_csv: str,
    bundle: ModelBundle,
    chunksize: int,
) -> Dict[str, Any]:
    overall = RunningConfusion()
    high_only = RunningConfusion()

    evil_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []

    rows = 0
    evil_count = 0

    extras: List[str] = []
    if bundle.user_col not in RAW_INPUT_COLS:
        extras.append(bundle.user_col)
    if bundle.host_col and bundle.host_col not in RAW_INPUT_COLS:
        extras.append(bundle.host_col)

    for chunk in _read_raw_chunks(test_csv, chunksize=chunksize, extra_cols=extras):
        if "evil" not in chunk.columns:
            raise ValueError(f"{test_csv}: missing label column 'evil'")
        keys = make_entity_keys(
            chunk,
            user_col=bundle.user_col,
            host_col=bundle.host_col,
            group_mode=bundle.group_mode,
        )

        feats = build_features_for_chunk(chunk, bundle.freq_tables)
        X = feats.to_numpy(dtype=np.float64, copy=False)
        Xs = bundle.scaler.transform(X)
        scores = bundle.model.decision_function(Xs)

        evil = chunk["evil"].to_numpy(dtype=np.int8, copy=False)
        evil_count += int(evil.sum())

        evil_all.append(evil)
        scores_all.append(scores)

        sev = np.zeros(len(scores), dtype=np.int8)
        for i in range(len(scores)):
            ent = str(keys[i])
            th = bundle.thresholds.get(ent)
            if th is None:
                low_thr = bundle.global_thresholds["low"]
                high_thr = bundle.global_thresholds["high"]
            else:
                low_thr = th["low"]
                high_thr = th["high"]
            sev[i] = severity_from_score(float(scores[i]), low_thr, high_thr)

        pred_any = (sev >= 1).astype(np.int8)
        pred_high = (sev >= 2).astype(np.int8)

        overall.update(evil, pred_any)
        high_only.update(evil, pred_high)

        rows += len(chunk)

    evil_vec = np.concatenate(evil_all) if evil_all else np.array([], dtype=np.int8)
    scores_vec = np.concatenate(scores_all) if scores_all else np.array([], dtype=np.float64)

    report: Dict[str, Any] = {
        "test_csv": test_csv,
        "rows": rows,
        "evil_count": evil_count,
        "policy": {
            "per_entity": True,
            "entity": bundle.group_mode,
            "user_col": bundle.user_col,
            "host_col": bundle.host_col,
            "low_percentile": bundle.low_percentile,
            "high_percentile": bundle.high_percentile,
        },
        "overall_low_or_high": {
            "confusion": {"tn": overall.tn, "fp": overall.fp, "fn": overall.fn, "tp": overall.tp},
            **overall.metrics(),
        },
        "high_only": {
            "confusion": {"tn": high_only.tn, "fp": high_only.fp, "fn": high_only.fn, "tp": high_only.tp},
            **high_only.metrics(),
        },
        "ranking": {
            "roc_auc_evil": float(roc_auc_score(evil_vec, -scores_vec)) if evil_vec.sum() > 0 else None,
            "avg_precision_evil": float(average_precision_score(evil_vec, -scores_vec)) if evil_vec.sum() > 0 else None,
        },
    }
    return report


# -----------------------------
# Commands
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
    print(f"[1/4] Building frequency tables (eventId, mountNamespace, processName) from benign training rows...")
    freq_tables = build_frequency_tables(args.train_csv, chunksize=args.chunksize)
    print(f"      eventId entries={len(freq_tables['eventId_freq_table'])}, "
          f"mountNamespace entries={len(freq_tables['mountNamespace_freq_table'])}, "
          f"processName entries={len(freq_tables['processName_freq_table'])}")

    print(f"[2/4] Streaming partial_fit StandardScaler on the {len(FEATURES)}-feature vector...")
    scaler = fit_scaler_on_benign(
        train_csv=args.train_csv,
        chunksize=args.chunksize,
        freq_tables=freq_tables,
    )

    print(f"[3/4] Reservoir-sampling up to {args.train_sample:,} benign rows for IsolationForest fit...")
    X_sample = sample_benign_for_training(
        train_csv=args.train_csv,
        scaler=scaler,
        chunksize=args.chunksize,
        train_sample=args.train_sample,
        random_state=args.random_state,
        freq_tables=freq_tables,
    )

    print(f"[3.5/4] Fitting IsolationForest (n_estimators={args.n_estimators}, "
          f"contamination={args.contamination}) on {X_sample.shape[0]:,} samples...")
    model = train_iforest(
        X_benign_scaled=X_sample,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    print(f"[4/4] Computing per-entity thresholds on validation (group_mode={args.group_mode})...")
    thresholds, global_thresholds, val_report = compute_thresholds_from_validation(
        val_csv=args.val_csv,
        scaler=scaler,
        model=model,
        chunksize=args.chunksize,
        user_col=args.user_col,
        host_col=args.host_col,
        group_mode=args.group_mode,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        max_scores_per_entity=args.max_scores_per_entity,
        random_state=args.random_state,
        freq_tables=freq_tables,
    )

    bundle = ModelBundle(
        scaler=scaler,
        model=model,
        features=FEATURES,
        user_col=args.user_col,
        host_col=args.host_col,
        group_mode=args.group_mode,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        thresholds=thresholds,
        global_thresholds=global_thresholds,
        freq_tables=freq_tables,
    )
    bundle.save(args.out_dir)

    out = {
        "saved_to": args.out_dir,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "train": {
            "train_csv": args.train_csv,
            "benign_sample_used": int(X_sample.shape[0]),
            "n_estimators": args.n_estimators,
            "contamination": args.contamination,
            "max_samples": args.max_samples,
            "n_jobs": args.n_jobs,
        },
        "freq_tables_size": {
            "eventId": len(freq_tables["eventId_freq_table"]),
            "mountNamespace": len(freq_tables["mountNamespace_freq_table"]),
            "processName": len(freq_tables["processName_freq_table"]),
        },
        "validation_thresholding": val_report,
        "entities_thresholded": len(thresholds),
        "global_thresholds": global_thresholds,
        "note": "Two-tier severity is per-entity percentiles on validation normality scores.",
    }
    print(json.dumps(out, indent=2))


def cmd_test(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.model_dir)
    report = evaluate_on_test_stream(
        test_csv=args.test_csv,
        bundle=bundle,
        chunksize=args.chunksize,
    )
    print(json.dumps(report, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.model_dir)

    extras: List[str] = []
    if bundle.user_col not in RAW_INPUT_COLS:
        extras.append(bundle.user_col)
    if bundle.host_col and bundle.host_col not in RAW_INPUT_COLS:
        extras.append(bundle.host_col)

    out_rows: List[pd.DataFrame] = []
    rows_written = 0

    for chunk in _read_raw_chunks(args.csv, chunksize=args.chunksize, extra_cols=extras):
        if bundle.user_col not in chunk.columns:
            raise ValueError(f"{args.csv}: missing user column '{bundle.user_col}'")
        keys = make_entity_keys(
            chunk,
            user_col=bundle.user_col,
            host_col=bundle.host_col,
            group_mode=bundle.group_mode,
        )

        feats = build_features_for_chunk(chunk, bundle.freq_tables)
        X = feats.to_numpy(dtype=np.float64, copy=False)
        Xs = bundle.scaler.transform(X)
        scores = bundle.model.decision_function(Xs)

        sev = np.zeros(len(scores), dtype=np.int8)
        low_thr_used = np.zeros(len(scores), dtype=np.float64)
        high_thr_used = np.zeros(len(scores), dtype=np.float64)

        for i in range(len(scores)):
            ent = str(keys[i])
            th = bundle.thresholds.get(ent)
            if th is None:
                low_thr = bundle.global_thresholds["low"]
                high_thr = bundle.global_thresholds["high"]
            else:
                low_thr = th["low"]
                high_thr = th["high"]
            low_thr_used[i] = low_thr
            high_thr_used[i] = high_thr
            sev[i] = severity_from_score(float(scores[i]), low_thr, high_thr)

        out_df = pd.DataFrame({
            "entity": [str(k) for k in keys],          # userId or host:userId
            "normality_score": scores,                 # higher = more normal
            "low_threshold": low_thr_used,
            "high_threshold": high_thr_used,
            "severity": sev,                           # 0/1/2
            "malicious_pred": (sev >= 1).astype(np.int8),
        })

        if args.attach_features:
            out_df = pd.concat([feats.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)

        out_rows.append(out_df)
        rows_written += len(out_df)

    final = pd.concat(out_rows, axis=0) if out_rows else pd.DataFrame()
    final.to_csv(args.out_csv, index=False)

    print(json.dumps({
        "wrote": args.out_csv,
        "rows": int(rows_written),
        "policy": {
            "per_entity": True,
            "two_tier": True,
            "group_mode": bundle.group_mode,
            "user_col": bundle.user_col,
            "host_col": bundle.host_col,
            "low_percentile": bundle.low_percentile,
            "high_percentile": bundle.high_percentile,
        }
    }, indent=2))


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BETH IsolationForest with per-user percentile + two-tier severity.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train benign-only iForest; compute per-user/host:user thresholds on validation.")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--val-csv", required=True)
    p_train.add_argument("--out-dir", required=True)

    p_train.add_argument("--chunksize", type=int, default=500_000)

    # grouping
    p_train.add_argument("--user-col", default="userId",
                         help="User identifier column name (default: userId).")
    p_train.add_argument("--host-col", default=None,
                         help="Optional host column. Only needed if group-mode=host_user.")
    p_train.add_argument("--group-mode", choices=["user", "host_user"], default="user",
                         help="How to build entity key: userId or host:userId.")

    # training + sampling
    # Defaults bumped for the ~22-feature model running on a 4090 Ti box with
    # 40GB RAM. Falling back to the old (600, 2M) values is one --flag away.
    p_train.add_argument("--train-sample", type=int, default=10_000_000,
                         help="Reservoir sample size of benign rows used to fit iForest. (default: 10M)")
    p_train.add_argument("--n-estimators", type=int, default=2000,
                         help="Number of trees in the IsolationForest. (default: 2000)")
    p_train.add_argument("--contamination", type=float, default=0.03,
                         help="iForest contamination (model internal). Does NOT set alert thresholds.")
    p_train.add_argument("--max-samples", default="auto")
    p_train.add_argument("--n-jobs", type=int, default=32)
    p_train.add_argument("--random-state", type=int, default=42)

    # per-entity thresholds
    p_train.add_argument("--low-percentile", type=float, default=2.0,
                         help="Bottom X%% normality scores => severity >= 1 (low).")
    p_train.add_argument("--high-percentile", type=float, default=0.2,
                         help="Bottom X%% normality scores => severity == 2 (high).")
    p_train.add_argument("--max-scores-per-entity", type=int, default=500_000,
                         help="Reservoir cap for storing validation scores per entity for percentile estimation.")

    p_test = sub.add_parser("test", help="Evaluate saved model on labeled test CSV (evil required).")
    p_test.add_argument("--model-dir", required=True)
    p_test.add_argument("--test-csv", required=True)
    p_test.add_argument("--chunksize", type=int, default=500_000)

    p_pred = sub.add_parser("predict", help="Score any CSV with the 7 feature columns; outputs severity tiers per user.")
    p_pred.add_argument("--model-dir", required=True)
    p_pred.add_argument("--csv", required=True)
    p_pred.add_argument("--out-csv", required=True)
    p_pred.add_argument("--chunksize", type=int, default=500_000)
    p_pred.add_argument("--attach-features", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        if args.high_percentile >= args.low_percentile:
            raise ValueError("--high-percentile must be smaller than --low-percentile (high tier is more strict).")
        if args.group_mode == "host_user" and not args.host_col:
            raise ValueError("--group-mode host_user requires --host-col <colname>.")
        cmd_train(args)
    elif args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
