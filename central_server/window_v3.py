"""
Sliding window anomaly tracker per (hostname, userId) — v3.

Extends window.py with four behavioral heuristics that catch recon tools
(e.g. LinPEAS) which the Isolation Forest scores as severity-1 only:

  H1 — Process diversity burst  : unique process names / window
  H2 — Exec frequency burst     : sched_process_exec events / window
  H3 — Severity-1 accumulation  : low-anomaly event volume / window
  H4 — LOLBAS / recon binary    : distinct known-recon process names / window

Pivot triggers on ANY of:
  - original: severity-2 count >= pivot_threshold
  - H1: unique_process_names >= process_diversity_threshold
  - H2: exec_events >= exec_frequency_threshold
  - H3: severity_1_events >= low_severity_threshold
  - H4: unique_recon_binaries >= recon_binary_threshold
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


RECON_BINARIES: Set[str] = {
    "id", "whoami", "hostname", "uname", "ifconfig", "netstat", "ss",
    "lsof", "arp", "route", "w", "who", "last", "lastlog", "crontab",
    "env", "printenv", "ps", "pstree", "find", "locate", "iptables",
    "ip6tables", "nmap", "curl", "wget", "nc", "ncat", "socat",
}


@dataclass
class UserWindow:
    hostname: str
    user_id: int
    # Each entry: (timestamp, severity)
    events: deque = field(default_factory=deque)
    pivoted: bool = False

    # H3: severity-1 events — (timestamp,)
    low_severity_events: deque = field(default_factory=deque)
    # H2: exec events — (timestamp,)
    exec_events: deque = field(default_factory=deque)
    # H1: exec process names — (timestamp, name)
    exec_process_names: deque = field(default_factory=deque)
    # H4: recon binary hits — (timestamp, name)
    recon_hits: deque = field(default_factory=deque)


class SlidingWindowTracker:
    """
    Maintains per-(hostname, userId) event windows.
    Call feed() for every scored event; check pivot_candidates() to get
    (hostname, userId) pairs that have crossed any threshold.
    """

    def __init__(
        self,
        pivot_threshold: int = 5,
        window_seconds: int = 60,
        min_severity: int = 2,
        whitelist_uids: Optional[Set[int]] = None,
        process_diversity_threshold: int = 20,
        exec_frequency_threshold: int = 100,
        low_severity_threshold: int = 30,
        recon_binary_threshold: int = 5,
    ):
        self._threshold = pivot_threshold
        self._window = window_seconds
        self._min_severity = min_severity
        self._whitelist_uids: Set[int] = whitelist_uids or set()

        self._process_diversity_threshold = process_diversity_threshold
        self._exec_frequency_threshold = exec_frequency_threshold
        self._low_severity_threshold = low_severity_threshold
        self._recon_binary_threshold = recon_binary_threshold

        self._windows: Dict[Tuple[str, int], UserWindow] = {}
        self._pending_pivots: List[Tuple[str, int]] = []

    def feed(
        self,
        hostname: str,
        user_id: int,
        severity: int,
        process_name: str = "",
        event_name: str = "",
    ):
        if user_id in self._whitelist_uids:
            return

        key = (hostname, user_id)
        if key not in self._windows:
            self._windows[key] = UserWindow(hostname=hostname, user_id=user_id)

        win = self._windows[key]
        if win.pivoted:
            return

        now = time.time()

        # Original severity-2 path
        if severity >= self._min_severity:
            win.events.append((now, severity))
            self._evict_old(win, now)

        # H3: track severity-1
        if severity == 1:
            win.low_severity_events.append((now,))

        # H2 + H1: track exec events
        if event_name == "sched_process_exec":
            win.exec_events.append((now,))
            if process_name:
                win.exec_process_names.append((now, process_name))

        # H4: track recon binaries
        proc_lower = process_name.lower() if process_name else ""
        if proc_lower in RECON_BINARIES:
            win.recon_hits.append((now, proc_lower))

        self._evict_heuristic_data(win, now)

        if len(win.events) >= self._threshold or self._check_heuristics(win):
            win.pivoted = True
            self._pending_pivots.append(key)

    def _evict_old(self, win: UserWindow, now: float):
        cutoff = now - self._window
        while win.events and win.events[0][0] < cutoff:
            win.events.popleft()

    def _evict_heuristic_data(self, win: UserWindow, now: float):
        cutoff = now - self._window
        for dq in (
            win.low_severity_events,
            win.exec_events,
            win.exec_process_names,
            win.recon_hits,
        ):
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def _check_heuristics(self, win: UserWindow) -> bool:
        # H3: severity-1 accumulation
        if len(win.low_severity_events) >= self._low_severity_threshold:
            return True

        # H2: exec frequency burst
        if len(win.exec_events) >= self._exec_frequency_threshold:
            return True

        # H1: process diversity burst
        unique_names = {name for _, name in win.exec_process_names}
        if len(unique_names) >= self._process_diversity_threshold:
            return True

        # H4: recon binary diversity
        unique_recon = {name for _, name in win.recon_hits}
        if len(unique_recon) >= self._recon_binary_threshold:
            return True

        return False

    def drain_pivots(self) -> List[Tuple[str, int]]:
        pending = self._pending_pivots[:]
        self._pending_pivots.clear()
        return pending

    def stats(self) -> Dict:
        return {
            "tracked_entities": len(self._windows),
            "pivoted": sum(1 for w in self._windows.values() if w.pivoted),
            "pending_pivots": len(self._pending_pivots),
            "heuristic_thresholds": {
                "process_diversity": self._process_diversity_threshold,
                "exec_frequency": self._exec_frequency_threshold,
                "low_severity_accumulation": self._low_severity_threshold,
                "recon_binary_diversity": self._recon_binary_threshold,
            },
        }
