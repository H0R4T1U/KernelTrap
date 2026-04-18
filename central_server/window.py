"""
Sliding window anomaly tracker per (hostname, userId).

Replaces the window logic that used to live in honeypot_integration.py.
Running on the central server means state accumulates across ALL batches
from ALL agents — the decision to pivot is never batch-boundary-dependent.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class UserWindow:
    hostname: str
    user_id: int
    # Each entry: (timestamp, severity)
    events: deque = field(default_factory=deque)
    pivoted: bool = False


class SlidingWindowTracker:
    """
    Maintains per-(hostname, userId) event windows.
    Call feed() for every scored event; check pivot_candidates() to get
    (hostname, userId) pairs that have crossed the threshold.
    """

    def __init__(
        self,
        pivot_threshold: int = 5,
        window_seconds: int = 60,
        min_severity: int = 2,
        whitelist_uids: Optional[Set[int]] = None,
    ):
        self._threshold = pivot_threshold
        self._window = window_seconds
        self._min_severity = min_severity
        self._whitelist_uids: Set[int] = whitelist_uids or set()

        # (hostname, user_id) -> UserWindow
        self._windows: Dict[Tuple[str, int], UserWindow] = {}
        # Pairs that crossed threshold but pivot not yet sent
        self._pending_pivots: List[Tuple[str, int]] = []

    def feed(self, hostname: str, user_id: int, severity: int):
        """Record a scored event. Only high-severity events advance the window."""
        if severity < self._min_severity:
            return
        if user_id in self._whitelist_uids:
            return

        key = (hostname, user_id)
        if key not in self._windows:
            self._windows[key] = UserWindow(hostname=hostname, user_id=user_id)

        win = self._windows[key]
        if win.pivoted:
            return

        now = time.time()
        win.events.append((now, severity))
        self._evict_old(win, now)

        if len(win.events) >= self._threshold:
            win.pivoted = True
            self._pending_pivots.append(key)

    def _evict_old(self, win: UserWindow, now: float):
        cutoff = now - self._window
        while win.events and win.events[0][0] < cutoff:
            win.events.popleft()

    def drain_pivots(self) -> List[Tuple[str, int]]:
        """Return and clear the list of (hostname, user_id) pairs ready to pivot."""
        pending = self._pending_pivots[:]
        self._pending_pivots.clear()
        return pending

    def stats(self) -> Dict:
        return {
            "tracked_entities": len(self._windows),
            "pivoted": sum(1 for w in self._windows.values() if w.pivoted),
            "pending_pivots": len(self._pending_pivots),
        }
