"""
Sliding window anomaly tracker per (hostname, userId).

Triggers a pivot when the number of severity-2 (high-anomaly) events
from the Isolation Forest exceeds `pivot_threshold` inside a rolling
`window_seconds` window.
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
    Call feed() for every scored event; call drain_pivots() to collect
    (hostname, userId, trigger) tuples that have crossed the threshold.
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

        self._windows: Dict[Tuple[str, int], UserWindow] = {}
        self._pending_pivots: List[Tuple[str, int, str]] = []

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

        if severity < self._min_severity:
            return

        now = time.time()
        win.events.append((now, severity))

        cutoff = now - self._window
        while win.events and win.events[0][0] < cutoff:
            win.events.popleft()

        if len(win.events) >= self._threshold:
            win.pivoted = True
            self._pending_pivots.append((hostname, user_id, "threshold"))

    def drain_pivots(self) -> List[Tuple[str, int, str]]:
        pending = self._pending_pivots[:]
        self._pending_pivots.clear()
        return pending

    def re_queue_pivots(self, pivots: List[Tuple[str, int, str]]):
        self._pending_pivots.extend(pivots)

    def user_states(self) -> List[Dict]:
        result = []
        for (hostname, user_id), win in list(self._windows.items()):
            result.append({
                "hostname": hostname,
                "user_id": user_id,
                "pivoted": win.pivoted,
                "window": {
                    "severity2_count": len(win.events),
                    "severity2_threshold": self._threshold,
                },
            })
        return result

    def stats(self) -> Dict:
        return {
            "tracked_entities": len(self._windows),
            "pivoted": sum(1 for w in self._windows.values() if w.pivoted),
            "pending_pivots": len(self._pending_pivots),
            "pivot_threshold": self._threshold,
            "window_seconds": self._window,
        }
