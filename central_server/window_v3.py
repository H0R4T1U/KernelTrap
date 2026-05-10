"""
Sliding window anomaly tracker per (hostname, userId).

Triggers a pivot when, in a rolling ``window_seconds`` window, BOTH:

  * the absolute count of severity-2 events crosses ``pivot_threshold``
    (a floor — at least N high-severity events seen recently)
  * those severity-2 events make up at least ``min_sev2_rate`` of all
    events the user generated in the window (a rate gate — proves the
    high-severity activity stands out from the user's baseline)

The rate gate is the important addition vs. the original (count-only) tracker:
on a busy system, 5 sev-2 events out of 5000 total is ~0.1% and is just
distribution shift; 5 sev-2 out of 10 is 50% and is genuinely suspicious.
The count floor protects against tiny-window false positives where 2/3
events are sev-2 but the user has barely done anything.

Tunables (all env-var overridable in main.py — defaults shown):
  PIVOT_THRESHOLD    = 5     # absolute floor on sev-2 count
  PIVOT_WINDOW_SEC   = 60
  MIN_SEV2_RATE      = 0.30  # 30% of events in window must be sev-2
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
    # Each entry: (timestamp, severity). Now stores ALL severities so we can
    # compute the sev-2 rate, not just the count.
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
        min_sev2_rate: float = 0.30,
        whitelist_uids: Optional[Set[int]] = None,
    ):
        self._threshold = pivot_threshold
        self._window = window_seconds
        self._min_severity = min_severity
        self._min_sev2_rate = min_sev2_rate
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

        # Store ALL events (severity 0/1/2) so we can compute the sev-2 rate.
        # The previous version stored only sev>=2 — that's fine for counting
        # but leaves us blind to the denominator (total user activity).
        now = time.time()
        win.events.append((now, severity))

        # Evict events older than the window.
        cutoff = now - self._window
        while win.events and win.events[0][0] < cutoff:
            win.events.popleft()

        # Trigger: both gates must pass.
        sev2_count = sum(1 for _, s in win.events if s >= self._min_severity)
        if sev2_count < self._threshold:
            return  # absolute floor not reached yet

        total = len(win.events)
        if total == 0:
            return  # impossible if sev2_count >= 1, but defensive
        rate = sev2_count / total
        if rate < self._min_sev2_rate:
            return  # high-severity activity not concentrated enough

        win.pivoted = True
        trigger = f"rate+threshold (sev2={sev2_count}/{total}={rate:.0%})"
        self._pending_pivots.append((hostname, user_id, trigger))

    def drain_pivots(self) -> List[Tuple[str, int, str]]:
        pending = self._pending_pivots[:]
        self._pending_pivots.clear()
        return pending

    def re_queue_pivots(self, pivots: List[Tuple[str, int, str]]):
        self._pending_pivots.extend(pivots)

    def user_states(self) -> List[Dict]:
        result = []
        for (hostname, user_id), win in list(self._windows.items()):
            sev2 = sum(1 for _, s in win.events if s >= self._min_severity)
            total = len(win.events)
            rate = (sev2 / total) if total else 0.0
            result.append({
                "hostname": hostname,
                "user_id": user_id,
                "pivoted": win.pivoted,
                "window": {
                    "severity2_count": sev2,
                    "total_count": total,
                    "severity2_rate": rate,
                    "severity2_threshold": self._threshold,
                    "min_severity2_rate": self._min_sev2_rate,
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
            "min_sev2_rate": self._min_sev2_rate,
        }
