"""
Sliding window anomaly tracker per (hostname, userId).

Triggers a pivot when, in a rolling ``window_seconds`` window, EITHER
detector fires (OR) — the two catch different attack shapes:

  * VOLUME: the count of severity-2 events reaches ``pivot_threshold``,
    regardless of concentration. Catches a loud scanner (e.g. linpeas)
    that floods the window with benign syscalls so its sev-2 *fraction*
    is tiny (~0%) yet its absolute sev-2 count is high.
  * CONCENTRATION: a smaller count (``conc_min_count``) that is also at
    least ``min_sev2_rate`` of all the user's events. Catches a stealthy
    attacker whose few actions are mostly anomalous. The ``conc_min_count``
    floor stops a single high-severity event (1/1 = 100%) from pivoting
    a quiet user.

Why OR (not AND): requiring BOTH (the old design) is the intersection of the
two shapes — too narrow, so a loud-but-diluted scanner like linpeas, whose
concentration is ~0%, slips through despite a high sev-2 count. OR is the
union: voluminous OR concentrated. The price is that the VOLUME branch is
count-only, so a genuinely very busy benign user who accumulates
``pivot_threshold`` sev-2 events can pivot — bounded by keeping the volume
floor high and the severity-2 score cutoff strict.

Tunables (all env-var overridable in main.py — defaults shown):
  PIVOT_THRESHOLD    = 10    # VOLUME floor on sev-2 count
  CONC_MIN_COUNT     = 5     # CONCENTRATION-branch floor on sev-2 count
  MIN_SEV2_RATE      = 0.30  # CONCENTRATION-branch rate: 30% of events must be sev-2
  PIVOT_WINDOW_SEC   = 60
"""

from __future__ import annotations

import time
from collections import deque
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
        pivot_threshold: int = 10,
        window_seconds: int = 60,
        min_severity: int = 2,
        min_sev2_rate: float = 0.30,
        conc_min_count: int = 5,
        whitelist_uids: Optional[Set[int]] = None,
    ):
        self._threshold = pivot_threshold          # VOLUME branch floor on sev-2 count
        self._window = window_seconds
        self._min_severity = min_severity
        self._min_sev2_rate = min_sev2_rate        # CONCENTRATION branch rate threshold
        self._conc_min_count = conc_min_count      # CONCENTRATION branch floor on sev-2 count
        self._whitelist_uids: Set[int] = whitelist_uids or set()

        self._windows: Dict[Tuple[str, int], UserWindow] = {}
        self._pending_pivots: List[Tuple[str, int, str]] = []

        # Periodic sweep so windows for users that went silent don't linger
        # forever (otherwise _windows grows once per distinct (host, uid) ever
        # seen). Sweeps about once per window length.
        self._last_prune: float = time.time()
        self._prune_interval: float = float(window_seconds)

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

        # Drop windows for users that have gone silent (cheap, amortized).
        if now - self._last_prune >= self._prune_interval:
            self._prune(now)
            self._last_prune = now

        # Trigger: pivot if EITHER detector fires (OR).
        sev2_count = sum(1 for _, s in win.events if s >= self._min_severity)
        total = len(win.events)
        if total == 0:
            return  # defensive; impossible once an event was appended

        # VOLUME detector: a sustained burst of sev-2, regardless of how diluted
        # — catches loud recon (e.g. linpeas) whose sev-2 fraction is ~0%.
        if sev2_count >= self._threshold:
            self._fire_pivot(win, hostname, user_id,
                             f"volume (sev2={sev2_count}>={self._threshold})")
            return

        # CONCENTRATION detector: fewer sev-2 events, but a large fraction of all
        # activity — catches a stealthy attacker. The count floor prevents a lone
        # high-severity event (1/1 = 100%) from pivoting a quiet user.
        rate = sev2_count / total
        if sev2_count >= self._conc_min_count and rate >= self._min_sev2_rate:
            self._fire_pivot(win, hostname, user_id,
                             f"concentration (sev2={sev2_count}/{total}={rate:.0%})")

    def _fire_pivot(self, win: "UserWindow", hostname: str, user_id: int, trigger: str):
        win.pivoted = True
        self._pending_pivots.append((hostname, user_id, trigger))
        # Pivoted users are ignored from here on (feed() returns early), so the
        # accumulated events serve no purpose — free them but keep the small
        # window entry so user_states() can still report the pivoted state.
        win.events.clear()

    def _prune(self, now: float):
        """Evict stale events and drop empty, non-pivoted windows."""
        cutoff = now - self._window
        stale: List[Tuple[str, int]] = []
        for key, win in self._windows.items():
            if win.pivoted:
                continue  # tiny (empty deque); retained for visibility
            while win.events and win.events[0][0] < cutoff:
                win.events.popleft()
            if not win.events:
                stale.append(key)
        for key in stale:
            del self._windows[key]

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
                    "severity2_threshold": self._threshold,    # VOLUME branch floor
                    "conc_min_count": self._conc_min_count,     # CONCENTRATION branch floor
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
            "conc_min_count": self._conc_min_count,
            "window_seconds": self._window,
            "min_sev2_rate": self._min_sev2_rate,
        }
