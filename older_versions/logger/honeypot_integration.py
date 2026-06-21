     #!/usr/bin/env python3
"""
KernelTrap Honeypot Integration

Integrates the syscall logger with the honeypot pivot mechanism.
When sustained anomalous behavior is detected from a user,
automatically pivots them into the honeypot container.

Usage:
    # Run with Tracee for real-time detection and automatic pivot:
    sudo tracee --output json | python honeypot_integration.py --model ../isolation_forest/beth_iforest_model

    # Test mode (no actual pivots):
    sudo tracee --output json | python honeypot_integration.py --model ./model --dry-run
"""

import argparse
import logging
import os
import pwd
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Import from our logger module
from syscall_logger import (
    SyscallEvent,
    SyscallLogger,
    IsolationForestScorer,
    TraceeParser,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('/var/log/kerneltrap/honeypot_integration.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyWindow:
    """Tracks anomalies for a user within a time window."""
    user_id: int
    username: str
    events: List[tuple]  # (timestamp, severity, event_name)
    last_alert: float = 0.0
    pivoted: bool = False


class HoneypotIntegration:
    """
    Monitors syscall anomalies and pivots suspicious users to honeypot.

    Uses a sliding window approach to detect sustained anomalous behavior
    before triggering a pivot, reducing false positives.
    """

    def __init__(
        self,
        model_dir: str,
        pivot_threshold: int = 5,
        pivot_window_seconds: int = 60,
        pivot_script: str = "/usr/local/sbin/hp_pivot_user",
        whitelist: Optional[Set[str]] = None,
        dry_run: bool = False,
        min_severity: int = 2,
    ):
        """
        Initialize honeypot integration.

        Args:
            model_dir: Path to trained Isolation Forest model
            pivot_threshold: Number of high-severity events to trigger pivot
            pivot_window_seconds: Time window for counting events
            pivot_script: Path to hp_pivot_user script
            whitelist: Set of usernames to never pivot
            dry_run: If True, don't actually pivot users
            min_severity: Minimum severity level to count (1=low, 2=high)
        """
        self.model_dir = model_dir
        self.pivot_threshold = pivot_threshold
        self.pivot_window_seconds = pivot_window_seconds
        self.pivot_script = pivot_script
        self.whitelist = whitelist or {"root", "admin"}
        self.dry_run = dry_run
        self.min_severity = min_severity

        # Track anomalies per user
        self.user_windows: Dict[int, AnomalyWindow] = {}

        # Track already pivoted users (don't pivot twice)
        self.pivoted_users: Set[int] = set()

        # UID to username cache
        self._uid_cache: Dict[int, str] = {}

        # Statistics
        self.stats = {
            "events_processed": 0,
            "anomalies_detected": 0,
            "pivots_triggered": 0,
            "pivots_blocked_whitelist": 0,
        }

        # Load the scorer
        self.scorer = IsolationForestScorer(model_dir)
        logger.info(f"Loaded Isolation Forest model from {model_dir}")

    def _get_username(self, uid: int) -> str:
        """Get username from UID (cached)."""
        if uid not in self._uid_cache:
            try:
                self._uid_cache[uid] = pwd.getpwuid(uid).pw_name
            except KeyError:
                self._uid_cache[uid] = f"uid_{uid}"
        return self._uid_cache[uid]

    def _clean_old_events(self, window: AnomalyWindow, current_time: float):
        """Remove events outside the sliding window."""
        cutoff = current_time - self.pivot_window_seconds
        window.events = [
            e for e in window.events
            if e[0] >= cutoff
        ]

    def _should_pivot(self, window: AnomalyWindow) -> bool:
        """Check if user should be pivoted based on anomaly count."""
        # Count events at or above minimum severity
        qualifying_events = sum(
            1 for _, severity, _ in window.events
            if severity >= self.min_severity
        )
        return qualifying_events >= self.pivot_threshold

    def _trigger_pivot(self, window: AnomalyWindow) -> bool:
        """Trigger honeypot pivot for user."""
        username = window.username

        # Check whitelist
        if username in self.whitelist:
            logger.warning(
                f"User '{username}' is whitelisted, skipping pivot"
            )
            self.stats["pivots_blocked_whitelist"] += 1
            return False

        # Check if already pivoted
        if window.user_id in self.pivoted_users:
            logger.debug(f"User '{username}' already pivoted")
            return False

        # Log the decision
        event_summary = ", ".join(
            f"{name}(sev={sev})"
            for _, sev, name in window.events[-5:]  # Last 5 events
        )
        logger.warning(
            f"PIVOT TRIGGERED for user '{username}' (UID: {window.user_id})\n"
            f"  Threshold: {self.pivot_threshold} events in {self.pivot_window_seconds}s\n"
            f"  Recent events: {event_summary}"
        )

        if self.dry_run:
            logger.info("[DRY-RUN] Would execute pivot, but dry-run mode is enabled")
            return True

        # Execute pivot
        try:
            if not os.path.exists(self.pivot_script):
                logger.error(f"Pivot script not found: {self.pivot_script}")
                return False

            result = subprocess.run(
                ["sudo", self.pivot_script, username],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info(f"Successfully pivoted user '{username}' to honeypot")
                self.pivoted_users.add(window.user_id)
                window.pivoted = True
                self.stats["pivots_triggered"] += 1
                return True
            else:
                logger.error(
                    f"Pivot failed for '{username}': {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Pivot timed out for user '{username}'")
            return False
        except Exception as e:
            logger.error(f"Pivot error for '{username}': {e}")
            return False

    def process_event(self, event: SyscallEvent) -> Optional[Dict]:
        """
        Process a single syscall event.

        Returns anomaly info if detected, None otherwise.
        """
        self.stats["events_processed"] += 1

        # Score the event
        score_result = self.scorer.score(event)

        # If not anomalous, nothing to do
        if not score_result["is_anomaly"]:
            return None

        self.stats["anomalies_detected"] += 1
        severity = score_result["severity"]
        current_time = time.time()

        # Get or create window for this user
        uid = event.userId
        if uid not in self.user_windows:
            username = self._get_username(uid)
            self.user_windows[uid] = AnomalyWindow(
                user_id=uid,
                username=username,
                events=[],
            )

        window = self.user_windows[uid]

        # Clean old events
        self._clean_old_events(window, current_time)

        # Add new event
        window.events.append((current_time, severity, event.eventName))

        # Log the anomaly (rate-limited)
        if current_time - window.last_alert >= 1.0:  # Max 1 alert/sec per user
            severity_label = "LOW" if severity == 1 else "HIGH"
            logger.warning(
                f"[{severity_label}] Anomaly detected: "
                f"user={window.username}, event={event.eventName}, "
                f"process={event.processName}, score={score_result['anomaly_score']:.4f}, "
                f"window_events={len(window.events)}"
            )
            window.last_alert = current_time

        # Check if we should pivot
        if self._should_pivot(window):
            self._trigger_pivot(window)

        return score_result

    def run_from_stdin(self):
        """Process Tracee JSON from stdin."""
        parser = TraceeParser()
        logger.info("Starting honeypot integration, reading from stdin...")
        logger.info(
            f"Pivot threshold: {self.pivot_threshold} events "
            f"in {self.pivot_window_seconds}s window"
        )

        try:
            for line in sys.stdin:
                event = parser.parse_line(line)
                if event:
                    self.process_event(event)

                # Print stats periodically
                if self.stats["events_processed"] % 1000 == 0:
                    self._print_stats()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._print_stats()

    def _print_stats(self):
        """Print current statistics."""
        logger.info(
            f"Stats: events={self.stats['events_processed']}, "
            f"anomalies={self.stats['anomalies_detected']}, "
            f"pivots={self.stats['pivots_triggered']}, "
            f"blocked={self.stats['pivots_blocked_whitelist']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="KernelTrap Honeypot Integration - Auto-pivot anomalous users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time monitoring with automatic pivot:
  sudo tracee --output json | python honeypot_integration.py -m ../isolation_forest/beth_iforest_model

  # Test mode (see what would happen without actually pivoting):
  sudo tracee --output json | python honeypot_integration.py -m ./model --dry-run

  # More sensitive detection (3 events in 30s):
  sudo tracee --output json | python honeypot_integration.py -m ./model --threshold 3 --window 30
        """
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained Isolation Forest model directory"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=5,
        help="Number of high-severity events to trigger pivot (default: 5)"
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=60,
        help="Time window in seconds for counting events (default: 60)"
    )
    parser.add_argument(
        "--pivot-script",
        default="/usr/local/sbin/hp_pivot_user",
        help="Path to hp_pivot_user script"
    )
    parser.add_argument(
        "--whitelist",
        nargs="+",
        default=["root", "admin"],
        help="Users to never pivot (default: root admin)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually pivot users, just log what would happen"
    )
    parser.add_argument(
        "--min-severity",
        type=int,
        choices=[1, 2],
        default=2,
        help="Minimum severity to count: 1=low, 2=high (default: 2)"
    )

    args = parser.parse_args()

    # Create log directory
    os.makedirs("/var/log/kerneltrap", exist_ok=True)

    # Initialize and run
    integration = HoneypotIntegration(
        model_dir=args.model,
        pivot_threshold=args.threshold,
        pivot_window_seconds=args.window,
        pivot_script=args.pivot_script,
        whitelist=set(args.whitelist),
        dry_run=args.dry_run,
        min_severity=args.min_severity,
    )

    integration.run_from_stdin()


if __name__ == "__main__":
    main()
