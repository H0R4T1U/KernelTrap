"""
Wraps the trained Isolation Forest model for scoring incoming syscall events.

Loads the model from the same directory structure produced by beth_iforest.py:
  model_dir/
    iforest.joblib
    scaler.joblib
    meta.json          (thresholds per entity + global fallback)

Severity levels:
  0 = benign
  1 = low anomaly   (score < low_threshold)
  2 = high anomaly  (score < high_threshold)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FEATURES = [
    "processId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "eventId",
    "argsNum",
    "returnValue",
]


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

        # Global fallback thresholds from meta (used when per-entity thresholds are absent)
        gt = self._meta.get("global_thresholds", {})
        self._global_low = float(gt.get("low", -0.02))
        self._global_high = float(gt.get("high", -0.10))

        # Per-entity thresholds keyed by str(entity)
        self._entity_thresholds: Dict[str, Dict[str, float]] = self._meta.get("thresholds", {})

        self._host_col: Optional[str] = self._meta.get("host_col")

    def _entity_key(self, event: Dict[str, Any], hostname: str) -> str:
        if self._host_col:
            return f"{hostname}:{event.get('userId', 0)}"
        return str(event.get("userId", 0))

    def _thresholds_for(self, entity_key: str):
        t = self._entity_thresholds.get(entity_key, {})
        low = float(t.get("low", self._global_low))
        high = float(t.get("high", self._global_high))
        return low, high

    def score_batch(self, events: List[Dict[str, Any]], hostname: str) -> List[Dict[str, Any]]:
        """
        Score a batch of event dicts (each matching the BETH CSV row format).
        Returns one result dict per event with keys: raw_score, severity, is_anomaly.
        """
        if not events:
            return []

        rows = np.array([
            [float(e.get(f, 0) or 0) for f in FEATURES]
            for e in events
        ])

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
