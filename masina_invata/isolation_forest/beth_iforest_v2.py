#!/usr/bin/env python3
"""
BETH Isolation Forest — real-world version:
- Train IsolationForest on BENIGN-only events (sus==0 & evil==0) from training CSV
- Compute per-host percentile thresholds on VALIDATION scores
- Two-tier severity per host:
    severity = 0 (benign)
    severity = 1 (low)  -> score < low_threshold(host)
    severity = 2 (high) -> score < high_threshold(host)

Supports:
- per-host thresholding via a host column (if present) OR by filename-as-host (default)
- large CSVs via chunked reads + sampling for training (reservoir sampling)
- evaluation on test CSV (if labels exist)
- scoring "outside logs" CSV (labels optional)

Expected feature columns:
processId, parentProcessId, userId, mountNamespace, eventId, argsNum, returnValue

Expected label columns (if evaluating):
sus, evil

Usage examples:

TRAIN (host inferred from filename):
  python beth_iforest_host2tier.py train \
    --train-csv labelled_training_data.csv \
    --val-csv labelled_validation_data.csv \
    --out-dir beth_iforest_model \
    --low-percentile 2.0 \
    --high-percentile 0.2 \
    --n-estimators 600 \
    --n-jobs 32

TEST (host inferred from filename):
  python beth_iforest_host2tier.py test \
    --model-dir beth_iforest_model \
    --test-csv labelled_testing_data.csv

PREDICT (outside logs):
  python beth_iforest_host2tier.py predict \
    --model-dir beth_iforest_model \
    --csv my_new_logs.csv \
    --out-csv my_new_logs_scored.csv \
    --attach-features

If you DO have a host column inside the CSV (e.g., "host", "hostname", "machine", etc.):
  add: --host-col host
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


FEATURES = [
    "processId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "eventId",
    "argsNum",
    "returnValue",
]

LABEL_COLS = ["sus", "evil"]


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


def infer_host_from_filename(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    # strip common suffix
    for suf in [".csv", ".CSV"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base


def reservoir_add(
    reservoir: List[np.ndarray],
    item: np.ndarray,
    seen: int,
    k: int,
    rng: np.random.Generator,
) -> None:
    """
    Classic reservoir sampling:
    - keep a uniform sample of size k from a stream
    """
    if k <= 0:
        return
    if len(reservoir) < k:
        reservoir.append(item)
        return
    j = rng.integers(0, seen)
    if j < k:
        reservoir[j] = item


def safe_percentile(arr: np.ndarray, p: float) -> float:
    # p in [0,100]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


# -----------------------------
# Model bundle
# -----------------------------

@dataclass
class ModelBundle:
    scaler: StandardScaler
    model: IsolationForest
    features: List[str]
    host_col: Optional[str]
    low_percentile: float
    high_percentile: float
    thresholds: Dict[str, Dict[str, float]]   # host -> {low, high}
    global_thresholds: Dict[str, float]       # fallback {low, high}

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        dump(self.scaler, os.path.join(out_dir, "scaler.joblib"))
        dump(self.model, os.path.join(out_dir, "iforest.joblib"))
        meta = {
            "features": self.features,
            "host_col": self.host_col,
            "low_percentile": float(self.low_percentile),
            "high_percentile": float(self.high_percentile),
            "thresholds": self.thresholds,
            "global_thresholds": self.global_thresholds,
            "notes": "normality_score = IsolationForest.decision_function (higher is more normal). "
                     "severity tiers are based on per-host score percentiles computed on validation set.",
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(model_dir: str) -> "ModelBundle":
        scaler = load(os.path.join(model_dir, "scaler.joblib"))
        model = load(os.path.join(model_dir, "iforest.joblib"))
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return ModelBundle(
            scaler=scaler,
            model=model,
            features=list(meta["features"]),
            host_col=meta.get("host_col"),
            low_percentile=float(meta["low_percentile"]),
            high_percentile=float(meta["high_percentile"]),
            thresholds=dict(meta["thresholds"]),
            global_thresholds=dict(meta["global_thresholds"]),
        )


# -----------------------------
# Streaming: scaler fit + reservoir sample for training
# -----------------------------

def fit_scaler_on_benign(
    train_csv: str,
    chunksize: int,
    host_col: Optional[str],
) -> StandardScaler:
    usecols = FEATURES + LABEL_COLS + ([host_col] if host_col else [])
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    if host_col:
        dtype_map[host_col] = "string"

    scaler = StandardScaler(with_mean=True, with_std=True)

    first = True
    benign_count = 0

    for chunk in read_csv_stream(train_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, train_csv)
        ensure_columns(chunk, LABEL_COLS, train_csv)

        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue

        Xb = benign[FEATURES].to_numpy(dtype=np.float64, copy=False)
        if first:
            scaler.partial_fit(Xb)
            first = False
        else:
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
    host_col: Optional[str],
) -> np.ndarray:
    """
    Second pass: reservoir sample of transformed benign rows, up to train_sample
    """
    usecols = FEATURES + LABEL_COLS + ([host_col] if host_col else [])
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    if host_col:
        dtype_map[host_col] = "string"

    rng = np.random.default_rng(random_state)
    reservoir: List[np.ndarray] = []
    seen = 0

    for chunk in read_csv_stream(train_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue

        Xb = benign[FEATURES].to_numpy(dtype=np.float64, copy=False)
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
# Validation: compute per-host two-tier thresholds
# -----------------------------

def compute_thresholds_from_validation(
    val_csv: str,
    scaler: StandardScaler,
    model: IsolationForest,
    chunksize: int,
    host_col: Optional[str],
    low_percentile: float,
    high_percentile: float,
    max_scores_per_host: int,
    random_state: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      thresholds: host -> {low, high}
      global_thresholds: {low, high}
      report: some stats (including if evil present)
    """
    usecols = FEATURES + ([host_col] if host_col else []) + LABEL_COLS
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    if host_col:
        dtype_map[host_col] = "string"

    rng = np.random.default_rng(random_state)

    # store scores per host (reservoir if large)
    host_scores: Dict[str, List[float]] = {}
    host_seen: Dict[str, int] = {}

    # global reservoir too
    global_scores: List[float] = []
    global_seen = 0

    # For report
    evil_all: List[int] = []
    scores_all_for_auc: List[float] = []
    have_labels = True
    evil_count = 0

    default_host = infer_host_from_filename(val_csv) if not host_col else None

    for chunk in read_csv_stream(val_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, val_csv)
        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        if "evil" not in chunk.columns:
            have_labels = False

        # host value per row
        if host_col and host_col in chunk.columns:
            hvals = chunk[host_col].astype("string").fillna("UNKNOWN").to_numpy()
        else:
            hvals = np.array([default_host] * len(chunk), dtype=object)

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
        Xs = scaler.transform(X)
        scores = model.decision_function(Xs)  # higher = more normal

        # labels for report if present
        if have_labels and "evil" in chunk.columns:
            evil = chunk["evil"].to_numpy(dtype=np.int8, copy=False)
            evil_count += int(evil.sum())
            evil_all.append(evil)
            scores_all_for_auc.append(scores)

        # per-host + global reservoirs
        for i in range(len(scores)):
            host = str(hvals[i])
            s = float(scores[i])

            # host reservoir
            if host not in host_scores:
                host_scores[host] = []
                host_seen[host] = 0
            host_seen[host] += 1
            if len(host_scores[host]) < max_scores_per_host:
                host_scores[host].append(s)
            else:
                j = rng.integers(0, host_seen[host])
                if j < max_scores_per_host:
                    host_scores[host][j] = s

            # global reservoir
            global_seen += 1
            if len(global_scores) < max_scores_per_host:
                global_scores.append(s)
            else:
                jg = rng.integers(0, global_seen)
                if jg < max_scores_per_host:
                    global_scores[jg] = s

    # compute thresholds per host
    thresholds: Dict[str, Dict[str, float]] = {}
    for host, lst in host_scores.items():
        arr = np.array(lst, dtype=np.float64)
        low_thr = safe_percentile(arr, low_percentile)
        high_thr = safe_percentile(arr, high_percentile)
        thresholds[host] = {"low": low_thr, "high": high_thr, "n_scores": int(arr.size)}

    # global fallback
    garr = np.array(global_scores, dtype=np.float64)
    global_thresholds = {
        "low": safe_percentile(garr, low_percentile),
        "high": safe_percentile(garr, high_percentile),
        "n_scores": int(garr.size),
    }

    report: Dict[str, Any] = {
        "validation_csv": val_csv,
        "hosts": len(thresholds),
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "global_thresholds": global_thresholds,
        "evil_count_in_validation": evil_count,
    }

    # If validation has evil labels, compute ranking metrics (not threshold-based)
    if evil_count > 0 and scores_all_for_auc:
        evil_vec = np.concatenate(evil_all)
        scores_vec = np.concatenate(scores_all_for_auc)
        mal_score = -scores_vec
        report["roc_auc_evil"] = float(roc_auc_score(evil_vec, mal_score))
        report["avg_precision_evil"] = float(average_precision_score(evil_vec, mal_score))
    else:
        report["roc_auc_evil"] = None
        report["avg_precision_evil"] = None

    return thresholds, global_thresholds, report


# -----------------------------
# Severity + Evaluation
# -----------------------------

def severity_from_score(score: float, low_thr: float, high_thr: float) -> int:
    # smaller score => more anomalous
    if np.isnan(low_thr) or np.isnan(high_thr):
        # fail-safe: if thresholds missing, never alert
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
        # y_true, y_pred: 0/1 arrays
        # tn: true 0 pred 0; fp: true 0 pred 1; fn: true 1 pred 0; tp: true 1 pred 1
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
    host_col = bundle.host_col
    usecols = FEATURES + LABEL_COLS + ([host_col] if host_col else [])
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    if host_col:
        dtype_map[host_col] = "string"

    default_host = infer_host_from_filename(test_csv) if not host_col else None

    overall = RunningConfusion()
    high_only = RunningConfusion()

    evil_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []

    rows = 0
    evil_count = 0

    for chunk in read_csv_stream(test_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, test_csv)
        ensure_columns(chunk, LABEL_COLS, test_csv)

        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        # hosts
        if host_col and host_col in chunk.columns:
            hvals = chunk[host_col].astype("string").fillna("UNKNOWN").to_numpy()
        else:
            hvals = np.array([default_host] * len(chunk), dtype=object)

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
        Xs = bundle.scaler.transform(X)
        scores = bundle.model.decision_function(Xs)

        evil = chunk["evil"].to_numpy(dtype=np.int8, copy=False)
        evil_count += int(evil.sum())

        # store for AUC/AP (ranking)
        evil_all.append(evil)
        scores_all.append(scores)

        # severity per row
        sev = np.zeros(len(scores), dtype=np.int8)
        for i in range(len(scores)):
            host = str(hvals[i])
            th = bundle.thresholds.get(host)
            if th is None:
                low_thr = bundle.global_thresholds["low"]
                high_thr = bundle.global_thresholds["high"]
            else:
                low_thr = th["low"]
                high_thr = th["high"]
            sev[i] = severity_from_score(float(scores[i]), low_thr, high_thr)

        # predictions
        pred_any = (sev >= 1).astype(np.int8)   # low OR high alerts count as malicious
        pred_high = (sev >= 2).astype(np.int8)  # only high alerts

        overall.update(evil, pred_any)
        high_only.update(evil, pred_high)

        rows += len(chunk)

    evil_vec = np.concatenate(evil_all) if evil_all else np.array([], dtype=np.int8)
    scores_vec = np.concatenate(scores_all) if scores_all else np.array([], dtype=np.float64)
    mal_score = -scores_vec

    report: Dict[str, Any] = {
        "test_csv": test_csv,
        "rows": rows,
        "evil_count": evil_count,
        "policy": {
            "per_host": True,
            "low_percentile": bundle.low_percentile,
            "high_percentile": bundle.high_percentile,
            "host_col": host_col,
            "host_inference": "filename" if not host_col else "host_col",
        },
        "overall_low_or_high": {
            "confusion": {"tn": overall.tn, "fp": overall.fp, "fn": overall.fn, "tp": overall.tp},
            **overall.metrics(),
        },
        "high_only": {
            "confusion": {"tn": high_only.tn, "fp": high_only.fp, "fn": high_only.fn, "tp": high_only.tp},
            **high_only.metrics(),
        },
    }

    if evil_vec.size > 0 and evil_vec.sum() > 0:
        report["ranking"] = {
            "roc_auc_evil": float(roc_auc_score(evil_vec, mal_score)),
            "avg_precision_evil": float(average_precision_score(evil_vec, mal_score)),
        }
    else:
        report["ranking"] = {"roc_auc_evil": None, "avg_precision_evil": None}

    return report


# -----------------------------
# Commands
# -----------------------------

def cmd_train(args: argparse.Namespace) -> None:
    # 1) scaler on benign-only
    scaler = fit_scaler_on_benign(
        train_csv=args.train_csv,
        chunksize=args.chunksize,
        host_col=args.host_col,
    )

    # 2) sample benign for training
    X_sample = sample_benign_for_training(
        train_csv=args.train_csv,
        scaler=scaler,
        chunksize=args.chunksize,
        train_sample=args.train_sample,
        random_state=args.random_state,
        host_col=args.host_col,
    )

    # 3) fit iforest
    model = train_iforest(
        X_benign_scaled=X_sample,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    # 4) compute per-host thresholds from validation
    thresholds, global_thresholds, val_report = compute_thresholds_from_validation(
        val_csv=args.val_csv,
        scaler=scaler,
        model=model,
        chunksize=args.chunksize,
        host_col=args.host_col,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        max_scores_per_host=args.max_scores_per_host,
        random_state=args.random_state,
    )

    bundle = ModelBundle(
        scaler=scaler,
        model=model,
        features=FEATURES,
        host_col=args.host_col,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        thresholds=thresholds,
        global_thresholds=global_thresholds,
    )
    bundle.save(args.out_dir)

    out = {
        "saved_to": args.out_dir,
        "train": {
            "train_csv": args.train_csv,
            "benign_sample_used": int(X_sample.shape[0]),
            "n_estimators": args.n_estimators,
            "contamination": args.contamination,
            "max_samples": args.max_samples,
            "n_jobs": args.n_jobs,
        },
        "validation_thresholding": val_report,
        "hosts_thresholded": len(thresholds),
        "global_thresholds": global_thresholds,
        "note": "Two-tier severity is per-host percentiles on validation normality scores.",
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
    host_col = bundle.host_col

    # outside logs may not have labels
    usecols = FEATURES + ([host_col] if host_col else [])
    dtype_map = {c: "float64" for c in FEATURES}
    if host_col:
        dtype_map[host_col] = "string"

    default_host = infer_host_from_filename(args.csv) if not host_col else None

    out_rows: List[pd.DataFrame] = []
    rows_written = 0

    for chunk in read_csv_stream(args.csv, usecols=usecols, chunksize=args.chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, args.csv)
        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        if host_col and host_col in chunk.columns:
            hvals = chunk[host_col].astype("string").fillna("UNKNOWN").to_numpy()
        else:
            hvals = np.array([default_host] * len(chunk), dtype=object)

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
        Xs = bundle.scaler.transform(X)
        scores = bundle.model.decision_function(Xs)

        sev = np.zeros(len(scores), dtype=np.int8)
        low_thr_used = np.zeros(len(scores), dtype=np.float64)
        high_thr_used = np.zeros(len(scores), dtype=np.float64)

        for i in range(len(scores)):
            host = str(hvals[i])
            th = bundle.thresholds.get(host)
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
            "host": [str(h) for h in hvals],
            "normality_score": scores,                 # higher = more normal
            "low_threshold": low_thr_used,
            "high_threshold": high_thr_used,
            "severity": sev,                           # 0/1/2
            "malicious_pred": (sev >= 1).astype(np.int8),
        })

        if args.attach_features:
            # re-attach features as seen
            feats = chunk[FEATURES].reset_index(drop=True)
            out_df = pd.concat([feats, out_df], axis=1)

        out_rows.append(out_df)
        rows_written += len(out_df)

    final = pd.concat(out_rows, axis=0) if out_rows else pd.DataFrame()
    final.to_csv(args.out_csv, index=False)

    print(json.dumps({
        "wrote": args.out_csv,
        "rows": int(rows_written),
        "policy": {
            "per_host": True,
            "two_tier": True,
            "low_percentile": bundle.low_percentile,
            "high_percentile": bundle.high_percentile,
            "host_col": host_col,
        }
    }, indent=2))


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BETH IsolationForest with per-host percentile + two-tier severity.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train benign-only iForest; compute per-host two-tier thresholds on validation.")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--val-csv", required=True)
    p_train.add_argument("--out-dir", required=True)

    p_train.add_argument("--host-col", default=None,
                         help="Optional host column inside CSV. If omitted, host is inferred from filename.")
    p_train.add_argument("--chunksize", type=int, default=500_000)

    # training + sampling
    p_train.add_argument("--train-sample", type=int, default=2_000_000,
                         help="Reservoir sample size of benign rows used to fit iForest.")
    p_train.add_argument("--n-estimators", type=int, default=600)
    p_train.add_argument("--contamination", type=float, default=0.03,
                         help="iForest contamination (model internal). Does NOT set alert thresholds.")
    p_train.add_argument("--max-samples", default="auto")
    p_train.add_argument("--n-jobs", type=int, default=32)
    p_train.add_argument("--random-state", type=int, default=42)

    # per-host two-tier thresholds (percentiles of normality scores)
    p_train.add_argument("--low-percentile", type=float, default=2.0,
                         help="Bottom X%% normality scores => severity >= 1 (low).")
    p_train.add_argument("--high-percentile", type=float, default=0.2,
                         help="Bottom X%% normality scores => severity == 2 (high).")
    p_train.add_argument("--max-scores-per-host", type=int, default=500_000,
                         help="Reservoir cap for storing validation scores per host for percentile estimation.")

    p_test = sub.add_parser("test", help="Evaluate saved model on labeled test CSV (evil required).")
    p_test.add_argument("--model-dir", required=True)
    p_test.add_argument("--test-csv", required=True)
    p_test.add_argument("--chunksize", type=int, default=500_000)

    p_pred = sub.add_parser("predict", help="Score any CSV with the 7 feature columns; outputs severity tiers.")
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
        cmd_train(args)
    elif args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
