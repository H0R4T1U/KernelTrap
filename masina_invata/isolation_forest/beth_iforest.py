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
            "notes": (
                "normality_score = IsolationForest.decision_function (higher is more normal). "
                "severity tiers are based on per-entity score percentiles computed on validation set. "
                "entity is userId by default (or host:user if group_mode=host_user)."
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
        )


# -----------------------------
# Streaming: scaler fit + reservoir sample for training
# -----------------------------

def fit_scaler_on_benign(train_csv: str, chunksize: int) -> StandardScaler:
    usecols = FEATURES + LABEL_COLS
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})

    scaler = StandardScaler(with_mean=True, with_std=True)

    benign_count = 0
    first = True

    for chunk in read_csv_stream(train_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, train_csv)
        ensure_columns(chunk, LABEL_COLS, train_csv)

        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        benign = chunk[(chunk["sus"] == 0) & (chunk["evil"] == 0)]
        if benign.empty:
            continue

        Xb = benign[FEATURES].to_numpy(dtype=np.float64, copy=False)
        scaler.partial_fit(Xb)
        benign_count += Xb.shape[0]
        first = False

    if benign_count == 0:
        raise ValueError("No benign rows found in training set (sus==0 & evil==0).")

    return scaler


def sample_benign_for_training(
    train_csv: str,
    scaler: StandardScaler,
    chunksize: int,
    train_sample: int,
    random_state: int,
) -> np.ndarray:
    usecols = FEATURES + LABEL_COLS
    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})

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
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Any]]:
    # Read only what we need
    usecols = FEATURES + LABEL_COLS
    if user_col not in usecols:
        usecols.append(user_col)
    if host_col and host_col not in usecols:
        usecols.append(host_col)

    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    # user/host can be string-ish
    dtype_map[user_col] = "string"
    if host_col:
        dtype_map[host_col] = "string"

    rng = np.random.default_rng(random_state)

    entity_scores: Dict[str, List[float]] = {}
    entity_seen: Dict[str, int] = {}

    global_scores: List[float] = []
    global_seen = 0

    evil_all: List[np.ndarray] = []
    scores_all_for_auc: List[np.ndarray] = []
    evil_count = 0

    for chunk in read_csv_stream(val_csv, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, val_csv)

        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        keys = make_entity_keys(chunk, user_col=user_col, host_col=host_col, group_mode=group_mode)

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
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
    # Ensure we can read user/host cols if needed for entity key
    usecols = FEATURES + LABEL_COLS
    if bundle.user_col not in usecols:
        usecols.append(bundle.user_col)
    if bundle.host_col and bundle.host_col not in usecols:
        usecols.append(bundle.host_col)

    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map.update({"sus": "int8", "evil": "int8"})
    dtype_map[bundle.user_col] = "string"
    if bundle.host_col:
        dtype_map[bundle.host_col] = "string"

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

        keys = make_entity_keys(
            chunk,
            user_col=bundle.user_col,
            host_col=bundle.host_col,
            group_mode=bundle.group_mode,
        )

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
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
    scaler = fit_scaler_on_benign(train_csv=args.train_csv, chunksize=args.chunksize)

    X_sample = sample_benign_for_training(
        train_csv=args.train_csv,
        scaler=scaler,
        chunksize=args.chunksize,
        train_sample=args.train_sample,
        random_state=args.random_state,
    )

    model = train_iforest(
        X_benign_scaled=X_sample,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

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

    usecols = FEATURES
    if bundle.user_col not in usecols:
        usecols.append(bundle.user_col)
    if bundle.host_col and bundle.host_col not in usecols:
        usecols.append(bundle.host_col)

    dtype_map = {c: "float64" for c in FEATURES}
    dtype_map[bundle.user_col] = "string"
    if bundle.host_col:
        dtype_map[bundle.host_col] = "string"

    out_rows: List[pd.DataFrame] = []
    rows_written = 0

    for chunk in read_csv_stream(args.csv, usecols=usecols, chunksize=args.chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, args.csv)

        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        keys = make_entity_keys(
            chunk,
            user_col=bundle.user_col,
            host_col=bundle.host_col,
            group_mode=bundle.group_mode,
        )

        X = chunk[FEATURES].to_numpy(dtype=np.float64, copy=False)
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
    p_train.add_argument("--train-sample", type=int, default=2_000_000,
                         help="Reservoir sample size of benign rows used to fit iForest.")
    p_train.add_argument("--n-estimators", type=int, default=600)
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
