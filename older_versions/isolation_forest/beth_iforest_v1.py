#!/usr/bin/env python3
"""
BETH Isolation Forest:
- Train on labelled_training_data.csv (benign-only fit)
- Tune threshold on labelled_validation_data.csv (optimize for 'evil' detection)
- Evaluate on labelled_testing_data.csv
- Predict on any new CSV with the same feature columns (outside logs)

Features used (common BETH kernel-process setup):
processId, parentProcessId, userId, mountNamespace, eventId, argsNum, returnValue
Targets expected (if available): sus, evil
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "processId",
    "parentProcessId",
    "userId",
    "mountNamespace",
    "eventId",
    "argsNum",
    "returnValue",
]

LABEL_COLS = ["sus", "evil"]  # may or may not exist in external logs


@dataclass
class ModelBundle:
    scaler: StandardScaler
    model: IsolationForest
    threshold: float
    features: List[str]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        dump(self.scaler, os.path.join(out_dir, "scaler.joblib"))
        dump(self.model, os.path.join(out_dir, "iforest.joblib"))
        meta = {
            "threshold": float(self.threshold),
            "features": self.features,
            "sklearn": "IsolationForest+StandardScaler",
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        scaler = load(os.path.join(path, "scaler.joblib"))
        model = load(os.path.join(path, "iforest.joblib"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return ModelBundle(
            scaler=scaler,
            model=model,
            threshold=float(meta["threshold"]),
            features=list(meta["features"]),
        )


def read_csv_stream(
    csv_path: str,
    usecols: List[str],
    chunksize: int,
    dtype_map: Optional[Dict[str, Any]] = None,
) -> Iterable[pd.DataFrame]:
    # Pandas chunked read to handle large BETH files comfortably.
    # Note: engine="c" is default; keep it fast.
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=dtype_map,
        low_memory=False,
    ):
        yield chunk


def ensure_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns: {missing}. Found columns: {list(df.columns)}")


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # BETH cols should already be numeric, but make robust for outside logs.
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def drop_bad_rows(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Remove rows with NaN in features
    return df.dropna(subset=cols)


def load_features_and_labels(
    csv_path: str,
    chunksize: int,
    need_labels: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    usecols = FEATURES + (LABEL_COLS if need_labels else [])
    dtype_map = {c: "float64" for c in FEATURES}
    if need_labels:
        dtype_map.update({"sus": "int8", "evil": "int8"})

    X_parts: List[np.ndarray] = []
    sus_parts: List[np.ndarray] = []
    evil_parts: List[np.ndarray] = []

    for chunk in read_csv_stream(csv_path, usecols=usecols, chunksize=chunksize, dtype_map=dtype_map):
        ensure_columns(chunk, FEATURES, where=csv_path)
        chunk = coerce_numeric(chunk, FEATURES)
        chunk = drop_bad_rows(chunk, FEATURES)

        X_parts.append(chunk[FEATURES].to_numpy(dtype=np.float64, copy=False))

        if need_labels:
            ensure_columns(chunk, LABEL_COLS, where=csv_path)
            sus_parts.append(chunk["sus"].to_numpy(dtype=np.int8, copy=False))
            evil_parts.append(chunk["evil"].to_numpy(dtype=np.int8, copy=False))

    X = np.vstack(X_parts) if X_parts else np.empty((0, len(FEATURES)), dtype=np.float64)
    if not need_labels:
        return X, None, None

    sus = np.concatenate(sus_parts) if sus_parts else np.empty((0,), dtype=np.int8)
    evil = np.concatenate(evil_parts) if evil_parts else np.empty((0,), dtype=np.int8)
    return X, sus, evil


def fit_benign_only_scaler(
    X_train: np.ndarray,
    sus: np.ndarray,
    evil: np.ndarray,
) -> Tuple[StandardScaler, np.ndarray]:
    benign_mask = (sus == 0) & (evil == 0)
    X_benign = X_train[benign_mask]
    if X_benign.shape[0] == 0:
        raise ValueError("No benign rows found in training set using sus==0 & evil==0.")

    scaler = StandardScaler()
    X_benign_scaled = scaler.fit_transform(X_benign)
    return scaler, X_benign_scaled


def train_iforest(
    X_benign_scaled: np.ndarray,
    n_estimators: int,
    contamination: float,
    max_samples: str,
    n_jobs: int,
    random_state: int,
) -> IsolationForest:
    # GPU is not used by sklearn iForest; your 32 cores matter here (n_jobs).
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_benign_scaled)
    return model


def scores_to_malicious_pred(scores: np.ndarray, threshold: float) -> np.ndarray:
    # Lower score => more anomalous => malicious
    return (scores < threshold).astype(np.int8)


def pick_threshold_on_validation(
    scores_val: np.ndarray,
    evil_val: np.ndarray,
    metric: str = "f1",
) -> float:
    # Sweep thresholds over score percentiles for stability
    percentiles = np.linspace(0.1, 50.0, 200)  # focus on anomalous tail
    candidates = np.percentile(scores_val, percentiles)

    best_thr = float(candidates[0])
    best = -1.0

    for thr in candidates:
        y_pred = scores_to_malicious_pred(scores_val, float(thr))
        # Treat evil==1 as positive class for "malicious"
        p, r, f1, _ = precision_recall_fscore_support(
            evil_val, y_pred, average="binary", zero_division=0
        )
        if metric == "f1":
            val = f1
        elif metric == "precision":
            val = p
        elif metric == "recall":
            val = r
        else:
            raise ValueError("metric must be one of: f1, precision, recall")

        if val > best:
            best = val
            best_thr = float(thr)

    return best_thr


def evaluate(
    name: str,
    scores: np.ndarray,
    evil: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_pred = scores_to_malicious_pred(scores, threshold)

    # Note: ROC-AUC expects "higher score => more positive".
    # Our "maliciousness" is inverse of normality score.
    mal_score = -scores

    out: Dict[str, Any] = {}
    try:
        out["roc_auc_evil"] = float(roc_auc_score(evil, mal_score))
    except Exception:
        out["roc_auc_evil"] = None

    try:
        out["avg_precision_evil"] = float(average_precision_score(evil, mal_score))
    except Exception:
        out["avg_precision_evil"] = None

    p, r, f1, _ = precision_recall_fscore_support(evil, y_pred, average="binary", zero_division=0)
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)

    cm = confusion_matrix(evil, y_pred, labels=[0, 1])
    # cm = [[tn, fp],[fn,tp]]
    out["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    out["threshold"] = float(threshold)
    out["n"] = int(evil.shape[0])
    out["name"] = name
    return out


def cmd_train(args: argparse.Namespace) -> None:
    # Load train + validation with labels
    X_train, sus_train, evil_train = load_features_and_labels(args.train_csv, args.chunksize, need_labels=True)
    X_val, sus_val, evil_val = load_features_and_labels(args.val_csv, args.chunksize, need_labels=True)

    # Fit scaler on benign-only training events
    scaler, X_benign_scaled = fit_benign_only_scaler(X_train, sus_train, evil_train)

    # Train model on benign-only
    model = train_iforest(
        X_benign_scaled=X_benign_scaled,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    # Threshold tuning on validation for evil detection
    X_val_scaled = scaler.transform(X_val)
    scores_val = model.decision_function(X_val_scaled)  # higher = more normal
    thr = pick_threshold_on_validation(scores_val, evil_val, metric=args.tune_metric)

    bundle = ModelBundle(scaler=scaler, model=model, threshold=thr, features=FEATURES)
    bundle.save(args.out_dir)

    report = evaluate("validation", scores_val, evil_val, thr)
    print(json.dumps({"saved_to": args.out_dir, "validation_report": report}, indent=2))


def cmd_test(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.model_dir)

    X_test, sus_test, evil_test = load_features_and_labels(args.test_csv, args.chunksize, need_labels=True)
    X_test_scaled = bundle.scaler.transform(X_test)
    scores_test = bundle.model.decision_function(X_test_scaled)

    report = evaluate("test", scores_test, evil_test, bundle.threshold)
    print(json.dumps(report, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = ModelBundle.load(args.model_dir)

    # outside logs might not have labels
    X, _, _ = load_features_and_labels(args.csv, args.chunksize, need_labels=False)
    X_scaled = bundle.scaler.transform(X)
    scores = bundle.model.decision_function(X_scaled)
    pred_mal = scores_to_malicious_pred(scores, bundle.threshold)

    out_df = pd.DataFrame({
        "anomaly_score_normality": scores,   # higher => more normal
        "malicious_pred": pred_mal,          # 1 => malicious
    })

    # If user wants to keep original columns too, we can re-read a small file without chunking.
    if args.attach_features:
        raw = pd.read_csv(args.csv, usecols=bundle.features, low_memory=False)
        raw = coerce_numeric(raw, bundle.features)
        raw = drop_bad_rows(raw, bundle.features)
        raw = raw.reset_index(drop=True)
        out_df = pd.concat([raw, out_df], axis=1)

    out_df.to_csv(args.out_csv, index=False)
    print(json.dumps({"wrote": args.out_csv, "rows": int(out_df.shape[0])}, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BETH Isolation Forest (train/test/predict).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on train.csv, tune threshold on val.csv, save model.")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--val-csv", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--chunksize", type=int, default=500_000)

    # Defaults mirror common iForest settings in BETH iForest writeups
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--contamination", type=float, default=0.01)
    p_train.add_argument("--max-samples", default="auto")
    p_train.add_argument("--n-jobs", type=int, default=32)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--tune-metric", choices=["f1", "precision", "recall"], default="f1")

    p_test = sub.add_parser("test", help="Evaluate saved model on test.csv (requires labels).")
    p_test.add_argument("--model-dir", required=True)
    p_test.add_argument("--test-csv", required=True)
    p_test.add_argument("--chunksize", type=int, default=500_000)

    p_pred = sub.add_parser("predict", help="Predict malicious on any CSV with the 7 feature columns.")
    p_pred.add_argument("--model-dir", required=True)
    p_pred.add_argument("--csv", required=True)
    p_pred.add_argument("--out-csv", required=True)
    p_pred.add_argument("--chunksize", type=int, default=500_000)
    p_pred.add_argument("--attach-features", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "test":
        cmd_test(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()
