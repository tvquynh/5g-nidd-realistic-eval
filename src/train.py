"""
Train + evaluate one (model, split, seed, feature_set) combination.
Outputs: metrics dict (PR-AUC, ROC-AUC, macro-F1, per-class recall, FPR, latency, model size).
"""
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix, recall_score,
)

from src.models import fit_predict, MODEL_REGISTRY


def evaluate(y_true, y_pred, y_proba, num_classes):
    """Compute metrics for multiclass classification."""
    metrics = {}
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    # Per-class recall + precision + f1
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    metrics["per_class"] = {
        int(i): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in range(num_classes)
    }
    # Binary aggregate (Benign=0 vs anything else)
    y_bin_true = (y_true != 0).astype(int)
    y_bin_pred = (y_pred != 0).astype(int)
    metrics["binary_f1"] = float(f1_score(y_bin_true, y_bin_pred))
    metrics["binary_recall"] = float(recall_score(y_bin_true, y_bin_pred))
    cm = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["binary_fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # ROC-AUC + PR-AUC (multiclass via OvR averaging)
    try:
        if y_proba is not None and y_proba.ndim == 2:
            y_oh = np.eye(num_classes)[y_true]
            metrics["macro_roc_auc"] = float(roc_auc_score(y_oh, y_proba, average="macro", multi_class="ovr"))
            metrics["macro_pr_auc"] = float(average_precision_score(y_oh, y_proba, average="macro"))
    except Exception as e:
        metrics["macro_roc_auc"] = None
        metrics["macro_pr_auc"] = None
        metrics["roc_auc_error"] = str(e)
    return metrics


def run_one(df: pd.DataFrame, feature_cols, train_idx, test_idx, model_name: str, seed: int):
    """Run one experiment: fit model on train_idx, eval on test_idx."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    BS_train = None
    if model_name == "stability_lgbm" and "BS" in df.columns:
        BS_train = df["BS"].values[train_idx]

    t0 = time.time()
    model, pred, proba = fit_predict(model_name, X_train, y_train, X_test,
                                      num_classes=num_classes, seed=seed, BS_train=BS_train)
    train_time = time.time() - t0

    t0 = time.time()
    # Re-predict for a coarse inference-timing figure. Every registered model is
    # routed through the same registry call, so no model can be left unmeasured;
    # an earlier version enumerated four model names and silently recorded a
    # zero inference time for the rest.
    #
    # This is a per-cell sanity figure, not the paper's latency measurement:
    # it times one pass over the whole test set without warm-up and reports no
    # percentiles. Deployment claims come from src/latency_profile.py.
    _, predict_fn = MODEL_REGISTRY[model_name]
    predict_fn(model, X_test)
    inference_time = time.time() - t0

    metrics = evaluate(y_test, pred, proba, num_classes)
    metrics.update({
        "train_time_sec": float(train_time),
        "inference_time_sec": float(inference_time),  # coarse; see latency_profile
        "inference_timing_method": "single_pass_no_warmup",
        "inference_throughput_per_sec": float(len(X_test) / inference_time) if inference_time > 0 else 0.0,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(feature_cols)),
        "num_classes": num_classes,
    })
    return metrics, model


def save_metrics(metrics: dict, out_path: Path, meta: dict):
    record = {**meta, **metrics}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path
