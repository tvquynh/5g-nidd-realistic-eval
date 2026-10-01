"""
Threshold-sensitivity sweep: train classifier once, score once per OOD method,
then evaluate at multiple threshold quantiles (90, 95, 97, 99). Fast because
no retrain.

Usage:
    python -m src.run_open_set_sensitivity --base lightgbm --method msp --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_experiment import load_paths, load_or_build_master, get_class_names
from src.splits import HOLDOUT_TRAIN_ATTACKS, HOLDOUT_TEST_ATTACKS, get_split
from src.features import get_feature_set
from src.models import fit_lightgbm, fit_xgboost
from src.open_set import (
    _msp_score, _energy_score, _mahalanobis_score, _knn_score,
    open_set_metrics, UNKNOWN_LABEL,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lightgbm", choices=["lightgbm", "xgboost"])
    ap.add_argument("--method", required=True, choices=["msp", "energy", "mahalanobis", "knn"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    if args.subsample > 0 and args.subsample < len(df):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample, random_state=args.seed)
        _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[sub_idx].reset_index(drop=True)
        if classes:
            df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("holdout_attack", df, args.seed)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    test_attack_indices = {cls_to_idx[c] for c in HOLDOUT_TEST_ATTACKS}
    y_test_open = np.where(np.isin(y_test, list(test_attack_indices)),
                            UNKNOWN_LABEL, y_test)

    # Train once
    t0 = time.time()
    if args.base == "lightgbm":
        model = fit_lightgbm(X_train, y_train, num_classes=num_classes, seed=args.seed)
        proba_train = model.predict(X_train)
        proba_test = model.predict(X_test)
    else:
        import xgboost as xgb
        model = fit_xgboost(X_train, y_train, num_classes=num_classes, seed=args.seed)
        proba_train = model.predict(xgb.DMatrix(X_train))
        proba_test = model.predict(xgb.DMatrix(X_test))
    train_time = time.time() - t0

    # Score once
    if args.method == "msp":
        score_train = _msp_score(proba_train)
        score_test = _msp_score(proba_test)
    elif args.method == "energy":
        score_train = _energy_score(proba_train)
        score_test = _energy_score(proba_test)
    elif args.method == "mahalanobis":
        score_train = _mahalanobis_score(proba_train, proba_train, y_train)
        score_test = _mahalanobis_score(proba_test, proba_train, y_train)
    elif args.method == "knn":
        score_train = _knn_score(proba_train, proba_train, k=50)
        score_test = _knn_score(proba_test, proba_train, k=50)

    # Sweep over thresholds
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]})
    quantiles = [0.90, 0.93, 0.95, 0.97, 0.99]
    results = {}
    for q in quantiles:
        threshold = float(np.quantile(score_train, q))
        pred = np.argmax(proba_test, axis=1)
        unknown_mask = score_test > threshold
        pred[unknown_mask] = UNKNOWN_LABEL
        m = open_set_metrics(y_test_open, pred, known_classes,
                             unknown_label_in_truth=UNKNOWN_LABEL)
        m["threshold"] = threshold
        m["threshold_quantile"] = q
        results[f"q{q}"] = m

    out = {
        "base_model": args.base, "method": args.method, "seed": args.seed,
        "features": args.features, "n_train": len(train_idx), "n_test": len(test_idx),
        "train_time_sec": train_time,
        "sensitivity_results": results,
    }

    out_path = args.out or (
        Path(paths["results"]) / "metrics"
        / f"sensitivity_{args.base}_{args.method}_full_seed{args.seed}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n=== SENSITIVITY ({args.base}/{args.method}/seed{args.seed}) ===")
    for q, r in results.items():
        print(f"  {q}: macro_f1_open={r['macro_f1_open']:.4f} novel_recall={r['novel_recall']:.4f} false_unknown={r['false_unknown_rate']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
