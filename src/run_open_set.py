"""
Run open-set detection experiments on attack-holdout split.

For each (model_base, ood_method, seed):
    1. Train base classifier on train_idx (5 known attacks + 50% benign)
    2. Compute softmax proba on train + test
    3. Apply open-set detector (MSP/Energy/Mahalanobis) with 95% threshold
    4. Relabel test set: samples whose true class is one of the 3 unseen attacks
       are mapped to UNKNOWN_LABEL = -1.
    5. Compute open-set metrics: macro_f1_open, novel_recall, false_unknown_rate.

Usage:
    python -m src.run_open_set --base lightgbm --method msp --seed 42 --features full
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
from src.features import get_feature_set, get_feature_columns
from src.models import fit_lightgbm, predict_lightgbm, fit_xgboost, predict_xgboost
from src.open_set import detect_open_set, open_set_metrics, UNKNOWN_LABEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lightgbm", choices=["lightgbm", "xgboost", "tabnet", "ftt"])
    ap.add_argument("--method", required=True, choices=["msp", "energy", "mahalanobis", "knn", "none"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full", choices=["full", "top50", "top20", "top20_stable"])
    ap.add_argument("--threshold-q", type=float, default=0.95)
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    # Subsample
    if args.subsample > 0 and args.subsample < len(df):
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample, random_state=args.seed)
        _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[sub_idx].reset_index(drop=True)
        if classes:
            df.attrs["_class_names"] = classes

    # Holdout-attack split
    train_idx, test_idx = get_split("holdout_attack", df, args.seed)

    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Relabel test: samples in HOLDOUT_TEST_ATTACKS classes -> UNKNOWN_LABEL
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    test_attack_indices = {cls_to_idx[c] for c in HOLDOUT_TEST_ATTACKS}
    y_test_open = np.where(np.isin(y_test, list(test_attack_indices)),
                            UNKNOWN_LABEL, y_test)

    # Fit base
    t0 = time.time()
    if args.base == "lightgbm":
        model = fit_lightgbm(X_train, y_train, num_classes=num_classes, seed=args.seed)
        proba_train = model.predict(X_train)
        proba_test = model.predict(X_test)
    elif args.base == "xgboost":
        import xgboost as xgb
        model = fit_xgboost(X_train, y_train, num_classes=num_classes, seed=args.seed)
        proba_train = model.predict(xgb.DMatrix(X_train))
        proba_test = model.predict(xgb.DMatrix(X_test))
    elif args.base == "tabnet":
        from src.deep_models import fit_tabnet, predict_tabnet
        model = fit_tabnet(X_train, y_train, num_classes=num_classes, seed=args.seed)
        _, proba_train = predict_tabnet(model, X_train)
        _, proba_test = predict_tabnet(model, X_test)
    elif args.base == "ftt":
        from src.deep_models import fit_fttransformer, predict_fttransformer
        model = fit_fttransformer(X_train, y_train, num_classes=num_classes, seed=args.seed)
        _, proba_train = predict_fttransformer(model, X_train)
        _, proba_test = predict_fttransformer(model, X_test)
    train_time = time.time() - t0

    # Apply open-set detection
    if args.method == "none":
        # baseline: argmax, no rejection
        pred = np.argmax(proba_test, axis=1)
        scores = np.zeros(len(pred))
        threshold = float("inf")
    else:
        result = detect_open_set(
            args.method, proba_test, proba_train=proba_train,
            y_train=y_train, threshold_quantile=args.threshold_q,
        )
        pred = result.pred
        scores = result.scores
        threshold = result.threshold

    # Compute open-set metrics
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS})
    known_classes.add(cls_to_idx["Benign"]) if isinstance(known_classes, set) else None
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]})

    metrics = open_set_metrics(y_test_open, pred, known_classes,
                               unknown_label_in_truth=UNKNOWN_LABEL)
    metrics["train_time_sec"] = float(train_time)
    metrics["threshold"] = float(threshold)
    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["n_features"] = int(len(feat_cols))
    metrics["base_model"] = args.base
    metrics["method"] = args.method
    metrics["seed"] = args.seed
    metrics["features"] = args.features

    # Also compute regular macro-F1 on multiclass test (treating Unknown as wrong)
    # this is to compare against baseline (no open-set) which has zero recall on novel attacks
    from sklearn.metrics import f1_score
    valid_mask = pred != UNKNOWN_LABEL
    if valid_mask.any():
        metrics["macro_f1_known_only"] = float(f1_score(
            y_test[valid_mask], pred[valid_mask],
            labels=list(known_classes), average="macro", zero_division=0,
        ))
    else:
        metrics["macro_f1_known_only"] = 0.0

    out_path = args.out
    if out_path is None:
        out_path = (
            Path(paths["results"]) / "metrics"
            / f"openset_{args.base}_{args.method}_{args.features}_seed{args.seed}.json"
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2))
    print(f"\n=== OPEN-SET RESULTS ({args.base}/{args.method}/seed{args.seed}) ===")
    print(f"macro_f1_open       : {metrics['macro_f1_open']:.4f}")
    print(f"novel_recall        : {metrics['novel_recall']:.4f}")
    print(f"false_unknown_rate  : {metrics['false_unknown_rate']:.4f}")
    print(f"known_class_acc     : {metrics['known_classification_acc']:.4f}")
    print(f"threshold           : {metrics['threshold']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
