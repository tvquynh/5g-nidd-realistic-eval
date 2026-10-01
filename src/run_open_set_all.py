"""Fit one base classifier and score every open-set rejection rule on it.

The earlier runner took the scoring method as an argument and refitted the base
classifier for each one, so two rules that are mathematically equivalent were
nonetheless evaluated on separately trained models. Any difference between them
then mixes the rule with training variance. Fitting once and scoring all rules
from the same probabilities removes that confound, and costs one fifth as much.

Usage:
    python -m src.run_open_set_all --base lightgbm --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.open_set import detect_open_set, open_set_metrics, UNKNOWN_LABEL
from src.run_experiment import get_class_names, load_or_build_master, load_paths, detect_profile
from src.splits import HOLDOUT_TEST_ATTACKS, HOLDOUT_TRAIN_ATTACKS, get_split

METHODS = ("none", "msp", "energy", "mahalanobis", "knn")
BASES = ("lightgbm", "xgboost", "tabnet", "ftt")


def fit_and_predict(base: str, X_train, y_train, X_test, num_classes: int, seed: int):
    """Fit the base classifier once and return train and test probabilities."""
    if base == "lightgbm":
        from src.models import fit_lightgbm
        model = fit_lightgbm(X_train, y_train, num_classes=num_classes, seed=seed)
        return model.predict(X_train), model.predict(X_test)

    if base == "xgboost":
        import xgboost as xgb
        from src.models import fit_xgboost
        model = fit_xgboost(X_train, y_train, num_classes=num_classes, seed=seed)
        return model.predict(xgb.DMatrix(X_train)), model.predict(xgb.DMatrix(X_test))

    if base == "tabnet":
        from src.deep_models import fit_tabnet, predict_tabnet
        model = fit_tabnet(X_train, y_train, num_classes=num_classes, seed=seed)
        return predict_tabnet(model, X_train)[1], predict_tabnet(model, X_test)[1]

    if base == "ftt":
        from src.deep_models import fit_fttransformer, predict_fttransformer
        model = fit_fttransformer(X_train, y_train, num_classes=num_classes, seed=seed)
        return (predict_fttransformer(model, X_train)[1],
                predict_fttransformer(model, X_test)[1])

    raise ValueError(f"unknown base classifier: {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full",
                    choices=["full", "top50", "top20", "top20_stable"])
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = load_paths()
    print(f"Profile: {detect_profile()}")

    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    df.attrs["_class_names"] = classes

    if 0 < args.subsample < len(df):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample,
                                     random_state=args.seed)
        _, idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[idx].reset_index(drop=True)
        df.attrs["_class_names"] = classes
        print(f"Subsample: {len(df):,} rows")

    train_idx, test_idx = get_split("holdout_attack", df, args.seed)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Attack-holdout: train {len(train_idx):,}  test {len(test_idx):,} "
          f"| features {len(feat_cols)}")

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    novel_indices = {cls_to_idx[c] for c in HOLDOUT_TEST_ATTACKS if c in cls_to_idx}
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]
                            if c in cls_to_idx})
    # Truth relabeling: every flow of a held-out attack type becomes Unknown.
    y_test_open = np.where(np.isin(y_test, list(novel_indices)), UNKNOWN_LABEL, y_test)

    print(f"Fitting {args.base} once for all {len(METHODS)} rules...")
    t0 = time.time()
    proba_train, proba_test = fit_and_predict(
        args.base, X_train, y_train, X_test, num_classes, args.seed)
    fit_seconds = time.time() - t0
    print(f"  fitted in {fit_seconds:.1f}s")

    out_dir = Path(args.out_dir) if args.out_dir else Path(paths["results"]) / "openset_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import f1_score

    for method in METHODS:
        t0 = time.time()
        if method == "none":
            pred = np.argmax(proba_test, axis=1)
            threshold = float("inf")
        else:
            result = detect_open_set(method, proba_test, proba_train=proba_train,
                                     y_train=y_train,
                                     threshold_quantile=args.threshold_quantile)
            pred, threshold = result.pred, float(result.threshold)
        score_seconds = time.time() - t0

        metrics = open_set_metrics(y_test_open, pred, known_classes,
                                   unknown_label_in_truth=UNKNOWN_LABEL)
        valid = pred != UNKNOWN_LABEL
        metrics["macro_f1_known_only"] = float(f1_score(
            y_test[valid], pred[valid], labels=list(known_classes),
            average="macro", zero_division=0)) if valid.any() else 0.0
        metrics["threshold"] = threshold
        record = {
            "base_model": args.base,
            "method": method,
            "seed": args.seed,
            "features": args.features,
            "n_features": len(feat_cols),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "threshold_quantile": args.threshold_quantile,
            "fit_seconds": float(fit_seconds),
            "score_seconds": float(score_seconds),
            "shared_base_fit": True,
            **metrics,
        }
        out = out_dir / f"openset_{args.base}_{method}_{args.features}_seed{args.seed}.json"
        out.write_text(json.dumps(record, indent=2, default=float))
        print(f"  {method:<12} macro-F1-open {metrics.get('macro_f1_open', float('nan')):.4f}"
              f"  novel-recall {metrics.get('novel_recall', float('nan')):.4f}"
              f"  false-unknown {metrics.get('false_unknown_rate', float('nan')):.4f}")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
