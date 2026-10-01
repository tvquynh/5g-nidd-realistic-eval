"""Per-attack rejectability and threshold sensitivity, from one base fit.

Two questions the aggregate grid cannot answer:

1. Aggregate novel-recall averages over three held-out attack types that behave
   very differently. Reported per type, it says which novel attacks a rejection
   rule can actually catch.
2. The 95th-percentile operating point is a choice. Sweeping it shows whether
   the choice is load-bearing.

Both are computed from the same fitted base classifier, so neither is
contaminated by training variance.

Usage:
    python -m src.run_open_set_detail --base xgboost --seed 42
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
from src.run_open_set_all import BASES, METHODS, fit_and_predict
from src.run_experiment import get_class_names, load_or_build_master, load_paths
from src.splits import HOLDOUT_TEST_ATTACKS, HOLDOUT_TRAIN_ATTACKS, get_split

QUANTILES = (0.90, 0.93, 0.95, 0.97, 0.99)
SCORING_RULES = tuple(m for m in METHODS if m != "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    df.attrs["_class_names"] = classes

    if 0 < args.subsample < len(df):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample,
                                     random_state=args.seed)
        _, idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[idx].reset_index(drop=True)
        df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("holdout_attack", df, args.seed)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    novel_names = [c for c in HOLDOUT_TEST_ATTACKS if c in cls_to_idx]
    novel_indices = {cls_to_idx[c] for c in novel_names}
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]
                            if c in cls_to_idx})
    y_test_open = np.where(np.isin(y_test, list(novel_indices)), UNKNOWN_LABEL, y_test)

    print(f"{args.base} seed {args.seed}: fitting once for "
          f"{len(SCORING_RULES)} rules x {len(QUANTILES)} operating points")
    t0 = time.time()
    proba_train, proba_test = fit_and_predict(
        args.base, X_train, y_train, X_test, num_classes, args.seed)
    fit_seconds = time.time() - t0
    print(f"  fitted in {fit_seconds:.1f}s")

    record = {
        "base_model": args.base, "seed": args.seed, "features": args.features,
        "fit_seconds": fit_seconds, "novel_attack_types": novel_names,
        "per_attack": {}, "threshold_sweep": {},
    }

    for rule in SCORING_RULES:
        # Per-attack rejectability at the default operating point.
        result = detect_open_set(rule, proba_test, proba_train=proba_train,
                                 y_train=y_train, threshold_quantile=0.95)
        flagged = result.pred == UNKNOWN_LABEL
        record["per_attack"][rule] = {
            name: float(np.mean(flagged[y_test == cls_to_idx[name]]))
            for name in novel_names
        }
        benign_idx = cls_to_idx.get("Benign")
        if benign_idx is not None:
            record["per_attack"][rule]["Benign (false unknown)"] = float(
                np.mean(flagged[y_test == benign_idx]))

        # Operating-point sweep.
        sweep = {}
        for q in QUANTILES:
            r = detect_open_set(rule, proba_test, proba_train=proba_train,
                                y_train=y_train, threshold_quantile=q)
            metrics = open_set_metrics(y_test_open, r.pred, known_classes,
                                       unknown_label_in_truth=UNKNOWN_LABEL)
            sweep[f"{q:.2f}"] = {
                "macro_f1_open": float(metrics.get("macro_f1_open", float("nan"))),
                "novel_recall": float(metrics.get("novel_recall", float("nan"))),
                "false_unknown_rate": float(metrics.get("false_unknown_rate", float("nan"))),
            }
        record["threshold_sweep"][rule] = sweep

        best = max(sweep, key=lambda k: sweep[k]["macro_f1_open"])
        spread = (max(v["macro_f1_open"] for v in sweep.values())
                  - min(v["macro_f1_open"] for v in sweep.values()))
        print(f"  {rule:<12} per-attack "
              f"{ {k: round(v, 3) for k, v in record['per_attack'][rule].items()} }")
        print(f"  {'':<12} sweep best q={best}, spread {spread:.4f}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(paths["results"]) / "openset_detail"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"detail_{args.base}_{args.features}_seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2, default=float))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
