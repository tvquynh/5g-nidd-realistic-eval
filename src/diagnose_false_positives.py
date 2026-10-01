"""Characterize the severe-direction false positives.

A false-positive rate near 0.69 that is nearly identical across detector
families invites the question of whether it is a defect in the pipeline. This
script answers it by naming the confusion: it reports which class the
misclassified benign flows are assigned to, and how the flagged benign flows
differ from the retained ones on the features that move most between stations.

Usage:
    python -m src.diagnose_false_positives --models lightgbm,rf,lr --train-bs 2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.models import MODEL_REGISTRY
from src.run_experiment import get_class_names, load_or_build_master, load_paths
from src.splits import get_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lightgbm,rf,lr")
    ap.add_argument("--train-bs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="full")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--top-features", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    df.attrs["_class_names"] = classes

    if 0 < args.subsample < len(df):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample,
                                     random_state=args.seed)
        _, idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[idx].reset_index(drop=True)
        df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("cross_station", df, args.seed,
                                    train_bs=args.train_bs)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    y_test = y[test_idx]
    benign = y_test == 0
    total_benign = int(benign.sum())

    test_bs = 2 if args.train_bs == 1 else 1
    print(f"Cross-station BS{args.train_bs} -> BS{test_bs}, seed {args.seed}")
    print(f"Benign flows in the scoring set: {total_benign:,}\n")

    # Features that move most between the stations, for the subpopulation check.
    shift_csv = results / "shift" / f"shift_bs{args.train_bs}_{args.features}_seed{args.seed}_features.csv"
    top_features = []
    if shift_csv.exists():
        top_features = pd.read_csv(shift_csv).head(args.top_features)["feature"].tolist()

    record = {
        "train_bs": args.train_bs, "test_bs": test_bs, "seed": args.seed,
        "features": args.features, "n_benign_scored": total_benign,
        "per_model": {},
    }

    frame = df.iloc[test_idx]
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        fit_fn, predict_fn = MODEL_REGISTRY[name]
        model = fit_fn(X[train_idx], y[train_idx], X_val=None, y_val=None,
                       num_classes=num_classes, seed=args.seed)
        pred, _ = predict_fn(model, X[test_idx])

        destinations = Counter(pred[benign])
        flagged = benign & (pred != 0)
        retained = benign & (pred == 0)
        fpr = float(flagged.sum() / total_benign)

        print(f"--- {name}: false-positive rate {fpr:.4f}")
        entry = {"false_positive_rate": fpr, "destinations": {}}
        for cls, count in destinations.most_common():
            share = count / total_benign
            entry["destinations"][classes[cls]] = {"count": int(count), "share": share}
            print(f"    {classes[cls]:<16} {count:>7,}  ({share:6.2%})")

        entry["flagged_vs_retained"] = {}
        for feature in top_features:
            if feature not in frame.columns:
                continue
            kept = float(frame.loc[retained, feature].mean())
            sent = float(frame.loc[flagged, feature].mean())
            entry["flagged_vs_retained"][feature] = {"retained_mean": kept,
                                                     "flagged_mean": sent}
        record["per_model"][name] = entry
        print()

    out = Path(args.out) if args.out else results / f"false_positive_profile_bs{args.train_bs}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
