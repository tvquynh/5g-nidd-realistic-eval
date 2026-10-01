"""Characterize the base-station shift and each detector's exposure to it.

Usage:
    python -m src.run_shift_analysis --seed 42 --train-bs 1 --features full \
        --models lightgbm,xgboost,rf,lr,mlp

Two questions are answered for one cross-station configuration:

1.  Which features move between the two base stations, pooled and within
    class, so that prior shift is separated from concept shift.
2.  How much of each detector's predictive weight sits on those moving
    features, and how concentrated that weight is.

Detectors are fitted on 80% of the training station and their reliance is
measured on the held-out 20% (in-station) and on the other station
(cross-station), so that reliance is never read off data the detector was
fitted on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.models import MODEL_REGISTRY
from src.run_experiment import get_class_names, load_or_build_master, load_paths, detect_profile
from src.shift_analysis import (
    concentration_stats,
    feature_shift_table,
    permutation_reliance,
    reliance_shift,
    shift_vs_reliance_alignment,
)
from src.splits import get_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-bs", type=int, default=1)
    ap.add_argument("--features", default="full",
                    choices=["full", "top50", "top20", "top20_stable"])
    ap.add_argument("--models", default="lightgbm,xgboost,rf,lr,mlp")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--reliance-subsample", type=int, default=20_000)
    ap.add_argument("--reliance-repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    print(f"Profile: {detect_profile()}")

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
        print(f"Subsample: {len(df):,} rows")

    train_idx, test_idx = get_split("cross_station", df, args.seed, train_bs=args.train_bs)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    test_bs = 2 if args.train_bs == 1 else 1
    print(f"Cross-station BS{args.train_bs} -> BS{test_bs}: "
          f"train={len(train_idx):,} test={len(test_idx):,} | features={len(feat_cols)}")

    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    X_station, y_station = X[train_idx], y[train_idx]
    X_other, y_other = X[test_idx], y[test_idx]
    num_classes = int(df["y_multi"].max()) + 1

    print("Computing per-feature shift statistics...")
    t0 = time.time()
    shift_table = feature_shift_table(
        X_station, X_other, feat_cols, y_ref=y_station, y_cmp=y_other, seed=args.seed
    )
    print(f"  done in {time.time() - t0:.1f}s")
    print(shift_table.head(10).to_string(index=False))

    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X_station, y_station, test_size=0.2, random_state=args.seed, stratify=y_station
    )

    per_model = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n--- {name}")
        t0 = time.time()
        fit_fn, predict_fn = MODEL_REGISTRY[name]
        model = fit_fn(X_fit, y_fit, X_val=None, y_val=None,
                       num_classes=num_classes, seed=args.seed)
        label_fn = lambda X, _p=predict_fn, _m=model: _p(_m, X)[0]
        fit_seconds = time.time() - t0

        f1_in = float(f1_score(y_holdout, label_fn(X_holdout), average="macro", zero_division=0))
        f1_cross = float(f1_score(y_other, label_fn(X_other), average="macro", zero_division=0))
        print(f"  macro-F1 in-station {f1_in:.4f} | cross-station {f1_cross:.4f} "
              f"| drop {f1_in - f1_cross:+.4f}")

        print("  permutation reliance (in-station)...")
        reliance_in = permutation_reliance(
            label_fn, X_holdout, y_holdout, feat_cols,
            n_repeats=args.reliance_repeats, subsample=args.reliance_subsample, seed=args.seed,
        )
        print("  permutation reliance (cross-station)...")
        reliance_cross = permutation_reliance(
            label_fn, X_other, y_other, feat_cols,
            n_repeats=args.reliance_repeats, subsample=args.reliance_subsample, seed=args.seed,
        )

        per_model[name] = {
            "fit_seconds": fit_seconds,
            "macro_f1_in_station": f1_in,
            "macro_f1_cross_station": f1_cross,
            "macro_f1_drop": f1_in - f1_cross,
            "concentration_in_station": concentration_stats(reliance_in),
            "concentration_cross_station": concentration_stats(reliance_cross),
            "reliance_shift": reliance_shift(reliance_in, reliance_cross),
            "alignment_with_feature_shift": shift_vs_reliance_alignment(shift_table, reliance_in),
            "top10_reliance_in_station": (
                reliance_in.sort_values(ascending=False).head(10).round(6).to_dict()
            ),
        }
        conc = per_model[name]["concentration_in_station"]
        align = per_model[name]["alignment_with_feature_shift"]
        print(f"  gini {conc['gini']:.3f} | effective features {conc['effective_features']:.1f} "
              f"| reliance-weighted shift {align['reliance_weighted_shift']:.4f}")

    record = {
        "seed": args.seed,
        "train_bs": args.train_bs,
        "test_bs": test_bs,
        "features": args.features,
        "n_features": len(feat_cols),
        "subsample": int(args.subsample),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "reliance_subsample": int(args.reliance_subsample),
        "reliance_repeats": int(args.reliance_repeats),
        "shift_summary": {
            "n_features_psi_above_0_25": int(
                (shift_table.get("psi_class_matched", shift_table["psi_pooled"]) > 0.25).sum()
            ),
            "n_features_psi_above_0_10": int(
                (shift_table.get("psi_class_matched", shift_table["psi_pooled"]) > 0.10).sum()
            ),
            "top10_shifted_features": shift_table.head(10).to_dict(orient="records"),
        },
        "per_model": per_model,
    }

    out_dir = Path(paths["results"]) / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shift_bs{args.train_bs}_{args.features}_seed{args.seed}"
    out_path = Path(args.out) if args.out else out_dir / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=float))
    shift_table.to_csv(out_dir / f"{stem}_features.csv", index=False)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {out_dir / f'{stem}_features.csv'}")


if __name__ == "__main__":
    main()
