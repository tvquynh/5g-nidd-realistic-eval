"""Fit one stacked ensemble and evaluate every fusion rule on it.

Usage:
    python -m src.run_stacking --base-set trees_mlp --split cross_station \
        --seed 42 --features full --train-bs 1

The base learners are fitted once --- out-of-fold blocks for the training set
and a refit for the test set --- and every meta-learner is then fitted on those
same blocks. A sweep over fusion rules therefore costs one round of base
fitting rather than one round per rule.

One JSON record is written per meta-learner. Each carries the same metric block
as the single-model experiments, so stacked results drop straight into the
existing aggregation, plus the fusion diagnostics: how the meta-learner
distributes weight across bases, how each base scores out-of-fold versus on the
test set, and the gap between the stack and its own best member.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.run_experiment import get_class_names, load_or_build_master, load_paths, detect_profile
from src.splits import get_split
from src.stacking import BASE_SETS, META_LEARNERS, StackedEnsemble
from src.train import evaluate

# Splits that are defined relative to a training base station.
STATION_SPLITS = {"cross_station", "cross_station_naive", "prior_control",
                  "cross_station_raw_raw", "cross_station_raw_matched",
                  "cross_station_matched_raw", "cross_station_matched_matched"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-set", default="trees_mlp", choices=list(BASE_SETS.keys()))
    ap.add_argument("--metas", default=",".join(META_LEARNERS),
                    help="comma-separated fusion rules to evaluate on the shared base fit")
    ap.add_argument("--split", required=True,
                    choices=["random", "temporal", "temporal_cross_day", "cross_station",
                             "cross_station_naive", "cross_station_raw_raw",
                             "cross_station_raw_matched", "cross_station_matched_raw",
                             "cross_station_matched_matched",
                             "prior_control", "holdout_attack"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full",
                    choices=["full", "top50", "top20", "top20_stable"])
    ap.add_argument("--train-bs", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    metas = [m.strip() for m in args.metas.split(",") if m.strip()]
    unknown = set(metas) - set(META_LEARNERS)
    if unknown:
        raise SystemExit(f"Unknown meta-learner(s): {sorted(unknown)}")

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

    split_kwargs = {"train_bs": args.train_bs} if args.split in STATION_SPLITS else {}
    train_idx, test_idx = get_split(args.split, df, args.seed, **split_kwargs)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    print(f"Split {args.split}: train={len(train_idx):,} test={len(test_idx):,} "
          f"| features={len(feat_cols)}")

    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    num_classes = int(df["y_multi"].max()) + 1

    base_names = BASE_SETS[args.base_set]
    print(f"Fitting bases {base_names} (k={args.k_folds})")
    t0 = time.time()
    shared = StackedEnsemble(
        num_classes=num_classes,
        seed=args.seed,
        base_names=base_names,
        meta_learner="lr",
        k_folds=args.k_folds,
    )
    oof_blocks = shared.fit_bases(X_train, y_train)
    base_seconds = time.time() - t0
    print(f"Bases fitted in {base_seconds:.1f}s")
    print(f"OOF macro-F1 per base: "
          f"{ {k: round(v, 4) for k, v in shared.diagnostics['oof_macro_f1_per_base'].items()} }")

    # Score the test set once; every fusion rule reuses these blocks.
    t0 = time.time()
    test_blocks = shared._test_blocks(X_test)
    base_inference_seconds = time.time() - t0

    out_dir = Path(args.out_dir) if args.out_dir else Path(paths["results"]) / "stacking"
    out_dir.mkdir(parents=True, exist_ok=True)
    bs_part = f"_bs{args.train_bs}" if args.split in STATION_SPLITS else ""

    written = []
    for meta in metas:
        print(f"\n--- meta: {meta}")
        stack = shared.clone_with_meta(meta)
        t0 = time.time()
        stack._fit_meta(oof_blocks, y_train)
        meta_seconds = time.time() - t0

        t0 = time.time()
        proba = stack.predict_proba(X_test, blocks=test_blocks)
        fusion_seconds = time.time() - t0
        pred = proba.argmax(axis=1)

        metrics = evaluate(y_test, pred, proba, num_classes)
        inference_time = base_inference_seconds + fusion_seconds
        metrics.update({
            "train_time_sec": float(base_seconds + meta_seconds),
            "base_fit_time_sec": float(base_seconds),
            "meta_fit_time_sec": float(meta_seconds),
            "inference_time_sec": float(inference_time),
            "inference_throughput_per_sec": (
                float(len(X_test) / inference_time) if inference_time > 0 else 0.0
            ),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(feat_cols)),
            "num_classes": num_classes,
        })

        diagnostics = stack.shift_diagnostics(X_test, y_test)
        diagnostics.update({k: v for k, v in stack.diagnostics.items()
                            if k != "oof_macro_f1_per_base"})

        record = {
            "model": "stack",
            "base_set": args.base_set,
            "base_names": list(base_names),
            "meta": meta,
            "split": args.split,
            "seed": args.seed,
            "features": args.features,
            "train_bs": args.train_bs if args.split in STATION_SPLITS else None,
            "subsample": int(args.subsample),
            "k_folds": int(args.k_folds),
            "fit_times": stack.fit_times,
            "diagnostics": diagnostics,
            **metrics,
        }

        out_path = out_dir / (f"stack_{args.base_set}_{meta}_{args.split}{bs_part}"
                              f"_{args.features}_seed{args.seed}.json")
        out_path.write_text(json.dumps(record, indent=2, default=float))
        written.append(out_path)

        best = diagnostics["best_base_on_test"]
        print(f"  macro-F1 {metrics['macro_f1']:.4f} | best base {best} "
              f"({diagnostics['test_macro_f1_per_base'][best]:.4f}) "
              f"| stack-minus-best {diagnostics['stack_minus_best_base']:+.4f} "
              f"| Spearman OOF/test {diagnostics['spearman_oof_vs_test_rank']:+.3f}")
        if "convex_weights" in diagnostics:
            print(f"  weights: { {k: round(v, 3) for k, v in diagnostics['convex_weights'].items()} }")
        if "meta_coefficient_mass" in diagnostics:
            print(f"  coefficient mass: "
                  f"{ {k: round(v, 3) for k, v in diagnostics['meta_coefficient_mass'].items()} }")

    print(f"\nWrote {len(written)} records to {out_dir}")


if __name__ == "__main__":
    main()
