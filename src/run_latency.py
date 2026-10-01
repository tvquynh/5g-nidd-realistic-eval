"""Profile inference latency, model size, and memory for one configuration.

Usage:
    python -m src.run_latency --model lightgbm --split cross_station --seed 42 \
        --features full --train-bs 1 --subsample 300000

Produces one JSON record per (model, split, feature set, seed) containing a
batch-size sweep with p50/p95/p99 latency, model complexity metrics, and --- for
XGBoost --- a breakdown that separates input-conversion overhead from the cost
of evaluating the trees.
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
from src.latency_profile import (
    DEFAULT_BATCH_SIZES,
    measure_latency,
    model_complexity,
    profile_batch_sweep,
    xgboost_overhead_breakdown,
)
from src.models import MODEL_REGISTRY
from src.run_experiment import get_class_names, load_or_build_master, load_paths, detect_profile
from src.splits import get_split
from src.stacking import BASE_SETS, StackedEnsemble


def build_predict_fn(model_name: str, model):
    """Return the callable a deployment would invoke per batch.

    The callable includes everything on the serving path for that model: input
    conversion, scaling where applicable, probability computation, and the
    argmax that turns probabilities into a label. Timing the full path is what
    makes the numbers comparable across model families.
    """
    if model_name == "stack":
        return lambda X: model.predict(X)
    _, predict_fn = MODEL_REGISTRY[model_name]
    return lambda X: predict_fn(model, X)

# Splits that are defined relative to a training base station.
STATION_SPLITS = {"cross_station", "cross_station_naive", "prior_control",
                  "cross_station_raw_raw", "cross_station_raw_matched",
                  "cross_station_matched_raw", "cross_station_matched_matched"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=list(MODEL_REGISTRY.keys()) + ["stack"])
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
    ap.add_argument("--stack-base-set", default="trees_mlp", choices=list(BASE_SETS.keys()))
    ap.add_argument("--stack-meta", default="lr")
    ap.add_argument("--batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES))
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

    split_kwargs = {"train_bs": args.train_bs} if args.split in STATION_SPLITS else {}
    train_idx, test_idx = get_split(args.split, df, args.seed, **split_kwargs)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    print(f"Split {args.split}: train={len(train_idx):,} test={len(test_idx):,} "
          f"| features={len(feat_cols)}")

    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = np.ascontiguousarray(X[test_idx])
    num_classes = int(df["y_multi"].max()) + 1

    t0 = time.time()
    if args.model == "stack":
        model = StackedEnsemble(
            num_classes=num_classes,
            seed=args.seed,
            base_names=BASE_SETS[args.stack_base_set],
            meta_learner=args.stack_meta,
        ).fit(X_train, y_train)
    else:
        fit_fn, _ = MODEL_REGISTRY[args.model]
        model = fit_fn(X_train, y_train, X_val=None, y_val=None,
                       num_classes=num_classes, seed=args.seed)
    fit_seconds = time.time() - t0
    print(f"Fitted in {fit_seconds:.1f}s")

    predict_fn = build_predict_fn(args.model, model)
    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(","))

    # Timing and memory sampling are taken in separate passes: the sampler
    # thread would otherwise perturb the very latencies being measured.
    print("Running batch sweep...")
    sweep = profile_batch_sweep(predict_fn, X_test, batch_sizes=batch_sizes)

    print("Sampling resident memory during inference...")
    memory_batch = max(b for b in batch_sizes if b <= len(X_test))
    memory_probe = measure_latency(
        predict_fn, X_test, batch_size=memory_batch, n_repeats=50, sample_memory=True
    )

    complexity = model_complexity(model, args.model)

    overhead = None
    if args.model == "xgboost":
        print("Measuring input-conversion overhead...")
        overhead = {
            str(b): xgboost_overhead_breakdown(model, X_test, b)
            for b in (1, 16, 256)
            if b <= len(X_test)
        }

    record = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "features": args.features,
        "n_features": len(feat_cols),
        "train_bs": args.train_bs if args.split in STATION_SPLITS else None,
        "stack_base_set": args.stack_base_set if args.model == "stack" else None,
        "stack_meta": args.stack_meta if args.model == "stack" else None,
        "subsample": int(args.subsample),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "fit_seconds": float(fit_seconds),
        "complexity": complexity,
        "latency_sweep": sweep,
        "memory_probe": {
            "batch_size": memory_probe["batch_size"],
            "peak_rss_bytes": memory_probe.get("peak_rss_bytes"),
            "baseline_rss_bytes": memory_probe.get("baseline_rss_bytes"),
        },
        "xgboost_overhead": overhead,
    }

    out_path = args.out
    if out_path is None:
        bs_part = f"_bs{args.train_bs}" if args.split in STATION_SPLITS else ""
        stack_part = f"_{args.stack_base_set}_{args.stack_meta}" if args.model == "stack" else ""
        out_path = (
            Path(paths["results"]) / "latency"
            / f"latency_{args.model}{stack_part}_{args.split}{bs_part}"
              f"_{args.features}_seed{args.seed}.json"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))

    print("\n=== LATENCY ===")
    for entry in sweep:
        print(f"  b={entry['batch_size']:>5}  "
              f"p50={entry['p50_per_sample_us']:9.2f} us/flow  "
              f"p95={entry['p95_per_sample_us']:9.2f}  "
              f"p99={entry['p99_per_sample_us']:9.2f}  "
              f"tail p99/p50={entry['tail_ratio_p99_p50']:.2f}")
    if complexity.get("param_count") is not None:
        print(f"Parameters: {complexity['param_count']:,} | "
              f"size: {complexity['size_native_mb']:.2f} MB")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
