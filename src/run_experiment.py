"""
Main entry point: run one experiment (model × split × seed × feature_set).

Usage:
    python -m src.run_experiment --model lightgbm --split random --seed 42 --features full

For batch:
    python -m src.run_experiment --batch-config configs/batch.yaml
"""
import argparse
import json
import os
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_5g_nidd_master
from src.preprocess import preprocess
from src.splits import get_split
from src.features import get_feature_set, get_feature_columns
from src.train import run_one, save_metrics


def detect_profile():
    return os.environ.get("NIDD_PROFILE", "local")


def load_paths():
    cfg_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg[detect_profile()]


def load_or_build_master(paths) -> pd.DataFrame:
    out = Path(paths["data_out"]) / "master_5g_nidd.parquet"
    if out.exists():
        print(f"Loading cached master: {out}")
        df = pd.read_parquet(out)
        # Reconstruct class names
        # We saved without class names attrs; rebuild from y_multi unique
        return df
    print("Building master from raw CSVs...")
    raw = load_5g_nidd_master()
    classes = sorted(raw["Attack Type"].astype(str).str.strip().unique())
    df = preprocess(raw)
    df.attrs["_class_names"] = classes
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)
    # Also save class names to sidecar JSON
    with open(out.with_suffix(".classes.json"), "w") as f:
        json.dump(classes, f)
    print(f"Saved master: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    return df


def get_class_names(paths):
    p = Path(paths["data_out"]) / "master_5g_nidd.classes.json"
    if p.exists():
        return json.loads(p.read_text())
    return None

# Splits that are defined relative to a training base station.
STATION_SPLITS = {"cross_station", "cross_station_naive", "prior_control",
                  "cross_station_raw_raw", "cross_station_raw_matched",
                  "cross_station_matched_raw", "cross_station_matched_matched"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["lightgbm", "xgboost", "rf", "lr", "mlp", "stability_lgbm",
                             "tabnet", "ftt"])
    ap.add_argument("--split", required=True, choices=["random", "temporal", "temporal_cross_day", "cross_station",
                             "cross_station_naive", "cross_station_raw_raw",
                             "cross_station_raw_matched", "cross_station_matched_raw",
                             "cross_station_matched_matched",
                             "prior_control", "holdout_attack"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full", choices=["full", "top50", "top20", "top20_stable"])
    ap.add_argument("--train-bs", type=int, default=1, help="for cross_station split: train BS (1 or 2)")
    ap.add_argument("--smoke", action="store_true", help="subsample 20k flows + cap iterations for smoke test")
    ap.add_argument("--subsample", type=int, default=0, help="stratified subsample size (0=full)")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    paths = load_paths()
    print(f"Profile: {detect_profile()}")

    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    target_size = 0
    if args.smoke:
        target_size = 20_000
        import os
        os.environ["SMOKE_MODE"] = "1"
    elif args.subsample > 0:
        target_size = args.subsample
    if target_size > 0:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=target_size, random_state=args.seed)
        _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[sub_idx].reset_index(drop=True)
        if classes:
            df.attrs["_class_names"] = classes
        print(f"Subsample: {len(df):,} rows ({'smoke' if args.smoke else 'subsample'})")

    # Build split
    split_kwargs = {}
    if args.split in STATION_SPLITS:
        split_kwargs["train_bs"] = args.train_bs
    train_idx, test_idx = get_split(args.split, df, args.seed, **split_kwargs)
    print(f"Split {args.split}: train={len(train_idx):,} | test={len(test_idx):,}")

    # Feature set
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    print(f"Feature set {args.features}: {len(feat_cols)} cols")

    # Run experiment
    metrics, _model = run_one(df, feat_cols, train_idx, test_idx, args.model, args.seed)

    meta = {
        "model": args.model,
        "split": args.split,
        "seed": args.seed,
        "features": args.features,
        "train_bs": args.train_bs if args.split in STATION_SPLITS else None,
        "smoke": args.smoke,
    }

    out_path = args.out
    if out_path is None:
        suffix = "_smoke" if args.smoke else ""
        bs_part = f"_bs{args.train_bs}" if args.split in STATION_SPLITS else ""
        out_path = (
            Path(paths["results"]) / "metrics"
            / f"{args.model}_{args.split}{bs_part}_{args.features}_seed{args.seed}{suffix}.json"
        )
    save_metrics(metrics, Path(out_path), meta)
    print(f"\n=== RESULTS ===")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Binary F1: {metrics['binary_f1']:.4f}")
    print(f"Binary FPR: {metrics.get('binary_fpr', 0):.4f}")
    print(f"Train time: {metrics['train_time_sec']:.1f}s")
    print(f"Inference throughput: {metrics['inference_throughput_per_sec']:.0f} flows/sec")
    print(f"Saved: {out_path}")
    return metrics


if __name__ == "__main__":
    main()
