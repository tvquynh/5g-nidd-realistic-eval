"""Aggregate latency, model-size, and memory profiles into paper-ready tables.

Produces the deployment table: per-sample
latency at the median *and* at the 95th and 99th percentiles, alongside the
model complexity metrics that a median latency figure alone does not convey ---
parameter count, serialized size, and peak resident memory during inference.

It also summarizes the XGBoost input-conversion breakdown, which explains the
single-flow latency gap between two histogram-based gradient-boosting
implementations that would otherwise look like a difference in model cost.

Usage:
    python -m src.aggregate_latency
    python -m src.aggregate_latency --batch 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

MEGABYTE = 1024.0 * 1024.0


def load_records(directory: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(directory.glob("latency_*.json")):
        record = json.loads(file.read_text())
        complexity = record.get("complexity", {})
        memory = record.get("memory_probe", {})
        label = record["model"]
        if record.get("stack_base_set"):
            label = f"stack[{record['stack_base_set']}/{record['stack_meta']}]"
        for entry in record.get("latency_sweep", []):
            rows.append({
                "model": label,
                "features": record["features"],
                "n_features": record.get("n_features"),
                "split": record["split"],
                "train_bs": record.get("train_bs") or 0,
                "seed": record["seed"],
                "batch_size": entry["batch_size"],
                "p50_us": entry["p50_per_sample_us"],
                "p95_us": entry["p95_per_sample_us"],
                "p99_us": entry["p99_per_sample_us"],
                "tail_ratio": entry["tail_ratio_p99_p50"],
                "throughput_p50": entry["throughput_p50_per_sec"],
                "n_repeats": entry["n_repeats"],
                "param_count": complexity.get("param_count"),
                "size_mb": complexity.get("size_native_mb"),
                "n_trees": complexity.get("n_trees"),
                "peak_rss_mb": (memory.get("peak_rss_bytes") / MEGABYTE
                                if memory.get("peak_rss_bytes") else None),
                "fit_seconds": record.get("fit_seconds"),
            })
    return pd.DataFrame(rows)


def load_overhead(directory: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(directory.glob("latency_xgboost_*.json")):
        record = json.loads(file.read_text())
        for batch, block in (record.get("xgboost_overhead") or {}).items():
            inplace = block.get("inplace_predict") or {}
            rows.append({
                "features": record["features"],
                "seed": record["seed"],
                "batch_size": int(batch),
                "dmatrix_p50_ms": block["dmatrix_construction"]["p50_ms"],
                "predict_p50_ms": block["booster_predict"]["p50_ms"],
                "end_to_end_p50_ms": block["end_to_end"]["p50_ms"],
                "inplace_p50_ms": inplace.get("p50_ms"),
                "dmatrix_share": block["dmatrix_share_of_end_to_end"],
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="show the detailed per-model table for this batch size only")
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "latency"

    frame = load_records(directory)
    if frame.empty:
        raise SystemExit(f"no latency records under {directory}")
    print(f"Loaded {len(frame)} (model, batch, seed) rows from {directory}")

    group_keys = ["model", "features", "batch_size"]
    summary = frame.groupby(group_keys).agg(
        n_seeds=("seed", "nunique"),
        p50_us_mean=("p50_us", "mean"),
        p95_us_mean=("p95_us", "mean"),
        p99_us_mean=("p99_us", "mean"),
        p99_us_max=("p99_us", "max"),
        tail_ratio_mean=("tail_ratio", "mean"),
        throughput_p50_mean=("throughput_p50", "mean"),
        param_count=("param_count", "mean"),
        size_mb=("size_mb", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).reset_index()

    batches = args.batch if args.batch else sorted(summary["batch_size"].unique())
    if isinstance(batches, int):
        batches = [batches]

    for batch in batches:
        block = summary[summary["batch_size"] == batch].sort_values("p50_us_mean")
        if block.empty:
            continue
        print(f"\n=== Batch {batch}: per-sample latency in microseconds ===")
        columns = ["model", "features", "n_seeds", "p50_us_mean", "p95_us_mean",
                   "p99_us_mean", "tail_ratio_mean", "throughput_p50_mean"]
        with pd.option_context("display.width", 220):
            print(block[columns].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n=== Model complexity (independent of batch size) ===")
    complexity = summary.groupby(["model", "features"]).agg(
        param_count=("param_count", "mean"),
        size_mb=("size_mb", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        fit_seconds=("fit_seconds_mean", "mean"),
    ).reset_index().sort_values("param_count")
    with pd.option_context("display.width", 220):
        print(complexity.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    overhead = load_overhead(directory)
    if not overhead.empty:
        print("\n=== XGBoost: input conversion versus tree evaluation ===")
        overhead_summary = overhead.groupby(["features", "batch_size"]).agg(
            dmatrix_p50_ms=("dmatrix_p50_ms", "mean"),
            predict_p50_ms=("predict_p50_ms", "mean"),
            end_to_end_p50_ms=("end_to_end_p50_ms", "mean"),
            inplace_p50_ms=("inplace_p50_ms", "mean"),
            dmatrix_share=("dmatrix_share", "mean"),
        ).reset_index()
        with pd.option_context("display.width", 220):
            print(overhead_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    prefix = Path(args.out_prefix) if args.out_prefix else results / "latency_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}.csv", index=False)
    complexity.to_csv(f"{prefix}_complexity.csv", index=False)
    if not overhead.empty:
        overhead.to_csv(f"{prefix}_xgboost_overhead.csv", index=False)
    print(f"\nSaved: {prefix}.csv, {prefix}_complexity.csv")


if __name__ == "__main__":
    main()
