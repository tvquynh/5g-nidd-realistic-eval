"""Compare two runs of the same experiment grid across environments.

Given two runs of the same grid on different machines, this script pairs them
cell by cell and reports the distribution of differences, so that stability
across environments can be stated from measurement rather than asserted.

Usage:
    python -m src.compare_environments
    python -m src.compare_environments --metric macro_f1 --top 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

KEY_FIELDS = ["model", "split", "train_bs", "features", "seed"]
COMPARED_METRICS = [
    "macro_f1",
    "weighted_f1",
    "binary_f1",
    "binary_recall",
    "binary_fpr",
    "macro_roc_auc",
    "macro_pr_auc",
]
STRUCTURAL_FIELDS = ["n_train", "n_test", "n_features", "num_classes"]


# Records that share the grid's key fields but are not grid cells: the
# full-data verification runs, the open-set sweep, and the auxiliary studies.
# Including them would collide on the merge key and duplicate paired rows.
NON_GRID_PREFIXES = ("fullsize_", "openset_", "sensitivity_", "stability_")


def load_dir(path: Path, label: str) -> pd.DataFrame:
    rows = []
    for file in sorted(path.glob("*.json")):
        if file.name.endswith("_smoke.json") or file.name.startswith(NON_GRID_PREFIXES):
            continue
        try:
            record = json.loads(file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "macro_f1" not in record:
            continue
        row = {field: record.get(field) for field in KEY_FIELDS}
        # Normalize the join keys: the two runs serialize an absent train_bs
        # differently (null vs 1.0), which would break the merge on dtype.
        row["train_bs"] = 0 if row.get("train_bs") in (None, "") else int(row["train_bs"])
        row["seed"] = int(row["seed"])
        row.update({field: record.get(field) for field in STRUCTURAL_FIELDS})
        row.update({metric: record.get(metric) for metric in COMPARED_METRICS})
        row["environment"] = label
        row["file"] = file.name
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None, help="directory of the published run")
    ap.add_argument("--candidate", default=None, help="directory of the re-run")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--top", type=int, default=10, help="largest differences to list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    reference_dir = Path(args.reference) if args.reference else results / "metrics"
    candidate_dir = Path(args.candidate) if args.candidate else results / "metrics_windows"

    reference = load_dir(reference_dir, "reference")
    candidate = load_dir(candidate_dir, "candidate")
    print(f"reference {reference_dir}: {len(reference)} cells")
    print(f"candidate {candidate_dir}: {len(candidate)} cells")
    if reference.empty or candidate.empty:
        raise SystemExit("one of the two directories has no comparable cells")

    merged = reference.merge(candidate, on=KEY_FIELDS, suffixes=("_ref", "_cand"))
    print(f"paired cells: {len(merged)}")
    unpaired_ref = len(reference) - len(merged)
    unpaired_cand = len(candidate) - len(merged)
    if unpaired_ref or unpaired_cand:
        print(f"  unpaired: {unpaired_ref} reference-only, {unpaired_cand} candidate-only")

    # Structural fields must agree exactly: a difference there means the two
    # runs did not evaluate the same data, which no tolerance would excuse.
    structural_problems = []
    for field in STRUCTURAL_FIELDS:
        mismatch = merged[merged[f"{field}_ref"] != merged[f"{field}_cand"]]
        if len(mismatch):
            structural_problems.append((field, len(mismatch)))
    if structural_problems:
        print("\nSTRUCTURAL MISMATCH (same cell, different data):")
        for field, count in structural_problems:
            print(f"  {field}: {count} cells differ")
    else:
        print("\nStructural fields identical in every paired cell "
              "(n_train, n_test, n_features, num_classes).")

    print(f"\n{'metric':<16}{'n':>6}{'mean |delta|':>15}{'max |delta|':>14}"
          f"{'p99 |delta|':>14}{'cells >1e-3':>13}")
    summary_rows = []
    for metric in COMPARED_METRICS:
        pair = merged[[f"{metric}_ref", f"{metric}_cand"]].dropna()
        if pair.empty:
            continue
        delta = (pair[f"{metric}_cand"] - pair[f"{metric}_ref"]).abs().values
        row = {
            "metric": metric,
            "n": len(delta),
            "mean_abs_delta": float(delta.mean()),
            "max_abs_delta": float(delta.max()),
            "p99_abs_delta": float(np.percentile(delta, 99)),
            "cells_above_1e-3": int((delta > 1e-3).sum()),
        }
        summary_rows.append(row)
        print(f"{metric:<16}{row['n']:>6}{row['mean_abs_delta']:>15.3e}"
              f"{row['max_abs_delta']:>14.3e}{row['p99_abs_delta']:>14.3e}"
              f"{row['cells_above_1e-3']:>13}")

    # Means over the ten seeds, not individual cells.
    # A per-seed difference that flips an early-stopping iteration can be large
    # while the quantity actually published barely moves, so the aggregate
    # comparison is the one that decides whether conclusions are affected.
    cell_keys = ["model", "split", "train_bs", "features"]
    aggregate_rows = []
    print()
    print(f"{'metric':<16}{'cells':>7}{'mean |delta of means|':>24}"
          f"{'max |delta of means|':>23}{'cells >1e-3':>13}")
    for metric in COMPARED_METRICS:
        subset = merged[cell_keys + [f"{metric}_ref", f"{metric}_cand"]].dropna()
        if subset.empty:
            continue
        grouped = subset.groupby(cell_keys, dropna=False).agg(
            ref_mean=(f"{metric}_ref", "mean"),
            cand_mean=(f"{metric}_cand", "mean"),
            n_seeds=(f"{metric}_ref", "size"),
        ).reset_index()
        # Only cells with the full seed set are comparable as means. The
        # required count is the modal one rather than the maximum, so that a
        # single over-populated cell cannot define the threshold.
        required = int(grouped["n_seeds"].mode().iat[0])
        dropped = int((grouped["n_seeds"] != required).sum())
        grouped = grouped[grouped["n_seeds"] == required]
        if dropped:
            print(f"  ({metric}: {dropped} cells excluded, seed count != {required})")
        if grouped.empty:
            continue
        delta = (grouped["cand_mean"] - grouped["ref_mean"]).abs().values
        row = {
            "metric": metric,
            "n_cells": int(len(grouped)),
            "seeds_per_cell": int(grouped["n_seeds"].max()),
            "mean_abs_delta_of_means": float(delta.mean()),
            "max_abs_delta_of_means": float(delta.max()),
            "cells_above_1e-3": int((delta > 1e-3).sum()),
        }
        aggregate_rows.append(row)
        print(f"{metric:<16}{row['n_cells']:>7}{row['mean_abs_delta_of_means']:>24.3e}"
              f"{row['max_abs_delta_of_means']:>23.3e}{row['cells_above_1e-3']:>13}")

    metric = args.metric
    merged["abs_delta"] = (merged[f"{metric}_cand"] - merged[f"{metric}_ref"]).abs()
    worst = merged.nlargest(args.top, "abs_delta")
    print(f"\nLargest {args.top} differences in {metric}:")
    for _, row in worst.iterrows():
        bs = f" bs{int(row['train_bs'])}" if row["train_bs"] else ""
        print(f"  {row['model']:<10} {row['split']}{bs:<4} {row['features']:<13} "
              f"seed {int(row['seed']):<5} "
              f"{row[f'{metric}_ref']:.6f} -> {row[f'{metric}_cand']:.6f} "
              f"({row[f'{metric}_cand'] - row[f'{metric}_ref']:+.2e})")

    out_path = Path(args.out) if args.out else results / "environment_comparison.json"
    out_path.write_text(json.dumps({
        # Directory names only: absolute paths identify the machine and
        # must not travel with a released artifact.
        "reference_dir": reference_dir.name,
        "candidate_dir": candidate_dir.name,
        "n_reference": int(len(reference)),
        "n_candidate": int(len(candidate)),
        "n_paired": int(len(merged)),
        "structural_mismatches": {field: count for field, count in structural_problems},
        "per_metric_per_seed": summary_rows,
        "per_metric_seed_means": aggregate_rows,
    }, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
