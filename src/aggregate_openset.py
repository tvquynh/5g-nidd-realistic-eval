"""Aggregate the shared-fit open-set grid.

Every rejection rule is scored on the same fitted base classifier, so a
difference between two rules is a property of the rule. The script reports the
grid, checks the two claims the design was built to test, and writes the
reported tables.

Usage:
    python -m src.aggregate_openset
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

METHOD_ORDER = ["none", "msp", "energy", "mahalanobis", "knn"]
METHOD_LABELS = {
    "none": "no rejection",
    "msp": "maximum softmax probability",
    "energy": "energy score",
    "mahalanobis": "Mahalanobis",
    "knn": "deep nearest neighbour",
}
BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {
    "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "tabnet": "TabNet", "ftt": "FT-Transformer",
}
METRICS = ["macro_f1_open", "novel_recall", "false_unknown_rate"]


def load(directory: Path) -> pd.DataFrame:
    rows = [json.loads(f.read_text()) for f in sorted(directory.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame["method"] = pd.Categorical(frame["method"], METHOD_ORDER, ordered=True)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "openset_v2"
    frame = load(directory)
    print(f"Loaded {len(frame)} records from {directory}\n")

    summary = (frame.groupby(["base_model", "method"], observed=True)[METRICS]
               .agg(["mean", "std", "count"]).round(4))
    print("=== Open-set grid ===")
    for base in BASE_ORDER:
        if base not in set(frame.base_model):
            continue
        print(f"\n{BASE_LABELS[base]}")
        print(summary.loc[base].to_string())

    # Claim 1: without a rejection rule, nothing is flagged as novel.
    print("\n\n=== Claim 1: no architecture rejects novelty on its own ===")
    baseline = frame[frame.method == "none"]
    for base in BASE_ORDER:
        block = baseline[baseline.base_model == base]
        if block.empty:
            continue
        print(f"  {BASE_LABELS[base]:<16} novel-recall "
              f"mean {block.novel_recall.mean():.6f}  "
              f"max {block.novel_recall.max():.6f}  "
              f"({block.seed.nunique()} seeds)")

    # Claim 2: in probability space the energy score is the MSP rule.
    print("\n=== Claim 2: energy score equals MSP under a probability-only constraint ===")
    equal_everywhere = True
    for base in BASE_ORDER:
        block = frame[frame.base_model == base]
        msp = block[block.method == "msp"].set_index("seed")[METRICS]
        energy = block[block.method == "energy"].set_index("seed")[METRICS]
        common = msp.index.intersection(energy.index)
        if len(common) == 0:
            continue
        delta = (msp.loc[common] - energy.loc[common]).abs().max().max()
        identical = delta == 0.0
        equal_everywhere &= identical
        print(f"  {BASE_LABELS[base]:<16} {len(common)} seeds, "
              f"max |difference| = {delta:.3e}  -> "
              f"{'identical' if identical else 'DIFFERENT'}")
    print(f"  verdict: {'identical on every base and seed' if equal_everywhere else 'NOT identical'}")

    # Which rule wins, per base.
    print("\n=== Best rule per base, by open-set macro-F1 ===")
    for base in BASE_ORDER:
        block = frame[(frame.base_model == base) & (frame.method != "none")]
        if block.empty:
            continue
        means = block.groupby("method", observed=True).macro_f1_open.mean()
        best = means.idxmax()
        gain = means[best] - frame[(frame.base_model == base)
                                   & (frame.method == "none")].macro_f1_open.mean()
        recall = block[block.method == best].novel_recall.mean()
        print(f"  {BASE_LABELS[base]:<16} {METHOD_LABELS[best]:<28} "
              f"macro-F1-open {means[best]:.4f} (+{gain:.4f} over no rejection), "
              f"novel-recall {recall:.4f}")

    prefix = Path(args.out_prefix) if args.out_prefix else results / "openset_summary"
    flat = summary.copy()
    flat.columns = ["_".join(c) for c in flat.columns]
    flat.reset_index().to_csv(f"{prefix}.csv", index=False)
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    print(f"\nSaved: {prefix}.csv and {prefix}_per_seed.csv")


if __name__ == "__main__":
    main()
