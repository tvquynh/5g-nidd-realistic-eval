"""Aggregate the probability-space energy temperature campaign.

Two questions the grid cannot answer. Does the energy score reduce to MSP over
the whole range of temperatures a practitioner might pick, or only in the limit?
And what does the textbook unit-temperature substitution actually produce?

Usage:
    python -m src.aggregate_openset_temperature
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
               "tabnet": "TabNet", "ftt": "FT-Transformer"}
DEGENERATE_SCALE = 1e-6


def load(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        for rule, metrics in record["rules"].items():
            low, high = metrics["score_train_range"]
            rows.append({
                "base_model": record["base_model"], "seed": record["seed"],
                "rule": rule,
                "temperature": None if rule == "msp" else float(rule.split("T")[1]),
                "max_abs_score": max(abs(low), abs(high)),
                **{k: v for k, v in metrics.items() if k != "score_train_range"},
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "openset_temperature"
    frame = load(directory)
    print(f"Loaded {len(frame)} records from {directory}")
    print(f"Seeds per base: {frame.groupby('base_model').seed.nunique().to_dict()}\n")

    summary = (frame.groupby(["base_model", "rule"])
               .agg(seeds=("seed", "nunique"),
                    novel_recall=("novel_recall", "mean"),
                    novel_recall_std=("novel_recall", "std"),
                    macro_f1_open=("macro_f1_open", "mean"),
                    flag_rate=("flag_rate", "mean"),
                    max_abs_score=("max_abs_score", "max"),
                    distinct_scores_min=("score_train_unique", "min"))
               .reindex([(b, r) for b in BASE_ORDER
                         for r in ["msp", "energy_T0.05", "energy_T0.1",
                                   "energy_T0.25", "energy_T0.5", "energy_T1"]])
               .dropna(how="all"))
    print("=== Temperature grid ===")
    print(summary.round(6).to_string())

    print("\n=== Claim: below unit temperature the rule is MSP ===")
    worst_exact, worst_any = 0.0, 0.0
    for base in frame.base_model.unique():
        block = frame[frame.base_model == base]
        reference = block[block.rule == "msp"].set_index("seed").novel_recall
        for rule in sorted(set(block.rule) - {"msp"}):
            temperature = float(rule.split("T")[1])
            if temperature >= 1.0:
                continue
            other = block[block.rule == rule].set_index("seed").novel_recall
            common = reference.index.intersection(other.index)
            delta = float((reference.loc[common] - other.loc[common]).abs().max())
            worst_any = max(worst_any, delta)
            if temperature <= 0.25:
                worst_exact = max(worst_exact, delta)
    print(f"  max |energy - MSP| novel-recall, T <= 0.25: {worst_exact:.3e}")
    print(f"  max |energy - MSP| novel-recall, any T < 1:  {worst_any:.3e}")
    print("  verdict:", "identical at T <= 0.25" if worst_exact == 0.0
          else "NOT identical, check the implementation")

    print("\n=== Claim: at unit temperature the score is numerically degenerate ===")
    for base in BASE_ORDER:
        key = (base, "energy_T1")
        if key not in summary.index:
            continue
        row = summary.loc[key]
        msp = summary.loc[(base, "msp")]
        degenerate = row.max_abs_score < DEGENERATE_SCALE
        print(f"  {BASE_LABELS[base]:<16} max|s| {row.max_abs_score:.2e} "
              f"(MSP {msp.max_abs_score:.2e}), "
              f"fewest distinct values {int(row.distinct_scores_min):,}, "
              f"flags {row.flag_rate:.4f}, novel-recall {row.novel_recall:.4f} "
              f"-> {'DEGENERATE' if degenerate else 'informative'}")

    prefix = (Path(args.out_prefix) if args.out_prefix
              else results / "openset_temperature_summary")
    summary.reset_index().to_csv(f"{prefix}.csv", index=False)
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    print(f"\nSaved: {prefix}.csv and {prefix}_per_seed.csv")


if __name__ == "__main__":
    main()
