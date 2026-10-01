"""Aggregate per-attack rejectability and the operating-point sweep.

Two questions the aggregate grid cannot answer: which novel attack types a
rejection rule actually catches, and whether the 95th-percentile operating
point is doing the work. Both are computed from one base fit per cell, so
neither is contaminated by training variance.

Usage:
    python -m src.aggregate_openset_detail
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

RULE_ORDER = ["msp", "energy", "mahalanobis", "knn"]
RULE_LABELS = {
    "msp": "MSP", "energy": "Energy",
    "mahalanobis": "Mahalanobis", "knn": "KNN-OOD",
}
BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
               "tabnet": "TabNet", "ftt": "FT-Transformer"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "openset_detail"
    records = [json.loads(f.read_text()) for f in sorted(directory.glob("*.json"))]
    print(f"Loaded {len(records)} records from {directory}\n")

    # --- per-attack rejectability -----------------------------------------
    rows = []
    for r in records:
        for rule, per_attack in r["per_attack"].items():
            for attack, recall in per_attack.items():
                rows.append({"base_model": r["base_model"], "seed": r["seed"],
                             "rule": rule, "attack": attack, "recall": recall})
    per_attack = pd.DataFrame(rows)

    print("=== Rejectability by held-out attack type (mean over seeds) ===")
    pivot = (per_attack.pivot_table(index=["base_model", "rule"], columns="attack",
                                    values="recall", aggfunc="mean")
             .reindex([(b, r) for b in BASE_ORDER for r in RULE_ORDER]))
    print(pivot.round(3).to_string())

    print("\n=== Same, pooled over bases: which attacks are rejectable ===")
    pooled = (per_attack[per_attack.rule != "none"]
              .groupby(["attack", "rule"])["recall"].mean().unstack()
              .reindex(columns=RULE_ORDER))
    pooled.columns = [RULE_LABELS[c] for c in pooled.columns]
    print(pooled.round(3).to_string())

    # --- operating-point sweep --------------------------------------------
    sweep_rows = []
    for r in records:
        for rule, sweep in r["threshold_sweep"].items():
            for q, metrics in sweep.items():
                sweep_rows.append({"base_model": r["base_model"], "seed": r["seed"],
                                   "rule": rule, "quantile": float(q), **metrics})
    sweep = pd.DataFrame(sweep_rows)

    print("\n=== Operating-point sweep: open-set macro-F1 by quantile ===")
    table = (sweep.pivot_table(index=["base_model", "rule"], columns="quantile",
                               values="macro_f1_open", aggfunc="mean")
             .reindex([(b, r) for b in BASE_ORDER for r in RULE_ORDER]))
    table["spread"] = table.max(axis=1) - table.min(axis=1)
    table["best_q"] = table.drop(columns="spread").idxmax(axis=1)
    print(table.round(4).to_string())

    worst = table["spread"].max()
    print(f"\nLargest spread across quantiles 0.90-0.99, any base and rule: {worst:.4f}")
    print("The operating point is not load-bearing."
          if worst < 0.05 else "The operating point matters and must be tuned.")

    prefix = Path(args.out_prefix) if args.out_prefix else results / "openset_detail_summary"
    pivot.to_csv(f"{prefix}_per_attack.csv")
    table.to_csv(f"{prefix}_sweep.csv")
    per_attack.to_csv(f"{prefix}_per_attack_per_seed.csv", index=False)
    print(f"\nSaved: {prefix}_per_attack.csv, {prefix}_sweep.csv")


if __name__ == "__main__":
    main()
