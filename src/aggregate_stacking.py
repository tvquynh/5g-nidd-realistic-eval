"""Aggregate the stacked-ensemble sweep into paper-ready tables.

Answers three questions about fusion:

1.  Does the stack beat its own best member, and is the difference larger than
    seed noise? Reported as a paired Wilcoxon signed-rank test over the seeds,
    together with the mean gap.
2.  Where does each fusion rule put its trust? Reported as the convex weights
    for the weighted rule and as the normalized coefficient mass for the
    regression-based rules.
3.  Why does fusion fail to exploit the member that transfers best? Reported as
    the rank correlation between the out-of-fold ordering of the bases, which
    is what the meta-learner is fitted on, and their ordering on the shifted
    test set, which is what deployment sees.

Usage:
    python -m src.aggregate_stacking
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

GROUP_KEYS = ["base_set", "meta", "split", "train_bs", "features"]


def load_records(directory: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(directory.glob("stack_*.json")):
        record = json.loads(file.read_text())
        diagnostics = record.get("diagnostics", {})
        row = {
            "base_set": record["base_set"],
            "meta": record["meta"],
            "split": record["split"],
            "train_bs": record.get("train_bs") or 0,
            "features": record["features"],
            "seed": record["seed"],
            "macro_f1": record["macro_f1"],
            "binary_f1": record.get("binary_f1"),
            "binary_recall": record.get("binary_recall"),
            "binary_fpr": record.get("binary_fpr"),
            "train_time_sec": record.get("train_time_sec"),
            "stack_minus_best_base": diagnostics.get("stack_minus_best_base"),
            "best_base_on_test": diagnostics.get("best_base_on_test"),
            "spearman_oof_vs_test": diagnostics.get("spearman_oof_vs_test_rank"),
        }
        best = diagnostics.get("best_base_on_test")
        per_base = diagnostics.get("test_macro_f1_per_base", {})
        row["best_base_macro_f1"] = per_base.get(best) if best else None
        for name, value in per_base.items():
            row[f"base_test_f1__{name}"] = value
        for name, value in (diagnostics.get("oof_macro_f1_per_base") or {}).items():
            row[f"base_oof_f1__{name}"] = value
        for name, value in (diagnostics.get("convex_weights") or {}).items():
            row[f"weight__{name}"] = value
        for name, value in (diagnostics.get("meta_coefficient_mass") or {}).items():
            row[f"coef_mass__{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(values: np.ndarray) -> Dict:
    """Wilcoxon signed-rank test of a paired difference against zero.

    With ten seeds the smallest attainable two-sided p-value is 0.001953, so a
    result at that value is the strongest the test can express at this sample
    size rather than an unusually strong effect.
    """
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    n = len(values)
    if n < 6 or np.allclose(values, 0):
        return {"n": int(n), "p_value": None, "note": "insufficient or degenerate sample"}
    statistic, p_value = wilcoxon(values, alternative="two-sided")
    spread = values.std(ddof=1)
    return {
        "n": int(n),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean": float(values.mean()),
        "std": float(spread),
        "cohens_d": float(values.mean() / spread) if spread > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "stacking"
    frame = load_records(directory)
    if frame.empty:
        raise SystemExit(f"no stacking records under {directory}")

    print(f"Loaded {len(frame)} records from {directory}")
    print(f"Configurations: {frame[GROUP_KEYS].drop_duplicates().shape[0]}")

    summary_rows: List[Dict] = []
    for keys, group in frame.groupby(GROUP_KEYS, dropna=False):
        record = dict(zip(GROUP_KEYS, keys))
        record["n_seeds"] = int(len(group))
        record["macro_f1_mean"] = float(group["macro_f1"].mean())
        record["macro_f1_std"] = float(group["macro_f1"].std(ddof=1)) if len(group) > 1 else 0.0
        record["best_base_macro_f1_mean"] = float(group["best_base_macro_f1"].mean())
        record["stack_minus_best_mean"] = float(group["stack_minus_best_base"].mean())
        record["spearman_oof_vs_test_mean"] = float(group["spearman_oof_vs_test"].mean())
        record["train_time_sec_mean"] = float(group["train_time_sec"].mean())

        test = paired_test(group["stack_minus_best_base"].values)
        record["wilcoxon_p_stack_vs_best_base"] = test.get("p_value")
        record["cohens_d_stack_vs_best_base"] = test.get("cohens_d")

        # Which base the fusion actually leaned on, averaged over seeds.
        for column in group.columns:
            if column.startswith(("weight__", "coef_mass__")):
                value = group[column].mean()
                if pd.notna(value):
                    record[f"{column}_mean"] = float(value)
        # How often each base was the strongest on the shifted test set.
        record["best_base_mode"] = group["best_base_on_test"].mode().iat[0]
        summary_rows.append(record)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["split", "train_bs", "base_set", "macro_f1_mean"], ascending=[True, True, True, False]
    )

    display_columns = [
        "split", "train_bs", "base_set", "meta", "n_seeds",
        "macro_f1_mean", "macro_f1_std", "best_base_macro_f1_mean",
        "stack_minus_best_mean", "wilcoxon_p_stack_vs_best_base",
        "spearman_oof_vs_test_mean", "best_base_mode",
    ]
    print("\n=== Stack versus its own best member ===")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary[display_columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    weight_columns = sorted(c for c in summary.columns if c.startswith("weight__"))
    if weight_columns:
        print("\n=== Convex weights placed on each base (weighted fusion) ===")
        weighted = summary[summary["meta"] == "weighted"]
        with pd.option_context("display.width", 200):
            print(weighted[["split", "train_bs", "base_set"] + weight_columns].to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))

    mass_columns = sorted(c for c in summary.columns if c.startswith("coef_mass__"))
    if mass_columns:
        print("\n=== Coefficient mass per base (regression fusion) ===")
        regression = summary[summary["meta"] == "lr"]
        with pd.option_context("display.width", 200):
            print(regression[["split", "train_bs", "base_set"] + mass_columns].to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))

    prefix = Path(args.out_prefix) if args.out_prefix else results / "stacking_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}.csv", index=False)
    print(f"\nSaved: {prefix}.csv and {prefix}_per_seed.csv")


if __name__ == "__main__":
    main()
