"""Paired tests for the open-set rule comparisons.

Every rule in a cell is scored on the same fitted base classifier, so seeds pair
exactly and a signed-rank test over them is the right instrument. Reports the
best rule against each alternative, per base classifier, with the effect size.

Usage:
    python -m src.openset_stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from scipy import stats as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.make_openset_tables import BASE_LABELS, BASE_ORDER, RULE_LABELS
from src.run_experiment import load_paths

METRICS = ("macro_f1_open", "novel_recall")
ALTERNATIVES = ("none", "msp", "mahalanobis")
REFERENCE = "knn"


def compare(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for base in BASE_ORDER:
        block = frame[frame.base_model == base]
        if block.empty:
            continue
        reference = block[block.method == REFERENCE].set_index("seed")[metric]
        for rule in ALTERNATIVES:
            other = block[block.method == rule].set_index("seed")[metric]
            common = sorted(reference.index.intersection(other.index))
            if len(common) < 2:
                continue
            a, b = reference.loc[common], other.loc[common]
            difference = a - b
            if (difference == 0).all():
                statistic, p = float("nan"), 1.0
            else:
                statistic, p = sp.wilcoxon(a, b)
            rows.append({
                "base_model": base, "reference": REFERENCE, "alternative": rule,
                "metric": metric, "seeds": len(common),
                "reference_mean": float(a.mean()), "alternative_mean": float(b.mean()),
                "delta": float(difference.mean()),
                "wins": int((difference > 0).sum()),
                # Seeds pair exactly under the shared-fit protocol, so the
                # paired estimator d_z is the one that matches the test. The
                # unpaired pooled-SD formula in src.stats is for designs where
                # the two samples are independent.
                "p_value": float(p),
                "cohens_dz": float(difference.mean() / difference.std(ddof=1))
                if difference.std(ddof=1) > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    frame = pd.read_csv(results / "openset_summary_per_seed.csv")

    tables = []
    for metric in METRICS:
        table = compare(frame, metric)
        tables.append(table)
        print(f"\n=== {REFERENCE.upper()} against each alternative, {metric} ===")
        for _, r in table.iterrows():
            print(f"  {BASE_LABELS[r.base_model]:<16} vs {RULE_LABELS[r.alternative]:<12} "
                  f"n={int(r.seeds):>2}  delta {r.delta:+.4f}  "
                  f"wins {int(r.wins)}/{int(r.seeds)}  "
                  f"p={r.p_value:.4f}  dz={r.cohens_dz:+.2f}")

    combined = pd.concat(tables, ignore_index=True)
    out = Path(args.out) if args.out else results / "openset_stats.csv"
    combined.to_csv(out, index=False)

    # The two statements the manuscript makes from this file.
    f1 = combined[combined.metric == "macro_f1_open"]
    print(f"\nKNN-OOD beats every alternative on mean open-set macro-F1: "
          f"{bool((f1.delta > 0).all())}")
    worst = f1.loc[f1.p_value.idxmax()]
    print(f"Largest p-value over all {len(f1)} comparisons: {worst.p_value:.4f} "
          f"({BASE_LABELS[worst.base_model]} vs {RULE_LABELS[worst.alternative]})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
