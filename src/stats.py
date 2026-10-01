"""
Statistical tests: Wilcoxon signed-rank, Cohen's d, 95% CI on summary parquet.

Usage:
    python -m src.stats --in results/summary.parquet --out results/stats.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else 0.0, 0.0, 0.0)
    se = sp.sem(x)
    h = se * sp.t.ppf(0.975, len(x) - 1)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def compare(df: pd.DataFrame, metric: str = "macro_f1"):
    """For each (model, feature) pair, compute Wilcoxon comparing splits."""
    out = []
    splits = df["split"].unique().tolist()
    for model in df["model"].unique():
        for feat in df["features"].unique():
            sub = df[(df["model"] == model) & (df["features"] == feat)]
            for s1 in splits:
                for s2 in splits:
                    if s1 >= s2:
                        continue
                    a = sub[sub["split"] == s1][metric].values
                    b = sub[sub["split"] == s2][metric].values
                    n = min(len(a), len(b))
                    if n < 3:
                        continue
                    a, b = a[:n], b[:n]
                    try:
                        w_stat, p_val = sp.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                    except Exception:
                        w_stat, p_val = np.nan, np.nan
                    d = cohens_d(a, b)
                    out.append({
                        "model": model, "features": feat,
                        "split_a": s1, "split_b": s2, "metric": metric,
                        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                        "diff": float(a.mean() - b.mean()),
                        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
                        "wilcoxon_p": float(p_val) if p_val == p_val else None,
                        "cohens_d": float(d),
                        "n_seeds": int(n),
                    })
    return out


def summary_table(df: pd.DataFrame, metric: str = "macro_f1"):
    """Mean ± CI95 per (model, split, features)."""
    rows = []
    for (model, split, feat), g in df.groupby(["model", "split", "features"]):
        m, lo, hi = ci95(g[metric].values)
        rows.append({
            "model": model, "split": split, "features": feat, "metric": metric,
            "mean": m, "ci95_lo": lo, "ci95_hi": hi, "n_seeds": int(len(g)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/summary.parquet")
    ap.add_argument("--out", default="results/stats.json")
    ap.add_argument("--metric", default="macro_f1")
    args = ap.parse_args()
    df = pd.read_parquet(args.in_path)
    out = {
        "metric": args.metric,
        "summary": summary_table(df, args.metric),
        "comparisons": compare(df, args.metric),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved stats: {args.out} ({len(out['summary'])} groups, {len(out['comparisons'])} comparisons)")


if __name__ == "__main__":
    main()
