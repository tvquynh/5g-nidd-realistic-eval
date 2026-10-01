"""
Generate figures + tables for paper from results/summary.parquet.

Output files (saved to results/figures/ and results/tables/):
- fig1_split_comparison.pdf      Macro F1 across 4 splits per model
- fig2_feature_set_tradeoff.pdf  Performance vs latency per feature set
- fig3_per_class_heatmap.pdf     Per-class recall under different splits
- fig4_cross_station.pdf         BS1->BS2 vs BS2->BS1 confusion
- table1_summary.tex             Mean ± CI per (model, split, features)
- table2_wilcoxon.tex            Pairwise Wilcoxon p-values
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


def fig_split_comparison(df: pd.DataFrame, out: Path):
    """Boxplot: macro_f1 per split per model."""
    fig, axes = plt.subplots(1, len(df["features"].unique()), figsize=(14, 5), sharey=True)
    if len(df["features"].unique()) == 1:
        axes = [axes]
    for ax, feat in zip(axes, sorted(df["features"].unique())):
        sub = df[df["features"] == feat]
        sns.boxplot(data=sub, x="split", y="macro_f1", hue="model", ax=ax)
        ax.set_title(f"Feature set: {feat}")
        ax.set_xlabel("")
        ax.set_ylabel("Macro F1")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_feature_tradeoff(df: pd.DataFrame, out: Path):
    """Scatter: macro_f1 vs inference latency per feature set, colored by model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = df[df["split"] == "random"]
    sns.scatterplot(
        data=sub, x="inference_time_sec", y="macro_f1",
        hue="model", style="features", s=80, ax=ax,
    )
    ax.set_xlabel("Inference time (sec on test set)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Performance vs latency trade-off (random split)")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_per_class_heatmap(df: pd.DataFrame, out: Path, classes_path: Path = None):
    """Heatmap: per-class recall, rows=class, cols=split."""
    classes = json.loads(classes_path.read_text()) if classes_path and classes_path.exists() else None
    main_model = df["model"].mode()[0] if len(df) else "lightgbm"
    sub = df[(df["model"] == main_model) & (df["features"] == "full")]
    if len(sub) == 0:
        return
    cls_cols = [c for c in sub.columns if c.startswith("class") and c.endswith("_recall")]
    if not cls_cols:
        return
    cls_cols = sorted(cls_cols, key=lambda x: int(x.replace("class", "").split("_")[0]))
    avg_per_split = sub.groupby("split")[cls_cols].mean().T
    if classes:
        avg_per_split.index = classes[: len(avg_per_split)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(avg_per_split, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Per-class recall by split ({main_model}, full features)")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def table_summary(df: pd.DataFrame, out: Path, metric: str = "macro_f1"):
    """LaTeX table: mean ± std per (model, split, features) with human-readable labels."""
    df = df.copy()
    split_pretty = {
        "random": "Random",
        "temporal": "Temporal",
        "cross_station": "Cross-station",
        "holdout_attack": "Holdout-attack",
    }
    feat_pretty = {
        "full": "full",
        "top20": "top-20",
        "top20_stable": "top-20 (stable)",
        "top50": "top-50",
    }
    df["split"] = df["split"].map(lambda s: split_pretty.get(s, s))
    df["features"] = df["features"].map(lambda f: feat_pretty.get(f, f))
    pivot = df.groupby(["model", "split", "features"])[metric].agg(["mean", "std"]).reset_index()
    pivot["cell"] = pivot.apply(lambda r: f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}", axis=1)
    table = pivot.pivot_table(index=["model", "features"], columns="split", values="cell", aggfunc="first")
    table.index.names = ["Model", "Features"]
    table.columns.name = "Split"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table.to_latex(escape=False, na_rep="--"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/summary.parquet")
    ap.add_argument("--out", dest="out_dir", default="results/figures")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--classes", default="data/master_5g_nidd.classes.json")
    args = ap.parse_args()
    df = pd.read_parquet(args.in_path)
    # Drop legacy ablation models that are not described in the manuscript.
    df = df[~df["model"].isin(["stability_lgbm"])].reset_index(drop=True)
    # Pretty model names for figures (match manuscript Table 7 / §4.2).
    pretty_names = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "rf": "RF",
        "lr": "LR",
        "mlp": "MLP",
        "tabnet": "TabNet",
        "ftt": "FTT",
    }
    df["model"] = df["model"].map(lambda m: pretty_names.get(m, m))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables)
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_split_comparison(df, out_dir / "fig1_split_comparison.pdf")
    fig_feature_tradeoff(df, out_dir / "fig2_feature_tradeoff.pdf")
    fig_per_class_heatmap(df, out_dir / "fig3_per_class_heatmap.pdf", Path(args.classes))
    table_summary(df, tables_dir / "table1_summary.tex")
    print(f"Figures -> {out_dir}")
    print(f"Tables -> {tables_dir}")


if __name__ == "__main__":
    main()
