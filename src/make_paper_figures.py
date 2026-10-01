"""Generate the manuscript figures from the result files.

Style follows the project convention: plain academic figures, no decorative
elements, a single blue family for series colour, English axis labels, vector
output. Every figure is regenerated from ``results/`` so that no figure can
drift from the numbers in the tables.

Usage:
    python -m src.make_paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

# Four-tone blue family, darkest to lightest.
BLUE = ["#1F3864", "#2E5C9A", "#5B9BD5", "#A9C9E8"]
GREY = "#595959"

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "ftt": "FT-Transformer",
}
MODEL_ORDER = ["lightgbm", "xgboost", "rf", "lr", "mlp", "tabnet", "ftt"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------

def figure_direction_asymmetry(cells: pd.DataFrame, out: Path) -> None:
    """False-positive rate by transfer direction, per detector."""
    matched = cells[(cells.train_balance == "matched") & (cells.test_balance == "matched")]
    models = [m for m in MODEL_ORDER if m in set(matched.model)]
    fpr1, fpr2 = [], []
    for model in models:
        block = matched[matched.model == model]
        a = block[block.train_bs == 1]
        b = block[block.train_bs == 2]
        fpr1.append(a.binary_fpr_mean.iat[0] if not a.empty else np.nan)
        fpr2.append(b.binary_fpr_mean.iat[0] if not b.empty else np.nan)

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    ax.bar(x - width / 2, fpr1, width, label=r"BS1 $\rightarrow$ BS2", color=BLUE[2])
    ax.bar(x + width / 2, fpr2, width, label=r"BS2 $\rightarrow$ BS1", color=BLUE[0])

    for xi, value in zip(x - width / 2, fpr1):
        if np.isfinite(value):
            ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
    for xi, value in zip(x + width / 2, fpr2):
        if np.isfinite(value):
            ax.text(xi, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=30, ha="right")
    ax.set_ylabel("Binary false-positive rate")
    ax.set_ylim(0, 0.82)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, out)


def figure_factorial(cells: pd.DataFrame, out: Path) -> None:
    """Binary F1 against false-positive rate across the four design cells."""
    block = cells[(cells.train_bs == 2) & cells.train_balance.isin(["raw", "matched"])]
    models = [m for m in MODEL_ORDER if m in set(block.model)]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, metric, label in zip(
        axes, ["binary_f1_mean", "binary_fpr_mean"],
        ["Binary F1", "Binary false-positive rate"],
    ):
        x = np.arange(len(models))
        width = 0.2
        for offset, (tr, te, colour) in zip(
            [-1.5, -0.5, 0.5, 1.5],
            [("raw", "raw", BLUE[0]), ("raw", "matched", BLUE[1]),
             ("matched", "raw", BLUE[2]), ("matched", "matched", BLUE[3])],
        ):
            values = []
            for model in models:
                row = block[(block.model == model) & (block.train_balance == tr)
                            & (block.test_balance == te)]
                values.append(row[metric].iat[0] if not row.empty else np.nan)
            ax.bar(x + offset * width, values, width, color=colour,
                   label=f"train {tr}, score {te}")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, fontsize=6.5, loc="lower left", ncol=1)
    fig.suptitle("")
    _save(fig, out)


def figure_mechanism(shift_dir: Path, out: Path) -> None:
    """Effective feature count against macro-F1 loss, by transfer direction."""
    rows = []
    for file in sorted(shift_dir.glob("*.json")):
        record = json.loads(file.read_text())
        for model, entry in record["per_model"].items():
            rows.append({
                "model": model,
                "train_bs": record["train_bs"],
                "effective_features": entry["concentration_in_station"]["effective_features"],
                "macro_f1_drop": entry["macro_f1_drop"],
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  (no shift records; skipping mechanism figure)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, bs, title in zip(axes, [1, 2],
                             [r"BS1 $\rightarrow$ BS2 (mild)",
                              r"BS2 $\rightarrow$ BS1 (severe)"]):
        block = frame[frame.train_bs == bs]
        for colour, model in zip(BLUE + [GREY], [m for m in MODEL_ORDER if m in set(block.model)]):
            sub = block[block.model == model]
            ax.scatter(sub["effective_features"], sub["macro_f1_drop"], s=14, color=colour,
                       label=MODEL_LABELS[model], alpha=0.85, edgecolors="none")
        if len(block) > 2:
            coefficients = np.polyfit(block["effective_features"], block["macro_f1_drop"], 1)
            xs = np.linspace(block["effective_features"].min(), block["effective_features"].max(), 50)
            ax.plot(xs, np.polyval(coefficients, xs), color=GREY, linewidth=1.0, linestyle="--")
        ax.set_xlabel("Effective feature count")
        ax.set_title(title)
        ax.grid(linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1 loss")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    _save(fig, out)


def figure_split_comparison(metrics_dir: Path, out: Path) -> None:
    """Macro-F1 by protocol, with the cross-station column split by direction."""
    rows = []
    for file in sorted(metrics_dir.glob("*.json")):
        if file.name.endswith("_smoke.json"):
            continue
        record = json.loads(file.read_text())
        if "macro_f1" not in record or record.get("features") != "full":
            continue
        split = record.get("split")
        if split == "cross_station":
            column = f"Cross-station\nBS{int(record.get('train_bs') or 1)}"
        elif split in {"random", "temporal", "holdout_attack"}:
            column = {"random": "Random", "temporal": "Temporal",
                      "holdout_attack": "Holdout-attack"}[split]
        else:
            continue
        rows.append({"model": record["model"], "column": column, "macro_f1": record["macro_f1"]})

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  (no grid records; skipping split comparison)")
        return

    order = ["Random", "Temporal", "Cross-station\nBS1", "Cross-station\nBS2", "Holdout-attack"]
    models = [m for m in MODEL_ORDER if m in set(frame.model)]
    x = np.arange(len(order))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    palette = (BLUE * 3)[:len(models)]
    for i, (model, colour) in enumerate(zip(models, palette)):
        means, errors = [], []
        for column in order:
            values = frame[(frame.model == model) & (frame.column == column)].macro_f1
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std() if len(values) else np.nan)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=1.5, color=colour,
               edgecolor="white", linewidth=0.3, label=MODEL_LABELS[model],
               error_kw={"linewidth": 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="lower left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, out)



def figure_per_class(metrics_dir: Path, class_names: list, out: Path) -> None:
    """Per-class recall under both cross-station directions."""
    rows = []
    for file in sorted(metrics_dir.glob("*.json")):
        if file.name.endswith("_smoke.json"):
            continue
        record = json.loads(file.read_text())
        if record.get("split") != "cross_station" or record.get("features") != "full":
            continue
        per_class = record.get("per_class") or {}
        for index, stats in per_class.items():
            rows.append({
                "model": record["model"],
                "train_bs": int(record.get("train_bs") or 1),
                "class_index": int(index),
                "recall": stats["recall"],
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  (no cross-station records; skipping per-class figure)")
        return

    models = [m for m in MODEL_ORDER if m in set(frame.model)]
    n_classes = int(frame.class_index.max()) + 1
    labels = class_names if len(class_names) == n_classes else [str(i) for i in range(n_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    for ax, bs, title in zip(axes, [1, 2],
                             [r"BS1 $\rightarrow$ BS2", r"BS2 $\rightarrow$ BS1"]):
        grid = np.full((len(models), n_classes), np.nan)
        for i, model in enumerate(models):
            for c in range(n_classes):
                values = frame[(frame.model == model) & (frame.train_bs == bs)
                               & (frame.class_index == c)].recall
                if len(values):
                    grid[i, c] = values.mean()
        image = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=7)
        ax.set_title(title)
        for i in range(len(models)):
            for c in range(n_classes):
                if np.isfinite(grid[i, c]):
                    ax.text(c, i, f"{grid[i, c]:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if grid[i, c] > 0.55 else "black")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Recall")
    _save(fig, out)


def figure_feature_budget(metrics_dir: Path, out: Path) -> None:
    """Macro-F1 against inference time across feature budgets, per direction."""
    rows = []
    for file in sorted(metrics_dir.glob("*.json")):
        if file.name.endswith("_smoke.json"):
            continue
        record = json.loads(file.read_text())
        if record.get("split") != "cross_station" or "macro_f1" not in record:
            continue
        rows.append({
            "model": record["model"],
            "features": record.get("features", "full"),
            "train_bs": int(record.get("train_bs") or 1),
            "macro_f1": record["macro_f1"],
            "throughput": record.get("inference_throughput_per_sec", np.nan),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  (no records; skipping feature-budget figure)")
        return

    markers = {"full": "o", "top50": "s", "top20": "^", "top20_stable": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, bs, title in zip(axes, [1, 2],
                             [r"BS1 $\rightarrow$ BS2", r"BS2 $\rightarrow$ BS1"]):
        block = frame[frame.train_bs == bs]
        for colour, model in zip((BLUE * 3), [m for m in MODEL_ORDER if m in set(block.model)]):
            for subset, marker in markers.items():
                sub = block[(block.model == model) & (block.features == subset)]
                if sub.empty:
                    continue
                ax.scatter(sub["throughput"].mean(), sub["macro_f1"].mean(),
                           s=22, color=colour, marker=marker, edgecolors="none")
        ax.set_xscale("log")
        ax.set_xlabel("Inference throughput (flows/s)")
        ax.set_title(title)
        ax.grid(linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1")

    handles = [plt.Line2D([], [], marker=m, linestyle="", color=GREY, label=l)
               for l, m in markers.items()]
    axes[1].legend(handles=handles, frameon=False, fontsize=6.5, loc="lower right")
    _save(fig, out)

def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    figures = Path(paths["workspace"]) / "paper" / "figures"

    print("Generating manuscript figures from result files:")

    cells_path = results / "decomposition_summary_cells.csv"
    if cells_path.exists():
        cells = pd.read_csv(cells_path)
        figure_direction_asymmetry(cells, figures / "fig_direction.pdf")
        figure_factorial(cells, figures / "fig_factorial.pdf")

    shift_dir = results / "shift"
    if shift_dir.exists():
        figure_mechanism(shift_dir, figures / "fig_mechanism.pdf")

    grid_dir = results / "metrics_windows"
    if grid_dir.exists():
        figure_split_comparison(grid_dir, figures / "fig_split_comparison.pdf")
        class_path = Path(paths["data_out"]) / "master_5g_nidd.classes.json"
        class_names = json.loads(class_path.read_text()) if class_path.exists() else []
        figure_per_class(grid_dir, class_names, figures / "fig_per_class.pdf")
        figure_feature_budget(grid_dir, figures / "fig_feature_budget.pdf")

    print("done")


if __name__ == "__main__":
    main()
