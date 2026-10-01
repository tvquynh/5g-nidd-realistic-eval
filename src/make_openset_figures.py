"""Generate the reported figures from the result files.

Plain academic style, single blue family, English axis labels, vector output.

Usage:
    python -m src.make_openset_figures
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
import os

BLUE = ["#1F3864", "#2E5C9A", "#5B9BD5", "#A9C9E8"]
GREY = "#595959"
BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
               "tabnet": "TabNet", "ftt": "FT-Transformer"}
RULE_ORDER = ["none", "msp", "mahalanobis", "knn"]
RULE_LABELS = {"none": "no rejection", "msp": "MSP / Energy",
               "mahalanobis": "Mahalanobis", "knn": "KNN-OOD"}

# cas-dc column and text widths, in inches.
COLUMN_WIDTH = 3.35
TEXT_WIDTH = 7.0

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight",
})


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"  wrote {out.name}")


def figure_novel_recall(grid: pd.DataFrame, out: Path) -> None:
    """Novel-attack recall by rule and base. Without a rule it is exactly zero."""
    stats = grid.groupby(["base_model", "method"])["novel_recall"].agg(["mean", "std"])
    x = np.arange(len(BASE_ORDER))
    width = 0.2

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    for i, (rule, colour) in enumerate(zip(RULE_ORDER, BLUE)):
        means, errors = [], []
        for base in BASE_ORDER:
            key = (base, rule)
            means.append(stats.loc[key, "mean"] if key in stats.index else np.nan)
            errors.append(stats.loc[key, "std"] if key in stats.index else np.nan)
        offset = (i - (len(RULE_ORDER) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=2,
               color=colour, label=RULE_LABELS[rule],
               error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([BASE_LABELS[b].replace("FT-", "FT-\n") for b in BASE_ORDER])
    ax.set_ylabel("Novel-attack recall")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, ncol=2, loc="upper center", fontsize=6.5,
              handlelength=1.2, columnspacing=1.0, borderpad=0.1)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
    ax.set_axisbelow(True)
    # A bar of height exactly zero draws nothing, which reads as missing data
    # rather than as the result. Mark the slot and label it.
    offset = -1.5 * width
    for xi in x:
        ax.plot([xi + offset - width / 2, xi + offset + width / 2], [0, 0],
                color=BLUE[0], linewidth=2.2, solid_capstyle="butt", zorder=3)
        ax.text(xi + offset, 0.03, "0", ha="center", fontsize=6,
                color=BLUE[0])
    _save(fig, out)


def figure_per_attack(per_attack: pd.DataFrame, out: Path) -> None:
    """Rejectability by held-out attack type: one class resists every rule."""
    order = ["SYNFlood", "TCPConnectScan", "SlowrateDoS", "Benign (false unknown)"]
    labels = ["SYN flood", "TCP connect\nscan", "Slow-rate\nDoS",
              "Benign\n(false alarm)"]
    rules = ["msp", "mahalanobis", "knn"]
    # Base classifiers weighted equally, matching every per-attack number in
    # the reported tables; see table_per_attack.
    pooled = (per_attack[per_attack.rule.isin(rules)]
              .groupby(["attack", "rule", "base_model"])["recall"].mean()
              .groupby(["attack", "rule"]).mean().unstack())

    x = np.arange(len(order))
    width = 0.26
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    for i, (rule, colour) in enumerate(zip(rules, [BLUE[0], BLUE[2], BLUE[1]])):
        values = [pooled.loc[a, rule] if a in pooled.index else np.nan for a in order]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, color=colour,
                      label=RULE_LABELS[rule])
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                        f"{value:.2f}", ha="center", fontsize=5.5, color=GREY)

    ax.axvspan(1.5, 2.5, color=BLUE[3], alpha=0.22, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Fraction flagged as unknown")
    ax.set_ylim(0, 1.22)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6,
              handlelength=1.2, columnspacing=0.9, borderpad=0.1)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, out)


def figure_temperature(temperature: pd.DataFrame, out: Path) -> None:
    """Probability-space energy across temperatures: MSP everywhere, then collapse."""
    bases = [b for b in BASE_ORDER if b in set(temperature.base_model)]
    stats = temperature.groupby(["base_model", "rule"]).agg(
        nr=("novel_recall", "mean"), max_abs=("max_abs_score", "max"))
    temps = sorted(float(r.split("T")[1])
                   for r in set(temperature.rule) if r != "msp")
    markers = ["o", "s", "^", "D"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5))
    fig.subplots_adjust(wspace=0.24)
    for base, colour, marker in zip(bases, BLUE, markers):
        recall = [stats.loc[(base, f"energy_T{t:g}"), "nr"] for t in temps]
        scale = [stats.loc[(base, f"energy_T{t:g}"), "max_abs"] for t in temps]
        left.plot(temps, recall, marker=marker, markersize=3.5, linewidth=1.1,
                  color=colour, label=BASE_LABELS[base])
        left.axhline(stats.loc[(base, "msp"), "nr"], color=colour,
                     linestyle=":", linewidth=0.8)
        right.plot(temps, scale, marker=marker, markersize=3.5, linewidth=1.1,
                   color=colour, label=BASE_LABELS[base])

    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("Temperature $T$")
        ax.set_xticks(temps)
        ax.set_xticklabels([f"{t:g}" for t in temps])
        ax.grid(linestyle=":", linewidth=0.5, color=GREY, alpha=0.5)
        ax.set_axisbelow(True)
    left.set_ylabel("Novel-attack recall")
    left.set_ylim(0, 0.8)
    left.legend(frameon=False, loc="lower left", ncol=1)
    right.set_yscale("log")
    right.set_ylabel(r"$\max|s|$ on calibration set")
    right.axhspan(1e-13, 1e-6, color=BLUE[3], alpha=0.25, zorder=0)
    right.text(0.055, 3e-11, "numerically\ndegenerate", fontsize=7, color=GREY)
    _save(fig, out)


def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    figures = Path(os.environ.get("NIDD_FIGURES_DIR",
                     Path(paths["workspace"]) / "artifacts" / "figures"))
    print("Generating open-set figures:")

    grid = pd.read_csv(results / "openset_summary_per_seed.csv")
    figure_novel_recall(grid, figures / "fig_novel_recall.pdf")

    per_attack = pd.read_csv(results / "openset_detail_summary_per_attack_per_seed.csv")
    figure_per_attack(per_attack, figures / "fig_per_attack.pdf")

    rows = []
    for f in sorted((results / "openset_temperature").glob("*.json")):
        record = json.loads(f.read_text())
        for rule, metrics in record["rules"].items():
            low, high = metrics["score_train_range"]
            rows.append({"base_model": record["base_model"], "rule": rule,
                         "novel_recall": metrics["novel_recall"],
                         "max_abs_score": max(abs(low), abs(high))})
    figure_temperature(pd.DataFrame(rows), figures / "fig_temperature.pdf")
    print("done")


if __name__ == "__main__":
    main()
