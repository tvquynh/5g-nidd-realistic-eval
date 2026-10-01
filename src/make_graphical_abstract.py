"""Generate the graphical abstract.

The publisher asks for 531 x 1328 pixels (height x width) or proportionally
more, readable at 5 x 13 cm. The figure carries one message: the same detector,
the same dataset, two transfer directions, and a false-alarm rate that differs
by two orders of magnitude while the headline metric moves the wrong way.

Usage:
    python -m src.make_graphical_abstract
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

BLUE_DARK = "#1F3864"
BLUE_MID = "#2E5C9A"
BLUE_LIGHT = "#5B9BD5"
GREY = "#595959"

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    cells = pd.read_csv(results / "decomposition_summary_cells.csv")
    matched = cells[(cells.train_balance == "matched") & (cells.test_balance == "matched")]
    light = matched[matched.model.isin(["lightgbm", "xgboost", "rf", "lr", "mlp"])]

    fpr = {bs: light[light.train_bs == bs].binary_fpr_mean.mean() for bs in (1, 2)}
    f1 = {bs: light[light.train_bs == bs].binary_f1_mean.mean() for bs in (1, 2)}

    # 1328 x 531 px at 200 dpi -> 6.64 x 2.655 inches.
    fig, axes = plt.subplots(1, 3, figsize=(6.64, 2.655), dpi=200,
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.25]})

    # Panel 1: the two directions, drawn as a schematic.
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Same dataset, two directions", fontsize=8.5, pad=6)
    for y, (label, colour) in zip(
        (0.68, 0.28),
        ((r"BS1 $\rightarrow$ BS2", BLUE_LIGHT), (r"BS2 $\rightarrow$ BS1", BLUE_DARK)),
    ):
        ax.annotate("", xy=(0.86, y), xytext=(0.14, y),
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.2))
        ax.text(0.5, y + 0.09, label, ha="center", fontsize=8, color=colour)
    ax.text(0.5, 0.90, "benign-heavy station trains", ha="center", fontsize=6.8, color=GREY)
    ax.text(0.5, 0.06, "attack-heavy station trains", ha="center", fontsize=6.8, color=GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: false-positive rate, log scale, the real failure.
    ax = axes[1]
    bars = ax.bar([0, 1], [fpr[1], fpr[2]], width=0.55,
                  color=[BLUE_LIGHT, BLUE_DARK])
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BS1→BS2", "BS2→BS1"], fontsize=7.5)
    ax.set_ylabel("False-positive rate (log)", fontsize=8)
    ax.set_title("The detector really does fail", fontsize=8.5, pad=6)
    for bar, value in zip(bars, (fpr[1], fpr[2])):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.35, f"{value:.3f}",
                ha="center", fontsize=7.5, color=GREY)
    ax.tick_params(labelsize=7)
    ax.text(0.5, 0.93, f"{fpr[2] / fpr[1]:.0f}$\\times$ more false alarms",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=BLUE_DARK)

    # Panel 3: the metric ranks them the wrong way round.
    ax = axes[2]
    bars = ax.bar([0, 1], [f1[1], f1[2]], width=0.55, color=[BLUE_LIGHT, BLUE_DARK])
    ax.set_ylim(0.80, 0.98)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BS1→BS2", "BS2→BS1"], fontsize=7.5)
    ax.set_ylabel("Binary F1", fontsize=8)
    ax.set_title("But aggregate F1 prefers the failure", fontsize=8.5, pad=6)
    for bar, value in zip(bars, (f1[1], f1[2])):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}",
                ha="center", fontsize=7.5, color=GREY)
    ax.tick_params(labelsize=7)
    ax.text(0.5, 0.93,
            "ranked higher despite 120$\\times$ the false alarms",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=BLUE_MID)

    fig.tight_layout(pad=0.6)
    out = Path(paths["workspace"]) / "paper" / "graphical_abstract.pdf"
    fig.savefig(out, format="pdf")
    png = out.with_suffix(".png")
    fig.savefig(png, format="png", dpi=200)
    plt.close(fig)

    from PIL import Image  # noqa: PLC0415  (optional check)
    with Image.open(png) as image:
        width, height = image.size
    print(f"  wrote {out.name} and {png.name} ({width} x {height} px)")
    if width < 1328 or height < 531:
        print("  WARNING: below the 1328 x 531 minimum")
    else:
        print("  meets the 1328 x 531 minimum")


if __name__ == "__main__":
    main()
