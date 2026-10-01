"""Generate the remaining reported LaTeX tables.

Four tables that draw on sources outside the main open-set grid: the split
composition, the per-rule-refit confound, the realized false-alarm budget, and
the temperature sweep.

Usage:
    python -m src.make_openset_tables2
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.make_openset_tables import ATTACK_LABELS, BASE_LABELS, BASE_ORDER, _write
from src.run_experiment import load_paths
import os

CLASS_LABELS = {
    "Benign": "Benign", "HTTPFlood": "HTTP flood", "ICMPFlood": "ICMP flood",
    "SYNScan": "SYN scan", "UDPFlood": "UDP flood", "UDPScan": "UDP scan",
    "SYNFlood": "SYN flood", "SlowrateDoS": "Slow-rate DoS",
    "TCPConnectScan": "TCP connect scan",
}
NOMINAL_QUANTILE = 0.95


def table_setup(composition: dict, out: Path) -> None:
    """Class composition of the training and held-out partitions."""
    cell = composition["per_seed"]["42"]
    train, test = cell["train_counts"], cell["test_counts"]
    n_train, n_test = cell["n_train"], cell["n_test"]
    novel = sum(v for k, v in test.items() if k != "Benign")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Training} & \multicolumn{2}{c}{Held-out test} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Class & Flows & Share & Flows & Share \\",
        r"\midrule",
    ]

    def row(name: str, bold: bool = False) -> str:
        label = CLASS_LABELS[name]
        if bold:
            label = f"\\textbf{{{label}}}"
        cells = []
        for counts, total in ((train, n_train), (test, n_test)):
            n = counts.get(name)
            cells.append(f"{n:,} & {n / total:.3f}" if n else "--- & ---")
        return f"{label} & " + " & ".join(cells) + r" \\"

    lines.append(row("Benign"))
    lines.append(r"\addlinespace")
    for name in ["UDPFlood", "HTTPFlood", "SYNScan", "UDPScan", "ICMPFlood"]:
        lines.append(row(name))
    lines.append(r"\addlinespace")
    for name in ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]:
        lines.append(row(name, bold=True))
    lines += [
        r"\midrule",
        f"Total & {n_train:,} & 1.000 & {n_test:,} & 1.000 \\\\",
        f"Novel (unknown) & --- & --- & {novel:,} & {novel / n_test:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(out, "\n".join(lines) + "\n")


def table_confound(legacy: pd.DataFrame, shared: pd.DataFrame, out: Path) -> None:
    """MSP against energy under the per-rule-refit design and the shared fit."""
    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Design & Base & MSP & Energy & Differ & Max $|\Delta|$ \\",
        r"\midrule",
    ]
    for name, frame in (("Refit per rule", legacy), ("Shared fit", shared)):
        for i, base in enumerate(["lightgbm", "xgboost"]):
            block = frame[frame.base_model == base]
            msp = block[block.method == "msp"].set_index("seed")["novel_recall"]
            energy = block[block.method == "energy"].set_index("seed")["novel_recall"]
            common = msp.index.intersection(energy.index)
            delta = (msp.loc[common] - energy.loc[common]).abs()
            design = name if i == 0 else ""
            lines.append(
                f"{design} & {BASE_LABELS[base]} & {msp.loc[common].mean():.4f} & "
                f"{energy.loc[common].mean():.4f} & {int((delta > 0).sum())} of "
                f"{len(common)} & {delta.max():.4f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


def table_budget(grid: pd.DataFrame, out: Path) -> None:
    """Nominal against realized benign false-alarm rate.

    The rightmost column carries the per-seed range as well as the mean: the
    recommendation this table supports is about what a single deployed fit
    spends, so the dispersion is the operational quantity.
    """
    rules = ["msp", "mahalanobis", "knn"]
    labels = {"msp": "MSP/En.", "mahalanobis": "Mahal.", "knn": "KNN"}
    grouped = grid.groupby(["base_model", "method"], observed=True)["false_unknown_rate"]
    stats = grouped.mean()
    spans = grouped.agg(["min", "max"])
    nominal = 1.0 - NOMINAL_QUANTILE

    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(rules) + "cc}",
        r"\toprule",
        r"Base & " + " & ".join(labels[r] for r in rules)
        + r" & Worst & Range \\",
        r"\midrule",
    ]
    ratios = []
    for base in BASE_ORDER:
        cells, base_ratios = [], []
        for rule in rules:
            key = (base, rule)
            if key not in stats.index:
                cells.append("---")
                continue
            value = stats.loc[key]
            base_ratios.append(value / nominal)
            cells.append(f"{value:.4f}")
        worst = max(base_ratios, key=lambda r: abs(np.log(r))) if base_ratios else np.nan
        ratios.extend(base_ratios)
        worst_rule = rules[base_ratios.index(worst)]
        low, high = spans.loc[(base, worst_rule), ["min", "max"]] / nominal
        lines.append(f"{BASE_LABELS[base]} & " + " & ".join(cells)
                     + f" & ${worst:.2f}\\times$ & ${low:.2f}$--${high:.2f}$ \\\\")
    lines += [
        r"\midrule",
        f"Nominal $1-q$ & \\multicolumn{{{len(rules)}}}{{c}}{{{nominal:.4f}}}"
        f" & $1.00\\times$ & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _write(out, "\n".join(lines) + "\n")
    print(f"    realized/nominal ratio range: {min(ratios):.2f}x to {max(ratios):.2f}x")


def _sci(value: float) -> str:
    """Scientific notation the way a LaTeX table wants it."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def table_temperature(temperature: pd.DataFrame, out: Path) -> None:
    """Probability-space energy across temperatures, against MSP.

    Two quantities per base classifier: the novel-recall the rule achieves, and
    the largest absolute score on the calibration set. The second column is what
    separates an informative score from a numerically degenerate one.
    """
    order = ["msp"] + [r for r in sorted(set(temperature.rule) - {"msp"},
                                         key=lambda r: float(r.split("T")[1]))]
    labels = {"msp": "MSP"}
    for rule in order[1:]:
        labels[rule] = f"Energy, $T={float(rule.split('T')[1]):g}$"

    bases = [b for b in BASE_ORDER if b in set(temperature.base_model)]
    # max|s| is a per-run maximum, so the campaign figure is the maximum over
    # seeds rather than their mean.
    stats = temperature.groupby(["base_model", "rule"]).agg(
        nr=("novel_recall", "mean"), max_abs=("max_abs_score", "max"),
        seeds=("seed", "nunique"))

    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "rl" * len(bases) + "}",
        r"\toprule",
        "& " + " & ".join(r"\multicolumn{2}{c}{" + BASE_LABELS[b] + "}"
                          for b in bases) + r" \\",
        " ".join(f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}"
                 for i in range(len(bases))),
        "Scoring rule & " + " & ".join(r"Recall & $\max|s|$" for _ in bases) + r" \\",
        r"\midrule",
    ]
    for rule in order:
        cells = []
        for base in bases:
            key = (base, rule)
            if key not in stats.index:
                cells += ["---", "---"]
                continue
            r = stats.loc[key]
            cells += [f"{r.nr:.4f}", _sci(r.max_abs)]
        label = labels[rule]
        if rule.endswith("T1"):
            label = f"\\textbf{{{label}}}"
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if rule == "msp":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


def table_stats(stats: pd.DataFrame, out: Path) -> None:
    """KNN-OOD against each alternative, paired over seeds."""
    rules = ["none", "mahalanobis", "msp"]
    labels = {"none": "no rule", "mahalanobis": "Mahal.", "msp": "MSP/Energy"}
    block = stats[stats.metric == "macro_f1_open"]

    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Base & Alternative & $\Delta$ & Wins & $p$ & $d_z$ \\",
        r"\midrule",
    ]
    for base in BASE_ORDER:
        first = True
        for rule in rules:
            row = block[(block.base_model == base) & (block.alternative == rule)]
            if row.empty:
                continue
            r = row.iloc[0]
            label = BASE_LABELS[base] if first else ""
            first = False
            lines.append(
                f"{label} & {labels[rule]} & {r.delta:+.4f} & "
                f"{int(r.wins)}/{int(r.seeds)} & "
                f"{r.p_value:.4f} & {r.cohens_dz:+.2f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


TRAINED_ATTACKS = ["HTTPFlood", "SYNScan", "UDPScan", "UDPFlood", "ICMPFlood"]
HELD_OUT = ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]
ATTACK_ORDER = ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]


def _destination_split(counts: dict) -> tuple:
    total = sum(counts.values())
    benign = counts.get("Benign", 0)
    untrained = sum(counts.get(a, 0) for a in HELD_OUT)
    trained = total - benign - untrained
    return benign / total, trained / total, untrained / total


def table_destinations(records: list, out: Path) -> None:
    """Where novel flows land when no rule rejects them, and how sure the base is."""
    rows = []
    for r in records:
        benign, trained, untrained = _destination_split(r["novel_destination_counts"])
        rows.append({"base_model": r["base_model"], "benign": benign,
                     "trained": trained, "untrained": untrained,
                     "conf_novel": r["confidence_novel"]["mean"],
                     "conf_known": r["confidence_known"]["mean"]})
    frame = pd.DataFrame(rows).groupby("base_model").mean()

    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Label received} "
        r"& \multicolumn{2}{c}{Mean $\max_k p_k$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"Base & Attack & Benign & Untr. & Novel & Known \\",
        r"\midrule",
    ]
    for base in BASE_ORDER:
        if base not in frame.index:
            continue
        r = frame.loc[base]
        lines.append(
            f"{BASE_LABELS[base]} & {r.trained:.3f} & {r.benign:.3f} & "
            f"{r.untrained:.3f} & {r.conf_novel:.3f} & {r.conf_known:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


def table_per_attack_destinations(records: list, out: Path) -> None:
    """Per held-out attack type: the label it gets, and the confidence behind it."""
    rows = []
    for r in records:
        for attack, block in r["per_attack_destinations"].items():
            benign, trained, untrained = _destination_split(block["counts"])
            total = sum(block["counts"].values())
            modal = max(((k, v) for k, v in block["counts"].items()
                         if k != "Benign" and k not in HELD_OUT),
                        key=lambda kv: kv[1], default=("---", 0))
            rows.append({"base_model": r["base_model"], "attack": attack,
                         "benign": benign, "trained": trained,
                         "modal": modal[0], "modal_share": modal[1] / total,
                         "confidence": block["confidence"]["mean"]})
    frame = pd.DataFrame(rows)
    numeric = (frame.groupby(["attack", "base_model"])
               [["benign", "trained", "modal_share", "confidence"]].mean()
               .groupby("attack").mean())
    modal = frame.groupby("attack").modal.agg(lambda s: s.value_counts().idxmax())

    lines = [
        r"\setlength{\tabcolsep}{2pt}",
        r"\footnotesize",
        r"\begin{tabular}{lrrlr}",
        r"\toprule",
        r"Held-out type & Attack & Benign & Most often & Conf. \\",
        r"\midrule",
    ]
    for attack in ATTACK_ORDER:
        if attack not in numeric.index:
            continue
        r = numeric.loc[attack]
        label = ATTACK_LABELS[attack]
        if attack == "SlowrateDoS":
            label = f"\\textbf{{{label}}}"
        lines.append(
            f"{label} & {r.trained:.3f} & {r.benign:.3f} & "
            f"{CLASS_LABELS[modal[attack]]} ({r.modal_share:.2f}) & "
            f"{r.confidence:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    tables = Path(os.environ.get("NIDD_TABLES_DIR",
                    Path(paths["workspace"]) / "artifacts" / "tables"))
    print("Generating open-set manuscript tables (second set):")

    composition = json.loads((results / "openset_composition.json").read_text())
    table_setup(composition, tables / "tab_setup.tex")

    legacy = pd.DataFrame([json.loads(Path(f).read_text())
                           for f in glob.glob(str(results / "metrics" / "openset_*.json"))])
    shared = pd.read_csv(results / "openset_summary_per_seed.csv")
    table_confound(legacy, shared, tables / "tab_confound.tex")
    table_budget(shared, tables / "tab_budget.tex")

    rows = []
    for f in sorted(glob.glob(str(results / "openset_temperature" / "*.json"))):
        record = json.loads(Path(f).read_text())
        for rule, metrics in record["rules"].items():
            low, high = metrics["score_train_range"]
            rows.append({"base_model": record["base_model"], "seed": record["seed"],
                         "rule": rule, "max_abs_score": max(abs(low), abs(high)),
                         **metrics})
    temperature = pd.DataFrame(rows)
    print(f"  temperature records: {len(temperature)} rows, "
          f"{temperature.groupby('base_model').seed.nunique().to_dict()}")
    table_temperature(temperature, tables / "tab_temperature.tex")

    stats = pd.read_csv(results / "openset_stats.csv")
    table_stats(stats, tables / "tab_stats.tex")

    destinations = [json.loads(Path(f).read_text()) for f in
                    sorted(glob.glob(str(results / "openset_destinations" / "*.json")))]
    print(f"  destination records: {len(destinations)}")
    table_destinations(destinations, tables / "tab_destinations.tex")
    table_per_attack_destinations(destinations,
                                  tables / "tab_attack_destinations.tex")
    print("done")


if __name__ == "__main__":
    main()
