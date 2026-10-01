"""Generate the reported LaTeX tables from the result files.

Every reported number must resolve to a file under
``results/``. The tables are emitted here and included by the manuscript rather
than typed into it.

Usage:
    python -m src.make_openset_tables
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths
import os

BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
               "tabnet": "TabNet", "ftt": "FT-Transformer"}
RULE_ORDER = ["none", "msp", "energy", "mahalanobis", "knn"]
RULE_LABELS = {"none": "no rejection", "msp": "MSP", "energy": "Energy",
               "mahalanobis": "Mahalanobis", "knn": "KNN-OOD"}
ATTACK_ORDER = ["SYNFlood", "TCPConnectScan", "SlowrateDoS", "Benign (false unknown)"]
ATTACK_LABELS = {"SYNFlood": "SYN flood", "TCPConnectScan": "TCP connect scan",
                 "SlowrateDoS": "Slow-rate DoS",
                 "Benign (false unknown)": "Benign (false unknown)"}


def _write(path: Path, body: str, compact: bool = False,
           colsep: str = "3pt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        body = ("\\setlength{\\tabcolsep}{" + colsep + "}\n\\footnotesize\n") + body
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.name}")


def table_grid(frame: pd.DataFrame, out: Path) -> None:
    """The full grid: every rule on every base, with seed counts."""
    stats = (frame.groupby(["base_model", "method"], observed=True)
             .agg(f1_mean=("macro_f1_open", "mean"), f1_std=("macro_f1_open", "std"),
                  nr_mean=("novel_recall", "mean"), nr_std=("novel_recall", "std"),
                  fu_mean=("false_unknown_rate", "mean"),
                  fu_std=("false_unknown_rate", "std"),
                  seeds=("seed", "nunique")))
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Base classifier & Rejection rule & Seeds & macro-F1-open & Novel recall & False unknown \\",
        r"\midrule",
    ]
    for base in BASE_ORDER:
        for i, rule in enumerate(RULE_ORDER):
            if (base, rule) not in stats.index:
                continue
            r = stats.loc[(base, rule)]
            label = BASE_LABELS[base] if i == 0 else ""
            # No rule is marked best: Section 6.6 shows the differences among
            # the three probability-space rules are not resolvable at this
            # number of base fits, so a bolded winner would contradict the
            # paper's own testing.
            lines.append(
                f"{label} & {RULE_LABELS[rule]} & {int(r.seeds)} & "
                f"{r.f1_mean:.4f} $\\pm$ {r.f1_std:.4f} & "
                f"{r.nr_mean:.4f} $\\pm$ {r.nr_std:.4f} & "
                f"{r.fu_mean:.4f} $\\pm$ {r.fu_std:.4f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


def table_equivalence(frame: pd.DataFrame, out: Path) -> None:
    """MSP against Energy, seed by seed, on identical probabilities."""
    metrics = ["macro_f1_open", "novel_recall", "false_unknown_rate"]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Base & Paired seeds & Max $|$difference$|$ & Identical \\",
        r"\midrule",
    ]
    for base in BASE_ORDER:
        block = frame[frame.base_model == base]
        msp = block[block.method == "msp"].set_index("seed")[metrics]
        energy = block[block.method == "energy"].set_index("seed")[metrics]
        common = msp.index.intersection(energy.index)
        if len(common) == 0:
            continue
        delta = float((msp.loc[common] - energy.loc[common]).abs().max().max())
        lines.append(
            f"{BASE_LABELS[base]} & {len(common)} & "
            f"{delta:.1e} & {'yes' if delta == 0.0 else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


def table_per_attack(per_attack: pd.DataFrame, out: Path) -> None:
    """Rejectability by held-out attack type, base classifiers weighted equally.

    The boosted ensembles contribute ten seeds each and the deep models five, so
    an unweighted mean over the 30 records would weight the boosted ensembles
    twice as heavily. Averaging within a base first and then across bases keeps
    the classifier a factor of the design rather than a sampling unit.
    """
    pooled = (per_attack[per_attack.rule != "none"]
              .groupby(["attack", "rule", "base_model"])["recall"].mean()
              .groupby(["attack", "rule"]).mean().unstack())
    rules = [r for r in RULE_ORDER if r in pooled.columns and r != "none"]
    lines = [
        r"\begin{tabular}{l" + "c" * len(rules) + "}",
        r"\toprule",
        "Held-out attack type & " + " & ".join(RULE_LABELS[r] for r in rules) + r" \\",
        r"\midrule",
    ]
    for attack in ATTACK_ORDER:
        if attack not in pooled.index:
            continue
        if attack.startswith("Benign"):
            lines.append(r"\midrule")
        cells = []
        for rule in rules:
            value = pooled.loc[attack, rule]
            worst = attack == "SlowrateDoS"
            cells.append(f"\\textbf{{{value:.3f}}}" if worst else f"{value:.3f}")
        lines.append(f"{ATTACK_LABELS[attack]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True, colsep="2pt")


def table_sweep(sweep: pd.DataFrame, out: Path) -> None:
    """Open-set macro-F1 across operating points."""
    table = sweep.pivot_table(index=["base_model", "rule"], columns="quantile",
                              values="macro_f1_open", aggfunc="mean")
    quantiles = sorted(table.columns)
    lines = [
        r"\begin{tabular}{ll" + "c" * len(quantiles) + "c}",
        r"\toprule",
        r"Base & Rule & " + " & ".join(f"$q={q:.2f}$" for q in quantiles)
        + r" & Spread \\",
        r"\midrule",
    ]
    for base in BASE_ORDER:
        first = True
        for rule in RULE_ORDER:
            if rule == "none" or (base, rule) not in table.index:
                continue
            row = table.loc[(base, rule)]
            spread = row.max() - row.min()
            label = BASE_LABELS[base] if first else ""
            first = False
            lines.append(f"{label} & {RULE_LABELS[rule]} & "
                         + " & ".join(f"{row[q]:.4f}" for q in quantiles)
                         + f" & {spread:.4f} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    tables = Path(os.environ.get("NIDD_TABLES_DIR",
                    Path(paths["workspace"]) / "artifacts" / "tables"))
    print("Generating open-set manuscript tables:")

    grid = pd.read_csv(results / "openset_summary_per_seed.csv")
    table_grid(grid, tables / "tab_grid.tex")
    table_equivalence(grid, tables / "tab_equivalence.tex")

    per_attack = pd.read_csv(results / "openset_detail_summary_per_attack_per_seed.csv")
    table_per_attack(per_attack, tables / "tab_per_attack.tex")

    detail = [json.loads(f.read_text())
              for f in sorted((results / "openset_detail").glob("*.json"))]
    rows = [{"base_model": r["base_model"], "rule": rule, "quantile": float(q), **m}
            for r in detail for rule, s in r["threshold_sweep"].items()
            for q, m in s.items()]
    table_sweep(pd.DataFrame(rows), tables / "tab_sweep.tex")
    print("done")


if __name__ == "__main__":
    main()
