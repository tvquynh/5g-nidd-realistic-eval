"""Generate the manuscript's LaTeX tables directly from the result files.

Every number in the paper must be traceable to a file under ``results/``.
Transcribing figures by hand into the manuscript is how discrepancies between
text and data appear, so the tables are emitted from the aggregated CSVs and
included by the manuscript rather than typed into it.

Usage:
    python -m src.make_paper_tables
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

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

# Abbreviations for tables whose column count leaves no room for full names.
SHORT_LABELS = {
    "lightgbm": "LGBM",
    "xgboost": "XGB",
    "rf": "RF",
    "lr": "LR",
    "mlp": "MLP",
    "tabnet": "TabNet",
    "ftt": "FTT",
}


def _fmt(value: float, digits: int = 4) -> str:
    return "---" if pd.isna(value) else f"{value:.{digits}f}"


def _write(path: Path, body: str, compact: bool = False,
           size: str = "footnotesize") -> None:
    """Write a tabular fragment, optionally with tightened spacing.

    Several of these tables carry more columns than the two-column layout
    accommodates at full size. Reducing the inter-column padding and the font
    one step keeps them inside the text block without rescaling, which would
    leave the table's type size inconsistent with the body text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        prefix = "\\setlength{\\tabcolsep}{2.5pt}\n\\" + size + "\n"
        body = prefix + body
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# Table 1: direction asymmetry
# ---------------------------------------------------------------------------

def table_direction_asymmetry(cells: pd.DataFrame, out: Path) -> None:
    """Binary rates per detector and direction under the class-matched protocol."""
    matched = cells[(cells.train_balance == "matched") & (cells.test_balance == "matched")]
    rows = []
    for model in MODEL_ORDER:
        block = matched[matched.model == model]
        if block.empty:
            continue
        entry = {"model": MODEL_LABELS[model]}
        for bs in (1, 2):
            side = block[block.train_bs == bs]
            if side.empty:
                entry[f"tpr{bs}"] = entry[f"fpr{bs}"] = float("nan")
                entry[f"n{bs}"] = 0
            else:
                entry[f"tpr{bs}"] = side.binary_recall_mean.iat[0]
                entry[f"fpr{bs}"] = side.binary_fpr_mean.iat[0]
                entry[f"n{bs}"] = int(side.n_seeds.iat[0])
        rows.append(entry)

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{BS1 $\rightarrow$ BS2} & \multicolumn{2}{c}{BS2 $\rightarrow$ BS1} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Detector & TPR & FPR & TPR & FPR \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['model']} & {_fmt(r['tpr1'], 3)} & {_fmt(r['fpr1'], 4)} & "
            f"{_fmt(r['tpr2'], 3)} & \\textbf{{{_fmt(r['fpr2'], 3)}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 2: the two-by-two design
# ---------------------------------------------------------------------------

def table_factorial(cells: pd.DataFrame, out: Path, train_bs: int = 2) -> None:
    """The four cells of the design for one direction, all detectors."""
    block = cells[(cells.train_bs == train_bs) & (cells.train_balance.isin(["raw", "matched"]))]
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Detector & Training set & \multicolumn{2}{c}{Scoring set raw} & \multicolumn{2}{c}{Scoring set matched} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"& & macro-F1 & binary F1 & macro-F1 & binary F1 \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        rows = block[block.model == model]
        if rows.empty:
            continue
        for balance in ("raw", "matched"):
            raw = rows[(rows.train_balance == balance) & (rows.test_balance == "raw")]
            mat = rows[(rows.train_balance == balance) & (rows.test_balance == "matched")]
            if raw.empty and mat.empty:
                continue
            label = MODEL_LABELS[model] if balance == "raw" else ""
            lines.append(
                f"{label} & {balance} & "
                f"{_fmt(raw.macro_f1_mean.iat[0], 3) if not raw.empty else '---'} & "
                f"{_fmt(raw.binary_f1_mean.iat[0], 3) if not raw.empty else '---'} & "
                f"{_fmt(mat.macro_f1_mean.iat[0], 3) if not mat.empty else '---'} & "
                f"{_fmt(mat.binary_f1_mean.iat[0], 3) if not mat.empty else '---'} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 3: main effects
# ---------------------------------------------------------------------------

def table_effects(effects: pd.DataFrame, out: Path, metric: str = "binary_f1") -> None:
    """Paired main effects on one metric: detectors as rows, effects as columns.

    Each cell gives the mean paired difference over the seeds with its Wilcoxon
    p-value beneath. Scoring-side effects occupy the left pair of columns and
    training-side effects the right pair, so the asymmetry between them is
    readable across a single row.
    """
    block = effects[effects.metric == metric]
    columns = [
        ("test composition (train raw)", r"tr.\,raw"),
        ("test composition (train matched)", r"tr.\,match"),
        ("train composition (test raw)", r"sc.\,raw"),
        ("train composition (test matched)", r"sc.\,match"),
    ]

    lines = [
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        r"Detector & " + " & ".join(label for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for bs in sorted(block.train_bs.unique()):
        other = 2 if bs == 1 else 1
        lines.append(
            r"\multicolumn{" + str(len(columns) + 1) + r"}{l}{\emph{BS"
            + f"{int(bs)}" + r"$\!\to\!$BS" + f"{int(other)}" + r"}} \\"
        )
        side = block[block.train_bs == bs]
        for model in MODEL_ORDER:
            rows = side[side.model == model]
            if rows.empty:
                continue
            cells = []
            for effect, _ in columns:
                match = rows[rows.effect == effect]
                if match.empty:
                    cells.append("---")
                    continue
                mean = match["mean"].iat[0]
                p_value = match["p_value"].iat[0]
                cells.append(f"{mean:+.3f}\\,({p_value:.3f})")
            lines.append(f"{SHORT_LABELS[model]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


# ---------------------------------------------------------------------------
# Table 4: metric sensitivity
# ---------------------------------------------------------------------------

def table_metric_sensitivity(effects: pd.DataFrame, out: Path) -> None:
    """How much the scoring composition moves each metric."""
    labels = {
        "binary_f1": "binary F1",
        "macro_f1": "macro-F1",
        "binary_recall": "binary TPR",
        "binary_fpr": "binary FPR",
    }
    test_effects = effects[effects["effect"].str.startswith("test composition")]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Mean effect & Max $|$effect$|$ & Significant \\",
        r"\midrule",
    ]
    for metric in ["binary_f1", "macro_f1", "binary_recall", "binary_fpr"]:
        rows = test_effects[test_effects.metric == metric]
        if rows.empty:
            continue
        lines.append(
            f"{labels[metric]} & {rows['mean'].mean():+.4f} & "
            f"{rows['mean'].abs().max():.4f} & "
            f"{int((rows['p_value'] < 0.05).sum())} of {len(rows)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table 5: mechanism
# ---------------------------------------------------------------------------

def table_mechanism(per_model: pd.DataFrame, mechanism: dict, out_model: Path,
                    out_corr: Path) -> None:
    """Detector reliance structure, and its correlation with degradation."""
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Detector & In-station & Cross-station & Drop & Effective features & Gini \\",
        r"\midrule",
    ]
    order = [m for m in MODEL_ORDER if m in set(per_model.model)]
    for model in sorted(order, key=lambda m: per_model[per_model.model == m].drop_mean.iat[0]):
        r = per_model[per_model.model == model].iloc[0]
        lines.append(
            f"{SHORT_LABELS[model]} & {_fmt(r.in_station_mean, 3)} & "
            f"{_fmt(r.cross_station_mean, 3)} & {_fmt(r.drop_mean, 3)} & "
            f"{_fmt(r.effective_features_mean, 1)} & {_fmt(r.gini_mean, 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_model, "\n".join(lines) + "\n", compact=True)

    by_run = mechanism.get("by_run", {})
    scope_labels = {
        "pooled": "Pooled",
        "train_bs1": "BS1 $\\rightarrow$ BS2 only",
        "train_bs2": "BS2 $\\rightarrow$ BS1 only",
    }
    pred_labels = {
        "effective_features": "Effective features",
        "gini": "Reliance Gini",
        "reliance_weighted_shift": "Reliance-wt. shift",
        "reliance_rank_stability": "Rank stability",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Scope & Predictor & $n$ & Spearman $\rho$ & $p$ \\",
        r"\midrule",
    ]
    for scope in ["train_bs1", "train_bs2", "pooled"]:
        block = by_run.get(scope, {})
        first = True
        for key, label in pred_labels.items():
            if key not in block:
                continue
            stats = block[key]
            lines.append(
                f"{scope_labels[scope] if first else ''} & {label} & {stats['n']} & "
                f"{stats['spearman_rho']:+.3f} & {stats['spearman_p']:.1e} \\\\"
            )
            first = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out_corr, "\n".join(lines) + "\n", compact=True)



FEATURE_LABELS = {
    "full": "full",
    "top50": "top-50",
    "top20": "top-20",
    "top20_stable": "top-20 (stable)",
}
FEATURE_ORDER = ["full", "top50", "top20", "top20_stable"]


def table_main_summary(metrics_dir: Path, out: Path) -> None:
    """Macro-F1 per detector, protocol and feature subset, over the seeds.

    The cross-station column is split by direction rather than averaged: the
    two directions behave differently, and a single pooled figure describes
    neither.
    """
    import json

    rows = []
    for file in sorted(metrics_dir.glob("*.json")):
        if file.name.endswith("_smoke.json"):
            continue
        record = json.loads(file.read_text())
        if "macro_f1" not in record:
            continue
        split = record.get("split")
        if split == "cross_station":
            column = f"Cross-station BS{int(record.get('train_bs') or 1)}"
        elif split == "random":
            column = "Random"
        elif split == "temporal":
            column = "Temporal"
        elif split == "holdout_attack":
            column = "Holdout-attack"
        else:
            continue
        rows.append({
            "model": record["model"],
            "features": record.get("features", "full"),
            "column": column,
            "macro_f1": record["macro_f1"],
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("  (no grid records found; skipping summary table)")
        return

    grouped = frame.groupby(["model", "features", "column"])["macro_f1"].agg(["mean", "std", "size"])

    columns = ["Random", "Temporal", "Cross-station BS1", "Cross-station BS2", "Holdout-attack"]
    headers = [
        "Random", "Temporal",
        r"BS1$\!\to\!$BS2", r"BS2$\!\to\!$BS1",
        "Holdout-attack",
    ]

    lines = [
        r"\begin{tabular}{ll" + "c" * len(columns) + "}",
        r"\toprule",
        r"Detector & Features & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model not in set(frame.model):
            continue
        subsets = [f for f in FEATURE_ORDER
                   if (model, f) in {(m, s) for m, s, _ in grouped.index}]
        for i, subset in enumerate(subsets):
            label = SHORT_LABELS[model] if i == 0 else ""
            cells = []
            for column in columns:
                key = (model, subset, column)
                if key in grouped.index:
                    stats = grouped.loc[key]
                    cells.append(f"{stats['mean']:.3f} $\\pm$ {stats['std']:.3f}")
                else:
                    cells.append("---")
            lines.append(f"{label} & {FEATURE_LABELS[subset]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


META_LABELS = {
    "lr": "logistic meta",
    "weighted": "convex weights",
    "confidence": "confidence-aware",
    "calibrated": "temperature + logistic",
    "best_single": "best single (by OOF)",
}
META_ORDER = ["lr", "weighted", "confidence", "calibrated", "best_single"]
PROTOCOL_LABELS = {
    ("random", 0): "Random (no shift)",
    ("temporal", 0): "Temporal (mild shift)",
    ("cross_station", 1): r"Cross-station BS1$\!\to\!$BS2",
    ("cross_station", 2): r"Cross-station BS2$\!\to\!$BS1",
}


def table_fusion(summary: pd.DataFrame, out: Path, base_set: str = "trees_mlp") -> None:
    """Fusion rules against the best base learner, by protocol.

    The last column is the quantity that governs the outcome: the rank
    correlation between the base ordering out-of-fold, on which every fusion
    rule is fitted, and their ordering on the scoring set.
    """
    block = summary[summary.base_set == base_set]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Protocol & Fusion rule & Stack & Best base & $\Delta$ & $p$ \\",
        r"\midrule",
    ]
    for key, label in PROTOCOL_LABELS.items():
        split, train_bs = key
        rows = block[(block.split == split) & (block.train_bs == train_bs)]
        if rows.empty:
            continue
        first = True
        for meta in META_ORDER:
            match = rows[rows.meta == meta]
            if match.empty:
                continue
            r = match.iloc[0]
            lines.append(
                f"{label if first else ''} & {META_LABELS[meta]} & "
                f"{r.macro_f1_mean:.4f} & {r.best_base_macro_f1_mean:.4f} & "
                f"{r.stack_minus_best_mean:+.4f} & {r.wilcoxon_p_stack_vs_best_base:.3f} \\\\"
            )
            first = False
        rho = rows.spearman_oof_vs_test_mean.iloc[0]
        lines.append(
            r"\multicolumn{6}{l}{\hspace{1em}\emph{Spearman correlation, "
            r"out-of-fold ordering versus scoring-set ordering: "
            f"{rho:+.2f}" + r"}} \\"
        )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True)


LATENCY_LABELS = dict(SHORT_LABELS)
LATENCY_LABELS["stack[trees_mlp/lr]"] = "Stack"


def _latency_label(name: str) -> str:
    return LATENCY_LABELS.get(name, name)


def table_latency(summary: pd.DataFrame, out: Path,
                  batches: tuple = (1, 256, 4096)) -> None:
    """Per-sample latency percentiles at selected batch sizes, full feature set."""
    block = summary[summary.features == "full"]
    order = block[block.batch_size == batches[0]].sort_values("p50_us_mean")["model"].tolist()

    header = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{$b={b}$}}" for b in batches
    )
    rules = " ".join(
        rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(batches))
    )
    lines = [
        r"\begin{tabular}{l" + "rrr" * len(batches) + "}",
        r"\toprule",
        f"& {header} \\\\",
        rules,
        "Detector & " + " & ".join(["p50 & p95 & p99"] * len(batches)) + r" \\",
        r"\midrule",
    ]
    for model in order:
        cells = []
        for b in batches:
            row = block[(block.model == model) & (block.batch_size == b)]
            if row.empty:
                cells += ["---"] * 3
                continue
            r = row.iloc[0]
            for column in ("p50_us_mean", "p95_us_mean", "p99_us_mean"):
                value = r[column]
                cells.append(f"{value:,.0f}" if value >= 1000 else f"{value:.1f}")
        lines.append(f"{_latency_label(model)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n", compact=True, size="scriptsize")


def table_complexity(complexity: pd.DataFrame, out: Path) -> None:
    """Parameter count, serialized size and peak resident memory."""
    block = complexity[complexity.features == "full"].sort_values("param_count")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Detector & Parameters & Size (MB) & Peak RSS (MB) \\",
        r"\midrule",
    ]
    for _, r in block.iterrows():
        params = "---" if pd.isna(r.param_count) else f"{r.param_count:,.0f}"
        lines.append(
            f"{_latency_label(r.model)} & {params} & {r.size_mb:.2f} & "
            f"{r.peak_rss_mb:,.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")


def table_xgboost_overhead(overhead: pd.DataFrame, out: Path) -> None:
    """Input conversion against tree evaluation for the XGBoost serving path."""
    block = (overhead[overhead.features == "full"]
             .groupby("batch_size", as_index=False)
             .mean(numeric_only=True)
             .sort_values("batch_size"))
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Batch & \texttt{DMatrix} & Tree eval. & End to end & "
        r"\texttt{inplace} & Conversion share \\",
        r"\midrule",
    ]
    for _, r in block.iterrows():
        lines.append(
            f"{int(r.batch_size)} & {r.dmatrix_p50_ms:.2f} & {r.predict_p50_ms:.2f} & "
            f"{r.end_to_end_p50_ms:.2f} & {r.inplace_p50_ms:.2f} & "
            f"{r.dmatrix_share:.0%} \\\\".replace("%", r"\%")
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out, "\n".join(lines) + "\n")

def main() -> None:
    paths = load_paths()
    results = Path(paths["results"])
    tables = Path(paths["workspace"]) / "paper" / "tables"

    print("Generating manuscript tables from result files:")

    cells_path = results / "decomposition_summary_cells.csv"
    effects_path = results / "decomposition_summary_effects.csv"
    if cells_path.exists():
        cells = pd.read_csv(cells_path)
        table_direction_asymmetry(cells, tables / "tab_direction.tex")
        table_factorial(cells, tables / "tab_factorial.tex", train_bs=2)
    if effects_path.exists():
        effects = pd.read_csv(effects_path)
        table_effects(effects, tables / "tab_effects.tex")
        table_metric_sensitivity(effects, tables / "tab_metric_sensitivity.tex")

    per_model_path = results / "shift_summary_per_model.csv"
    mechanism_path = results / "shift_summary_mechanism.json"
    if per_model_path.exists() and mechanism_path.exists():
        table_mechanism(
            pd.read_csv(per_model_path),
            json.loads(mechanism_path.read_text()),
            tables / "tab_mechanism_models.tex",
            tables / "tab_mechanism_correlations.tex",
        )

    grid_dir = results / "metrics_windows"
    if grid_dir.exists():
        table_main_summary(grid_dir, tables / "table1_summary.tex")

    fusion_path = results / "stacking_summary.csv"
    if fusion_path.exists():
        table_fusion(pd.read_csv(fusion_path), tables / "tab_fusion.tex")

    latency_path = results / "latency_summary.csv"
    if latency_path.exists():
        table_latency(pd.read_csv(latency_path), tables / "tab_latency.tex")
    complexity_path = results / "latency_summary_complexity.csv"
    if complexity_path.exists():
        table_complexity(pd.read_csv(complexity_path), tables / "tab_complexity.tex")
    overhead_path = results / "latency_summary_xgboost_overhead.csv"
    if overhead_path.exists():
        table_xgboost_overhead(pd.read_csv(overhead_path), tables / "tab_overhead.tex")

    print("done")


if __name__ == "__main__":
    main()
