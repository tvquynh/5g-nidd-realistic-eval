"""Aggregate the base-station shift analysis into paper-ready tables.

Produces three things:

1.  A per-feature shift table averaged over seeds and both directions, in the
    class-matched form that isolates concept shift from prior shift.
2.  A per-detector table pairing the cross-station accuracy drop with how
    concentrated that detector's reliance is and how much of it sits on
    features that move between stations.
3.  The correlation across detectors between reliance-weighted shift and the
    accuracy drop. This is the quantitative form of the explanation: detectors
    that lean on unstable features are the ones that lose accuracy.

Usage:
    python -m src.aggregate_shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths


def load_model_rows(directory: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for file in sorted(directory.glob("shift_bs*_seed*.json")):
        record = json.loads(file.read_text())
        for model, block in record.get("per_model", {}).items():
            concentration = block.get("concentration_in_station", {})
            alignment = block.get("alignment_with_feature_shift", {})
            reliance = block.get("reliance_shift", {})
            rows.append({
                "model": model,
                "train_bs": record["train_bs"],
                "test_bs": record["test_bs"],
                "seed": record["seed"],
                "macro_f1_in_station": block.get("macro_f1_in_station"),
                "macro_f1_cross_station": block.get("macro_f1_cross_station"),
                "macro_f1_drop": block.get("macro_f1_drop"),
                "gini": concentration.get("gini"),
                "top1_share": concentration.get("top1_share"),
                "top5_share": concentration.get("top5_share"),
                "effective_features": concentration.get("effective_features"),
                "reliance_weighted_shift": alignment.get("reliance_weighted_shift"),
                "unweighted_mean_shift": alignment.get("unweighted_mean_shift"),
                "reliance_rank_stability": reliance.get("spearman_rho"),
            })
    return pd.DataFrame(rows)


def load_feature_tables(directory: Path) -> pd.DataFrame:
    frames = []
    for file in sorted(directory.glob("shift_bs*_features.csv")):
        table = pd.read_csv(file)
        stem = file.stem  # shift_bs{N}_{features}_seed{S}_features
        table["train_bs"] = int(stem.split("_bs")[1][0])
        table["seed"] = int(stem.split("seed")[1].split("_")[0])
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def mechanism_correlations(runs: pd.DataFrame) -> dict:
    """Correlate reliance structure with observed degradation, per direction.

    Reported at run level rather than detector level: with five detector
    families no correlation can reach significance whatever its size. The two
    transfer directions are kept apart because they behave differently, and
    pooling them cancels the effect.
    """
    from scipy.stats import pearsonr, spearmanr

    predictors = {
        "effective_features": "number of features the detector effectively uses",
        "gini": "concentration of feature reliance",
        "reliance_weighted_shift": "share of reliance sitting on features that move",
        "reliance_rank_stability": "survival of the reliance ordering across stations",
    }

    out = {}
    for scope, frame in [("pooled", runs)] + [
        (f"train_bs{int(bs)}", g) for bs, g in runs.groupby("train_bs")
    ]:
        scope_out = {}
        for column, description in predictors.items():
            if column not in frame.columns:
                continue
            valid = frame[[column, "macro_f1_drop"]].dropna()
            if len(valid) < 6:
                continue
            rho, rho_p = spearmanr(valid[column], valid["macro_f1_drop"])
            r, r_p = pearsonr(valid[column], valid["macro_f1_drop"])
            scope_out[column] = {
                "description": description,
                "n": int(len(valid)),
                "spearman_rho": float(rho),
                "spearman_p": float(rho_p),
                "pearson_r": float(r),
                "pearson_p": float(r_p),
            }
        out[scope] = scope_out
    return out


def print_mechanism(correlations: dict) -> None:
    print("\n=== Reliance structure versus observed degradation ===")
    print("Run level (detector x seed x direction). Directions reported apart:")
    print("pooling them averages two regimes that behave differently.\n")
    header = f"{'scope':<14}{'predictor':<26}{'n':>5}{'Spearman':>11}{'p':>12}{'Pearson':>10}{'p':>12}"
    print(header)
    for scope, block in correlations.items():
        for column, stats in block.items():
            print(f"{scope:<14}{column:<26}{stats['n']:>5}"
                  f"{stats['spearman_rho']:>+11.3f}{stats['spearman_p']:>12.2e}"
                  f"{stats['pearson_r']:>+10.3f}{stats['pearson_p']:>12.2e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "shift"

    models = load_model_rows(directory)
    if models.empty:
        raise SystemExit(f"no shift records under {directory}")
    print(f"Loaded {len(models)} model-level rows from {directory}")

    per_model = models.groupby("model").agg(
        n=("seed", "size"),
        in_station_mean=("macro_f1_in_station", "mean"),
        cross_station_mean=("macro_f1_cross_station", "mean"),
        cross_station_std=("macro_f1_cross_station", "std"),
        drop_mean=("macro_f1_drop", "mean"),
        drop_std=("macro_f1_drop", "std"),
        gini_mean=("gini", "mean"),
        effective_features_mean=("effective_features", "mean"),
        reliance_weighted_shift_mean=("reliance_weighted_shift", "mean"),
        unweighted_mean_shift_mean=("unweighted_mean_shift", "mean"),
        reliance_rank_stability_mean=("reliance_rank_stability", "mean"),
    ).reset_index().sort_values("drop_mean")

    print("\n=== Detector exposure to the base-station shift ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(per_model.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # The mechanism claim, stated as a correlation across detector families.
    mechanism = {}
    if len(per_model) >= 3:
        x = per_model["reliance_weighted_shift_mean"].values
        y = per_model["drop_mean"].values
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 3:
            rho = spearmanr(x[finite], y[finite])
            r = pearsonr(x[finite], y[finite])
            mechanism = {
                "n_models": int(finite.sum()),
                "spearman_rho": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
                "pearson_r": float(r.statistic),
                "pearson_p": float(r.pvalue),
            }
            print(f"\nReliance-weighted shift versus cross-station drop across "
                  f"{mechanism['n_models']} detectors: "
                  f"Spearman rho = {mechanism['spearman_rho']:+.3f} "
                  f"(p = {mechanism['spearman_p']:.4f}), "
                  f"Pearson r = {mechanism['pearson_r']:+.3f} "
                  f"(p = {mechanism['pearson_p']:.4f})")

    features = load_feature_tables(directory)
    run_correlations = mechanism_correlations(models)
    print_mechanism(run_correlations)

    feature_summary = pd.DataFrame()
    if not features.empty:
        shift_column = ("psi_class_matched" if "psi_class_matched" in features.columns
                        else "psi_pooled")
        aggregations = {
            "psi_pooled_mean": ("psi_pooled", "mean"),
            "ks_pooled_mean": ("ks_pooled", "mean"),
        }
        if "psi_class_matched" in features.columns:
            aggregations["psi_class_matched_mean"] = ("psi_class_matched", "mean")
            aggregations["ks_class_matched_mean"] = ("ks_class_matched", "mean")
        feature_summary = (
            features.groupby("feature").agg(**aggregations).reset_index()
            .sort_values(f"{shift_column}_mean", ascending=False)
        )
        above_025 = int((feature_summary[f"{shift_column}_mean"] > 0.25).sum())
        above_010 = int((feature_summary[f"{shift_column}_mean"] > 0.10).sum())
        print(f"\n=== Per-feature shift ({shift_column}, averaged over seeds and directions) ===")
        print(f"{above_025} of {len(feature_summary)} features exceed 0.25 (major shift); "
              f"{above_010} exceed 0.10 (moderate)")
        with pd.option_context("display.width", 200):
            print(feature_summary.head(args.top).to_string(
                index=False, float_format=lambda v: f"{v:.4f}"))

    prefix = Path(args.out_prefix) if args.out_prefix else results / "shift_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    models.to_csv(f"{prefix}_per_seed.csv", index=False)
    per_model.to_csv(f"{prefix}_per_model.csv", index=False)
    if not feature_summary.empty:
        feature_summary.to_csv(f"{prefix}_per_feature.csv", index=False)
    Path(f"{prefix}_mechanism.json").write_text(json.dumps(
        {"by_detector_family": mechanism, "by_run": run_correlations}, indent=2))
    print(f"\nSaved: {prefix}_per_model.csv, {prefix}_per_feature.csv, {prefix}_mechanism.json")


if __name__ == "__main__":
    main()
