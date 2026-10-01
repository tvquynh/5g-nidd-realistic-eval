"""
SHAP analysis: identify which features Random Forest vs LightGBM rely on
under cross-station evaluation. Goal: explain why RF macro F1 = 0.87
while LightGBM macro F1 = 0.45 cross-station.

Output:
    results/figures/fig4_shap_rf_vs_lgbm_crossstation.pdf
    results/shap_analysis.json (top features per model + transferability scores)
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_experiment import load_paths, load_or_build_master, get_class_names
from src.splits import get_split
from src.features import get_feature_columns
from src.models import fit_lightgbm, fit_rf


def explain_one(model_name: str, df, feature_cols, train_idx, test_idx, n_explain=500, seed=42):
    """Train model on train_idx, compute SHAP on n_explain test samples."""
    X_train = df.iloc[train_idx][feature_cols].values
    y_train = df.iloc[train_idx]["y_multi"].values
    X_test = df.iloc[test_idx][feature_cols].values
    y_test = df.iloc[test_idx]["y_multi"].values
    n_class = int(y_train.max()) + 1

    if model_name == "lightgbm":
        model = fit_lightgbm(X_train, y_train, num_classes=n_class, seed=seed)
        explainer = shap.TreeExplainer(model)
    elif model_name == "rf":
        model = fit_rf(X_train, y_train, seed=seed)
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unknown model {model_name}")

    n_explain = min(n_explain, len(X_test))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_sample = X_test[sample_idx]

    shap_values = explainer.shap_values(X_sample)
    # multi-class: shap_values is list of arrays, one per class
    if isinstance(shap_values, list):
        # average abs SHAP over all classes -> per-feature importance
        global_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes)
            global_imp = np.abs(shap_values).mean(axis=(0, 2))
        else:
            global_imp = np.abs(shap_values).mean(axis=0)

    return model, X_sample, shap_values, global_imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=200_000)
    ap.add_argument("--n-explain", type=int, default=300)
    ap.add_argument("--out-fig", default="results/figures/fig4_shap_rf_vs_lgbm.pdf")
    ap.add_argument("--out-json", default="results/shap_analysis.json")
    args = ap.parse_args()

    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    # Subsample for tractability
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample, random_state=args.seed)
    _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
    df = df.iloc[sub_idx].reset_index(drop=True)
    if classes:
        df.attrs["_class_names"] = classes

    feat_cols = get_feature_columns(df)
    train_idx, test_idx = get_split("cross_station", df, args.seed, train_bs=1)
    print(f"Cross-station BS1->BS2: train={len(train_idx)} test={len(test_idx)} features={len(feat_cols)}")

    print("\n=== Explaining LightGBM ===")
    _, X_lgbm, sv_lgbm, imp_lgbm = explain_one("lightgbm", df, feat_cols, train_idx, test_idx,
                                                n_explain=args.n_explain, seed=args.seed)

    print("\n=== Explaining Random Forest ===")
    _, X_rf, sv_rf, imp_rf = explain_one("rf", df, feat_cols, train_idx, test_idx,
                                          n_explain=args.n_explain, seed=args.seed)

    # Top-10 features per model
    top_n = 15
    lgbm_top = pd.Series(imp_lgbm, index=feat_cols).sort_values(ascending=False).head(top_n)
    rf_top = pd.Series(imp_rf, index=feat_cols).sort_values(ascending=False).head(top_n)
    print("\nLightGBM top-15 SHAP features:")
    print(lgbm_top.round(4).to_string())
    print("\nRandom Forest top-15 SHAP features:")
    print(rf_top.round(4).to_string())

    # Side-by-side bar plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False)
    ax_l, ax_r = axes
    lgbm_top[::-1].plot(kind="barh", ax=ax_l, color="#1f77b4")
    ax_l.set_title("LightGBM top-15 SHAP features (cross-station BS1→BS2)\nmacro F1 ≈ 0.45")
    ax_l.set_xlabel("Mean |SHAP value|")
    rf_top[::-1].plot(kind="barh", ax=ax_r, color="#2ca02c")
    ax_r.set_title("Random Forest top-15 SHAP features (cross-station BS1→BS2)\nmacro F1 ≈ 0.87")
    ax_r.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out}")

    # Save JSON summary
    summary = {
        "lightgbm_top15": {k: float(v) for k, v in lgbm_top.items()},
        "rf_top15": {k: float(v) for k, v in rf_top.items()},
        "lgbm_only_features": sorted(set(lgbm_top.index) - set(rf_top.index)),
        "rf_only_features": sorted(set(rf_top.index) - set(lgbm_top.index)),
        "shared_features": sorted(set(lgbm_top.index) & set(rf_top.index)),
        "n_explain": args.n_explain,
        "subsample": args.subsample,
        "seed": args.seed,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
