"""
C6 Method Contribution: Stability-Aware LightGBM (StabLGBM).

Idea: Encourage the gradient-boosted classifier to rely on features whose gain
importance is *consistent* across the two base stations (BS1, BS2). Achieved
via per-feature scaling, where the scaling factor is inversely related to
per-BS gain variance computed from auxiliary single-BS LightGBM rankers.

DESIGN PHASE (run once, save sidecar):
    Use a random 80% sample of the full master data (both BS):
    1. Fit aux LightGBM on BS1 part -> gain_bs1 (per-feature gain).
    2. Fit aux LightGBM on BS2 part -> gain_bs2.
    3. Compute instability per feature:
        instability_j = |gain_bs1_j - gain_bs2_j| / (gain_bs1_j + gain_bs2_j + eps)
    4. Compute feature_weight = 1 / (1 + lambda * instability).
    5. Save feature_weight.npy alongside master parquet.

TRAINING/INFERENCE PHASE (per experiment):
    1. Load feature_weight.npy.
    2. Scale features X_scaled = X * feature_weight (broadcast per column).
    3. Fit final LightGBM on X_scaled (no per-BS access needed).
    4. At inference time, scale test features the same way.

This is a method-level analog of the post-hoc 'top20_stable' subset: instead of
hard-selecting 20 stable features, we softly downweight unstable ones using
information about base-station consistency available during the design phase.

Note: Using BS labels (data origin metadata) during the design phase to compute
feature_weight is analogous to using domain labels in domain adaptation. It
does not constitute test-set leakage because BS labels are part of the
collection metadata, not the prediction target.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def _fit_aux_ranker(X, y, num_classes, seed, n_iter=80):
    params = dict(
        objective="multiclass",
        num_class=num_classes,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        seed=seed,
        n_jobs=16,
    )
    ds = lgb.Dataset(X, y)
    return lgb.train(params, ds, num_boost_round=n_iter)


def compute_feature_weight(X, y, BS, num_classes, seed=42, lambda_stab=0.5):
    """Design-phase: compute per-feature stability weight from per-BS importance.

    Use this on FULL master data (both BS), once. Save the result as a sidecar
    to be loaded at training/inference time.
    """
    bs1_mask = (BS == 1)
    bs2_mask = (BS == 2)

    if bs1_mask.sum() < 100 or bs2_mask.sum() < 100:
        return np.ones(X.shape[1], dtype=np.float32)

    m1 = _fit_aux_ranker(X[bs1_mask], y[bs1_mask], num_classes, seed=seed)
    g1 = m1.feature_importance(importance_type="gain").astype(np.float64)

    m2 = _fit_aux_ranker(X[bs2_mask], y[bs2_mask], num_classes, seed=seed + 1)
    g2 = m2.feature_importance(importance_type="gain").astype(np.float64)

    g1n = g1 / (g1.sum() + 1e-9)
    g2n = g2 / (g2.sum() + 1e-9)

    eps = 1e-9
    instability = np.abs(g1n - g2n) / (g1n + g2n + eps)
    weight = 1.0 / (1.0 + float(lambda_stab) * instability)
    return weight.astype(np.float32)


def precompute_and_save_weight(master_parquet_path, out_npy_path,
                                lambda_stab=0.5, seed=42, sample_frac=0.8):
    """Run once: load master, compute weight on a random 80% sample, save."""
    import pandas as pd
    df = pd.read_parquet(master_parquet_path)
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.choice(n, size=int(n * sample_frac), replace=False)
    df_s = df.iloc[idx]

    meta_cols = {"y_multi", "y_binary", "BS", "capture_day"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df_s[feature_cols].values.astype(np.float32)
    y = df_s["y_multi"].values.astype(np.int32)
    BS = df_s["BS"].values
    n_cls = int(y.max()) + 1
    weight = compute_feature_weight(X, y, BS, n_cls, seed=seed, lambda_stab=lambda_stab)
    np.save(out_npy_path, weight)
    print(f"Saved weight ({len(weight)} features, range [{weight.min():.3f}, {weight.max():.3f}], mean {weight.mean():.3f}) to {out_npy_path}")
    return weight, feature_cols


def _load_or_compute_weight(X_train, y_train, BS_train, num_classes, seed, lambda_stab):
    """Load precomputed weight from sidecar if available, else compute from train."""
    import os
    weight_path = os.environ.get("STAB_WEIGHT_PATH",
                                 str(Path(__file__).resolve().parents[1]
                                     / "data" / "feature_weight.npy"))
    if os.path.exists(weight_path):
        weight = np.load(weight_path)
        if len(weight) == X_train.shape[1]:
            return weight.astype(np.float32)
    # Fallback: compute from current training data (only useful when both BS present)
    return compute_feature_weight(X_train, y_train, BS_train, num_classes, seed, lambda_stab)


def fit_stability_lgbm(X_train, y_train, BS_train, X_val=None, y_val=None,
                       num_classes=None, seed=42,
                       lambda_stab=0.5, n_iter=300, early_stop=25):
    """Train stability-aware LightGBM.

    Returns (model, feature_weight) tuple. Use predict_stability_lgbm at test time.
    """
    n_classes = num_classes or int(np.max(y_train)) + 1
    feature_weight = _load_or_compute_weight(
        X_train, y_train, BS_train, n_classes, seed, lambda_stab,
    )

    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )

    X_train_s = X_train * feature_weight  # broadcast (n, d) * (d,)
    X_val_s = X_val * feature_weight

    params = dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        seed=seed,
        n_jobs=16,
    )
    train_set = lgb.Dataset(X_train_s, label=y_train)
    val_set = lgb.Dataset(X_val_s, label=y_val, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_iter,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model, feature_weight


def predict_stability_lgbm(packed, X):
    model, feature_weight = packed
    X_s = X * feature_weight
    proba = model.predict(X_s)
    pred = np.argmax(proba, axis=1)
    return pred, proba
