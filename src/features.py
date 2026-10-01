"""
Feature set builder for paper:
- F_full: all numeric columns after preprocess (~95 cols)
- F_top50: top 50 by LightGBM importance (gain) on random split
- F_top20: top 20 deployment-friendly = top by importance ∩ cross-station-stable
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb


META_COLS = {"y_multi", "y_binary", "BS", "capture_day"}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """All feature columns (exclude meta + label)."""
    return [c for c in df.columns if c not in META_COLS]


def f_full(df: pd.DataFrame) -> List[str]:
    return get_feature_columns(df)


def rank_by_lgbm_importance(df: pd.DataFrame, train_idx: np.ndarray, seed: int = 42) -> pd.Series:
    """Rank features by LightGBM gain importance on a random subset."""
    feat_cols = get_feature_columns(df)
    X = df.iloc[train_idx][feat_cols].values
    y = df.iloc[train_idx]["y_multi"].values
    n_class = int(np.max(y)) + 1
    params = dict(
        objective="multiclass",
        num_class=n_class,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        seed=seed,
        n_jobs=16,  # avoid 60-core hang
    )
    train_set = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_set, num_boost_round=100)
    imp = model.feature_importance(importance_type="gain")
    return pd.Series(imp, index=feat_cols, name="gain").sort_values(ascending=False)


def f_top_k(df: pd.DataFrame, train_idx: np.ndarray, k: int, seed: int = 42) -> List[str]:
    """Top-k features by LightGBM gain on train_idx."""
    ranked = rank_by_lgbm_importance(df, train_idx, seed=seed)
    return ranked.index[:k].tolist()


def f_top_stable(df: pd.DataFrame, k: int = 20, seed: int = 42) -> List[str]:
    """Top-k features stable across BS1 and BS2.
    Compute gain on BS1 and BS2 separately, rank intersection by mean(rank).
    """
    bs1_idx = df.index[df["BS"] == 1].values
    bs2_idx = df.index[df["BS"] == 2].values
    rank_bs1 = rank_by_lgbm_importance(df, bs1_idx, seed=seed).rank(ascending=False)
    rank_bs2 = rank_by_lgbm_importance(df, bs2_idx, seed=seed).rank(ascending=False)
    common = rank_bs1.index.intersection(rank_bs2.index)
    mean_rank = (rank_bs1.loc[common] + rank_bs2.loc[common]) / 2
    return mean_rank.sort_values().index[:k].tolist()


def get_feature_set(name: str, df: pd.DataFrame, train_idx: Optional[np.ndarray] = None, seed: int = 42) -> List[str]:
    if name == "full":
        return f_full(df)
    if name == "top50":
        if train_idx is None:
            raise ValueError("top50 needs train_idx")
        return f_top_k(df, train_idx, k=50, seed=seed)
    if name == "top20":
        if train_idx is None:
            raise ValueError("top20 needs train_idx")
        return f_top_k(df, train_idx, k=20, seed=seed)
    if name == "top20_stable":
        return f_top_stable(df, k=20, seed=seed)
    raise ValueError(f"Unknown feature set: {name}")
