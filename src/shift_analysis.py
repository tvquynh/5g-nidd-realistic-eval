"""Feature-level characterization of the shift between base stations.

The cross-station experiments establish *that* detectors degrade when trained
on one base station and evaluated on the other. This module supplies the
evidence for *why*, on two axes:

**Distribution shift.** Population stability index and two-sample
Kolmogorov-Smirnov statistics per feature, computed both over the pooled data
and within each class. The class-matched variant is the one that isolates
concept shift, because the pooled variant also absorbs the difference in class
priors between the two stations.

**Reliance shift.** How each detector distributes its predictive weight across
features, measured model-agnostically by permutation importance, and
summarized by concentration statistics. A detector that concentrates its
reliance on a few features is exposed to exactly those features moving between
stations; a detector that spreads reliance is not.

Together these let the degradation ordering across model families be explained
rather than merely reported.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import f1_score

EPS = 1e-6


# ---------------------------------------------------------------------------
# Distribution shift
# ---------------------------------------------------------------------------

def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10
) -> float:
    """Population stability index between two samples of one feature.

    Bin edges are quantiles of the reference sample, so the reference is
    uniform across bins by construction and the index measures how far the
    comparison sample departs from it. Conventional reading: below 0.1 is
    negligible, 0.1 to 0.25 is moderate, above 0.25 is a major shift.
    """
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    comparison = comparison[np.isfinite(comparison)]
    if len(reference) == 0 or len(comparison) == 0:
        return float("nan")

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 3:
        # Degenerate feature (constant or near-constant on the reference).
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)
    ref_share = ref_counts / max(ref_counts.sum(), 1)
    cmp_share = cmp_counts / max(cmp_counts.sum(), 1)
    ref_share = np.clip(ref_share, EPS, None)
    cmp_share = np.clip(cmp_share, EPS, None)
    return float(np.sum((cmp_share - ref_share) * np.log(cmp_share / ref_share)))


def _ks_statistic(reference: np.ndarray, comparison: np.ndarray, max_n: int, rng) -> float:
    """Two-sample KS statistic, subsampling large inputs for tractability."""
    if len(reference) > max_n:
        reference = rng.choice(reference, size=max_n, replace=False)
    if len(comparison) > max_n:
        comparison = rng.choice(comparison, size=max_n, replace=False)
    if len(reference) < 2 or len(comparison) < 2:
        return float("nan")
    return float(ks_2samp(reference, comparison).statistic)


def feature_shift_table(
    X_ref: np.ndarray,
    X_cmp: np.ndarray,
    feature_names: Sequence[str],
    y_ref: Optional[np.ndarray] = None,
    y_cmp: Optional[np.ndarray] = None,
    n_bins: int = 10,
    ks_max_n: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-feature shift statistics between a reference and a comparison set.

    When labels are supplied, class-matched statistics are added: the shift is
    computed within each class and averaged with weights given by the class
    frequency in the reference set. This removes the contribution of differing
    class priors, leaving the shift in the features themselves.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    class_weights: Dict[int, float] = {}
    if y_ref is not None and y_cmp is not None:
        shared = sorted(set(np.unique(y_ref)).intersection(np.unique(y_cmp)))
        total = sum(int((y_ref == c).sum()) for c in shared)
        class_weights = {int(c): int((y_ref == c).sum()) / max(total, 1) for c in shared}

    for j, name in enumerate(feature_names):
        ref_col = X_ref[:, j]
        cmp_col = X_cmp[:, j]
        row = {
            "feature": name,
            "psi_pooled": population_stability_index(ref_col, cmp_col, n_bins),
            "ks_pooled": _ks_statistic(ref_col, cmp_col, ks_max_n, rng),
            "mean_ref": float(np.nanmean(ref_col)) if len(ref_col) else float("nan"),
            "mean_cmp": float(np.nanmean(cmp_col)) if len(cmp_col) else float("nan"),
            "std_ref": float(np.nanstd(ref_col)) if len(ref_col) else float("nan"),
            "std_cmp": float(np.nanstd(cmp_col)) if len(cmp_col) else float("nan"),
        }

        if class_weights:
            psi_parts, ks_parts = 0.0, 0.0
            for cls, weight in class_weights.items():
                ref_cls = ref_col[y_ref == cls]
                cmp_cls = cmp_col[y_cmp == cls]
                if len(ref_cls) < 2 or len(cmp_cls) < 2:
                    continue
                psi_parts += weight * population_stability_index(ref_cls, cmp_cls, n_bins)
                ks_parts += weight * _ks_statistic(ref_cls, cmp_cls, ks_max_n, rng)
            row["psi_class_matched"] = float(psi_parts)
            row["ks_class_matched"] = float(ks_parts)

        rows.append(row)

    table = pd.DataFrame(rows)
    sort_key = "psi_class_matched" if "psi_class_matched" in table.columns else "psi_pooled"
    return table.sort_values(sort_key, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reliance shift
# ---------------------------------------------------------------------------

def permutation_reliance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 3,
    subsample: int = 20000,
    seed: int = 42,
) -> pd.Series:
    """Model-agnostic feature reliance measured by permutation importance.

    Importance of a feature is the mean drop in macro-F1 when that feature's
    column is shuffled. Values are floored at zero: a negative drop means the
    feature carried no usable signal on this sample.
    """
    rng = np.random.default_rng(seed)
    if len(X) > subsample:
        idx = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[idx], y[idx]

    X = np.ascontiguousarray(X)
    baseline = f1_score(y, predict_fn(X), average="macro", zero_division=0)

    drops = np.zeros(len(feature_names), dtype=np.float64)
    for j in range(len(feature_names)):
        original = X[:, j].copy()
        acc = 0.0
        for _ in range(n_repeats):
            X[:, j] = rng.permutation(original)
            acc += baseline - f1_score(y, predict_fn(X), average="macro", zero_division=0)
        X[:, j] = original
        drops[j] = acc / n_repeats

    return pd.Series(np.clip(drops, 0.0, None), index=list(feature_names), name="reliance")


def concentration_stats(importance: pd.Series) -> Dict[str, float]:
    """Summarize how concentrated a reliance distribution is.

    ``gini`` is 0 when every feature is relied on equally and approaches 1 when
    a single feature carries all the weight. ``effective_features`` is the
    exponential of the Shannon entropy of the normalized distribution and reads
    as the number of features the detector is effectively using.
    """
    values = np.asarray(importance.values, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0:
        return {
            "gini": float("nan"),
            "top1_share": float("nan"),
            "top5_share": float("nan"),
            "effective_features": float("nan"),
        }

    share = values / total
    ordered = np.sort(share)
    n = len(ordered)
    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * ordered)) / (n * np.sum(ordered)) - (n + 1.0) / n)

    descending = np.sort(share)[::-1]
    entropy = float(-np.sum(np.clip(share, EPS, None) * np.log(np.clip(share, EPS, None))))
    return {
        "gini": gini,
        "top1_share": float(descending[0]),
        "top5_share": float(descending[:5].sum()),
        "effective_features": float(np.exp(entropy)),
    }


def reliance_shift(
    reliance_ref: pd.Series, reliance_cmp: pd.Series
) -> Dict[str, float]:
    """Compare two reliance distributions over the same feature set."""
    common = reliance_ref.index.intersection(reliance_cmp.index)
    a = reliance_ref.loc[common].values.astype(np.float64)
    b = reliance_cmp.loc[common].values.astype(np.float64)
    a_share = a / a.sum() if a.sum() > 0 else a
    b_share = b / b.sum() if b.sum() > 0 else b

    corr = spearmanr(a, b) if len(common) >= 3 else None
    return {
        "n_features": int(len(common)),
        "spearman_rho": float(corr.statistic) if corr is not None else float("nan"),
        "spearman_p_value": float(corr.pvalue) if corr is not None else float("nan"),
        "total_variation_distance": float(0.5 * np.abs(a_share - b_share).sum()),
    }


def shift_vs_reliance_alignment(
    shift_table: pd.DataFrame, reliance: pd.Series, shift_column: str = "psi_class_matched"
) -> Dict[str, float]:
    """Quantify how much of a detector's reliance sits on unstable features.

    ``reliance_weighted_shift`` is the reliance-weighted mean of the per-feature
    shift statistic: high when a detector leans on exactly those features that
    move between stations. This is the quantity that separates detector
    families whose accuracy survives the shift from those whose does not.
    """
    column = shift_column if shift_column in shift_table.columns else "psi_pooled"
    shift = shift_table.set_index("feature")[column]
    common = shift.index.intersection(reliance.index)
    if len(common) == 0:
        return {"reliance_weighted_shift": float("nan"), "spearman_rho": float("nan")}

    weights = reliance.loc[common].values.astype(np.float64)
    values = shift.loc[common].values.astype(np.float64)
    finite = np.isfinite(values)
    weights, values = weights[finite], values[finite]
    total = weights.sum()

    corr = spearmanr(weights, values) if len(weights) >= 3 else None
    return {
        "reliance_weighted_shift": float(np.sum(weights * values) / total) if total > 0 else float("nan"),
        "unweighted_mean_shift": float(values.mean()) if len(values) else float("nan"),
        "spearman_rho": float(corr.statistic) if corr is not None else float("nan"),
        "spearman_p_value": float(corr.pvalue) if corr is not None else float("nan"),
    }
