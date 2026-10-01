"""Unit tests for the base-station shift characterization."""
import numpy as np
import pandas as pd
import pytest

from src.shift_analysis import (
    concentration_stats,
    feature_shift_table,
    permutation_reliance,
    population_stability_index,
    reliance_shift,
    shift_vs_reliance_alignment,
)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    b = rng.normal(size=20000)
    assert population_stability_index(a, b) < 0.01


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=20000)
    b = rng.normal(loc=1.5, size=20000)
    assert population_stability_index(a, b) > 0.25


def test_psi_is_monotone_in_the_size_of_the_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20000)
    small = population_stability_index(reference, rng.normal(loc=0.2, size=20000))
    large = population_stability_index(reference, rng.normal(loc=1.0, size=20000))
    assert small < large


def test_psi_handles_constant_feature():
    constant = np.zeros(1000)
    assert population_stability_index(constant, constant) == 0.0


def test_feature_shift_table_separates_prior_shift_from_concept_shift():
    """A feature that only differs because class priors differ is not concept shift."""
    rng = np.random.default_rng(3)
    n = 6000
    y_ref = np.repeat([0, 1], [n // 2, n // 2])
    y_cmp = np.repeat([0, 1], [n // 6, 5 * n // 6])  # same classes, different mix

    def feature_for(labels):
        return np.where(labels == 0, rng.normal(0.0, 0.3, len(labels)),
                        rng.normal(4.0, 0.3, len(labels)))

    X_ref = feature_for(y_ref).reshape(-1, 1)
    X_cmp = feature_for(y_cmp).reshape(-1, 1)

    table = feature_shift_table(X_ref, X_cmp, ["f0"], y_ref=y_ref, y_cmp=y_cmp)
    row = table.iloc[0]
    assert row["psi_pooled"] > 0.25          # pooled view sees a large shift
    assert row["psi_class_matched"] < 0.05   # within class, nothing moved


def test_feature_shift_table_is_sorted_by_shift():
    rng = np.random.default_rng(4)
    X_ref = rng.normal(size=(4000, 3))
    X_cmp = X_ref + np.array([0.0, 0.5, 2.0])
    table = feature_shift_table(X_ref, X_cmp, ["stable", "mild", "severe"])
    assert list(table["feature"]) == ["severe", "mild", "stable"]


def test_concentration_stats_on_uniform_reliance():
    reliance = pd.Series(np.ones(20), index=[f"f{i}" for i in range(20)])
    stats = concentration_stats(reliance)
    assert stats["gini"] == pytest.approx(0.0, abs=1e-6)
    assert stats["effective_features"] == pytest.approx(20.0, rel=1e-6)
    assert stats["top1_share"] == pytest.approx(0.05, rel=1e-6)


def test_concentration_stats_on_single_dominant_feature():
    values = np.zeros(20)
    values[0] = 1.0
    reliance = pd.Series(values, index=[f"f{i}" for i in range(20)])
    stats = concentration_stats(reliance)
    assert stats["gini"] > 0.9
    assert stats["top1_share"] == pytest.approx(1.0)
    assert stats["effective_features"] == pytest.approx(1.0, rel=1e-3)


def test_concentration_stats_on_all_zero_reliance():
    reliance = pd.Series(np.zeros(5), index=[f"f{i}" for i in range(5)])
    stats = concentration_stats(reliance)
    assert np.isnan(stats["gini"])


def test_permutation_reliance_finds_the_informative_feature():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = (X[:, 2] > 0).astype(np.int32)

    def predict_fn(Xq):
        return (Xq[:, 2] > 0).astype(np.int32)

    reliance = permutation_reliance(predict_fn, X, y, ["a", "b", "c", "d"],
                                    n_repeats=2, subsample=600, seed=0)
    assert reliance.idxmax() == "c"
    assert reliance["c"] > 0.2
    assert reliance.drop("c").max() < 0.05


def test_permutation_reliance_does_not_mutate_input():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    original = X.copy()
    permutation_reliance(lambda Xq: (Xq[:, 0] > 0).astype(np.int32), X, y,
                         ["a", "b", "c"], n_repeats=1, subsample=200, seed=0)
    assert np.array_equal(X, original)


def test_reliance_shift_reports_perfect_agreement():
    reliance = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    out = reliance_shift(reliance, reliance)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["total_variation_distance"] == pytest.approx(0.0, abs=1e-9)


def test_alignment_weights_shift_by_reliance():
    """A detector leaning on the unstable feature must show a higher weighted shift."""
    shift_table = pd.DataFrame({
        "feature": ["stable", "unstable"],
        "psi_class_matched": [0.01, 0.90],
        "psi_pooled": [0.01, 0.90],
    })
    leans_stable = pd.Series([1.0, 0.0], index=["stable", "unstable"])
    leans_unstable = pd.Series([0.0, 1.0], index=["stable", "unstable"])

    a = shift_vs_reliance_alignment(shift_table, leans_stable)["reliance_weighted_shift"]
    b = shift_vs_reliance_alignment(shift_table, leans_unstable)["reliance_weighted_shift"]
    assert a == pytest.approx(0.01)
    assert b == pytest.approx(0.90)
    assert a < b
