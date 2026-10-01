"""Unit tests for the stacked ensemble and its fusion rules."""
import os

import numpy as np
import pytest

from src.stacking import (
    BASE_SETS,
    StackedEnsemble,
    _expand_proba,
    _fit_convex_weights,
    _fit_temperature,
    _apply_temperature,
    _confidence_features,
    _safe_kfold,
)


def test_expand_proba_scatters_into_the_global_layout():
    """A model that never saw class 1 must not have its columns shifted."""
    proba = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    out = _expand_proba(proba, classes=np.array([0, 2]), num_classes=3)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 0], [0.7, 0.2])
    assert np.allclose(out[:, 1], [0.0, 0.0])
    assert np.allclose(out[:, 2], [0.3, 0.8])


def test_expand_proba_passes_through_complete_layout():
    proba = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    out = _expand_proba(proba, classes=np.array([0, 1, 2]), num_classes=3)
    assert np.allclose(out, proba)


def test_safe_kfold_caps_folds_at_the_rarest_class():
    y = np.array([0] * 100 + [1] * 3)
    assert _safe_kfold(y, k_folds=5, seed=0).n_splits == 3
    y_plenty = np.array([0] * 100 + [1] * 100)
    assert _safe_kfold(y_plenty, k_folds=5, seed=0).n_splits == 5


def test_convex_weights_concentrate_on_the_accurate_base():
    rng = np.random.default_rng(1)
    n, c = 2000, 3
    y = rng.integers(0, c, size=n)

    good = np.full((n, c), 0.05, dtype=np.float32)
    good[np.arange(n), y] = 0.90
    bad = np.full((n, c), 1.0 / c, dtype=np.float32)

    weights = _fit_convex_weights([good, bad], y)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > 0.9


def test_convex_weights_sum_to_one_with_equal_bases():
    rng = np.random.default_rng(2)
    n, c = 500, 3
    y = rng.integers(0, c, size=n)
    block = np.full((n, c), 0.2, dtype=np.float32)
    block[np.arange(n), y] = 0.6
    weights = _fit_convex_weights([block, block.copy()], y)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0)


def test_temperature_scaling_softens_an_overconfident_block():
    rng = np.random.default_rng(3)
    n, c = 3000, 3
    y = rng.integers(0, c, size=n)
    # Confident but wrong on a third of the rows: the fitted temperature should
    # be greater than one, that is, it should soften the distribution.
    proba = np.full((n, c), 0.005, dtype=np.float32)
    proba[np.arange(n), y] = 0.99
    flip = rng.random(n) < 0.33
    proba[flip] = np.roll(proba[flip], 1, axis=1)
    proba /= proba.sum(axis=1, keepdims=True)

    temperature = _fit_temperature(proba, y)
    assert temperature > 1.0

    scaled = _apply_temperature(proba, temperature)
    assert np.allclose(scaled.sum(axis=1), 1.0, atol=1e-5)
    assert scaled.max(axis=1).mean() < proba.max(axis=1).mean()


def test_apply_temperature_of_one_is_a_no_op():
    proba = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], dtype=np.float32)
    assert np.allclose(_apply_temperature(proba, 1.0), proba, atol=1e-6)


def test_confidence_features_add_two_columns_per_base():
    blocks = [np.random.default_rng(4).random((10, 3)).astype(np.float32) for _ in range(2)]
    blocks = [b / b.sum(axis=1, keepdims=True) for b in blocks]
    feats = _confidence_features(blocks)
    assert feats.shape == (10, 2 * (3 + 2))


# ---------------------------------------------------------------------------
# End-to-end on a small synthetic problem
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_problem():
    os.environ["SMOKE_MODE"] = "1"  # caps boosting rounds in src/models.py
    rng = np.random.default_rng(7)
    n, c = 1200, 3
    y = rng.integers(0, c, size=n).astype(np.int32)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    X[:, 0] += y * 2.0
    X[:, 1] -= y * 1.5
    split = int(0.7 * n)
    yield X[:split], y[:split], X[split:], y[split:], c
    os.environ.pop("SMOKE_MODE", None)


def test_stack_produces_global_probability_layout(toy_problem):
    X_train, y_train, X_test, _, c = toy_problem
    stack = StackedEnsemble(num_classes=c, seed=0, base_names=("lightgbm", "lr"),
                            meta_learner="lr", k_folds=3).fit(X_train, y_train)
    proba = stack.predict_proba(X_test)
    assert proba.shape == (len(X_test), c)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_best_single_matches_its_selected_base(toy_problem):
    X_train, y_train, X_test, _, c = toy_problem
    stack = StackedEnsemble(num_classes=c, seed=0, base_names=("lightgbm", "lr"),
                            meta_learner="best_single", k_folds=3).fit(X_train, y_train)
    selected = stack.base_names[stack.selected_base]
    assert stack.diagnostics["selected_base"] == selected

    from src.stacking import _predict_proba_global

    expected = _predict_proba_global(selected, stack.bases[stack.selected_base], X_test, c)
    assert np.allclose(stack.predict_proba(X_test), expected)


def test_weighted_meta_records_readable_weights(toy_problem):
    X_train, y_train, X_test, _, c = toy_problem
    stack = StackedEnsemble(num_classes=c, seed=0, base_names=("lightgbm", "lr"),
                            meta_learner="weighted", k_folds=3).fit(X_train, y_train)
    weights = stack.diagnostics["convex_weights"]
    assert set(weights) == {"lightgbm", "lr"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert np.allclose(stack.predict_proba(X_test).sum(axis=1), 1.0, atol=1e-5)


def test_coefficient_mass_is_a_distribution_over_bases(toy_problem):
    X_train, y_train, _, _, c = toy_problem
    stack = StackedEnsemble(num_classes=c, seed=0, base_names=("lightgbm", "lr"),
                            meta_learner="lr", k_folds=3).fit(X_train, y_train)
    mass = stack.diagnostics["meta_coefficient_mass"]
    assert set(mass) == {"lightgbm", "lr"}
    assert sum(mass.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in mass.values())


def test_shift_diagnostics_report_the_gap_to_the_best_base(toy_problem):
    X_train, y_train, X_test, y_test, c = toy_problem
    stack = StackedEnsemble(num_classes=c, seed=0, base_names=("lightgbm", "lr"),
                            meta_learner="lr", k_folds=3).fit(X_train, y_train)
    diag = stack.shift_diagnostics(X_test, y_test)

    assert set(diag["test_macro_f1_per_base"]) == {"lightgbm", "lr"}
    best = diag["best_base_on_test"]
    expected_gap = diag["stack_macro_f1"] - diag["test_macro_f1_per_base"][best]
    assert diag["stack_minus_best_base"] == pytest.approx(expected_gap)
    assert diag["stack_minus_best_base"] <= 1e-9 or diag["stack_macro_f1"] > 0


def test_base_sets_are_registered():
    assert BASE_SETS["trees_mlp"] == ("lightgbm", "xgboost", "mlp")
    assert "rf" in BASE_SETS["diverse"] and "lr" in BASE_SETS["diverse"]
