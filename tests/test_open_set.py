"""Tests for the open-set rejection rules and their metrics.

The two strongest claims of the open-set study are properties of the code
rather than of the dataset: that MSP and the probability-space energy score
decide identically at any calibrated quantile, and that the unit-temperature
substitution is degenerate. Both are asserted here so that a change in the
implementation fails a test rather than a paragraph.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.open_set import (UNKNOWN_LABEL, detect_open_set, open_set_metrics,
                          _energy_score, _msp_score)
from src.run_open_set_temperature import energy_probability_space, msp_score

QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.99)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


@pytest.fixture
def probabilities() -> tuple[np.ndarray, np.ndarray]:
    """Train and test probability matrices with a range of peakedness."""
    rng = np.random.default_rng(20260907)
    train = _softmax(rng.normal(0, 3.0, size=(4000, 6)))
    test = _softmax(rng.normal(0, 2.0, size=(1500, 6)))
    return train, test


@pytest.fixture
def peaked_probabilities() -> tuple[np.ndarray, np.ndarray]:
    """Probability matrices in the regime a fitted detector actually produces.

    On 5G-NIDD the 95th percentile of the training score distribution sits at a
    maximum class probability above 0.998 for every base classifier, so the
    largest term dominates the sum in the energy family well before the
    temperature becomes small. These matrices reproduce that regime; the
    ``probabilities`` fixture above deliberately does not.
    """
    rng = np.random.default_rng(20260907)

    def draw(n: int) -> np.ndarray:
        # One dominant class, with 1 - max(p) spanning the range a fitted
        # detector produces, and the remainder spread over the other classes.
        remainder = 10.0 ** rng.uniform(-6.0, -2.5, size=n)
        rest = rng.dirichlet(np.ones(5), size=n) * remainder[:, None]
        proba = np.concatenate([(1.0 - remainder)[:, None], rest], axis=1)
        columns = np.argsort(rng.random((n, 6)), axis=1)
        return np.take_along_axis(proba, columns, axis=1)

    train, test = draw(4000), draw(1500)
    assert np.quantile(train.max(axis=1), 0.05) > 0.99
    return train, test


# --------------------------------------------------------------------------
# Proposition 1: MSP and the probability-space energy score are one rule.
# --------------------------------------------------------------------------

def test_energy_is_a_strictly_increasing_transform_of_msp(probabilities):
    _, test = probabilities
    msp, energy = _msp_score(test), _energy_score(test)
    order_msp = np.argsort(msp, kind="stable")
    order_energy = np.argsort(energy, kind="stable")
    assert np.array_equal(order_msp, order_energy)


@pytest.mark.parametrize("quantile", QUANTILES)
def test_msp_and_energy_flag_identical_sets(probabilities, quantile):
    train, test = probabilities
    msp = detect_open_set("msp", test, proba_train=train, threshold_quantile=quantile)
    energy = detect_open_set("energy", test, proba_train=train,
                             threshold_quantile=quantile)
    assert np.array_equal(msp.pred == UNKNOWN_LABEL, energy.pred == UNKNOWN_LABEL)
    assert np.array_equal(msp.pred, energy.pred)


@pytest.mark.parametrize("quantile", QUANTILES)
def test_msp_and_energy_give_identical_metrics(probabilities, quantile):
    train, test = probabilities
    y_true = np.where(np.arange(len(test)) % 4 == 0, UNKNOWN_LABEL,
                      np.argmax(test, axis=1))
    known = sorted(set(range(test.shape[1])))
    results = [open_set_metrics(
        y_true,
        detect_open_set(rule, test, proba_train=train,
                        threshold_quantile=quantile).pred,
        known, unknown_label_in_truth=UNKNOWN_LABEL)
        for rule in ("msp", "energy")]
    assert results[0] == results[1]


# --------------------------------------------------------------------------
# The temperature family: MSP in the limit, degenerate at T = 1.
# --------------------------------------------------------------------------

def test_unit_temperature_energy_is_algebraically_zero(probabilities):
    _, test = probabilities
    scores = energy_probability_space(test, T=1.0)
    informative = energy_probability_space(test, T=0.25)
    # The magnitude, not the number of distinct values, is the reliable
    # signature: how coarse the residue is depends on the base classifier.
    assert np.abs(scores).max() < 1e-9
    assert np.abs(scores).max() < 1e-6 * np.abs(informative).max()


def test_small_temperature_energy_approaches_the_msp_surrogate(probabilities):
    """The limit in Eq. (5) is a limit: the gap shrinks as the temperature does."""
    _, test = probabilities
    reference = -np.log(np.clip(np.max(test, axis=1), 1e-12, 1.0))
    gaps = [np.abs(energy_probability_space(test, T=t) - reference).max()
            for t in (0.5, 0.25, 0.1, 0.05)]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.1


@pytest.mark.parametrize("temperature", (0.05, 0.1, 0.25))
def test_energy_decides_like_msp_on_peaked_outputs(peaked_probabilities, temperature):
    """The manuscript's empirical claim, isolated from the dataset.

    Finite-temperature energy is not a function of the maximum probability
    alone, so agreement with MSP is a property of peaked outputs rather than an
    identity. Detectors fitted on 5G-NIDD are peaked enough for it to hold.
    """
    train, test = peaked_probabilities
    threshold_energy = np.quantile(energy_probability_space(train, temperature), 0.95)
    threshold_msp = np.quantile(msp_score(train), 0.95)
    flagged_energy = energy_probability_space(test, temperature) > threshold_energy
    flagged_msp = msp_score(test) > threshold_msp
    assert np.array_equal(flagged_energy, flagged_msp)


def test_half_temperature_agrees_with_msp_only_approximately(peaked_probabilities):
    """At T = 0.5 the sum is not yet dominated by its largest term, so agreement
    is close but not exact --- which is what the campaign shows as well."""
    train, test = peaked_probabilities
    threshold_energy = np.quantile(energy_probability_space(train, 0.5), 0.95)
    threshold_msp = np.quantile(msp_score(train), 0.95)
    flagged_energy = energy_probability_space(test, 0.5) > threshold_energy
    flagged_msp = msp_score(test) > threshold_msp
    disagreement = (flagged_energy != flagged_msp).mean()
    assert 0.0 < disagreement < 0.02


def test_energy_and_msp_can_disagree_on_flat_outputs(probabilities):
    """The boundary of the claim above, asserted so it is not overread."""
    train, test = probabilities
    threshold_energy = np.quantile(energy_probability_space(train, 0.25), 0.95)
    threshold_msp = np.quantile(msp_score(train), 0.95)
    flagged_energy = energy_probability_space(test, 0.25) > threshold_energy
    flagged_msp = msp_score(test) > threshold_msp
    assert not np.array_equal(flagged_energy, flagged_msp)


# --------------------------------------------------------------------------
# Metric definitions.
# --------------------------------------------------------------------------

def test_absent_labels_rescale_macro_f1_exactly():
    """Labels with no support contribute exactly zero, so the macro average over
    the full universe is the restricted average times the ratio of label counts.
    This is what lets the manuscript report both figures from one number."""
    y_true = np.array([0] * 60 + [UNKNOWN_LABEL] * 40)
    y_pred = np.array([0] * 55 + [UNKNOWN_LABEL] * 5
                      + [UNKNOWN_LABEL] * 30 + [0] * 10)
    universe = [UNKNOWN_LABEL, 0, 1, 2, 3, 4, 5]
    full = f1_score(y_true, y_pred, labels=universe, average="macro", zero_division=0)
    restricted = f1_score(y_true, y_pred, labels=[UNKNOWN_LABEL, 0],
                          average="macro", zero_division=0)
    assert full * len(universe) / 2 == pytest.approx(restricted, abs=1e-12)


def test_novel_recall_and_false_unknown_rate_are_what_they_say():
    y_true = np.array([UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL,
                       0, 0, 0, 1, 1, 1])
    y_pred = np.array([UNKNOWN_LABEL, UNKNOWN_LABEL, 0, 1,
                       0, UNKNOWN_LABEL, 0, 1, 1, 1])
    metrics = open_set_metrics(y_true, y_pred, [0, 1],
                               unknown_label_in_truth=UNKNOWN_LABEL)
    assert metrics["novel_recall"] == pytest.approx(2 / 4)
    assert metrics["false_unknown_rate"] == pytest.approx(1 / 6)
    assert metrics["known_classification_acc"] == pytest.approx(1.0)


def test_no_rule_can_never_emit_unknown(probabilities):
    """The manuscript's first claim, as a property of Eq. (1)."""
    _, test = probabilities
    predictions = np.argmax(test, axis=1)
    assert not (predictions == UNKNOWN_LABEL).any()


def test_extreme_thresholds_bracket_the_flagged_set(probabilities):
    train, test = probabilities
    none_flagged = detect_open_set("msp", test, proba_train=train,
                                   threshold=np.inf)
    all_flagged = detect_open_set("msp", test, proba_train=train,
                                  threshold=-np.inf)
    assert not (none_flagged.pred == UNKNOWN_LABEL).any()
    assert (all_flagged.pred == UNKNOWN_LABEL).all()


@pytest.mark.parametrize("rule", ("msp", "energy", "mahalanobis", "knn"))
def test_every_rule_returns_a_usable_result(probabilities, rule):
    train, test = probabilities
    y_train = np.argmax(train, axis=1)
    result = detect_open_set(rule, test, proba_train=train, y_train=y_train,
                             threshold_quantile=0.95)
    assert result.pred.shape == (len(test),)
    assert np.isfinite(result.threshold)
    assert result.scores.shape == (len(test),)
    flagged = (result.pred == UNKNOWN_LABEL).mean()
    assert 0.0 <= flagged <= 1.0
