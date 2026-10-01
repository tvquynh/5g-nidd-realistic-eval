"""Unit tests for the three cross-station protocols.

The decomposition claim of the paper rests on these three splits measuring
what they say they measure, so each property is asserted directly:

- ``cross_station_naive`` uses every flow of both stations and therefore
  carries both prior shift and concept shift.
- ``cross_station`` matches per-class counts and therefore carries concept
  shift only.
- ``prior_control`` keeps both sides on the same station, so no concept shift
  is possible, and gives the test side the other station's class priors.
"""
import numpy as np
import pandas as pd
import pytest

from src.splits import (
    get_split,
    split_cross_station,
    split_cross_station_naive,
    split_prior_control,
)


@pytest.fixture
def two_station_frame():
    """Two stations with deliberately different class priors."""
    rng = np.random.default_rng(0)
    rows = []
    # BS1: benign-heavy. BS2: attack-heavy. Same three classes on both sides.
    for bs, counts in ((1, {0: 6000, 1: 2000, 2: 400}), (2, {0: 1000, 1: 5000, 2: 900})):
        for label, count in counts.items():
            rows.append(pd.DataFrame({
                "BS": bs,
                "y_multi": label,
                "f0": rng.normal(label, 1.0, count),
                "f1": rng.normal(0.0, 1.0, count),
            }))
    frame = pd.concat(rows, ignore_index=True)
    return frame.sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_naive_uses_every_flow_of_both_stations(two_station_frame):
    df = two_station_frame
    train_idx, test_idx = split_cross_station_naive(df, train_bs=1)

    assert len(train_idx) == int((df["BS"] == 1).sum())
    assert len(test_idx) == int((df["BS"] == 2).sum())
    assert set(df["BS"].values[train_idx]) == {1}
    assert set(df["BS"].values[test_idx]) == {2}
    assert not set(train_idx) & set(test_idx)


def test_naive_preserves_the_prior_difference(two_station_frame):
    """The whole point of the naive protocol: priors are left unmatched."""
    df = two_station_frame
    train_idx, test_idx = split_cross_station_naive(df, train_bs=1)
    train_share = pd.Series(df["y_multi"].values[train_idx]).value_counts(normalize=True)
    test_share = pd.Series(df["y_multi"].values[test_idx]).value_counts(normalize=True)
    # Class 0 dominates BS1 but not BS2.
    assert train_share[0] > 0.6
    assert test_share[0] < 0.2


def test_class_matched_split_equalizes_per_class_counts(two_station_frame):
    df = two_station_frame
    train_idx, test_idx = split_cross_station(df, train_bs=1, seed=42)
    train_counts = pd.Series(df["y_multi"].values[train_idx]).value_counts().sort_index()
    test_counts = pd.Series(df["y_multi"].values[test_idx]).value_counts().sort_index()
    assert train_counts.equals(test_counts)
    # Each class is capped by the smaller of the two stations.
    assert train_counts[0] == 1000  # min(6000, 1000)
    assert train_counts[1] == 2000  # min(2000, 5000)
    assert train_counts[2] == 400   # min(400, 900)


def test_prior_control_keeps_both_sides_on_the_training_station(two_station_frame):
    df = two_station_frame
    train_idx, test_idx = split_prior_control(df, train_bs=1, seed=42)
    assert set(df["BS"].values[train_idx]) == {1}
    assert set(df["BS"].values[test_idx]) == {1}
    assert not set(train_idx) & set(test_idx)


def test_prior_control_test_side_matches_the_other_station_priors(two_station_frame):
    df = two_station_frame
    _, test_idx = split_prior_control(df, train_bs=1, seed=42)

    achieved = pd.Series(df["y_multi"].values[test_idx]).value_counts(normalize=True).sort_index()
    other = df[df["BS"] == 2]["y_multi"].value_counts(normalize=True).sort_index()
    for label in other.index:
        assert achieved[label] == pytest.approx(other[label], abs=0.01)


def test_prior_control_is_reproducible(two_station_frame):
    df = two_station_frame
    a_train, a_test = split_prior_control(df, train_bs=1, seed=7)
    b_train, b_test = split_prior_control(df, train_bs=1, seed=7)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_test, b_test)


def test_prior_control_differs_between_seeds(two_station_frame):
    df = two_station_frame
    _, a_test = split_prior_control(df, train_bs=1, seed=7)
    _, b_test = split_prior_control(df, train_bs=1, seed=8)
    assert not np.array_equal(np.sort(a_test), np.sort(b_test))


def test_both_directions_are_supported(two_station_frame):
    df = two_station_frame
    for train_bs, other in ((1, 2), (2, 1)):
        train_idx, test_idx = split_cross_station_naive(df, train_bs=train_bs)
        assert set(df["BS"].values[train_idx]) == {train_bs}
        assert set(df["BS"].values[test_idx]) == {other}

        train_idx, test_idx = split_prior_control(df, train_bs=train_bs, seed=1)
        assert set(df["BS"].values[train_idx]) == {train_bs}
        assert set(df["BS"].values[test_idx]) == {train_bs}


def test_dispatcher_routes_the_new_names(two_station_frame):
    df = two_station_frame
    naive = get_split("cross_station_naive", df, seed=42, train_bs=1)
    direct = split_cross_station_naive(df, train_bs=1)
    assert np.array_equal(naive[0], direct[0])
    assert np.array_equal(naive[1], direct[1])

    control = get_split("prior_control", df, seed=42, train_bs=2)
    assert set(df["BS"].values[control[1]]) == {2}
