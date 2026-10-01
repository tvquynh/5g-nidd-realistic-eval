"""Unit tests for the latency and model-complexity profiler."""
import time

import numpy as np
import pytest

from src.latency_profile import (
    _choose_repeats,
    measure_latency,
    model_complexity,
    profile_batch_sweep,
)


def _identity_predict(X):
    return np.zeros(len(X), dtype=np.int32)


def test_measure_latency_reports_ordered_percentiles():
    X = np.random.default_rng(0).random((512, 4)).astype(np.float32)
    result = measure_latency(_identity_predict, X, batch_size=32, n_warmup=2, n_repeats=50)

    assert result["batch_size"] == 32
    assert result["n_repeats"] == 50
    assert result["min_ms_per_batch"] <= result["p50_ms_per_batch"]
    assert result["p50_ms_per_batch"] <= result["p95_ms_per_batch"]
    assert result["p95_ms_per_batch"] <= result["p99_ms_per_batch"]
    assert result["p99_ms_per_batch"] <= result["max_ms_per_batch"]
    assert result["tail_ratio_p99_p50"] >= 1.0


def test_per_sample_scaling_matches_batch_latency():
    X = np.random.default_rng(1).random((256, 3)).astype(np.float32)
    result = measure_latency(_identity_predict, X, batch_size=64, n_warmup=1, n_repeats=30)

    expected = result["p50_ms_per_batch"] * 1000.0 / 64
    assert result["p50_per_sample_us"] == pytest.approx(expected, rel=1e-9)
    expected_throughput = 64 * 1000.0 / result["p50_ms_per_batch"]
    assert result["throughput_p50_per_sec"] == pytest.approx(expected_throughput, rel=1e-9)


def test_measure_latency_detects_injected_tail():
    """A predictor that stalls on one call in ten must show it at p95."""
    state = {"calls": 0}

    def stalling_predict(X):
        state["calls"] += 1
        if state["calls"] % 10 == 0:
            time.sleep(0.004)
        return np.zeros(len(X), dtype=np.int32)

    X = np.random.default_rng(2).random((128, 2)).astype(np.float32)
    result = measure_latency(stalling_predict, X, batch_size=8, n_warmup=0, n_repeats=100)

    assert result["p95_ms_per_batch"] > result["p50_ms_per_batch"] * 2


def test_batch_smaller_than_dataset_wraps_without_error():
    X = np.random.default_rng(3).random((10, 2)).astype(np.float32)
    result = measure_latency(_identity_predict, X, batch_size=64, n_warmup=1, n_repeats=5)
    # Batch size is capped at the dataset size rather than failing.
    assert result["batch_size"] == 10


def test_choose_repeats_is_bounded():
    assert _choose_repeats(1e-9) == 500
    assert _choose_repeats(10.0) == 30
    assert 30 <= _choose_repeats(0.01) <= 500


def test_profile_batch_sweep_returns_one_entry_per_batch():
    X = np.random.default_rng(4).random((256, 2)).astype(np.float32)
    sweep = profile_batch_sweep(_identity_predict, X, batch_sizes=(1, 16, 64),
                                n_warmup=1, n_repeats=20)
    assert [entry["batch_size"] for entry in sweep] == [1, 16, 64]


def test_model_complexity_counts_logistic_regression_parameters():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(5)
    X = rng.random((200, 6))
    y = (X[:, 0] > 0.5).astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=200).fit(scaler.transform(X), y)

    info = model_complexity((clf, scaler), "lr")
    # 6 coefficients + 1 intercept + 6 means + 6 scales
    assert info["param_count"] == 6 + 1 + 6 + 6
    assert info["size_native_bytes"] > 0
    assert info["size_native_mb"] == pytest.approx(info["size_native_bytes"] / 1048576.0)


def test_model_complexity_counts_mlp_parameters():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(6)
    X = rng.random((200, 4))
    y = (X[:, 0] > 0.5).astype(int)
    scaler = StandardScaler().fit(X)
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=20, random_state=0)
    clf.fit(scaler.transform(X), y)

    info = model_complexity((clf, scaler), "mlp")
    expected = 4 * 8 + 8 * 1 + 8 + 1 + 4 + 4
    assert info["param_count"] == expected
    assert info["hidden_layer_sizes"] == [8]
