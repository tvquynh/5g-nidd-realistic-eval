"""Latency, model-size, and memory profiling for the deployment analysis.

This module supports the deployment-oriented claims of the paper. It differs
from the coarse end-to-end timing in ``src/train.py`` in three ways:

1.  It reports **tail** latency (p95, p99) in addition to the median, using
    enough repetitions per batch size for the high percentiles to be
    meaningful. A latency service-level objective cannot be assessed from a
    median alone.
2.  It separates **framework overhead** from model evaluation cost. For
    XGBoost the construction of a ``DMatrix`` dominates single-flow latency,
    which would otherwise be misread as an intrinsic model cost.
3.  It records deployment-relevant **model complexity**: parameter count,
    serialized model size, and peak resident memory observed during
    inference.

All timings use ``time.perf_counter`` and are taken after warm-up passes.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

# Batch sizes swept for the deployment figure. Batch 1 is the in-line
# per-flow case; 4096 is a batched SIEM-style case.
DEFAULT_BATCH_SIZES = (1, 16, 64, 256, 1024, 4096)

# Target wall-clock budget per (model, batch size) measurement. Repetitions
# are chosen adaptively so that cheap batches get enough samples for a stable
# p99 without letting expensive batches run unbounded.
TARGET_SECONDS = 2.0
MIN_REPEATS = 30
MAX_REPEATS = 500


# ---------------------------------------------------------------------------
# Resident-memory sampling
# ---------------------------------------------------------------------------

def _read_rss_bytes() -> Optional[int]:
    """Current resident set size in bytes, or None if unavailable."""
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        # Linux fallback: statm reports pages; field 1 is resident.
        with open("/proc/self/statm", "r") as fh:
            resident_pages = int(fh.read().split()[1])
        import resource

        return resident_pages * resource.getpagesize()
    except Exception:
        return None


class RSSSampler:
    """Sample resident set size in a background thread.

    Used as a context manager around an inference loop to capture the peak
    resident memory of the process while that loop runs.
    """

    def __init__(self, interval_sec: float = 0.005):
        self.interval_sec = interval_sec
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _read_rss_bytes()
            if rss is not None:
                self.samples.append(rss)
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "RSSSampler":
        if _read_rss_bytes() is None:
            return self  # sampling unsupported on this platform
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_bytes(self) -> Optional[int]:
        return max(self.samples) if self.samples else None

    @property
    def baseline_bytes(self) -> Optional[int]:
        return min(self.samples) if self.samples else None


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def _choose_repeats(seconds_per_call: float) -> int:
    """Pick a repetition count that fits TARGET_SECONDS, within bounds."""
    if seconds_per_call <= 0:
        return MAX_REPEATS
    est = int(TARGET_SECONDS / seconds_per_call)
    return int(np.clip(est, MIN_REPEATS, MAX_REPEATS))


def measure_latency(
    predict_fn: Callable[[np.ndarray], object],
    X: np.ndarray,
    batch_size: int,
    n_warmup: int = 10,
    n_repeats: Optional[int] = None,
    sample_memory: bool = False,
) -> Dict:
    """Time ``predict_fn`` on a fixed batch and return latency percentiles.

    Each repetition scores a different slice of ``X`` (cycling through the
    array) so that the measurement is not dominated by cache effects on a
    single batch.

    Parameters
    ----------
    predict_fn:
        Callable that accepts a 2-D feature array and returns predictions.
    X:
        Feature matrix to draw batches from.
    batch_size:
        Number of rows scored per call.
    n_warmup:
        Untimed calls executed before measurement.
    n_repeats:
        Fixed repetition count. When ``None``, the count is chosen adaptively
        from a pilot measurement so that the total budget is TARGET_SECONDS.
    sample_memory:
        When True, sample resident set size during the timed loop and report
        the peak.
    """
    n = len(X)
    if n == 0:
        raise ValueError("X is empty")
    batch_size = int(min(batch_size, n))

    def batch_at(offset: int) -> np.ndarray:
        start = offset % n
        end = start + batch_size
        if end <= n:
            return X[start:end]
        # Wrap around the end of the array without reallocating the whole set.
        return np.concatenate([X[start:], X[: end - n]], axis=0)

    for i in range(n_warmup):
        predict_fn(batch_at(i * batch_size))

    if n_repeats is None:
        pilot_start = time.perf_counter()
        predict_fn(batch_at(0))
        pilot_sec = time.perf_counter() - pilot_start
        n_repeats = _choose_repeats(pilot_sec)

    times_ms: List[float] = []
    sampler = RSSSampler() if sample_memory else None
    if sampler is not None:
        sampler.__enter__()
    try:
        for i in range(n_repeats):
            batch = batch_at(i * batch_size)
            start = time.perf_counter()
            predict_fn(batch)
            times_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        if sampler is not None:
            sampler.__exit__()

    arr = np.asarray(times_ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    out = {
        "batch_size": batch_size,
        "n_repeats": int(n_repeats),
        "p50_ms_per_batch": p50,
        "p95_ms_per_batch": p95,
        "p99_ms_per_batch": p99,
        "mean_ms_per_batch": float(arr.mean()),
        "std_ms_per_batch": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms_per_batch": float(arr.min()),
        "max_ms_per_batch": float(arr.max()),
        "p50_per_sample_us": p50 * 1000.0 / batch_size,
        "p95_per_sample_us": p95 * 1000.0 / batch_size,
        "p99_per_sample_us": p99 * 1000.0 / batch_size,
        "throughput_p50_per_sec": batch_size * 1000.0 / p50 if p50 > 0 else float("inf"),
        "tail_ratio_p99_p50": p99 / p50 if p50 > 0 else float("nan"),
    }
    if sampler is not None:
        out["peak_rss_bytes"] = sampler.peak_bytes
        out["baseline_rss_bytes"] = sampler.baseline_bytes
    return out


def profile_batch_sweep(
    predict_fn: Callable[[np.ndarray], object],
    X: np.ndarray,
    batch_sizes: tuple = DEFAULT_BATCH_SIZES,
    **kwargs,
) -> List[Dict]:
    """Run :func:`measure_latency` across a sweep of batch sizes."""
    return [measure_latency(predict_fn, X, b, **kwargs) for b in batch_sizes]


# ---------------------------------------------------------------------------
# Framework-overhead breakdown
# ---------------------------------------------------------------------------

def xgboost_overhead_breakdown(
    booster,
    X: np.ndarray,
    batch_size: int,
    n_warmup: int = 10,
    n_repeats: int = 100,
) -> Dict:
    """Split XGBoost single-batch latency into input conversion and scoring.

    XGBoost's Python binding requires a ``DMatrix``. At batch 1 the cost of
    building that object can exceed the cost of evaluating the trees, which
    makes end-to-end per-flow latency look far worse than the model itself is.
    This function reports the two components separately, plus the
    ``inplace_predict`` path that bypasses ``DMatrix`` entirely.
    """
    import xgboost as xgb

    batch_size = int(min(batch_size, len(X)))
    batch = np.ascontiguousarray(X[:batch_size])

    for _ in range(n_warmup):
        booster.predict(xgb.DMatrix(batch))

    dmatrix_ms, predict_ms, end_to_end_ms, inplace_ms = [], [], [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        dmat = xgb.DMatrix(batch)
        t1 = time.perf_counter()
        booster.predict(dmat)
        t2 = time.perf_counter()
        dmatrix_ms.append((t1 - t0) * 1000.0)
        predict_ms.append((t2 - t1) * 1000.0)
        end_to_end_ms.append((t2 - t0) * 1000.0)

    supports_inplace = hasattr(booster, "inplace_predict")
    if supports_inplace:
        for _ in range(n_warmup):
            booster.inplace_predict(batch)
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            booster.inplace_predict(batch)
            inplace_ms.append((time.perf_counter() - t0) * 1000.0)

    def _summary(values: List[float]) -> Dict:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
        return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "mean_ms": float(arr.mean())}

    dmat_summary = _summary(dmatrix_ms)
    pred_summary = _summary(predict_ms)
    e2e_summary = _summary(end_to_end_ms)
    overhead_share = (
        dmat_summary["p50_ms"] / e2e_summary["p50_ms"]
        if e2e_summary.get("p50_ms", 0) > 0
        else float("nan")
    )
    return {
        "batch_size": batch_size,
        "n_repeats": int(n_repeats),
        "dmatrix_construction": dmat_summary,
        "booster_predict": pred_summary,
        "end_to_end": e2e_summary,
        "inplace_predict": _summary(inplace_ms) if supports_inplace else None,
        "dmatrix_share_of_end_to_end": overhead_share,
    }


# ---------------------------------------------------------------------------
# Model complexity
# ---------------------------------------------------------------------------

def _joblib_size_bytes(obj) -> int:
    """Serialized size of an arbitrary Python model object."""
    import joblib

    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=0)
    return int(buf.tell())


def model_complexity(model, model_name: str) -> Dict:
    """Return parameter count and serialized size for a fitted model.

    ``param_count`` is the number of learned quantities the model must carry
    at inference time: split nodes plus leaf values for tree ensembles,
    weights plus biases for parametric models. ``size_native_bytes`` is the
    size in the format a deployment would actually ship (the library's own
    serialization for boosted trees, joblib otherwise).
    """
    info: Dict = {"model": model_name}

    if model_name == "lightgbm":
        # Parse the serialized model rather than calling dump_model(): a
        # multiclass booster with several thousand trees produces a very large
        # nested dictionary, while the text form is needed for the size figure
        # in any case.
        import re

        text = model.model_to_string()
        per_tree_leaves = [int(m) for m in re.findall(r"^num_leaves=(\d+)$", text, re.MULTILINE)]
        n_trees = int(model.num_trees())
        if per_tree_leaves:
            n_leaves = int(sum(per_tree_leaves))
            n_trees = len(per_tree_leaves)
        else:  # pragma: no cover - serialization format fallback
            trees = model.dump_model().get("tree_info", [])
            n_leaves = int(sum(t.get("num_leaves", 0) for t in trees))
            n_trees = len(trees)
        # Each internal node stores a split feature and a threshold; a tree
        # with L leaves has L-1 internal nodes.
        n_internal = max(n_leaves - n_trees, 0)
        info.update(
            {
                "n_trees": n_trees,
                "n_leaves": n_leaves,
                "param_count": n_leaves + 2 * n_internal,
                "size_native_bytes": len(text.encode("utf-8")),
            }
        )

    elif model_name == "xgboost":
        dump = model.get_dump(dump_format="text")
        n_leaves = sum(entry.count("leaf=") for entry in dump)
        n_internal = sum(entry.count("[f") for entry in dump)
        info.update(
            {
                "n_trees": int(len(dump)),
                "n_leaves": int(n_leaves),
                "param_count": int(n_leaves + 2 * n_internal),
                "size_native_bytes": int(len(model.save_raw())),
            }
        )

    elif model_name == "rf":
        node_counts = [est.tree_.node_count for est in model.estimators_]
        n_leaves = int(sum(int((est.tree_.children_left == -1).sum()) for est in model.estimators_))
        n_internal = int(sum(node_counts) - n_leaves)
        info.update(
            {
                "n_trees": int(len(model.estimators_)),
                "n_leaves": n_leaves,
                "param_count": int(n_leaves * model.n_classes_ + 2 * n_internal),
                "size_native_bytes": _joblib_size_bytes(model),
            }
        )

    elif model_name == "lr":
        clf, scaler = model
        n_params = int(clf.coef_.size + clf.intercept_.size)
        n_params += int(scaler.mean_.size + scaler.scale_.size)
        info.update({"param_count": n_params, "size_native_bytes": _joblib_size_bytes(model)})

    elif model_name == "mlp":
        clf, scaler = model
        n_params = int(sum(w.size for w in clf.coefs_))
        n_params += int(sum(b.size for b in clf.intercepts_))
        n_params += int(scaler.mean_.size + scaler.scale_.size)
        info.update(
            {
                "param_count": n_params,
                "hidden_layer_sizes": list(clf.hidden_layer_sizes),
                "size_native_bytes": _joblib_size_bytes(model),
            }
        )

    else:
        # Stacked ensembles and deep tabular models fall through to a generic
        # serialized-size measurement.
        info.update({"param_count": None, "size_native_bytes": _joblib_size_bytes(model)})

    info["size_native_mb"] = info["size_native_bytes"] / (1024.0 * 1024.0)
    return info
