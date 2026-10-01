"""
Open-Set Detection for novel-attack-class flagging.

Wraps a trained multiclass classifier with a novelty-detection head that
predicts a special "Unknown" class when the model's confidence is low or
when the input is far from any seen-class distribution in feature space.

Every rule here reads class probabilities and nothing else. That is the
constraint a wrapper around a deployed classifier normally works under, and it
is not free: rules defined on logits or on penultimate features have to be
reformulated, and Energy in particular collapses onto MSP under it.

Methods implemented:
    - MSP   (Maximum Softmax Probability, Hendrycks & Gimpel 2017): simplest;
            mark prediction Unknown if max softmax prob < threshold.
    - Energy (Liu et al. 2020): free-energy score E(x) = -T * logsumexp(z/T),
            defined on logits z. Under probability-only access the usable
            surrogate is -log max_k p_k, which is a strictly increasing
            transform of the MSP score, so at any quantile-calibrated
            threshold the two rules flag identical sets. Substituting log p
            for z directly at T = 1 is degenerate: the score is then
            -log(sum_k p_k) = 0 for every input.
    - Mahalanobis (Lee et al. 2018, simplified): per-class centroid +
            shared covariance in softmax-probability space; threshold the
            distance to predicted class centroid. The covariance is singular
            on the simplex, so it is regularized as cov + 1e-3 * I.
    - KNN-OOD (Sun et al. 2022, adapted): distance to the k-th nearest
            training probability vector, k = 50.

Evaluation: open-set macro-F1 on the attack-holdout split, where 3 attack types
are unseen during training. Without a rejection rule the prediction reduces to
argmax_k p_k, whose range is the known classes, so novel recall is exactly zero
by construction: every held-out attack flow is given some known label. Measured
on 5G-NIDD, most of those flows land on a known ATTACK class rather than on
Benign; see novel_to_benign_rate in the records written by
src/run_open_set_destinations.py.

Threshold calibration: the `threshold_quantile` quantile (0.95 by default) of
the score distribution over the FULL pooled training set passed as proba_train.
No held-out calibration slice is used. The calibration population is therefore
whatever the training mixture is, which is why the realized false-alarm rate on
any one class need not match the nominal budget 1 - q.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score, recall_score


UNKNOWN_LABEL = -1  # we map the "Unknown" class to -1 so it does not collide with class IDs


@dataclass
class OpenSetResult:
    pred: np.ndarray          # class predictions, with UNKNOWN_LABEL for novelties
    proba: np.ndarray         # softmax probabilities (n, n_known_classes)
    scores: np.ndarray        # method-specific OOD scores (higher = more novel)
    threshold: float          # threshold used to flag unknown
    method: str


def _logits_from_lgbm_proba(proba: np.ndarray) -> np.ndarray:
    """LightGBM returns softmax probabilities; recover unnormalised logits up to a constant."""
    eps = 1e-12
    return np.log(np.clip(proba, eps, 1.0))


def _msp_score(proba: np.ndarray) -> np.ndarray:
    """Score = 1 - max softmax probability. Higher score = more uncertain = more OOD."""
    return 1.0 - np.max(proba, axis=1)


def _energy_score(proba: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score under probability-only access. Higher = more OOD.

    The canonical score is E(x) = -T * logsumexp(z / T), defined on logits z.
    Without z the only available substitution is z ~ log p, and at T = 1 that
    is degenerate: -logsumexp(log p) = -log(sum_k p_k) = 0 for every input,
    because probabilities sum to one. Implementations therefore fall back to
    the surrogate below, which is the T -> 0 limit of the family
    -T * log sum_k p_k^(1/T).

    That surrogate is -log(max_k p_k), a strictly increasing transform of the
    MSP score 1 - max_k p_k. Quantile calibration commutes with strictly
    increasing maps, so this rule and MSP flag identical sets at any quantile.
    They are one rule, not two. See src/run_open_set_temperature.py for the
    measurement, and note that the T argument is unused here for exactly this
    reason.
    """
    return -np.log(np.clip(np.max(proba, axis=1), 1e-12, 1.0))


def _knn_score(proba_test: np.ndarray, proba_train: np.ndarray, k: int = 50) -> np.ndarray:
    """Deep-Nearest-Neighbour OOD score (Sun et al. ICML 2022).

    For each test sample, compute distance to its k-th nearest neighbour in
    the training-set softmax-probability space. Far from training mass = OOD.
    Higher score = more OOD.

    The Sun et al. paper uses penultimate-layer features; we adapt to softmax
    probabilities for compatibility with our tree-based and deep classifiers
    that all expose the same softmax interface.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(proba_train)
    k_eff = min(k, max(1, n - 1))
    # Subsample train to bound runtime if very large
    if n > 30_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=30_000, replace=False)
        train_sub = proba_train[idx]
    else:
        train_sub = proba_train
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1).fit(train_sub)
    distances, _ = nn.kneighbors(proba_test)
    return distances[:, -1]  # distance to k-th NN


def _mahalanobis_score(proba_test: np.ndarray, proba_train: np.ndarray,
                        y_train: np.ndarray) -> np.ndarray:
    """Per-class centroid + shared covariance in softmax-probability space.
    Score for x is min over classes of d_M(x, c_k); then we threshold.
    """
    classes = np.unique(y_train)
    n_dim = proba_train.shape[1]
    centroids = {}
    for c in classes:
        centroids[c] = proba_train[y_train == c].mean(axis=0)
    # Pooled within-class covariance
    diffs = []
    for c in classes:
        m = proba_train[y_train == c]
        diffs.append(m - centroids[c])
    pooled = np.vstack(diffs)
    cov = np.cov(pooled.T) + 1e-3 * np.eye(n_dim)
    cov_inv = np.linalg.pinv(cov)

    # min Mahalanobis distance to any class centroid
    n = len(proba_test)
    dists = np.zeros((n, len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        diff = proba_test - centroids[c]
        d = np.einsum("nd,de,ne->n", diff, cov_inv, diff)
        dists[:, i] = d
    return dists.min(axis=1)


def detect_open_set(method: str, proba_test: np.ndarray,
                    proba_train: np.ndarray = None, y_train: np.ndarray = None,
                    threshold_quantile: float = 0.95,
                    threshold: float = None) -> OpenSetResult:
    """Compute OOD scores + flag unknowns.

    Args:
        method: 'msp', 'energy', or 'mahalanobis'.
        proba_test: softmax probabilities on test set, shape (n_test, n_classes).
        proba_train: train softmax probabilities (used for threshold calibration
                     and Mahalanobis statistics).
        y_train: train labels (multi-class int) for Mahalanobis.
        threshold_quantile: percentile of TRAIN scores to set threshold.
                            Higher = more conservative (fewer Unknown flags).
        threshold: override automatic threshold.

    Returns OpenSetResult with pred (with UNKNOWN_LABEL for flagged), proba,
    scores, threshold, method.
    """
    if method == "msp":
        score_test = _msp_score(proba_test)
        if threshold is None:
            assert proba_train is not None
            score_train = _msp_score(proba_train)
            threshold = float(np.quantile(score_train, threshold_quantile))
    elif method == "energy":
        score_test = _energy_score(proba_test)
        if threshold is None:
            assert proba_train is not None
            score_train = _energy_score(proba_train)
            threshold = float(np.quantile(score_train, threshold_quantile))
    elif method == "mahalanobis":
        assert proba_train is not None and y_train is not None
        score_test = _mahalanobis_score(proba_test, proba_train, y_train)
        if threshold is None:
            score_train = _mahalanobis_score(proba_train, proba_train, y_train)
            threshold = float(np.quantile(score_train, threshold_quantile))
    elif method == "knn":
        assert proba_train is not None
        score_test = _knn_score(proba_test, proba_train, k=50)
        if threshold is None:
            # Use train distance-to-self k-NN (excluding self) for calibration
            score_train = _knn_score(proba_train, proba_train, k=50)
            threshold = float(np.quantile(score_train, threshold_quantile))
    else:
        raise ValueError(f"Unknown open-set method: {method}")

    pred = np.argmax(proba_test, axis=1)
    unknown_mask = score_test > threshold
    pred[unknown_mask] = UNKNOWN_LABEL
    return OpenSetResult(pred=pred, proba=proba_test, scores=score_test,
                         threshold=threshold, method=method)


def open_set_metrics(y_true: np.ndarray, y_pred_with_unknown: np.ndarray,
                     known_classes: list, unknown_label_in_truth: int = None) -> Dict[str, float]:
    """Evaluate open-set predictions.

    For attack-holdout: y_true samples that belong to *unseen* test attacks should
    be flagged as Unknown by the detector. Caller should pass
    unknown_label_in_truth = -1 for those samples (after relabeling).

    Returns:
        - macro_f1_open: macro-F1 across (known classes + Unknown)
        - novel_recall: recall of Unknown class only (= correct novelty detection rate)
        - known_classification_acc: accuracy on known-class samples that were not flagged
        - false_unknown_rate: fraction of known-class samples wrongly flagged as Unknown
    """
    metrics = {}
    # Build the "open" label space: known_classes + UNKNOWN_LABEL
    unknown_in_truth = unknown_label_in_truth if unknown_label_in_truth is not None else UNKNOWN_LABEL
    label_universe = sorted(set(known_classes) | {unknown_in_truth})

    # Map predictions: keep known class predictions as-is; UNKNOWN_LABEL stays UNKNOWN_LABEL
    metrics["macro_f1_open"] = float(f1_score(
        y_true, y_pred_with_unknown, labels=label_universe, average="macro", zero_division=0
    ))
    # Novel recall: of true Unknown samples, how many flagged Unknown?
    if (y_true == unknown_in_truth).any():
        metrics["novel_recall"] = float(
            ((y_pred_with_unknown == UNKNOWN_LABEL) & (y_true == unknown_in_truth)).sum()
            / max((y_true == unknown_in_truth).sum(), 1)
        )
    else:
        metrics["novel_recall"] = 0.0
    # False unknown rate on known-class samples
    known_mask = y_true != unknown_in_truth
    if known_mask.any():
        metrics["false_unknown_rate"] = float(
            (y_pred_with_unknown[known_mask] == UNKNOWN_LABEL).mean()
        )
    else:
        metrics["false_unknown_rate"] = 0.0
    # Classification accuracy on known-class samples that were not flagged
    not_flagged = (y_pred_with_unknown != UNKNOWN_LABEL) & known_mask
    if not_flagged.any():
        metrics["known_classification_acc"] = float(
            (y_pred_with_unknown[not_flagged] == y_true[not_flagged]).mean()
        )
    else:
        metrics["known_classification_acc"] = 0.0
    return metrics
