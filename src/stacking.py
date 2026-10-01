"""Heterogeneous stacking with leakage-free out-of-fold features.

The stack is built in two stages:

1.  For every base learner, stratified K-fold cross-validation over the
    training set produces out-of-fold (OOF) class probabilities. Each base is
    then refitted on the full training set to score the test set. Holding the
    base predictions to OOF on the training set is what keeps the meta-learner
    from being fitted on predictions the bases have already memorized.
2.  A meta-learner is fitted on the horizontally concatenated OOF probability
    blocks and applied to the corresponding test blocks.

Five meta-learners are provided so that the fusion rule itself can be
ablated rather than assumed:

``lr``
    Multinomial logistic regression on the concatenated probability blocks.
``weighted``
    Convex combination of the base probability matrices, with weights fitted
    on the OOF blocks by minimizing multiclass log loss on the simplex. The
    weights are directly readable as the trust the fusion places in each base.
``confidence``
    Logistic regression on the probability blocks augmented with per-base
    confidence descriptors (maximum probability and predictive entropy).
``calibrated``
    Per-base temperature scaling fitted on the OOF blocks, followed by the
    ``lr`` meta-learner on the calibrated probabilities.
``best_single``
    Selects the single base with the highest OOF macro-F1 and uses it alone.
    This is the control that shows whether fusion helps at all.

The module also records diagnostics that explain *why* a fusion rule behaves
the way it does under distribution shift: how the meta-learner distributes
weight across bases, and how the OOF ranking of the bases compares with their
ranking on the shifted test set.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

from src.models import MODEL_REGISTRY

# Base sets. ``trees_mlp`` reproduces the conventional gradient-boosting plus
# shallow-network stack; ``diverse`` additionally admits the two learners that
# transfer best across base stations, so that the fusion rule has the option
# of relying on them.
BASE_SETS: Dict[str, Tuple[str, ...]] = {
    "trees_mlp": ("lightgbm", "xgboost", "mlp"),
    "diverse": ("lightgbm", "xgboost", "rf", "lr", "mlp"),
}

META_LEARNERS: Tuple[str, ...] = ("lr", "weighted", "confidence", "calibrated", "best_single")

EPS = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sklearn_classes(model) -> Optional[np.ndarray]:
    """Return ``classes_`` for a fitted estimator, unwrapping (model, scaler)."""
    inner = model[0] if isinstance(model, tuple) else model
    return getattr(inner, "classes_", None)


def _expand_proba(proba: np.ndarray, classes: Optional[np.ndarray], num_classes: int) -> np.ndarray:
    """Place a probability matrix into the global class layout.

    Under the cross-station and attack-holdout splits a training set need not
    contain every class. Scikit-learn estimators then emit one column per
    observed class, which must be scattered back into the global ordering
    before blocks from different bases can be concatenated.
    """
    proba = np.asarray(proba, dtype=np.float32)
    if proba.shape[1] == num_classes and classes is None:
        return proba
    if classes is not None and len(classes) == num_classes:
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=np.float32)
    if classes is None:
        out[:, : proba.shape[1]] = proba
        return out
    for col, cls in enumerate(classes):
        out[:, int(cls)] = proba[:, col]
    return out


def _predict_proba_global(name: str, model, X: np.ndarray, num_classes: int) -> np.ndarray:
    """Predict class probabilities in the global class layout."""
    _, predict_fn = MODEL_REGISTRY[name]
    _, proba = predict_fn(model, X)
    return _expand_proba(proba, _sklearn_classes(model), num_classes)


def _fit_base(name: str, X: np.ndarray, y: np.ndarray, num_classes: int, seed: int):
    fit_fn, _ = MODEL_REGISTRY[name]
    return fit_fn(X, y, X_val=None, y_val=None, num_classes=num_classes, seed=seed)


def _safe_kfold(y: np.ndarray, k_folds: int, seed: int) -> StratifiedKFold:
    """Stratified K-fold with the fold count capped by the rarest class."""
    _, counts = np.unique(y, return_counts=True)
    k = int(max(2, min(k_folds, counts.min())))
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def _entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _log_loss_from_proba(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba, EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _fit_convex_weights(blocks: Sequence[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Fit simplex weights over base probability matrices by log loss."""
    n_bases = len(blocks)
    stacked = np.stack(blocks, axis=0)  # [B, N, C]

    def objective(w: np.ndarray) -> float:
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            return 1e6
        mixed = np.tensordot(w / total, stacked, axes=(0, 0))
        return _log_loss_from_proba(mixed, y)

    x0 = np.full(n_bases, 1.0 / n_bases)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_bases,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 200, "ftol": 1e-8},
    )
    w = np.clip(result.x, 0.0, None)
    return w / w.sum() if w.sum() > 0 else x0


def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Fit a scalar temperature that minimizes log loss of a probability block."""
    log_p = np.log(np.clip(proba, EPS, 1.0))

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        scaled = log_p / t
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return _log_loss_from_proba(exp / exp.sum(axis=1, keepdims=True), y)

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    log_p = np.log(np.clip(proba, EPS, 1.0)) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)
    exp = np.exp(log_p)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _confidence_features(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate probability blocks with per-base confidence descriptors."""
    parts: List[np.ndarray] = []
    for block in blocks:
        parts.append(block)
        parts.append(block.max(axis=1, keepdims=True).astype(np.float32))
        parts.append(_entropy(block).reshape(-1, 1).astype(np.float32))
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Stacked ensemble
# ---------------------------------------------------------------------------

@dataclass
class StackedEnsemble:
    """Out-of-fold stack with a selectable fusion rule."""

    num_classes: int
    seed: int
    base_names: Tuple[str, ...] = BASE_SETS["trees_mlp"]
    meta_learner: str = "lr"
    k_folds: int = 5

    bases: List[object] = field(default_factory=list)
    meta: Optional[object] = None
    weights: Optional[np.ndarray] = None
    temperatures: Optional[List[float]] = None
    selected_base: Optional[int] = None
    fit_times: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedEnsemble":
        oof_blocks = self.fit_bases(X, y)
        t0 = time.time()
        self._fit_meta(oof_blocks, y)
        self.fit_times["meta"] = time.time() - t0
        return self

    def clone_with_meta(self, meta_learner: str) -> "StackedEnsemble":
        """Return a view of this stack with a different fusion rule.

        The fitted base learners and their timings are shared, so a sweep over
        fusion rules costs one round of base fitting rather than one per rule.
        """
        clone = StackedEnsemble(
            num_classes=self.num_classes,
            seed=self.seed,
            base_names=self.base_names,
            meta_learner=meta_learner,
            k_folds=self.k_folds,
        )
        clone.bases = self.bases
        clone.fit_times = dict(self.fit_times)
        clone.diagnostics = {
            "oof_macro_f1_per_base": self.diagnostics.get("oof_macro_f1_per_base", {})
        }
        return clone

    def fit_bases(self, X: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Fit every base learner and return their out-of-fold probability blocks."""
        if self.meta_learner not in META_LEARNERS:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner}")

        skf = _safe_kfold(y, self.k_folds, self.seed)
        n, c = len(y), self.num_classes
        oof_blocks: List[np.ndarray] = []

        for name in self.base_names:
            oof = np.zeros((n, c), dtype=np.float32)
            t0 = time.time()
            for fold_id, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
                fold_model = _fit_base(name, X[tr_idx], y[tr_idx], c, self.seed + fold_id)
                oof[vl_idx] = _predict_proba_global(name, fold_model, X[vl_idx], c)
            self.fit_times[f"{name}_oof"] = time.time() - t0

            t0 = time.time()
            self.bases.append(_fit_base(name, X, y, c, self.seed))
            self.fit_times[f"{name}_full"] = time.time() - t0
            oof_blocks.append(oof)

        self.diagnostics["oof_macro_f1_per_base"] = {
            name: float(f1_score(y, block.argmax(axis=1), average="macro", zero_division=0))
            for name, block in zip(self.base_names, oof_blocks)
        }
        return oof_blocks

    # -- meta fitting ------------------------------------------------------

    def _fit_meta(self, oof_blocks: List[np.ndarray], y: np.ndarray) -> None:
        if self.meta_learner == "best_single":
            oof_f1 = self.diagnostics["oof_macro_f1_per_base"]
            best = max(range(len(self.base_names)), key=lambda i: oof_f1[self.base_names[i]])
            self.selected_base = best
            self.diagnostics["selected_base"] = self.base_names[best]
            return

        if self.meta_learner == "weighted":
            self.weights = _fit_convex_weights(oof_blocks, y)
            self.diagnostics["convex_weights"] = {
                name: float(w) for name, w in zip(self.base_names, self.weights)
            }
            return

        if self.meta_learner == "calibrated":
            self.temperatures = [_fit_temperature(block, y) for block in oof_blocks]
            self.diagnostics["temperatures"] = {
                name: float(t) for name, t in zip(self.base_names, self.temperatures)
            }
            oof_blocks = [
                _apply_temperature(block, t) for block, t in zip(oof_blocks, self.temperatures)
            ]

        features = (
            _confidence_features(oof_blocks)
            if self.meta_learner == "confidence"
            else np.hstack(oof_blocks)
        )
        self.meta = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=self.seed,
        )
        self.meta.fit(features, y)
        self.diagnostics["meta_coefficient_mass"] = self._coefficient_mass()

    def _coefficient_mass(self) -> Dict[str, float]:
        """Share of meta-learner coefficient magnitude attributable to each base.

        Reported as a normalized distribution so that it can be read directly
        as how the fusion rule allocates trust across the bases.
        """
        if self.meta is None:
            return {}
        coef = np.abs(np.atleast_2d(self.meta.coef_))
        per_base = self.num_classes + (2 if self.meta_learner == "confidence" else 0)
        mass: Dict[str, float] = {}
        for i, name in enumerate(self.base_names):
            block = coef[:, i * per_base : (i + 1) * per_base]
            mass[name] = float(block.sum())
        total = sum(mass.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in mass.items()}

    # -- inference ---------------------------------------------------------

    def _test_blocks(self, X: np.ndarray) -> List[np.ndarray]:
        return [
            _predict_proba_global(name, model, X, self.num_classes)
            for name, model in zip(self.base_names, self.bases)
        ]

    def predict_proba(self, X: np.ndarray, blocks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        blocks = self._test_blocks(X) if blocks is None else blocks

        if self.meta_learner == "best_single":
            return blocks[self.selected_base]

        if self.meta_learner == "weighted":
            return np.tensordot(self.weights, np.stack(blocks, axis=0), axes=(0, 0))

        if self.meta_learner == "calibrated":
            blocks = [_apply_temperature(b, t) for b, t in zip(blocks, self.temperatures)]

        features = (
            _confidence_features(blocks)
            if self.meta_learner == "confidence"
            else np.hstack(blocks)
        )
        # The meta-learner only ever sees the classes present in the training
        # station, so its output is scattered back into the global layout.
        return _expand_proba(
            self.meta.predict_proba(features), self.meta.classes_, self.num_classes
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # -- diagnostics -------------------------------------------------------

    def shift_diagnostics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
        """Compare the OOF ranking of the bases with their ranking on the test set.

        When the test set comes from a different base station, a fusion rule
        fitted on OOF predictions inherits the ordering of the bases as it was
        on the *training* station. If that ordering does not survive the shift,
        the fusion is systematically misallocated no matter how it is fitted.
        A low or negative rank correlation is therefore the mechanism behind a
        stack that fails to reach its own best member.
        """
        blocks = self._test_blocks(X_test)
        test_f1 = {
            name: float(f1_score(y_test, block.argmax(axis=1), average="macro", zero_division=0))
            for name, block in zip(self.base_names, blocks)
        }
        oof_f1 = self.diagnostics.get("oof_macro_f1_per_base", {})
        names = list(self.base_names)
        rho, p_value = (float("nan"), float("nan"))
        if len(names) >= 3:
            corr = spearmanr([oof_f1[n] for n in names], [test_f1[n] for n in names])
            rho, p_value = float(corr.statistic), float(corr.pvalue)

        stack_proba = self.predict_proba(X_test, blocks=blocks)
        stack_f1 = float(
            f1_score(y_test, stack_proba.argmax(axis=1), average="macro", zero_division=0)
        )
        best_name = max(test_f1, key=test_f1.get)
        return {
            "test_macro_f1_per_base": test_f1,
            "oof_macro_f1_per_base": oof_f1,
            "spearman_oof_vs_test_rank": rho,
            "spearman_p_value": p_value,
            "stack_macro_f1": stack_f1,
            "best_base_on_test": best_name,
            "stack_minus_best_base": stack_f1 - test_f1[best_name],
        }
