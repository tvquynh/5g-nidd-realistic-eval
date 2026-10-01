"""Probability-space energy score as a function of temperature.

The canonical energy score is defined on logits. A deployment that can read only
class probabilities must substitute pseudo-logits ``log p``, and at ``T = 1`` that
substitution is degenerate: ``-logsumexp(log p) = -log(sum_k p_k) = 0`` for every
input. The one-parameter family

    E_T(x) = -T * log sum_k p_k^(1/T)

is not degenerate for ``T != 1`` and converges to ``-log max_k p_k`` --- the
surrogate that is a monotone transform of the MSP score --- as ``T -> 0``. This
runner measures where between those two ends a given temperature falls.

One base fit per (base, seed); every temperature is scored on the same
probabilities, so a difference between temperatures is a property of the rule.

Usage:
    python -m src.run_open_set_temperature --base lightgbm --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.open_set import open_set_metrics, UNKNOWN_LABEL
from src.run_open_set_all import BASES, fit_and_predict
from src.run_experiment import get_class_names, load_or_build_master, load_paths
from src.splits import HOLDOUT_TEST_ATTACKS, HOLDOUT_TRAIN_ATTACKS, get_split

TEMPERATURES = (0.05, 0.10, 0.25, 0.50, 1.00)
EPS = 1e-12


def energy_probability_space(proba: np.ndarray, T: float) -> np.ndarray:
    """E_T(x) = -T * log sum_k p_k^(1/T), computed in the log domain.

    Higher score means more out-of-distribution, matching the convention used by
    the other rules in :mod:`src.open_set`.
    """
    log_p = np.log(np.clip(proba, EPS, 1.0))
    return -T * logsumexp(log_p / T, axis=1)


def msp_score(proba: np.ndarray) -> np.ndarray:
    return 1.0 - np.max(proba, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--threshold-quantile", type=float, default=0.95)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    df.attrs["_class_names"] = classes

    if 0 < args.subsample < len(df):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.subsample,
                                     random_state=args.seed)
        _, idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
        df = df.iloc[idx].reset_index(drop=True)
        df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("holdout_attack", df, args.seed)
    feat_cols = get_feature_set(args.features, df, train_idx=train_idx, seed=args.seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    novel_indices = {cls_to_idx[c] for c in HOLDOUT_TEST_ATTACKS if c in cls_to_idx}
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]
                            if c in cls_to_idx})
    y_test_open = np.where(np.isin(y_test, list(novel_indices)), UNKNOWN_LABEL, y_test)

    print(f"{args.base} seed {args.seed}: one fit, "
          f"{len(TEMPERATURES)} temperatures plus MSP")
    t0 = time.time()
    proba_train, proba_test = fit_and_predict(
        args.base, X_train, y_train, X_test, num_classes, args.seed)
    fit_seconds = time.time() - t0

    record = {"base_model": args.base, "seed": args.seed, "features": args.features,
              "threshold_quantile": args.threshold_quantile,
              "fit_seconds": fit_seconds, "n_train": int(len(train_idx)),
              "n_test": int(len(test_idx)), "rules": {}}

    def score_and_flag(name: str, s_train: np.ndarray, s_test: np.ndarray) -> None:
        threshold = float(np.quantile(s_train, args.threshold_quantile))
        pred = np.argmax(proba_test, axis=1)
        pred[s_test > threshold] = UNKNOWN_LABEL
        metrics = open_set_metrics(y_test_open, pred, known_classes,
                                   unknown_label_in_truth=UNKNOWN_LABEL)
        record["rules"][name] = {
            "threshold": threshold,
            "score_train_range": [float(s_train.min()), float(s_train.max())],
            "score_train_unique": int(np.unique(np.round(s_train, 12)).size),
            "flag_rate": float((s_test > threshold).mean()),
            "macro_f1_open": float(metrics["macro_f1_open"]),
            "novel_recall": float(metrics["novel_recall"]),
            "false_unknown_rate": float(metrics["false_unknown_rate"]),
        }
        r = record["rules"][name]
        print(f"  {name:<14} threshold {threshold: .6g}  flag-rate {r['flag_rate']:.4f}"
              f"  novel-recall {r['novel_recall']:.4f}"
              f"  macro-F1-open {r['macro_f1_open']:.4f}")

    score_and_flag("msp", msp_score(proba_train), msp_score(proba_test))
    for T in TEMPERATURES:
        score_and_flag(f"energy_T{T:g}",
                       energy_probability_space(proba_train, T),
                       energy_probability_space(proba_test, T))

    out_dir = Path(args.out_dir) if args.out_dir else Path(paths["results"]) / "openset_temperature"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"temp_{args.base}_{args.features}_seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2, default=float))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
