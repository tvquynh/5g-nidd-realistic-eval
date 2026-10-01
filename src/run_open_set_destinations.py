"""Where do novel-attack flows go when nothing rejects them?

Novel-recall counts flows flagged \"unknown\". Without a rejection rule
Eq. (1) reduces to argmax over the known classes, so that count is zero by
construction on any architecture. The quantity that is not fixed by
construction is the destination: a novel flow that is not flagged is still
given a label, and whether that label is Benign or a known attack class decides
whether the unwrapped detector stays silent or alerts under the wrong name.

This runner measures the destination histogram, per held-out attack type, and
the confidence the base classifier assigns to novel flows against known ones.
The second is what the phrase \"a learned representation would place unfamiliar
inputs somewhere unfamiliar\" actually predicts, and it is testable.

One fit per (classifier, seed), same protocol as the grid, so the campaign is
its own shared-fit run rather than a re-scoring of stored predictions.

Usage:
    python -m src.run_open_set_destinations --base lightgbm --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import get_feature_set
from src.open_set import open_set_metrics, UNKNOWN_LABEL
from src.run_open_set_all import BASES, fit_and_predict
from src.run_experiment import get_class_names, load_or_build_master, load_paths
from src.splits import HOLDOUT_TEST_ATTACKS, HOLDOUT_TRAIN_ATTACKS, get_split

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _confidence(proba: np.ndarray) -> dict:
    """Summary of the maximum class probability over a set of flows."""
    peak = np.max(proba, axis=1)
    return {
        "mean": float(peak.mean()),
        "quantiles": {f"{q:.2f}": float(np.quantile(peak, q)) for q in QUANTILES},
        "fraction_above_0.99": float((peak > 0.99).mean()),
        "fraction_below_0.50": float((peak < 0.50).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="full")
    ap.add_argument("--subsample", type=int, default=300_000)
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
    novel_names = [c for c in HOLDOUT_TEST_ATTACKS if c in cls_to_idx]
    novel_indices = {cls_to_idx[c] for c in novel_names}
    known_classes = sorted({cls_to_idx[c] for c in HOLDOUT_TRAIN_ATTACKS + ["Benign"]
                            if c in cls_to_idx})
    benign_idx = cls_to_idx["Benign"]
    y_test_open = np.where(np.isin(y_test, list(novel_indices)), UNKNOWN_LABEL, y_test)

    print(f"{args.base} seed {args.seed}: one fit, destination and confidence census")
    t0 = time.time()
    proba_train, proba_test = fit_and_predict(
        args.base, X_train, y_train, X_test, num_classes, args.seed)
    fit_seconds = time.time() - t0
    print(f"  fitted in {fit_seconds:.1f}s")

    pred = np.argmax(proba_test, axis=1)
    novel_mask = np.isin(y_test, list(novel_indices))
    known_mask = ~novel_mask
    n_novel = int(novel_mask.sum())

    counts = np.bincount(pred[novel_mask], minlength=num_classes)
    to_benign = int(counts[benign_idx])
    record = {
        "base_model": args.base, "seed": args.seed, "features": args.features,
        "fit_seconds": fit_seconds,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_novel": n_novel, "n_known": int(known_mask.sum()),
        "novel_attack_types": novel_names,
        "novel_destination_counts": {classes[i]: int(c)
                                     for i, c in enumerate(counts) if c},
        "novel_to_benign": to_benign,
        "novel_to_benign_rate": to_benign / n_novel,
        "novel_to_attack_rate": (n_novel - to_benign) / n_novel,
        "per_attack_destinations": {},
        "confidence_novel": _confidence(proba_test[novel_mask]),
        "confidence_known": _confidence(proba_test[known_mask]),
        "confidence_train": _confidence(proba_train),
    }

    for name in novel_names:
        rows = y_test == cls_to_idx[name]
        per = np.bincount(pred[rows], minlength=num_classes)
        record["per_attack_destinations"][name] = {
            "n": int(rows.sum()),
            "to_benign_rate": float(per[benign_idx] / max(rows.sum(), 1)),
            "counts": {classes[i]: int(c) for i, c in enumerate(per) if c},
            "confidence": _confidence(proba_test[rows]),
        }

    # The no-rejection metrics, recomputed here, must match the grid campaign.
    metrics = open_set_metrics(y_test_open, pred.copy(), known_classes,
                               unknown_label_in_truth=UNKNOWN_LABEL)
    record["macro_f1_open_no_rejection"] = float(metrics["macro_f1_open"])
    record["novel_recall_no_rejection"] = float(metrics["novel_recall"])
    record["known_classification_acc"] = float(metrics["known_classification_acc"])

    print(f"  novel flows: {n_novel:,}  -> Benign {to_benign:,} "
          f"({record['novel_to_benign_rate']:.4f}), "
          f"-> known attack {record['novel_to_attack_rate']:.4f}")
    print(f"  destinations: {record['novel_destination_counts']}")
    print(f"  mean max-probability  novel {record['confidence_novel']['mean']:.4f}  "
          f"known {record['confidence_known']['mean']:.4f}")
    print(f"  novel-recall without a rule: {record['novel_recall_no_rejection']:.6f}")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(paths["results"]) / "openset_destinations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dest_{args.base}_{args.features}_seed{args.seed}.json"
    out.write_text(json.dumps(record, indent=2, default=float))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
