"""Aggregate where unflagged novel flows land, and how confident the base is.

Answers two questions the grid cannot. When no rejection rule is applied, the
novel-recall of zero is a property of the output alphabet; the destination of
those flows is not. And the claim that a learned representation would place
unfamiliar inputs somewhere unfamiliar predicts low confidence on novel flows,
which is measurable.

Usage:
    python -m src.aggregate_openset_destinations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

BASE_ORDER = ["lightgbm", "xgboost", "tabnet", "ftt"]
BASE_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost",
               "tabnet": "TabNet", "ftt": "FT-Transformer"}
ATTACK_ORDER = ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]
TRAINED_ATTACKS = ["HTTPFlood", "SYNScan", "UDPScan", "UDPFlood", "ICMPFlood"]
HELD_OUT = ["SlowrateDoS", "TCPConnectScan", "SYNFlood"]


def _split(counts: dict) -> dict:
    """Three-way destination split.

    The classifier has one output slot per label in the dataset, but only six
    of the nine carry training data. A prediction landing on one of the three
    empty slots is neither a benign verdict nor a known-attack verdict, so it
    is counted separately rather than folded into either.
    """
    total = sum(counts.values())
    benign = counts.get("Benign", 0)
    trained = sum(counts.get(a, 0) for a in TRAINED_ATTACKS)
    untrained = sum(counts.get(a, 0) for a in HELD_OUT)
    assert benign + trained + untrained == total, counts
    return {"to_benign": benign / total, "to_trained_attack": trained / total,
            "to_untrained_slot": untrained / total}


def load(directory: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(directory.glob("*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    directory = Path(args.dir) if args.dir else results / "openset_destinations"
    records = load(directory)
    present = [b for b in BASE_ORDER if any(r["base_model"] == b for r in records)]
    print(f"Loaded {len(records)} records from {directory}")
    print("seeds per base:",
          {b: len({r['seed'] for r in records if r['base_model'] == b})
           for b in present})

    # --- destination split --------------------------------------------------
    rows = [{"base_model": r["base_model"], "seed": r["seed"],
             **_split(r["novel_destination_counts"]),
             "novel_recall": r["novel_recall_no_rejection"],
             "macro_f1_open": r["macro_f1_open_no_rejection"],
             "conf_novel": r["confidence_novel"]["mean"],
             "conf_known": r["confidence_known"]["mean"],
             "novel_above_099": r["confidence_novel"]["fraction_above_0.99"],
             "known_above_099": r["confidence_known"]["fraction_above_0.99"]}
            for r in records]
    frame = pd.DataFrame(rows)

    print("\n=== Where novel flows go when nothing rejects them ===")
    columns = ["to_benign", "to_trained_attack", "to_untrained_slot"]
    split = frame.groupby("base_model")[columns].agg(["mean", "std"])
    print(split.reindex(present).round(4).to_string())
    means = frame.groupby("base_model")[columns].mean().reindex(present)
    print(f"\nLabeled as a known attack type: "
          f"{means.to_trained_attack.min():.1%} to "
          f"{means.to_trained_attack.max():.1%} depending on the base")
    print(f"Labeled Benign:                 "
          f"{means.to_benign.min():.1%} to {means.to_benign.max():.1%}")
    print(f"Landing on an untrained slot:   "
          f"{means.to_untrained_slot.min():.1%} to "
          f"{means.to_untrained_slot.max():.1%}")
    worst = means.to_untrained_slot.idxmax()
    per_seed = frame[frame.base_model == worst].to_untrained_slot
    print(f"  the untrained-slot share is concentrated in {BASE_LABELS[worst]}, "
          f"per-seed {per_seed.min():.1%} to {per_seed.max():.1%}")

    print("\n=== Novel-recall without a rule, recomputed here ===")
    print(f"  maximum over all records: {frame.novel_recall.max():.6f}")

    print("\n=== Confidence: is the base uncertain about novelty? ===")
    conf = frame.groupby("base_model")[["conf_novel", "conf_known",
                                        "novel_above_099", "known_above_099"]].mean()
    print(conf.reindex(present).round(4).to_string())

    # --- destination histogram, pooled with bases weighted equally ----------
    hist_rows = []
    for r in records:
        total = sum(r["novel_destination_counts"].values())
        for label, count in r["novel_destination_counts"].items():
            hist_rows.append({"base_model": r["base_model"], "seed": r["seed"],
                              "destination": label, "share": count / total})
    hist = pd.DataFrame(hist_rows)
    pooled = (hist.groupby(["destination", "base_model"]).share.mean()
              .groupby("destination").mean().sort_values(ascending=False))
    print("\n=== Which known class absorbs them, bases weighted equally ===")
    print(pooled.round(4).to_string())

    # --- per held-out attack type ------------------------------------------
    attack_rows = []
    for r in records:
        for attack, block in r["per_attack_destinations"].items():
            attack_rows.append({
                "base_model": r["base_model"], "seed": r["seed"], "attack": attack,
                **_split(block["counts"]),
                "confidence": block["confidence"]["mean"]})
    per_attack = pd.DataFrame(attack_rows)
    print("\n=== Per held-out attack type, bases weighted equally ===")
    summary = (per_attack.groupby(["attack", "base_model"])
               [["to_benign", "to_trained_attack", "to_untrained_slot",
                 "confidence"]].mean()
               .groupby("attack").mean())
    print(summary.reindex(ATTACK_ORDER).round(4).to_string())

    print("\n=== Which known class each held-out type is labeled as ===")
    dest_rows = []
    for r in records:
        for attack, block in r["per_attack_destinations"].items():
            total = sum(block["counts"].values())
            for label, count in block["counts"].items():
                dest_rows.append({"base_model": r["base_model"], "attack": attack,
                                  "destination": label, "share": count / total})
    dest = (pd.DataFrame(dest_rows)
            .groupby(["attack", "destination", "base_model"]).share.mean()
            .groupby(["attack", "destination"]).mean().unstack().fillna(0.0))
    print(dest.reindex(ATTACK_ORDER).round(4).to_string())

    prefix = (Path(args.out_prefix) if args.out_prefix
              else results / "openset_destinations_summary")
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}_per_attack.csv")
    pooled.to_frame("share").to_csv(f"{prefix}_histogram.csv")
    print(f"\nSaved: {prefix}_per_seed.csv, {prefix}_per_attack.csv, "
          f"{prefix}_histogram.csv")


if __name__ == "__main__":
    main()
