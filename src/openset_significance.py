"""
Paired significance statistics for the open-set grid:
  - Wilcoxon signed-rank, KNN-OOD against MSP per base classifier, paired by
    seed.
  - Per-class novel recall for KNN-OOD on the attack-holdout split, reported per
    held-out attack type.

Output: results/openset_significance.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.run_experiment import load_paths, load_or_build_master, get_class_names
from src.splits import HOLDOUT_TRAIN_ATTACKS, HOLDOUT_TEST_ATTACKS, get_split
from src.features import get_feature_set
from src.models import fit_lightgbm, fit_xgboost
from src.open_set import (
    _msp_score, _energy_score, _mahalanobis_score, _knn_score,
    UNKNOWN_LABEL,
)


def compute_wilcoxon(in_dir):
    """For each base classifier, paired Wilcoxon KNN-OOD vs MSP across seeds.
    Pairing: same seed for both methods.
    """
    files = list(Path(in_dir).glob("openset_*.json"))
    by_base_method = defaultdict(dict)  # (base, method) -> {seed: row}
    for f in files:
        rec = json.loads(f.read_text())
        by_base_method[(rec["base_model"], rec["method"])][rec["seed"]] = rec

    out = {}
    metrics_to_test = ["macro_f1_open", "novel_recall"]
    for base in ["lightgbm", "xgboost", "tabnet", "ftt"]:
        knn = by_base_method.get((base, "knn"), {})
        msp = by_base_method.get((base, "msp"), {})
        common = sorted(set(knn.keys()) & set(msp.keys()))
        if len(common) < 3:
            continue
        knn_arr_f1 = np.array([knn[s]["macro_f1_open"] for s in common])
        msp_arr_f1 = np.array([msp[s]["macro_f1_open"] for s in common])
        knn_arr_nr = np.array([knn[s]["novel_recall"] for s in common])
        msp_arr_nr = np.array([msp[s]["novel_recall"] for s in common])

        try:
            stat_f1, p_f1 = wilcoxon(knn_arr_f1, msp_arr_f1, alternative="greater")
        except Exception:
            stat_f1, p_f1 = None, None
        try:
            stat_nr, p_nr = wilcoxon(knn_arr_nr, msp_arr_nr, alternative="greater")
        except Exception:
            stat_nr, p_nr = None, None

        out[base] = {
            "n_seeds_paired": len(common),
            "knn_f1_mean": float(knn_arr_f1.mean()),
            "msp_f1_mean": float(msp_arr_f1.mean()),
            "delta_f1": float(knn_arr_f1.mean() - msp_arr_f1.mean()),
            "wilcoxon_f1_p_one_sided": float(p_f1) if p_f1 is not None else None,
            "knn_novel_recall_mean": float(knn_arr_nr.mean()),
            "msp_novel_recall_mean": float(msp_arr_nr.mean()),
            "delta_novel_recall": float(knn_arr_nr.mean() - msp_arr_nr.mean()),
            "wilcoxon_novel_recall_p_one_sided": float(p_nr) if p_nr is not None else None,
        }
    return out


def compute_per_class_novel_recall(seed=42):
    """Re-run XGBoost+KNN-OOD and FTT+KNN-OOD on seed 42 attack-holdout,
    then break down novel-recall per held-out attack class (TCPConnectScan,
    SYNFlood, SlowrateDoS).

    For runtime, we limit to the same 300k subsample as the main experiments
    and re-use cached models if present.
    """
    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=300_000, random_state=seed)
    _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
    df = df.iloc[sub_idx].reset_index(drop=True)
    if classes:
        df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("holdout_attack", df, seed)
    feat_cols = get_feature_set("full", df, train_idx=train_idx, seed=seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    test_attack_indices = {cls_to_idx[c]: c for c in HOLDOUT_TEST_ATTACKS}

    out = {}
    # XGBoost only (faster); skip FTT (too slow per request)
    for base_name, base_fn, predict_fn in [
        ("xgboost", "xgboost", "xgboost"),
        ("lightgbm", "lightgbm", "lightgbm"),
    ]:
        if base_name == "xgboost":
            import xgboost as xgb
            model = fit_xgboost(X_train, y_train, num_classes=num_classes, seed=seed)
            proba_train = model.predict(xgb.DMatrix(X_train))
            proba_test = model.predict(xgb.DMatrix(X_test))
        else:
            model = fit_lightgbm(X_train, y_train, num_classes=num_classes, seed=seed)
            proba_train = model.predict(X_train)
            proba_test = model.predict(X_test)

        # KNN-OOD score
        score_train = _knn_score(proba_train, proba_train, k=50)
        score_test = _knn_score(proba_test, proba_train, k=50)
        threshold = float(np.quantile(score_train, 0.95))

        pred = np.argmax(proba_test, axis=1)
        unknown_mask = score_test > threshold
        pred[unknown_mask] = UNKNOWN_LABEL

        per_class = {}
        for cls_idx, cls_name in test_attack_indices.items():
            mask = (y_test == cls_idx)
            n_total = int(mask.sum())
            n_flagged = int(((pred == UNKNOWN_LABEL) & mask).sum())
            recall = n_flagged / max(n_total, 1)
            per_class[cls_name] = {
                "n_total": n_total,
                "n_flagged_unknown": n_flagged,
                "novel_recall": float(recall),
            }
        out[base_name] = {
            "method": "knn",
            "seed": seed,
            "threshold_q": 0.95,
            "threshold": threshold,
            "per_class": per_class,
        }
    return out


def benchmark_inference_latency(seed=42):
    """ Measure MSP and KNN-OOD inference latency per flow on a 60k test set.
    Reports flows/sec for each scoring method, plus base-classifier prediction cost.
    """
    paths = load_paths()
    df = load_or_build_master(paths)
    classes = get_class_names(paths)
    if classes:
        df.attrs["_class_names"] = classes

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=300_000, random_state=seed)
    _, sub_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
    df = df.iloc[sub_idx].reset_index(drop=True)
    if classes:
        df.attrs["_class_names"] = classes

    train_idx, test_idx = get_split("holdout_attack", df, seed)
    feat_cols = get_feature_set("full", df, train_idx=train_idx, seed=seed)
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_multi"].values.astype(np.int32)
    num_classes = int(y.max()) + 1
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    n_test = len(X_test)

    import xgboost as xgb
    model = fit_xgboost(X_train, y_train, num_classes=num_classes, seed=seed)

    # Base classifier softmax prediction (warm-up then time)
    dm_test = xgb.DMatrix(X_test)
    _ = model.predict(dm_test)  # warm-up
    t0 = time.time()
    proba_test = model.predict(dm_test)
    t_proba = time.time() - t0

    # Get train probabilities (one-shot, not measured per-query)
    proba_train = model.predict(xgb.DMatrix(X_train))

    # MSP score (no train data needed)
    _ = _msp_score(proba_test[:1000])  # warm-up
    t0 = time.time()
    msp_scores = _msp_score(proba_test)
    t_msp = time.time() - t0

    # KNN-OOD score (one-shot index build + per-query)
    # Build index (one-shot, deployment time)
    from sklearn.neighbors import NearestNeighbors
    n = len(proba_train)
    if n > 30_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=30_000, replace=False)
        train_sub = proba_train[idx]
    else:
        train_sub = proba_train
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=50, algorithm="auto", n_jobs=-1).fit(train_sub)
    t_knn_index = time.time() - t0

    # Query (warm-up then time)
    _ = nn.kneighbors(proba_test[:1000])
    t0 = time.time()
    distances, _ = nn.kneighbors(proba_test)
    knn_scores = distances[:, -1]
    t_knn = time.time() - t0

    return {
        "n_test": n_test,
        "n_train_index": len(train_sub),
        "n_classes": int(num_classes),
        "base_proba_time_sec": t_proba,
        "base_proba_throughput_flows_per_sec": n_test / t_proba,
        "msp_score_time_sec": t_msp,
        "msp_score_throughput_flows_per_sec": n_test / t_msp,
        "knn_index_build_time_sec": t_knn_index,
        "knn_score_time_sec": t_knn,
        "knn_score_throughput_flows_per_sec": n_test / t_knn,
        "msp_overhead_pct_of_base": 100 * t_msp / t_proba,
        "knn_overhead_pct_of_base": 100 * t_knn / t_proba,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="results/metrics")
    ap.add_argument("--out", default="results/openset_significance.json")
    ap.add_argument("--skip-perclass", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    out = {}

    print("Computing  Wilcoxon KNN vs MSP...")
    out["wilcoxon_knn_vs_msp"] = compute_wilcoxon(args.in_dir)
    for base, r in out["wilcoxon_knn_vs_msp"].items():
        print(f"  {base}: dF1={r['delta_f1']:+.4f} p_one_sided={r['wilcoxon_f1_p_one_sided']:.4g}; "
              f"d_novel={r['delta_novel_recall']:+.4f} p_one_sided={r['wilcoxon_novel_recall_p_one_sided']:.4g}")

    if not args.skip_perclass:
        print("\nComputing  per-class novel-recall...")
        out["per_class_novel_recall"] = compute_per_class_novel_recall(seed=42)
        for base, r in out["per_class_novel_recall"].items():
            print(f"  {base} seed42 KNN-OOD:")
            for cls_name, m in r["per_class"].items():
                print(f"    {cls_name}: {m['n_flagged_unknown']}/{m['n_total']} = {m['novel_recall']:.4f}")

    if not args.skip_latency:
        print("\nComputing  KNN-OOD vs MSP latency...")
        out["latency_benchmark"] = benchmark_inference_latency(seed=42)
        r = out["latency_benchmark"]
        print(f"  Base proba    : {r['base_proba_throughput_flows_per_sec']:.0f} flows/sec")
        print(f"  MSP scoring   : {r['msp_score_throughput_flows_per_sec']:.0f} flows/sec")
        print(f"  KNN scoring   : {r['knn_score_throughput_flows_per_sec']:.0f} flows/sec")
        print(f"  MSP overhead  : {r['msp_overhead_pct_of_base']:.2f}% of base")
        print(f"  KNN overhead  : {r['knn_overhead_pct_of_base']:.2f}% of base")
        print(f"  KNN index build (one-shot, deployment time): {r['knn_index_build_time_sec']:.2f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
