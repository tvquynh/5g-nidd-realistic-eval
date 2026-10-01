"""
4 split strategies for 5G-NIDD evaluation:
- random: stratified 80/20 random split
- temporal: train Day 1 (Aug 26) -> test Day 2 (Aug 29)
- cross_station: train BS1 -> test BS2 (and reverse), class-matched
- cross_station_naive: same, without rebalancing (prior + concept shift)
- prior_control: same station both sides, test priors of the other station
- holdout_attack: train 5/8 attacks, test 3 unseen (or LOAO 8 folds)

Each split returns (train_idx, test_idx) numpy arrays.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Held-out attack groups, chosen for class balance and category diversity.
# Group A (train): SYNScan, UDPScan, ICMPFlood, UDPFlood, HTTPFlood + Benign
# Group B (test):  TCPConnectScan, SYNFlood, SlowrateDoS
# Reasoning: keep at least 1 type per class category in train (scan, flood, dos),
# test has 1 unseen scan (TCPConnect), 1 unseen flood (SYN), 1 unseen DoS (Slowrate)
HOLDOUT_TRAIN_ATTACKS = ["SYNScan", "UDPScan", "ICMPFlood", "UDPFlood", "HTTPFlood"]
HOLDOUT_TEST_ATTACKS = ["TCPConnectScan", "SYNFlood", "SlowrateDoS"]


def split_random(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 80/20 random split on y_multi."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(df)), df["y_multi"].values))
    return train_idx, test_idx


def split_temporal(df: pd.DataFrame, seed: int = None, test_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Within-day temporal split: for each (BS, attack_type) group, sort by Seq
    (Argus sequence number, monotonically increasing within a capture session),
    take the last `test_ratio` portion as test. This isolates true temporal drift
    from attack novelty (test attacks are the SAME attack types as train).

    Note: original cross-day Day1->Day2 split is captured by `temporal_cross_day`.
    """
    train_idx = []
    test_idx = []
    df_with_idx = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"}) if "_orig_idx" not in df.columns else df
    sort_col = "Seq" if "Seq" in df.columns else None
    for (bs, atk), group in df.groupby(["BS", "y_multi"]):
        if sort_col is not None and sort_col in group.columns:
            sorted_group = group.sort_values(sort_col)
        else:
            sorted_group = group.sort_index()
        n = len(sorted_group)
        if n < 4:
            train_idx.append(sorted_group.index.values)
            continue
        n_test = max(1, int(n * test_ratio))
        train_idx.append(sorted_group.iloc[:-n_test].index.values)
        test_idx.append(sorted_group.iloc[-n_test:].index.values)
    if not test_idx:
        return np.concatenate(train_idx), np.array([], dtype=int)
    return np.concatenate(train_idx), np.concatenate(test_idx)


def split_temporal_cross_day(df: pd.DataFrame, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Original cross-day temporal: train Day 1 -> test Day 2. Confounded with
    attack novelty in 5G-NIDD because Day 1 and Day 2 have disjoint attacks.
    Kept for reference; not used in main paper.
    """
    train_mask = df["capture_day"].values == 1
    test_mask = df["capture_day"].values == 2
    return np.where(train_mask)[0], np.where(test_mask)[0]


def split_cross_station(df: pd.DataFrame, train_bs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Train on train_bs (1 or 2) -> Test on the other.
    Stratified rebalance per class: subsample BOTH train and test so they have
    same per-class size = min(class count BS1, class count BS2).
    """
    test_bs = 2 if train_bs == 1 else 1
    rng = np.random.default_rng(seed)

    # Per-class min count between BS1 and BS2 to balance prior
    classes = df["y_multi"].unique()
    train_indices = []
    test_indices = []
    for c in classes:
        m_train = (df["BS"].values == train_bs) & (df["y_multi"].values == c)
        m_test = (df["BS"].values == test_bs) & (df["y_multi"].values == c)
        idx_train_all = np.where(m_train)[0]
        idx_test_all = np.where(m_test)[0]
        if len(idx_train_all) == 0 or len(idx_test_all) == 0:
            continue
        n = min(len(idx_train_all), len(idx_test_all))
        train_indices.append(rng.choice(idx_train_all, size=n, replace=False))
        test_indices.append(rng.choice(idx_test_all, size=n, replace=False))

    return np.concatenate(train_indices), np.concatenate(test_indices)


def split_holdout_attack(df: pd.DataFrame, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Train on 5/8 attacks + Benign -> Test on 3 unseen attacks + Benign.
    Train benign and test benign are disjoint (random 50/50 split for benign).
    """
    rng = np.random.default_rng(seed if seed is not None else 42)
    y_str_col = df["y_label_str"] if "y_label_str" in df.columns else None
    # Use class string directly (need to reconstruct from cls_to_int if absent)
    # We assume df came from preprocess which dropped y_label_str.
    # Workaround: rebuild from attrs
    if "_class_names" in df.attrs:
        idx_to_cls = {i: c for i, c in enumerate(df.attrs["_class_names"])}
    else:
        # Fallback: use Attack Type column if not yet preprocessed
        raise ValueError("split_holdout_attack needs class mapping. Call after preprocess and pass df with attrs['_class_names']")

    cls_strings = df["y_multi"].map(idx_to_cls).values

    # Train: benign 50% + 5 train attacks all
    train_attack_mask = np.isin(cls_strings, HOLDOUT_TRAIN_ATTACKS)
    test_attack_mask = np.isin(cls_strings, HOLDOUT_TEST_ATTACKS)
    benign_mask = (cls_strings == "Benign")

    benign_idx = np.where(benign_mask)[0]
    rng.shuffle(benign_idx)
    half = len(benign_idx) // 2
    train_benign = benign_idx[:half]
    test_benign = benign_idx[half:]

    train_idx = np.concatenate([np.where(train_attack_mask)[0], train_benign])
    test_idx = np.concatenate([np.where(test_attack_mask)[0], test_benign])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def split_cross_station_naive(df: pd.DataFrame, train_bs: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Train on one base station, test on the other, with no rebalancing.

    This is the protocol used in the published cross-station literature: every
    flow of the training station trains, every flow of the other station tests.
    Because the two stations carry very different class priors, the resulting
    degradation mixes prior shift with concept shift and cannot separate them.
    It is included so that the mixed number can be measured on the same
    pipeline as the class-matched number in :func:`split_cross_station`, which
    isolates concept shift.
    """
    test_bs = 2 if train_bs == 1 else 1
    train_idx = np.where(df["BS"].values == train_bs)[0]
    test_idx = np.where(df["BS"].values == test_bs)[0]
    return train_idx, test_idx


def split_prior_control(df: pd.DataFrame, train_bs: int, seed: int,
                        test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Same station on both sides, with the test priors of the other station.

    Train and test flows are drawn from the *same* base station, so the
    feature-label relationship is unchanged and no concept shift is present.
    The held-out part is then resampled so that its class proportions match
    those of the other station. Any degradation observed here is attributable
    to the change in class priors alone, which makes this the control that
    turns the naive-versus-class-matched difference into an attributable
    decomposition.
    """
    test_bs = 2 if train_bs == 1 else 1
    rng = np.random.default_rng(seed)
    y = df["y_multi"].values
    bs = df["BS"].values

    station_idx = np.where(bs == train_bs)[0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    inner_train, inner_test = next(sss.split(np.zeros(len(station_idx)), y[station_idx]))
    train_idx = station_idx[inner_train]
    pool_idx = station_idx[inner_test]

    # Target priors: the class proportions of the other station.
    other_idx = np.where(bs == test_bs)[0]
    classes, other_counts = np.unique(y[other_idx], return_counts=True)
    target_share = other_counts / other_counts.sum()

    available = {int(c): pool_idx[y[pool_idx] == c] for c in classes}
    # The achievable test-set size is set by the class that runs out first.
    feasible = [len(available[int(c)]) / share
                for c, share in zip(classes, target_share)
                if share > 0 and len(available[int(c)]) > 0]
    if not feasible:
        raise ValueError("prior_control: no shared classes between the two stations")
    total = int(min(feasible))

    test_parts: List[np.ndarray] = []
    for c, share in zip(classes, target_share):
        n = int(round(total * share))
        pool = available[int(c)]
        if n <= 0 or len(pool) == 0:
            continue
        n = min(n, len(pool))
        test_parts.append(rng.choice(pool, size=n, replace=False))

    if not test_parts:
        raise ValueError("prior_control: empty test set after resampling")
    return train_idx, np.concatenate(test_parts)


def split_cross_station_factorial(df: pd.DataFrame, train_bs: int, seed: int,
                                  train_balance: str = "raw",
                                  test_balance: str = "raw") -> Tuple[np.ndarray, np.ndarray]:
    """Cross-station split with the two class compositions varied independently.

    Rebalancing a cross-station experiment changes two things at once: what the
    detector is trained on, and what it is scored against. Reporting only the
    unbalanced and the fully balanced cell therefore confounds the two. This
    function exposes the full two-by-two design.

    ``train_balance`` and ``test_balance`` each take ``"raw"`` (every flow of
    that station) or ``"matched"`` (per class, the smaller of the two station
    counts). The four cells are:

    ===================  ==============================================
    (raw, raw)           the protocol used in the published literature
    (raw, matched)       training composition unchanged, scoring balanced
    (matched, raw)       training composition balanced, scoring unchanged
    (matched, matched)   the class-matched protocol, concept shift only
    ===================  ==============================================

    The fully matched cell delegates to :func:`split_cross_station` so that it
    is index-identical to the previously reported configuration.
    """
    if train_balance not in {"raw", "matched"} or test_balance not in {"raw", "matched"}:
        raise ValueError("balance arguments must be 'raw' or 'matched'")
    if train_balance == "matched" and test_balance == "matched":
        return split_cross_station(df, train_bs, seed)

    test_bs = 2 if train_bs == 1 else 1
    rng = np.random.default_rng(seed)
    y = df["y_multi"].values
    bs = df["BS"].values

    def station_class_indices(station: int, cls: int) -> np.ndarray:
        return np.where((bs == station) & (y == cls))[0]

    def collect(station: int, balance: str) -> np.ndarray:
        parts: List[np.ndarray] = []
        for cls in np.unique(y):
            own = station_class_indices(station, int(cls))
            if len(own) == 0:
                continue
            if balance == "raw":
                parts.append(own)
                continue
            other = station_class_indices(2 if station == 1 else 1, int(cls))
            if len(other) == 0:
                continue  # class absent from one station cannot be matched
            n = min(len(own), len(other))
            parts.append(rng.choice(own, size=n, replace=False))
        if not parts:
            raise ValueError(f"cross_station_factorial: empty side for BS{station}")
        return np.concatenate(parts)

    return collect(train_bs, train_balance), collect(test_bs, test_balance)


def get_split(name: str, df: pd.DataFrame, seed: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to split function by name."""
    if name == "random":
        return split_random(df, seed)
    if name == "temporal":
        return split_temporal(df, seed)
    if name == "temporal_cross_day":
        return split_temporal_cross_day(df, seed)
    if name == "cross_station":
        train_bs = kwargs.get("train_bs", 1)
        return split_cross_station(df, train_bs, seed)
    if name == "cross_station_naive":
        return split_cross_station_naive(df, kwargs.get("train_bs", 1), seed)
    if name.startswith("cross_station_") and name.count("_") == 3:
        # cross_station_<train_balance>_<test_balance>
        _, _, train_balance, test_balance = name.split("_")
        return split_cross_station_factorial(
            df, kwargs.get("train_bs", 1), seed,
            train_balance=train_balance, test_balance=test_balance,
        )
    if name == "prior_control":
        return split_prior_control(df, kwargs.get("train_bs", 1), seed)
    if name == "holdout_attack":
        return split_holdout_attack(df, seed)
    raise ValueError(f"Unknown split: {name}")
