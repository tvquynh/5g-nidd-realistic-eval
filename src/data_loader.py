"""
Load 5G-NIDD master data from author-provided Encoded.csv (Stage 7a).

This file is the author's ML-ready version per their dataload.txt:
    https://github.com/yushan1986/5g-nidd/blob/main/dataload.txt

Schema (96 cols):
    [0]  Unnamed: 0           (pandas index from author save; drop)
    [1-43] 43 numeric raw features (Seq, Dur, RunTime, Mean, sTtl, ..., AckDat)
    [44] Label                 (binary; we use only as fallback target)
    [45] Attack Type           (multiclass target)
    [46] Attack Tool           (one-to-one with Attack Type; drop)
    [47-60] Flgs one-hot       (14 cols, e.g. ' e        ', ' *    f   ')
    [61-68] Proto one-hot      (arp, icmp, ipv6-icmp, llc, lldp, sctp, tcp, udp)
    [69-82] State one-hot      (ACC, CON, ECO, FIN, INT, NRS, REQ, RSP, RST, TST, URP, Shutdown, Start, Status)
    [83-95] sTos/sDSb one-hot  (DSCP names: 39, 4, 52, 54, af11, af12, af41, cs0, cs4, cs6, cs7, ef, nan)

We additionally drop sVid, dVid, '54' per the author's dataload.txt:
    df.drop(columns=['Attack Tool', 'Label', 'sVid', 'dVid', '54'])

We reconstruct the per-base-station marker from row position. Verified by
comparing boundary rows: rows 0..728315 in Encoded.csv exactly match BTS_1.csv
(728316 rows), and rows 728316..1215889 match BTS_2.csv (487574 rows).

We add capture_day (1=2022-08-26, 2=2022-08-29) from Attack Type per
Descriptor TABLE III + IV.
"""
from pathlib import Path
import pandas as pd
import os

BS1_ROW_COUNT = 728_316  # verified from BTS_1.csv
BS2_ROW_COUNT = 487_574  # verified from BTS_2.csv
TOTAL_ROWS = BS1_ROW_COUNT + BS2_ROW_COUNT  # 1,215,890

# Day mapping (Descriptor TABLE III + IV)
ATTACK_TO_DAY = {
    "Benign":          1,
    "ICMPFlood":       1,
    "SYNScan":         1,
    "TCPConnectScan":  1,
    "UDPFlood":        1,
    "UDPScan":         1,
    "HTTPFlood":       2,
    "SYNFlood":        2,
    "SlowrateDoS":     2,
}


def load_encoded(encoded_path: str) -> pd.DataFrame:
    """Load Encoded.csv with author's preprocessing + reconstructed BS marker.

    Returns DataFrame with:
        - 89 feature cols (43 numeric + 49 one-hot - 3 dropped sVid/dVid/'54')
        - 'Attack Type' column (target, multiclass string)
        - 'BS' column (1 or 2, reconstructed from row position)
        - 'capture_day' column (1 or 2, from Attack Type mapping)
    """
    df = pd.read_csv(encoded_path, low_memory=False)
    n = len(df)
    if n != TOTAL_ROWS:
        raise ValueError(f"Encoded.csv expected {TOTAL_ROWS} rows, got {n}")

    # Reconstruct BS marker from row position (verified order BTS_1 -> BTS_2)
    bs = (df.index >= BS1_ROW_COUNT).astype("int8") + 1  # 1 for first BS1_ROW_COUNT, then 2
    df["BS"] = bs

    # Author preprocessing per dataload.txt
    df = df.iloc[:, 1:]  # drop Unnamed: 0 (col 0)

    # Drop columns per author + drop redundant Attack Tool, Label
    drop_cols = []
    for col in ("Attack Tool", "Label", "sVid", "dVid", "54"):
        if col in df.columns:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)

    # NaN -> median (author's choice)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Add capture_day from Attack Type mapping
    df["capture_day"] = df["Attack Type"].map(ATTACK_TO_DAY).fillna(1).astype("int8")
    return df


def load_5g_nidd_master(encoded_path: str = None) -> pd.DataFrame:
    """Top-level loader. encoded_path defaults to configs/paths.yaml resolution."""
    if encoded_path is None:
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        profile = os.environ.get("NIDD_PROFILE", "local")
        configured = cfg[profile].get("encoded_csv")
        encoded_path = str(configured) if configured else str(
            Path(cfg[profile]["raw_csv_root"]) / "stage5" / "Encoded.csv"
        )
    return load_encoded(encoded_path)


if __name__ == "__main__":
    df = load_5g_nidd_master()
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols")
    print(f"BS distribution: {df['BS'].value_counts().to_dict()}")
    print(f"Day distribution: {df['capture_day'].value_counts().to_dict()}")
    print(f"\nAttack Type:")
    print(df["Attack Type"].value_counts())
    print(f"\nClass x BS x Day:")
    print(df.groupby(['capture_day', 'BS', 'Attack Type']).size().unstack('Attack Type', fill_value=0))
