"""
Minimal preprocessing on Encoded.csv (already pre-processed by authors).

Steps:
1. data_loader.load_encoded() handles: drop Unnamed:0/Attack Tool/Label/sVid/dVid/'54',
   median fillna, add BS marker + capture_day.
2. Here we encode multiclass label (Attack Type -> y_multi int code, alphabetical).
3. Encode binary label (y_binary = (Attack Type != 'Benign')).
4. Coerce remaining cols to numeric float32.
5. Save parquet.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os


def encode_label_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    atk = df["Attack Type"].astype(str).str.strip()
    classes = sorted(atk.unique().tolist())
    cls_to_int = {c: i for i, c in enumerate(classes)}
    df["y_multi"] = atk.map(cls_to_int).astype("int32")
    df["y_binary"] = (atk != "Benign").astype("int8")
    df.attrs["classes"] = classes
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Do NOT strip column names: Encoded.csv has Flgs one-hot cols like ' e        '
    # and 'e        ' which would collide if stripped.
    df = encode_label_multiclass(df)

    # Drop string Attack Type after encoding
    df = df.drop(columns=["Attack Type"], errors="ignore")

    # Coerce features to numeric (everything except metadata)
    meta_cols = {"y_multi", "y_binary", "BS", "capture_day"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0).astype("float32")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import load_5g_nidd_master
    import yaml

    cfg_path = Path(__file__).parent.parent / "configs" / "paths.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    profile = os.environ.get("NIDD_PROFILE", "local")
    paths = cfg[profile]

    print("Loading Encoded.csv...")
    raw = load_5g_nidd_master()
    print(f"Raw: {len(raw):,} rows x {len(raw.columns)} cols")
    classes = raw.attrs.get("classes")

    print("Preprocessing...")
    clean = preprocess(raw)
    new_classes = clean.attrs.get("classes") or classes
    print(f"Clean: {len(clean):,} rows x {len(clean.columns)} cols")
    print(f"Classes ({len(new_classes)}): {new_classes}")

    out = Path(paths["data_out"]) / "master_5g_nidd.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out, compression="snappy", index=False)
    print(f"Saved: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    if new_classes:
        cls_path = out.with_suffix(".classes.json")
        cls_path.write_text(json.dumps(new_classes))
        print(f"Saved classes: {cls_path}")
