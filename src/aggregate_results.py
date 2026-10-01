"""
Aggregate all metric JSON files in results/metrics/ into a single parquet table.

Usage:
    python -m src.aggregate_results --in results/metrics/ --out results/summary.parquet
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def flatten_metrics(record: dict) -> dict:
    """Flatten per_class nested dict into wide columns."""
    flat = {k: v for k, v in record.items() if k != "per_class"}
    pc = record.get("per_class", {})
    for cls_idx, m in pc.items():
        for metric_name, val in m.items():
            flat[f"class{cls_idx}_{metric_name}"] = val
    return flat


def aggregate(in_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(in_dir.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
            rec["_filename"] = f.name
            rows.append(flatten_metrics(rec))
        except Exception as e:
            print(f"Skip {f.name}: {e}")
    if not rows:
        raise FileNotFoundError(f"No JSON metrics found in {in_dir}")
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="results/metrics")
    ap.add_argument("--out", default="results/summary.parquet")
    args = ap.parse_args()
    df = aggregate(Path(args.in_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy", index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"Aggregated {len(df)} runs from {args.in_dir} -> {out}")
    print(f"Cols: {list(df.columns)[:15]}...")
    print(f"Models: {df['model'].unique() if 'model' in df.columns else 'N/A'}")
    print(f"Splits: {df['split'].unique() if 'split' in df.columns else 'N/A'}")


if __name__ == "__main__":
    main()
