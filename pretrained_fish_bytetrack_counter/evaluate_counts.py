from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare predicted video counts against manual counts.")
    parser.add_argument("--pred", required=True, help="outputs/.../video_counts.csv")
    parser.add_argument("--manual", required=True, help="CSV with columns: source,manual_count")
    parser.add_argument("--out", default="outputs/count_error_report.csv")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred)
    manual = pd.read_csv(args.manual)
    df = pred.merge(manual, on="source", how="left")
    df["absolute_error"] = (df["predicted_count"] - df["manual_count"]).abs()
    df["signed_error"] = df["predicted_count"] - df["manual_count"]
    df["percent_error"] = df.apply(
        lambda r: None if r["manual_count"] == 0 else 100.0 * abs(r["predicted_count"] - r["manual_count"]) / r["manual_count"],
        axis=1,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Count evaluation")
    print("================")
    print(f"Videos: {len(df)}")
    print(f"MAE: {df['absolute_error'].mean():.3f}")
    if df["percent_error"].notna().any():
        print(f"MAPE: {df['percent_error'].dropna().mean():.2f}%")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
