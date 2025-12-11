#!/usr/bin/env python3
"""
Lab 5 – Dimensionality exploration (NEW DATASET)
Daily Granularity
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart

DATAFILE = "TrafficTwoMonth.csv"
OUTPUT_DIR = Path("images/CORRECT")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Total"   # Your target column


# ---------------------------------------------------
# Helper: build a valid timestamp
# ---------------------------------------------------
def build_timestamp(df):
    """
    Creates a continuous 15-minute timestamp series for the dataset.
    """
    df = df.copy()

    base = pd.date_range(
        start="2023-12-06",   # same assumption as before
        periods=len(df),
        freq="15min"
    )

    df["timestamp"] = base
    return df


# ---------------------------------------------------
# Main script
# ---------------------------------------------------
def main():

    print("\n=== LAB 5 — DIMENSIONALITY (NEW DATASET, DAILY) ===\n")

    # Load data
    df = pd.read_csv(DATAFILE)

    # Build usable timestamp
    df = build_timestamp(df)
    df = df.sort_values("timestamp")

    # Create DAILY time series from Total traffic counts
    ts = df.set_index("timestamp")[TARGET].resample("D").sum()
    ts.name = "total_traffic_per_day"

    # Console info
    print("Series shape:", ts.shape)
    print("Start:", ts.index.min())
    print("End:  ", ts.index.max())
    print("Total days:", len(ts))
    print("Zero-count days:", (ts == 0).sum())
    print("Percentage zeros:", (ts == 0).mean() * 100, "%")

    # Plot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()

    plot_line_chart(
        ts.index,
        ts.values,
        title="Traffic – Daily Total Counts (Dimensionality)",
        xlabel="time",
        ylabel="total traffic per day",
        ax=ax
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dimensionality_daily.png")
    plt.close()

    print("\nSaved: images/CORRECT/dimensionality_daily.png")
    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
