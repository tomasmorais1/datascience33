#!/usr/bin/env python3
"""
Lab 5 – Granularity Exploration
NEW DATASET: TrafficTwoMonth.csv
Two extra granularities: HOURLY and WEEKLY
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart

DATAFILE = "TrafficTwoMonth.csv"
OUTPUT_DIR = Path("images/CORRECT")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Total"   # traffic count column


# ---------------------------------------------------
# Helper: unified timestamp builder
# ---------------------------------------------------
def build_timestamp(df):
    """
    Dataset has 5952 rows recorded every 15 minutes.
    We reconstruct a continuous timestamp.
    """
    df = df.copy()
    df["timestamp"] = pd.date_range(
        start="2023-12-06",   # arbitrary but valid start date
        periods=len(df),
        freq="15min"
    )
    return df


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    print("\n=== LAB 5 — GRANULARITY (NEW DATASET) ===\n")

    # Load and prepare data
    df = pd.read_csv(DATAFILE)
    df = build_timestamp(df)
    df = df.sort_values("timestamp")

    # Build base time series (15-minute data)
    base_ts = df.set_index("timestamp")[TARGET]

    # Aggregations
    hourly_ts = base_ts.resample("H").sum()
    weekly_ts = base_ts.resample("W").sum()

    print("Base points (15 min):", len(base_ts))
    print("Hourly points:", len(hourly_ts))
    print("Weekly points:", len(weekly_ts))

    # Plot: HOURLY
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        hourly_ts.index,
        hourly_ts.values,
        title="Traffic – Hourly Total Counts",
        xlabel="time",
        ylabel="traffic per hour",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_hourly.png")
    plt.close()
    print("Saved: images/CORRECT/granularity_hourly.png")

    # Plot: WEEKLY
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        weekly_ts.index,
        weekly_ts.values,
        title="Traffic – Weekly Total Counts",
        xlabel="time",
        ylabel="traffic per week",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_weekly.png")
    plt.close()
    print("Saved: images/CORRECT/granularity_weekly.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
