#!/usr/bin/env python3
"""
Lab 5 – Dimensionality exploration ONLY
Using dslabs_functions for plotting
"""

from pathlib import Path
import pandas as pd
from dslabs_functions import plot_line_chart
import matplotlib.pyplot as plt

DATAFILE = "traffic_accidents.csv"
TIMESTAMP = "crash_date"

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():

    print("\n=== LAB 5 — DIMENSIONALITY (USING DSLABS FUNCTIONS) ===\n")

    # Load data
    df = pd.read_csv(DATAFILE)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP]).sort_values(TIMESTAMP)

    # Weekly crash counts
    ts = df.set_index(TIMESTAMP).resample("W").size()
    ts.name = "crashes_per_week"

    # Console exploration
    print("Series shape:", ts.shape)
    print("Start:", ts.index.min())
    print("End:  ", ts.index.max())
    print("Total weeks:", len(ts))
    print("Zero-count weeks:", (ts == 0).sum())
    print("Percentage zeros:", (ts == 0).mean() * 100, "%")

    # Plot using DSLABS
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        ts.index,
        ts.values,
        title="Traffic Accidents – Weekly Crash Counts (Dimensionality)",
        xlabel="time",
        ylabel="crashes per week",
        ax=ax
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dimensionality_weekly.png")
    plt.close()

    print("\nSaved: images/dimensionality_weekly.png")
    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()
