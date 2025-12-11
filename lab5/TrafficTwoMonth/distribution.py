#!/usr/bin/env python3
"""
Lab 5 – Distribution + Autocorrelation Exploration
Dataset: TrafficTwoMonth.csv
Granularities: Hourly, Daily, Weekly
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import HEIGHT, plot_multiline_chart, set_chart_labels

DATAFILE = "TrafficTwoMonth.csv"
OUTPUT_DIR = Path("images/CORRECT")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Total"


# ----------------- Build Timestamp -----------------
def build_timestamp(df):
    """Create continuous timestamps every 15 minutes."""
    df = df.copy()
    base = pd.date_range(
        start="2023-12-06",
        periods=len(df),
        freq="15min"
    )
    df["timestamp"] = base
    return df


# ----------------- Lag helper -----------------
def lagged_series(series, max_lag: int):
    lags = {"original": series}
    for lag in range(1, max_lag + 1):
        lags[f"lag_{lag}"] = series.shift(lag)
    return lags


# ----------------- Main -----------------
def main():

    print("\n=== LAB 5 — DISTRIBUTION + AUTOCORRELATION (NEW DATASET) ===\n")

    # Load dataset
    df = pd.read_csv(DATAFILE)
    df = build_timestamp(df)
    df = df.sort_values("timestamp")

    # Base series (15-min)
    base_ts = df.set_index("timestamp")[TARGET]

    # Aggregations
    granularities = {
        "Hourly": base_ts.resample("H").sum(),
        "Daily": base_ts.resample("D").sum(),
        "Weekly": base_ts.resample("W").sum()
    }

    names = list(granularities.keys())
    series_list = list(granularities.values())

    # ---------------- Boxplots ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        axs[i].boxplot(series)
        set_chart_labels(axs[i], title=f"{names[i]} Boxplot")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_boxplots_all.png")
    plt.close()
    print("Saved: distribution_boxplots_all.png")

    # ---------------- Histograms ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        axs[i].hist(series.values, bins=20, edgecolor="black")
        set_chart_labels(axs[i], title=f"{names[i]} Histogram", xlabel="Counts", ylabel="Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_histograms_all.png")
    plt.close()
    print("Saved: distribution_histograms_all.png")

    # ---------------- Lag Plots ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        lags = lagged_series(series, max_lag=5)
        plot_multiline_chart(
            series.index,
            lags,
            ax=axs[i],
            title=f"{names[i]} Lag Plots"
        )

        # Make all lines thinner
        for line in axs[i].lines:
            line.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_plots_all.png")
    plt.close()
    print("Saved: lag_plots_all.png")

    # ---------------- Autocorrelation ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        pd.plotting.autocorrelation_plot(series, ax=axs[i])
        axs[i].set_title(f"{names[i]} Autocorrelation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autocorr_all.png")
    plt.close()
    print("Saved: autocorr_all.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
