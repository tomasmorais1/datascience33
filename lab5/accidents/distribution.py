#!/usr/bin/env python3
"""
Lab 5 – Distribution + Autocorrelation Exploration
Traffic accidents dataset
All granularities in one figure per plot type
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import HEIGHT, plot_multiline_chart, set_chart_labels

DATAFILE = "traffic_accidents.csv"
TIMESTAMP = "crash_date"

OUTPUT_DIR = Path("images/distribution")
OUTPUT_DIR.mkdir(exist_ok=True)

def aggregate_ts(series, granularity: str, agg_func: str = "sum"):
    """Aggregate a Pandas Series with a time index"""
    return series.resample(granularity).agg(agg_func)

def lagged_series(series, max_lag: int):
    """Return a dict of lagged series for plotting"""
    lags = {"original": series}
    for lag in range(1, max_lag + 1):
        lags[f"lag_{lag}"] = series.shift(lag)
    return lags

def main():
    print("\n=== LAB 5 — DISTRIBUTION + AUTOCORRELATION EXPLORATION ===\n")

    # Load dataset
    df = pd.read_csv(DATAFILE)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP]).sort_values(TIMESTAMP)

    # Base hourly series
    hourly_ts = df.set_index(TIMESTAMP).resample("H").size()

    # Aggregations
    granularities = {
        "Weekly": aggregate_ts(hourly_ts, "W"),
        "Monthly": aggregate_ts(hourly_ts, "M"),
        "Quarterly": aggregate_ts(hourly_ts, "Q"),
    }

    names = list(granularities.keys())
    series_list = list(granularities.values())

    # ---------------- Boxplots ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3*HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        set_chart_labels(axs[i], title=f"{names[i]} Boxplot")
        axs[i].boxplot(series)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_boxplots_all.png")
    plt.close()
    print("Saved boxplots: distribution_boxplots_all.png")

    # ---------------- Histograms ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3*HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        axs[i].hist(series.values, bins=20, color="skyblue", edgecolor="black")
        set_chart_labels(axs[i], title=f"{names[i]} Histogram", xlabel="Crash counts", ylabel="Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_histograms_all.png")
    plt.close()
    print("Saved histograms: distribution_histograms_all.png")

    # Lag plots
    fig, axs = plt.subplots(1, 3, figsize=(3*HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        lags = lagged_series(series, max_lag=5)
        plot_multiline_chart(series.index, lags, ax=axs[i], title=f"{names[i]} Lag Plots")
        
        # Make lines thinner
        for line in axs[i].lines:
            line.set_linewidth(0.8)
            
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_plots_all.png")
    plt.close()
    print("Saved lag plots: lag_plots_all.png")



    # ---------------- Autocorrelation plots ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3*HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        pd.plotting.autocorrelation_plot(series, ax=axs[i])
        axs[i].set_title(f"{names[i]} Autocorrelation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autocorr_all.png")
    plt.close()
    print("Saved autocorrelation plots: autocorr_all.png")

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()
