#!/usr/bin/env python3
"""
Lab 5 – Stationarity Exploration
NEW TrafficTwoMonth dataset
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import plot_line_chart

DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"

OUTPUT_DIR = Path("images/CORRECT")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------
# Build proper timestamp from 15-minute sequence
# --------------------------------------------------------
def build_timestamp(df):
    """
    Dataset has no real date, only day number + time.
    We rebuild a continuous timestamp for the 5952 rows.
    """
    df = df.copy()
    base = pd.date_range(
        start="2023-12-06",
        periods=len(df),
        freq="15min"
    )
    df["timestamp"] = base
    return df


# --------------------------------------------------------
# ADF test helper
# --------------------------------------------------------
def eval_adf(series):
    series = series.dropna()   # ADF cannot run with NaN

    result = adfuller(series, autolag='AIC')

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "1%": result[4]["1%"],
        "5%": result[4]["5%"],
        "10%": result[4]["10%"],
        "stationary": (result[1] <= 0.05)
    }


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():

    print("\n=== LAB 5 — STATIONARITY (NEW DATASET) ===\n")

    # Load + timestamp
    df = pd.read_csv(DATAFILE)
    df = build_timestamp(df).sort_values("timestamp")

    # Base HOURLY series
    ts_hourly = df.set_index("timestamp")[TARGET].resample("H").sum()

    # Additional granularities
    ts_daily = ts_hourly.resample("D").sum()
    ts_weekly = ts_hourly.resample("W").sum()

    granularities = {
        "Hourly": ts_hourly,
        "Daily": ts_daily,
        "Weekly": ts_weekly
    }

    # Run ADF + Plot
    for name, series in granularities.items():

        print(f"\n--- {name} ---")
        res = eval_adf(series)

        print(f"ADF Statistic: {res['ADF Statistic']:.4f}")
        print(f"p-value: {res['p-value']:.6f}")
        print(f"Critical Values:")
        print(f"   1%:  {res['1%']}")
        print(f"   5%:  {res['5%']}")
        print(f"   10%: {res['10%']}")
        print(f"Stationary (p<=0.05)?  {res['stationary']}")

        # Plot series + mean line
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

        plot_line_chart(
            series.index,
            series.values,
            title=f"{name} Series (Stationarity Check)",
            xlabel="time",
            ylabel=TARGET,
            ax=ax
        )

        # Mean line
        ax.axhline(series.mean(), color="red", linestyle="--", label="Mean")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"stationarity_{name.lower()}.png")
        plt.close()

        print(f"Saved: stationarity_{name.lower()}.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
