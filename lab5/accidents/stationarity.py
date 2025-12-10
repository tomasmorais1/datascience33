#!/usr/bin/env python3
"""
Lab 5 – Stationarity Exploration
Traffic Accidents dataset
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import plot_line_chart, HEIGHT

DATAFILE = "traffic_accidents.csv"
TIMESTAMP = "crash_date"

OUTPUT_DIR = Path("images/stationarity")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------
def aggregate_ts(series, granularity: str, agg_func: str = "sum"):
    """
    Aggregate series by granularity: "W", "M", "Q", etc.
    """
    return series.resample(granularity).agg(agg_func)

def eval_adf(series):
    """
    Run Augmented Dickey-Fuller test and return result dict.
    """
    result = adfuller(series, autolag='AIC')
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "1%": result[4]['1%'],
        "5%": result[4]['5%'],
        "10%": result[4]['10%'],
        "stationary": result[1] <= 0.05
    }

def main():

    print("\n=== LAB 5 — STATIONARITY EXPLORATION ===\n")

    # 1) Load dataset
    df = pd.read_csv(DATAFILE)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP]).sort_values(TIMESTAMP)

    # 2) Build base hourly crash count
    ts_hourly = df.set_index(TIMESTAMP).resample("H").size()
    ts_hourly.name = "crashes_per_hour"

    # 3) Compute granularities: weekly, monthly, trimestral
    granularities = {"Weekly": "W", "Monthly": "M", "Trimestral": "Q"}
    ts_gran = {name: aggregate_ts(ts_hourly, code, "sum") for name, code in granularities.items()}

    # 4) Plot series with mean line and run ADF test
    for name, series in ts_gran.items():
        print(f"\n--- {name} Series ---")
        result = eval_adf(series)
        print(f"ADF Statistic: {result['ADF Statistic']:.3f}")
        print(f"p-value: {result['p-value']:.3f}")
        print("Critical Values: 1%={1%}, 5%={5%}, 10%={10%}".format(**result))
        print(f"Stationary: {result['stationary']}")

        # Plot series + mean
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
        plot_line_chart(
            series.index,
            series.values,
            title=f"Traffic Accidents – {name} Crash Counts (Stationarity)",
            xlabel="Time",
            ylabel="Crashes",
            ax=ax
        )
        mean_series = [series.mean()] * len(series)
        ax.plot(series.index, mean_series, "r--", label="Mean")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"stationarity_{name.lower()}.png")
        plt.close()
        print(f"Saved: stationarity_{name.lower()}.png")

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()
