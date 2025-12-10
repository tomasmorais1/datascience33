#!/usr/bin/env python3
"""
Lab 5 – Granularity Exploration
Using dslabs_functions




-> ATENCAO, UMA DAS GRANURALIDADES JA TEM IMAGEM CRIADA NA DIMENSIONALITY





"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart

DATAFILE = "traffic_accidents.csv"
TIMESTAMP = "crash_date"

OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)

def aggregate_ts(series, granularity: str, agg_func: str = "sum"):
    """
    Aggregates a Pandas Series with a time index.
    granularity: "W", "M", "Q", etc.
    agg_func: "sum", "mean", etc.
    """
    return series.resample(granularity).agg(agg_func)

def main():
    print("\n=== LAB 5 — GRANULARITY (USING DSLABS FUNCTIONS) ===\n")

    # Load dataset
    df = pd.read_csv(DATAFILE)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP]).sort_values(TIMESTAMP)

    # Base hourly crash count
    hourly_ts = df.set_index(TIMESTAMP).resample("H").size()
    hourly_ts.name = "crashes_per_hour"

    # Aggregations
    weekly_ts = aggregate_ts(hourly_ts, "W", "sum")
    monthly_ts = aggregate_ts(hourly_ts, "M", "sum")
    quarterly_ts = aggregate_ts(hourly_ts, "Q", "sum")

    print("Hourly points:", len(hourly_ts))
    print("Weekly points:", len(weekly_ts))
    print("Monthly points:", len(monthly_ts))
    print("Quarterly points:", len(quarterly_ts))

    # Plotting
    for ts_data, name, ylabel in [
        (weekly_ts, "weekly", "crashes per week"),
        (monthly_ts, "monthly", "crashes per month"),
        (quarterly_ts, "quarterly", "crashes per quarter")
    ]:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()
        plot_line_chart(
            ts_data.index,
            ts_data.values,
            title=f"Traffic Accidents – {name.capitalize()} Crash Counts",
            xlabel="time",
            ylabel=ylabel,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"granularity_{name}.png")
        plt.close()
        print(f"Saved: images/granularity_{name}.png")

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()

