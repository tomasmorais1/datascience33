#!/usr/bin/env python3
"""
Lab 5 – Components & Stationarity Study
DATASET: TrafficTwoMonth.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import plot_line_chart, set_chart_labels, HEIGHT

DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"
OUTPUT_DIR = Path("images/components_traffic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# 1. Helper: Timestamp Builder (Traffic specific)
# --------------------------------------------------------
def build_timestamp(df):
    df = df.copy()
    # Começa a 10 de Outubro de 2023 (Terça-feira)
    df["timestamp"] = pd.date_range(
        start="2023-10-10 00:00:00",
        periods=len(df),
        freq="15min"
    )
    return df

# --------------------------------------------------------
# 2. Components Decomposition Plot (Código do Stor)
# --------------------------------------------------------
def plot_components(series, title="", x_label="time", y_label=""):
    # model='add' assume Sazonalidade Aditiva (Amplitude constante)
    # period=24 porque estamos a usar dados Horários e queremos ver o ciclo Diário
    decomposition = seasonal_decompose(series, model="add", period=24)
    
    components = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    
    rows = len(components)
    fig, axs = plt.subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    
    i = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    
    plt.tight_layout()
    return fig

# --------------------------------------------------------
# 3. ADF Test (Código do Stor adaptado)
# --------------------------------------------------------
def eval_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05

# --------------------------------------------------------
# 4. Stationarity Visual Study (Média Segmentada)
# --------------------------------------------------------
def plot_stationary_study(series, title=""):
    n = len(series)
    
    # 1. Plot com Média Global
    fig1 = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=series.name,
        title=f"{title} (Global Mean)",
        name="original",
    )
    plt.plot(series.index, [series.mean()] * n, "r-", label="mean", linewidth=3)
    plt.legend()
    
    # 2. Plot com Média Segmentada (BINS)
    BINS = 10
    mean_line = []
    for i in range(BINS):
        segment = series[i * n // BINS : (i + 1) * n // BINS]
        mean_value = [segment.mean()] * (n // BINS)
        mean_line += mean_value
    mean_line += [mean_line[-1]] * (n - len(mean_line))
    
    fig2 = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(
        series.index.to_list(),
        series.to_list(),
        xlabel=series.index.name,
        ylabel=series.name,
        title=f"{title} (Segmented Mean)",
        name="original",
    )
    plt.plot(series.index, mean_line, "r-", label="segmented mean", linewidth=3)
    plt.legend()
    
    return fig1, fig2

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== COMPONENTS & STATIONARITY (TRAFFIC) ===\n")

    # Load & Prepare
    df = pd.read_csv(DATAFILE)
    df = build_timestamp(df).sort_values("timestamp")
    
    # Vamos usar HOURLY para a decomposição (menos ruído que 15min)
    # A soma faz sentido para tráfego (total de carros na hora)
    ts = df.set_index("timestamp")[TARGET].resample("H").sum().dropna()
    ts.name = TARGET
    ts.index.name = "Date"

    # --- 1. Components Decomposition ---
    print("Generating Decomposition...")
    fig_decomp = plot_components(ts, title="Traffic Components (Hourly)", x_label="Date", y_label=TARGET)
    fig_decomp.savefig(OUTPUT_DIR / "traffic_components.png")
    plt.close()

    # --- 2. Stationarity Visuals ---
    print("Generating Stationarity Plots...")
    fig_glob, fig_seg = plot_stationary_study(ts, title="Traffic Stationarity")
    fig_glob.savefig(OUTPUT_DIR / "traffic_stationarity_global.png")
    fig_seg.savefig(OUTPUT_DIR / "traffic_stationarity_segmented.png")
    plt.close()
    plt.close()

    # --- 3. ADF Test ---
    print("\n--- ADF Test Results (Traffic) ---")
    is_stationary = eval_stationarity(ts)
    print(f"The series {('is' if is_stationary else 'is NOT')} stationary.")

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()