#!/usr/bin/env python3
"""
Lab 5 – Components & Stationarity Exploration
DATASET: economic_indicators_dataset_2010_2023.csv
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import plot_line_chart, set_chart_labels, HEIGHT

DATAFILE = "economic_indicators_dataset_2010_2023.csv"
TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"

# --- CORREÇÃO DE CAMINHO ---
# Isto garante que a pasta 'images' é criada AO LADO deste script .py,
# independentemente de onde executas o comando no terminal.
OUTPUT_DIR = Path(__file__).parent / "images/components_economic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Functions
# --------------------------------------------------------
def plot_components(series, title="", x_label="time", y_label="", period=30):
    decomposition = seasonal_decompose(series, model="add", period=period)
    
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

def eval_stationarity(series):
    series = series.dropna()
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05

def plot_stationary_study(series, title=""):
    n = len(series)
    
    # 1. Global Mean
    fig1 = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(series.index.to_list(), series.to_list(),
                    xlabel=series.index.name, ylabel=series.name,
                    title=f"{title} (Global Mean)")
    plt.plot(series.index, [series.mean()] * n, "r-", label="mean", linewidth=3)
    plt.legend()
    
    # 2. Segmented Mean
    BINS = 10
    mean_line = []
    step = n // BINS
    if step < 1: step = 1
        
    for i in range(BINS):
        segment = series[i * step : (i + 1) * step]
        if len(segment) > 0:
            mean_value = [segment.mean()] * len(segment)
            mean_line += mean_value
            
    if len(mean_line) < n:
        mean_line += [mean_line[-1]] * (n - len(mean_line))
    
    fig2 = plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(series.index.to_list(), series.to_list(),
                    xlabel=series.index.name, ylabel=series.name,
                    title=f"{title} (Segmented Mean)")
    plt.plot(series.index, mean_line, "r-", label="segmented mean", linewidth=3)
    plt.legend()
    
    return fig1, fig2

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== COMPONENTS & STATIONARITY (ECONOMIC) ===\n")
    print(f"A guardar imagens em: {OUTPUT_DIR.resolve()}") # Debug Path

    # Load
    try:
        # Tenta carregar usando o caminho relativo ao script também
        file_path = Path(__file__).parent / DATAFILE
        if not file_path.exists():
            file_path = DATAFILE # Tenta no diretório atual se falhar
            
        df = pd.read_csv(file_path, parse_dates=['Date'])
    except Exception as e:
        print(f"Erro ao carregar ficheiro: {e}")
        return

    # Filter & Sort
    df = df[df['Country'] == COUNTRY_FILTER].sort_values("Date")
    ts = df.set_index("Date")[TARGET].dropna()
    ts.name = TARGET

    if ts.empty:
        print("Série vazia.")
        return

    print(f"Número de observações: {len(ts)}")

    # --- 1. Components Decomposition ---
    print("Generating Decomposition...")
    try:
        # Period=12 para dados mensais/poucos dados
        fig_decomp = plot_components(ts, title=f"Inflation {COUNTRY_FILTER}", period=12)
        save_path = OUTPUT_DIR / "economic_components.png"
        fig_decomp.savefig(save_path)
        plt.close()
        print(f" -> Guardado: {save_path.name}")
    except ValueError as e:
        print(f"Erro no Decomposition: {e}")

    # --- 2. Stationarity Visuals ---
    print("Generating Stationarity Plots...")
    fig_glob, fig_seg = plot_stationary_study(ts, title=f"Inflation {COUNTRY_FILTER}")
    
    fig_glob.savefig(OUTPUT_DIR / "economic_stationarity_global.png")
    fig_seg.savefig(OUTPUT_DIR / "economic_stationarity_segmented.png")
    plt.close()
    plt.close()
    print(" -> Gráficos de estacionaridade guardados.")

    # --- 3. ADF Test ---
    print("\n--- ADF Test Results (Economic) ---")
    is_stationary = eval_stationarity(ts)
    print(f"The series {('is' if is_stationary else 'is NOT')} stationary.")

    print(f"\n=== DONE ===\nVerifica a pasta: {OUTPUT_DIR.resolve()}\n")

# --- IMPORTANTE: ISTO FAZ O CÓDIGO CORRER ---
if __name__ == "__main__":
    main()