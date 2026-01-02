#!/usr/bin/env python3
"""
Lab 5 – Autocorrelation Line Charts (Lagged Series)
DATASET: Economic Indicators
Granularity: Daily
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_multiline_chart, HEIGHT

DATAFILE = "economic_indicators_dataset_2010_2023.csv"
TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"
OUTPUT_DIR = Path("images/autocorr_economic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# 1. Função do Professor
# --------------------------------------------------------
def get_lagged_series(series, max_lag: int, delta: int = 1):
    lagged_series = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series

# --------------------------------------------------------
# 2. Main
# --------------------------------------------------------
def main():
    print("\n=== LAG LINE CHARTS (ECONOMIC) ===\n")

    # Load
    try:
        df = pd.read_csv(DATAFILE, parse_dates=['Date'])
        df = df[df['Country'] == COUNTRY_FILTER].sort_values("Date")
        ts = df.set_index("Date")[TARGET].dropna()
    except Exception as e:
        print(f"Erro: {e}")
        return

    # --- Configuração dos Lags ---
    # Como temos poucos dados (57 pontos), vamos ver lags curtos:
    # Lag 1 (período anterior) e Lag 6 e 12 (meio ano e ano completo, se for mensal)
    lags = get_lagged_series(ts, max_lag=12, delta=6)

    # Plot
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_multiline_chart(
        ts.index.to_list(), 
        lags, 
        title="Inflation USA - Lag Comparison", 
        xlabel="Date", 
        ylabel=TARGET
    )
    
    save_path = OUTPUT_DIR / "economic_lagged_lines.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f" -> Guardado: {save_path}")

if __name__ == "__main__":
    main()