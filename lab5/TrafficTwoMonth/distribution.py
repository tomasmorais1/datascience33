#!/usr/bin/env python3
"""
Lab 5 – Autocorrelation Study
DATASET: TrafficTwoMonth.csv
Granularity: HOURLY
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dslabs_functions import HEIGHT

DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"

# Configuração de output
OUTPUT_DIR = Path("images/autocorr_traffic")
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
# 2. Autocorrelation Study Function (Estilo do Stor)
# --------------------------------------------------------
def autocorrelation_study(series, max_lag, delta=1, file_tag="traffic"):
    """
    Gera Lag-Plots (Scatter) e Correlograma.
    """
    k = int(max_lag / delta)
    fig = plt.figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values = series.tolist()
    
    # 1. Lag Plots (Scatter) - Linha de cima
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        # Scatter: x=t-lag, y=t
        ax.scatter(series.shift(lag).tolist(), series_values, alpha=0.5, s=10)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
        ax.set_title(f"Lag {lag}")

    # 2. Correlogram (Autocorrelation) - Linha de baixo (Ocupa toda a largura)
    ax_corr = fig.add_subplot(gs[1, :])
    # maxlags define até onde vai o gráfico
    ax_corr.acorr(series.astype(float), maxlags=max_lag, usevlines=True, normed=True)
    ax_corr.set_title(f"Autocorrelation Correlogram ({file_tag})")
    ax_corr.set_xlabel("Lags")
    ax_corr.grid(True)
    
    return fig

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== AUTOCORRELATION STUDY (TRAFFIC) ===\n")

    # Load & Prepare
    try:
        df = pd.read_csv(DATAFILE)
    except FileNotFoundError:
        print("Erro: Ficheiro não encontrado.")
        return

    df = build_timestamp(df).sort_values("timestamp")
    
    # Usar HOURLY para ver ciclos diários (24h)
    ts_hourly = df.set_index("timestamp")[TARGET].resample("H").sum().dropna()
    
    print(f"Série Horária: {len(ts_hourly)} pontos.")

    # --- FIGURE 73 & 75 GENERATION ---
    # Max Lag = 24 (para ver o ciclo completo de um dia)
    # Delta = 4 (mostra lags 4, 8, 12, 16, 20, 24) para não encher demasiado
    print("Generating Autocorrelation Plots...")
    
    fig = autocorrelation_study(ts_hourly, max_lag=24, delta=4, file_tag="Traffic Hourly")
    
    save_path = OUTPUT_DIR / "traffic_autocorrelation.png"
    fig.savefig(save_path)
    plt.close()
    
    print(f" -> Guardado: {save_path}")
    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()