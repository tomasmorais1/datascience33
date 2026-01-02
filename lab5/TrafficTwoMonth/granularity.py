#!/usr/bin/env python3
"""
Lab 5 – Granularity Exploration
DATASET: TrafficTwoMonth.csv
Granularities: 15-Min (Atomic), HOURLY, DAILY
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart

DATAFILE = "TrafficTwoMonth.csv"
OUTPUT_DIR = Path("images/CORRECT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Total"  # A coluna que queremos analisar (Soma de carros/bikes/etc)

# ---------------------------------------------------
# Helper: criar timestamp contínuo
# ---------------------------------------------------
def build_timestamp(df):
    """
    O dataset tem registos a cada 15 minutos.
    A coluna 'Date' só tem o dia (ex: 10), o que causa confusão entre meses.
    Vamos recriar um índice temporal contínuo para garantir que o gráfico fica perfeito.
    
    Começamos em 2023-10-10 porque no CSV o dia 10 é uma 'Tuesday', 
    e 10 de Outubro de 2023 foi uma Terça-feira.
    """
    df = df.copy()
    df["timestamp"] = pd.date_range(
        start="2023-10-10 00:00:00", 
        periods=len(df),
        freq="15min"
    )
    return df

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    print("\n=== LAB 5 — GRANULARITY (Atomic -> Hourly -> Daily) ===\n")

    # 1. Carregar dados
    try:
        # Lemos o ficheiro sem fazer parse de datas, pois vamos gerá-las
        df = pd.read_csv(DATAFILE)
    except FileNotFoundError:
        print(f"Erro: Ficheiro {DATAFILE} não encontrado.")
        return

    # 2. Construir timestamps e ordenar
    df = build_timestamp(df)
    df = df.sort_values("timestamp")

    # 3. Criar Série Base (15-minutos - Atomic)
    base_ts = df.set_index("timestamp")[TARGET]

    # 4. Agregações (Resampling)
    # Usamos .sum() porque 'Total' é um volume de tráfego. 
    # Queremos saber quantos carros passaram no total naquela hora/dia.
    hourly_ts = base_ts.resample("H").sum().dropna()
    daily_ts = base_ts.resample("D").sum().dropna()

    print(f"Registos de Tráfego:")
    print(f" - 15-Min (Atomic): {len(base_ts)}")
    print(f" - Hourly:          {len(hourly_ts)}")
    print(f" - Daily:           {len(daily_ts)}")
    print("---")

    # --- PLOT 1: 15-MIN (Atomic) ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        base_ts.index,
        base_ts.values,
        title="Traffic – 15-Min Intervals (Atomic)",
        xlabel="Time",
        ylabel="Traffic Count",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_1_15min.png")
    plt.close()
    print("Saved: granularity_1_15min.png")

    # --- PLOT 2: HOURLY ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        hourly_ts.index,
        hourly_ts.values,
        title="Traffic – Hourly Total Counts",
        xlabel="Time",
        ylabel="Traffic per Hour",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_2_hourly.png")
    plt.close()
    print("Saved: granularity_2_hourly.png")

    # --- PLOT 3: DAILY ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        daily_ts.index,
        daily_ts.values,
        title="Traffic – Daily Total Counts",
        xlabel="Time",
        ylabel="Traffic per Day",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_3_daily.png")
    plt.close()
    print("Saved: granularity_3_daily.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()