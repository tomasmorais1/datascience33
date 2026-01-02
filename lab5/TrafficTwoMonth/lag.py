#!/usr/bin/env python3
"""
Lab 5 – Autocorrelation Line Charts (Specific Lags)
DATASET: TrafficTwoMonth.csv
Granularity: HOURLY
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_multiline_chart, HEIGHT

DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"
OUTPUT_DIR = Path("images/autocorr_traffic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== LAG LINE CHARTS (TRAFFIC - CUSTOM) ===\n")

    # Load & Timestamp
    df = pd.read_csv(DATAFILE)
    df["timestamp"] = pd.date_range(start="2023-10-10", periods=len(df), freq="15min")
    
    # Granularidade Horária
    ts = df.set_index("timestamp")[TARGET].resample("H").sum()

    # --- ESCOLHA MANUAL DOS LAGS ---
    # Vamos construir o dicionário "à mão" para escolher apenas os importantes.
    # Se usássemos o ciclo 'for', teríamos de escolher entre ver tudo (1,2,3...) ou saltar muito.
    lags = {
        "Original": ts,
        "Lag 1 (1h)": ts.shift(1),    # Imediato
        "Lag 2 (2h)": ts.shift(2),    # Curto prazo
        "Lag 24 (1 Dia)": ts.shift(24) # Sazonalidade (Ontem à mesma hora)
    }

    # Plot
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_multiline_chart(
        ts.index.to_list(), 
        lags, 
        title="Traffic Hourly - Lags (1h, 2h, 24h)", 
        xlabel="Date", 
        ylabel=TARGET
    )
    
    save_path = OUTPUT_DIR / "traffic_lagged_lines_custom.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f" -> Guardado: {save_path}")

if __name__ == "__main__":
    main()