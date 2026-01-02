#!/usr/bin/env python3
"""
Lab 5 – Granularity Exploration
NEW DATASET: economic_indicators_dataset_2010_2023.csv
Granularities: DAILY (Atomic), MONTHLY, QUARTERLY, ANNUAL
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart, HEIGHT  # Importar HEIGHT se disponível, senão definir manualmente

DATAFILE = "economic_indicators_dataset_2010_2023.csv"
OUTPUT_DIR = Path("images_profiling")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"

def main():
    print("\n=== LAB 5 — GRANULARITY STUDY ===\n")

    # 1. Load Data
    try:
        df = pd.read_csv(DATAFILE, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Erro: O ficheiro {DATAFILE} não foi encontrado.")
        return
    except KeyError:
        print("Erro: A coluna 'Date' não foi encontrada.")
        return
            
    # 2. Filter by Country and Sort
    df_filtered = df[df['Country'] == COUNTRY_FILTER].copy()
    if df_filtered.empty:
        print(f"Aviso: Sem dados para '{COUNTRY_FILTER}'.")
        return

    df_filtered = df_filtered.sort_values("Date")
    
    # 3. Define Base Series (Most Atomic - Daily)
    # Removemos NaNs da base para limpar os dados brutos
    base_ts = df_filtered.set_index("Date")[TARGET].dropna()
        
    if base_ts.empty:
        print("Erro: Série temporal vazia.")
        return
        
    # 4. Aggregations
    # O .dropna() aqui é CRUCIAL para evitar os cortes na linha do gráfico
    # se houver meses/trimestres sem dados.
    monthly_ts = base_ts.resample("M").mean().dropna()
    quarterly_ts = base_ts.resample("Q").mean().dropna()
    annual_ts = base_ts.resample("A").mean().dropna()

    print(f"Records for {COUNTRY_FILTER}:")
    print(f" - Daily (Atomic): {len(base_ts)}")
    print(f" - Monthly:        {len(monthly_ts)}")
    print(f" - Quarterly:      {len(quarterly_ts)}")
    print(f" - Annual:         {len(annual_ts)}")
    print("---")

    # --- PLOT 1: DAILY (Atomic) ---
    plt.figure(figsize=(12, 4))
    plot_line_chart(
        base_ts.index.to_list(),
        base_ts.to_list(),
        title=f"{COUNTRY_FILTER} - {TARGET} (Daily/Atomic)",
        xlabel="Date",
        ylabel=TARGET
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_1_daily.png")
    plt.close()
    print("Generated: daily chart")

    # --- PLOT 2: MONTHLY ---
    plt.figure(figsize=(12, 4))
    plot_line_chart(
        monthly_ts.index.to_list(),
        monthly_ts.to_list(),
        title=f"{COUNTRY_FILTER} - {TARGET} (Monthly Average)",
        xlabel="Date",
        ylabel=TARGET
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_2_monthly.png")
    plt.close()
    print("Generated: monthly chart")

    # --- PLOT 3: QUARTERLY ---
    plt.figure(figsize=(12, 4))
    plot_line_chart(
        quarterly_ts.index.to_list(),
        quarterly_ts.to_list(),
        title=f"{COUNTRY_FILTER} - {TARGET} (Quarterly Average)",
        xlabel="Date",
        ylabel=TARGET
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_3_quarterly.png")
    plt.close()
    print("Generated: quarterly chart")

    # --- PLOT 4: ANNUAL ---
    plt.figure(figsize=(12, 4))
    plot_line_chart(
        annual_ts.index.to_list(),
        annual_ts.to_list(),
        title=f"{COUNTRY_FILTER} - {TARGET} (Annual Average)",
        xlabel="Date",
        ylabel=TARGET
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_4_annual.png")
    plt.close()
    print("Generated: annual chart")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()