#!/usr/bin/env python3
"""
Lab 5 – Granularity Exploration
NEW DATASET: economic_indicators_dataset_2010_2023.csv
Three extra granularities: ANNUAL, MONTHLY, and QUARTERLY
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dslabs_functions import plot_line_chart # Assumindo que esta função está disponível

DATAFILE = "economic_indicators_dataset_2010_2023.csv"
OUTPUT_DIR = Path("images_profiling")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Inflation Rate (%)"  # Nova coluna target
COUNTRY_FILTER = "USA"        # País que queremos analisar


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():

    print("\n=== LAB 5 — GRANULARITY (NEW DATASET: Monthly, Quarterly, Annual) ===\n")

    # Load and prepare data
    try:
        # Carregar e garantir que a coluna Date é um datetime
        df = pd.read_csv(DATAFILE, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Erro: O ficheiro {DATAFILE} não foi encontrado.")
        return
    except KeyError:
        print("Erro: A coluna 'Date' não foi encontrada no ficheiro.")
        return
            
    # 1. Filtrar pelo País (USA)
    df_filtered = df[df['Country'] == COUNTRY_FILTER].copy()

    if df_filtered.empty:
        print(f"Aviso: Não foram encontrados dados para o país '{COUNTRY_FILTER}'.")
        return

    # 2. Ordenar e definir a série temporal base (Diária)
    df_filtered = df_filtered.sort_values("Date")
    # Remover valores nulos antes de agregar
    base_ts = df_filtered.set_index("Date")[TARGET].dropna()
        
    if base_ts.empty:
        print("Erro: A série temporal base está vazia após a filtragem de NaNs.")
        return
        
    # Aggregations: M = Monthly, Q = Quarterly, A = Annual
    # Usamos .mean() para taxas
    monthly_ts = base_ts.resample("M").mean()
    quarterly_ts = base_ts.resample("Q").mean() # NOVO: Trimestral
    annual_ts = base_ts.resample("A").mean()

    print(f"Base points (Daily, {COUNTRY_FILTER}):", len(base_ts))
    print("Monthly points:", len(monthly_ts))
    print("Quarterly points:", len(quarterly_ts))
    print("Annual points:", len(annual_ts))
    print("---")

    # --- Plot: MONTHLY ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        monthly_ts.index,
        monthly_ts.values,
        title=f"Inflation Rate ({COUNTRY_FILTER}) – Monthly Average",
        xlabel="Time (Monthly)",
        ylabel="Avg. Inflation Rate (%)",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_monthly_inflation.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/granularity_monthly_inflation.png")
    
    # --- Plot: QUARTERLY (NOVO) ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        quarterly_ts.index,
        quarterly_ts.values,
        title=f"Inflation Rate ({COUNTRY_FILTER}) – Quarterly Average",
        xlabel="Time (Quarterly)",
        ylabel="Avg. Inflation Rate (%)",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_quarterly_inflation.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/granularity_quarterly_inflation.png")

    # --- Plot: ANNUAL ---
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()
    plot_line_chart(
        annual_ts.index,
        annual_ts.values,
        title=f"Inflation Rate ({COUNTRY_FILTER}) – Annual Average",
        xlabel="Time (Annual)",
        ylabel="Avg. Inflation Rate (%)",
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "granularity_annual_inflation.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/granularity_annual_inflation.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
