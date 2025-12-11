#!/usr/bin/env python3
"""
Lab 5 – Dimensionality exploration (NEW DATASET)
Daily Granularity
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Assumindo que a dslabs_functions e a plot_line_chart estão disponíveis
from dslabs_functions import plot_line_chart

# ALTEARÇÕES AQUI
DATAFILE = "economic_indicators_dataset_2010_2023.csv"
OUTPUT_DIR = Path("images_profiling")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Inflation Rate (%)"  # Nova coluna target
COUNTRY_FILTER = "USA"       # País que queremos analisar


# ---------------------------------------------------
# Main script
# ---------------------------------------------------
def main():

    print("\n=== LAB 5 — DIMENSIONALITY (NEW DATASET, DAILY) ===\n")

    # Load data
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
    # Assumindo que a coluna do país se chama 'Country'
    df_filtered = df[df['Country'] == COUNTRY_FILTER].copy()

    if df_filtered.empty:
        print(f"Aviso: Não foram encontrados dados para o país '{COUNTRY_FILTER}'.")
        return
        
    # 2. Ordenar e preparar a série temporal
    df_filtered = df_filtered.sort_values("Date")
    
    # Criar a série temporal (ts) a partir da coluna TARGET e usar a Date como índice
    # Como assumimos que os dados já são diários, não precisamos do .resample("D").sum()
    # Apenas definimos o índice e selecionamos a coluna.
    ts = df_filtered.set_index("Date")[TARGET]
    ts.name = f"{TARGET} for {COUNTRY_FILTER}"

    # Console info
    print("Series shape:", ts.shape)
    print("Start:", ts.index.min())
    print("End:  ", ts.index.max())
    print("Total days:", len(ts))
    
    # Verificação de valores nulos (pode ser mais relevante que zeros para % de inflação)
    print("NaN-count days:", ts.isna().sum())
    print("Percentage NaNs:", ts.isna().mean() * 100, "%")


    # Plot
    fig = plt.figure(figsize=(12, 4))
    ax = fig.gca()

    plot_line_chart(
        ts.index,
        ts.values,
        title=f"{TARGET} - {COUNTRY_FILTER} (Dimensionality)", # Título ajustado
        xlabel="Time",
        ylabel=TARGET, # Label ajustado
        ax=ax
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"dimensionality_daily_{COUNTRY_FILTER}_inflation.png")
    plt.close()

    print(f"\nSaved: {OUTPUT_DIR / f'dimensionality_daily_{COUNTRY_FILTER}_inflation.png'}")
    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
