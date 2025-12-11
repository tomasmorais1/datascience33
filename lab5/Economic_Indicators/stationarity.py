#!/usr/bin/env python3
"""
Lab 5 – Stationarity Exploration
NEW MacroeconomicData dataset
Granularities: Monthly, Quarterly, Annual
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from dslabs_functions import plot_line_chart # Assumindo que esta função está disponível

# --- VARIÁVEIS DO NOVO DATASET ---
DATAFILE = "economic_indicators_dataset_2010_2023.csv"
TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"

OUTPUT_DIR = Path("images_profiling")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------
# Removida a função build_timestamp (já usamos a coluna 'Date')
# --------------------------------------------------------

# --------------------------------------------------------
# ADF test helper (Mantida, pois é genérica)
# --------------------------------------------------------
def eval_adf(series):
    # O teste ADF é sensível a NaNs, por isso usamos dropna()
    series = series.dropna()

    # Se a série estiver vazia após dropna, retorna um resultado indicativo de não-estacionariedade
    if series.empty:
        return {
            "ADF Statistic": float('nan'),
            "p-value": 1.0,
            "1%": float('nan'),
            "5%": float('nan'),
            "10%": float('nan'),
            "stationary": False
        }

    result = adfuller(series, autolag='AIC')

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "1%": result[4]["1%"],
        "5%": result[4]["5%"],
        "10%": result[4]["10%"],
        # Uma série é considerada estacionária se p-value <= 0.05
        "stationary": (result[1] <= 0.05)
    }


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():

    print("\n=== LAB 5 — STATIONARITY (MACRO DATA) ===\n")

    # Load + filter
    try:
        # Carregar e garantir que a coluna Date é um datetime
        df = pd.read_csv(DATAFILE, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"Erro: O ficheiro {DATAFILE} não foi encontrado.")
        return
    except KeyError:
        print("Erro: A coluna 'Date' ou 'Country' não foi encontrada no ficheiro.")
        return

    # Filtrar pelo País (USA)
    df_filtered = df[df['Country'] == COUNTRY_FILTER].copy()
    if df_filtered.empty:
        print(f"Aviso: Não foram encontrados dados para o país '{COUNTRY_FILTER}'.")
        return

    # Base series (Diária) e preparação
    df_filtered = df_filtered.sort_values("Date")
    base_ts = df_filtered.set_index("Date")[TARGET].dropna() # Remove NaNs da base

    if base_ts.empty:
        print("Erro: A série temporal base está vazia após a filtragem de NaNs.")
        return

    # Aggregations (M = Monthly, Q = Quarterly, A = Annual)
    # Usamos .mean() para taxas como a Inflação
    ts_monthly = base_ts.resample("M").mean()
    ts_quarterly = base_ts.resample("Q").mean()
    ts_annual = base_ts.resample("A").mean()

    granularities = {
        "Monthly": ts_monthly,
        "Quarterly": ts_quarterly,
        "Annual": ts_annual
    }

    # Run ADF + Plot
    for name, series in granularities.items():

        # Os gráficos que viu antes com grandes saltos são causados por NaNs.
        # Para o teste ADF e para a plotagem, a série deve ser tratada.
        # Vamos usar .interpolate() apenas para a plotagem, para preencher os gaps visuais.
        series_filled_for_plot = series.interpolate(method='linear')
        
        print(f"\n--- {name} ---")
        # O teste ADF é executado na série original (resample, mas sem interpolação)
        res = eval_adf(series)

        print(f"ADF Statistic: {res['ADF Statistic']:.4f}")
        print(f"p-value: {res['p-value']:.6f}")
        print(f"Critical Values:")
        print(f"  1%:  {res['1%']}")
        print(f"  5%:  {res['5%']}")
        print(f"  10%: {res['10%']}")
        print(f"Stationary (p<=0.05)?  {res['stationary']}")

        # Plot series + mean line
        fig = plt.figure(figsize=(12, 4))
        ax = fig.gca()

        plot_line_chart(
            series_filled_for_plot.index, # Usar a série interpolada para a visualização
            series_filled_for_plot.values,
            title=f"Inflation Rate ({COUNTRY_FILTER}) – {name} Series (Stationarity Check)",
            xlabel="Time",
            ylabel=TARGET,
            ax=ax
        )

        # Mean line
        ax.axhline(series_filled_for_plot.mean(), color="red", linestyle="--", label="Mean")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"stationarity_inflation_{name.lower()}.png")
        plt.close()

        print(f"Saved: {OUTPUT_DIR}/stationarity_inflation_{name.lower()}.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
