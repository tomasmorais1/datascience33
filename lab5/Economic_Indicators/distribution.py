#!/usr/bin/env python3
"""
Lab 5 – Distribution + Autocorrelation Exploration
Dataset: economic_indicators_dataset_2010_2023.csv
Granularities: Monthly, Quarterly, Annual
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Assumindo que dslabs_functions, HEIGHT, plot_multiline_chart, set_chart_labels estão disponíveis
from dslabs_functions import HEIGHT, plot_multiline_chart, set_chart_labels

# --- VARIÁVEIS DO NOVO DATASET ---
DATAFILE = "economic_indicators_dataset_2010_2023.csv"
OUTPUT_DIR = Path("images_profiling") # Mantendo o diretório que usou no exemplo anterior
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"

# Removida a função build_timestamp (já usamos a coluna 'Date')


# ----------------- Lag helper -----------------
def lagged_series(series, max_lag: int):
    lags = {"original": series}
    for lag in range(1, max_lag + 1):
        # O nome do lag deve refletir a granularidade (ex: lag_1M, lag_1Q, etc.)
        lags[f"lag_{lag}"] = series.shift(lag)
    return lags


# ----------------- Main -----------------
def main():

    print("\n=== LAB 5 — DISTRIBUTION + AUTOCORRELATION (MACRO DATA) ===\n")

    # Load dataset
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

    # 2. Base series e remoção de NaNs
    df_filtered = df_filtered.sort_values("Date")
    base_ts = df_filtered.set_index("Date")[TARGET].dropna()


    # 3. Aggregations (M = Monthly, Q = Quarterly, A = Annual)
    # Usamos .mean() para taxas como a Inflação
    granularities = {
        "Monthly": base_ts.resample("M").mean(),
        "Quarterly": base_ts.resample("Q").mean(),
        "Annual": base_ts.resample("A").mean()
    }

    names = list(granularities.keys())
    series_list = list(granularities.values())

    # ---------------- Boxplots ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        axs[i].boxplot(series.dropna()) # Usar dropna() para boxplots
        set_chart_labels(axs[i], title=f"{names[i]} Inflation Boxplot ({COUNTRY_FILTER})",
                         ylabel="Inflation Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_boxplots_inflation.png")
    plt.close()
    print("Saved: images_profiling/distribution_boxplots_inflation.png")

    # ---------------- Histograms ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        axs[i].hist(series.dropna().values, bins=10, edgecolor="black") # Ajustei bins
        set_chart_labels(axs[i],
                         title=f"{names[i]} Inflation Histogram ({COUNTRY_FILTER})",
                         xlabel="Inflation Rate (%)",
                         ylabel="Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_histograms_inflation.png")
    plt.close()
    print("Saved: images_profiling/distribution_histograms_inflation.png")

    # ---------------- Lag Plots (Series no Tempo) ----------------
    # O lag plot para visualização no tempo não é o Scatter Plot (Autocorrelação),
    # é um gráfico de linhas de séries temporais atrasadas.
    # Usamos max_lag=2 para evitar sobreposição excessiva para visualização
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        # É importante usar dropna() após o shift
        lags = lagged_series(series.dropna(), max_lag=2)
        plot_multiline_chart(
            lags["original"].index, # O índice é sempre da série original (sem shift)
            lags,
            ax=axs[i],
            title=f"{names[i]} Lag Plots ({COUNTRY_FILTER})",
            ylabel="Inflation Rate (%)",
        )

        # Make all lines thinner
        for line in axs[i].lines:
            line.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_plots_line_inflation.png")
    plt.close()
    print("Saved: images_profiling/lag_plots_line_inflation.png")

    # ---------------- Autocorrelation (ACF Plot) ----------------
    # Plot Scatter (ACF) - Aqui é onde se mede a autocorrelação
    fig, axs = plt.subplots(1, 3, figsize=(3 * HEIGHT, HEIGHT))
    for i, series in enumerate(series_list):
        # autocorrelation_plot usa a série temporal e plota os coeficientes ACF
        pd.plotting.autocorrelation_plot(series.dropna(), ax=axs[i])
        axs[i].set_title(f"{names[i]} Autocorrelation ({COUNTRY_FILTER})")
        # Ajustar o limite de lags (para séries anuais, um lag de 10-15 anos é suficiente)
        if names[i] == "Annual":
            max_lags_plot = 15
            axs[i].set_xlim(0, max_lags_plot)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autocorr_acf_inflation.png")
    plt.close()
    print("Saved: images_profiling/autocorr_acf_inflation.png")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
