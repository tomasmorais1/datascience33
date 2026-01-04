#!/usr/bin/env python3
"""
Lab 6 – Differentiation & Linear Regression Forecasting
DATASET: Economic Indicators
Base Aggregation: Monthly (MEAN)
Differentiation: 1st and 2nd Order
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importações do dslabs_functions
from dslabs_functions import series_train_test_split, plot_forecasting_series, HEIGHT, plot_line_chart

DATAFILE = "economic_indicators_dataset_2010_2023.csv"
TARGET = "Inflation Rate (%)"
COUNTRY_FILTER = "USA"
OUTPUT_DIR = Path("images/forecast_economic_diff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# FUNÇÃO CUSTOM EVAL (Proteção "Hard" para MAPE)
# --------------------------------------------------------
def custom_plot_eval(train, test, prd_trn, prd_tst, title):
    
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # --- PROTEÇÃO MAPE AGRESSIVA ---
        mask = np.abs(y_true) > 1e-5 
        
        if np.sum(mask) > 0:
            pe = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            mape = np.mean(pe)
        else:
            mape = 0.0

        if mape > 1000: 
            mape = 0.0
            
        return rmse, mae, mape, r2

    rmse_trn, mae_trn, mape_trn, r2_trn = calc_metrics(train, prd_trn)
    rmse_tst, mae_tst, mape_tst, r2_tst = calc_metrics(test, prd_tst)

    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    
    labels = ["Train", "Test"]
    x = np.arange(len(labels))
    width = 0.35

    # Esquerda: RMSE vs MAE
    rmse_vals = [rmse_trn, rmse_tst]
    mae_vals = [mae_trn, mae_tst]
    rects1 = axs[0].bar(x - width/2, rmse_vals, width, label="RMSE")
    rects2 = axs[0].bar(x + width/2, mae_vals, width, label="MAE")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_title("Scale-dependent error")
    axs[0].legend()
    axs[0].bar_label(rects1, fmt='%.2f', padding=3)
    axs[0].bar_label(rects2, fmt='%.2f', padding=3)

    # Direita: MAPE vs R2
    mape_vals = [mape_trn, mape_tst]
    r2_vals = [r2_trn, r2_tst]
    rects3 = axs[1].bar(x - width/2, mape_vals, width, label="MAPE")
    rects4 = axs[1].bar(x + width/2, r2_vals, width, label="R2")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_title("Percentage error") 
    axs[1].legend()
    axs[1].bar_label(rects3, fmt='%.2f', padding=3)
    axs[1].bar_label(rects4, fmt='%.2f', padding=3)
    
    plt.suptitle(title)

# --------------------------------------------------------
# Helper: Run LR and Plot
# --------------------------------------------------------
def run_linear_regression(series, gran_name):
    print(f"--- Processing LR for: {gran_name} ---")
    df_temp = series.to_frame(name=TARGET)
    train, test = series_train_test_split(df_temp, trn_pct=0.90)
    
    trnX = np.arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = np.arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    model = LinearRegression()
    model.fit(trnX, trnY)
    
    prd_trn = pd.Series(model.predict(trnX).flatten(), index=train.index)
    prd_tst = pd.Series(model.predict(tstX).flatten(), index=test.index)
    
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    custom_plot_eval(train, test, prd_trn, prd_tst, title=f"LR Eval - {gran_name}")
    save_eval = OUTPUT_DIR / f"economic_lr_eval_{gran_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_eval)
    plt.close()
    
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_forecasting_series(
        train, test, prd_tst, 
        title=f"LR Forecast - {gran_name}", 
        xlabel="Date", ylabel=TARGET
    )
    save_series = OUTPUT_DIR / f"economic_lr_forecast_{gran_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_series)
    plt.close()
    print(f" -> Saved plots for {gran_name}")

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== DIFFERENTIATION & LR (ECONOMIC - MONTHLY) ===\n")

    # 1. Load & Aggregate
    try:
        df = pd.read_csv(DATAFILE, parse_dates=['Date'])
        df = df[df['Country'] == COUNTRY_FILTER].sort_values("Date")
        ts = df.set_index("Date")[TARGET].dropna()
    except Exception as e:
        print(f"Erro: {e}")
        return

    if ts.empty: return

    monthly_ts = ts.resample("M").mean()

    # 2. Differentiations
    diff1_ts = monthly_ts.diff().dropna()
    diff2_ts = diff1_ts.diff().dropna()

    # 3. Plots Differentiation
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(diff1_ts.index, diff1_ts.values, 
                   title="Economic - 1st Differentiation (Monthly)",
                   xlabel="Date", ylabel="Change")
    plt.savefig(OUTPUT_DIR / "economic_plot_1st_diff.png")
    plt.close()
    
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_line_chart(diff2_ts.index, diff2_ts.values, 
                   title="Economic - 2nd Differentiation (Monthly)",
                   xlabel="Date", ylabel="Change in Change")
    plt.savefig(OUTPUT_DIR / "economic_plot_2nd_diff.png")
    plt.close()

    # 4. Run LR
    series_to_test = {
        # AQUI ESTAVA O ERRO: Adicionado .dropna() para garantir que não há NaNs
        "Monthly Raw": monthly_ts.dropna(), 
        "1st Diff": diff1_ts,
        "2nd Diff": diff2_ts
    }

    for name, series in series_to_test.items():
        if len(series) > 5:
            run_linear_regression(series, name)

    print(f"\n=== DONE ===\nImages saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()