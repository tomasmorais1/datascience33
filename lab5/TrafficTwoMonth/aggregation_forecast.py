#!/usr/bin/env python3
"""
Lab 6 – Aggregation & Linear Regression Forecasting
DATASET: TrafficTwoMonth.csv
Aggregations: 15-Min (Atomic), Hourly, Daily
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# IMPORTANTE: Métricas para calcular os valores exatos
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Importações do dslabs_functions
from dslabs_functions import series_train_test_split, plot_forecasting_series, HEIGHT

DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"
OUTPUT_DIR = Path("images/forecast_traffic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Helper: Timestamp Builder
# --------------------------------------------------------
def build_timestamp(df):
    df = df.copy()
    # Usamos a data correta: 10 Out 2023 (Terça-feira)
    df["timestamp"] = pd.date_range(start="2023-10-10", periods=len(df), freq="15min")
    return df

# --------------------------------------------------------
# NOVA FUNÇÃO: Custom Eval (Eixo X = Train/Test, Barras = Métricas)
# --------------------------------------------------------
def custom_plot_eval(train, test, prd_trn, prd_tst, title):
    
    # 1. Calcular Métricas
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, mape, r2

    rmse_trn, mae_trn, mape_trn, r2_trn = calc_metrics(train, prd_trn)
    rmse_tst, mae_tst, mape_tst, r2_tst = calc_metrics(test, prd_tst)

    # 2. Configurar Subplots (1 linha, 2 colunas)
    fig, axs = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    
    # Dados comuns para o Eixo X (Train, Test)
    labels = ["Train", "Test"]
    x = np.arange(len(labels))
    width = 0.35

    # --- GRÁFICO DA ESQUERDA: Scale-dependent error (Barras: RMSE vs MAE) ---
    # Valores de RMSE para Train e Test
    rmse_vals = [rmse_trn, rmse_tst]
    # Valores de MAE para Train e Test
    mae_vals = [mae_trn, mae_tst]
    
    rects1 = axs[0].bar(x - width/2, rmse_vals, width, label="RMSE")
    rects2 = axs[0].bar(x + width/2, mae_vals, width, label="MAE")
    
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels) # Eixo X diz Train e Test
    axs[0].set_title("Scale-dependent error")
    axs[0].legend()
    
    # FORÇAR 2 CASAS DECIMAIS
    axs[0].bar_label(rects1, fmt='%.2f', padding=3)
    axs[0].bar_label(rects2, fmt='%.2f', padding=3)

    # --- GRÁFICO DA DIREITA: Percentage error (Barras: MAPE vs R2) ---
    mape_vals = [mape_trn, mape_tst]
    r2_vals = [r2_trn, r2_tst]
    
    rects3 = axs[1].bar(x - width/2, mape_vals, width, label="MAPE")
    rects4 = axs[1].bar(x + width/2, r2_vals, width, label="R2")
    
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels) # Eixo X diz Train e Test
    axs[1].set_title("Percentage error") 
    axs[1].legend()
    
    # FORÇAR 2 CASAS DECIMAIS
    axs[1].bar_label(rects3, fmt='%.2f', padding=3)
    axs[1].bar_label(rects4, fmt='%.2f', padding=3)
    
    plt.suptitle(title)

# --------------------------------------------------------
# Helper: Run LR and Plot
# --------------------------------------------------------
def run_linear_regression(series, gran_name):
    print(f"--- Processing {gran_name} ---")
    
    # Converter Series -> DataFrame
    df_temp = series.to_frame(name=TARGET)
    
    # 1. Split Train/Test
    train, test = series_train_test_split(df_temp, trn_pct=0.90)
    
    # 2. Prepare X (Time Index) and Y (Target)
    trnX = np.arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = np.arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    # 3. Train Model
    model = LinearRegression()
    model.fit(trnX, trnY)
    
    # 4. Predict
    prd_trn = pd.Series(model.predict(trnX).flatten(), index=train.index)
    prd_tst = pd.Series(model.predict(tstX).flatten(), index=test.index)
    
    # 5. Plot 1: Forecasting Evaluation (Custom)
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    # Usamos a função customizada em vez da do dslabs
    custom_plot_eval(train, test, prd_trn, prd_tst, title=f"LR Eval - {gran_name}")
    
    save_eval = OUTPUT_DIR / f"traffic_lr_eval_{gran_name.lower()}.png"
    plt.savefig(save_eval)
    plt.close()
    
    # 6. Plot 2: Forecasting Series (Line Chart)
    plt.figure(figsize=(3 * HEIGHT, HEIGHT))
    plot_forecasting_series(
        train, test, prd_tst, 
        title=f"LR Forecast - {gran_name}", 
        xlabel="Date", ylabel=TARGET
    )
    save_series = OUTPUT_DIR / f"traffic_lr_forecast_{gran_name.lower()}.png"
    plt.savefig(save_series)
    plt.close()
    
    print(f" -> Saved: {save_eval.name} & {save_series.name}")

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n=== AGGREGATION & LINEAR REGRESSION (TRAFFIC) ===\n")

    # Load & Timestamp
    df = pd.read_csv(DATAFILE)
    df = build_timestamp(df)
    
    # Base Series (15-min)
    base_ts = df.set_index("timestamp")[TARGET]
    
    # Define Aggregations (SUM para volume de tráfego)
    aggregations = {
        "15-Min": base_ts.resample("15min").sum(),
        "Hourly": base_ts.resample("H").sum(),
        "Daily":  base_ts.resample("D").sum()
    }
    
    # Run loop
    for name, series in aggregations.items():
        run_linear_regression(series.dropna(), name)

    print(f"\n=== DONE ===\nImages saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()