#!/usr/bin/env python3
"""
Lab 5 – AGGREGATION PREPARATION (MACRO DATA)
Target: Inflation Rate (%) - Aggregation: Mean
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dslabs_functions import plot_bar_chart

# --- CONFIGURAÇÃO ---
INPUT_TRAIN = "prepared_data/scaling_train.csv"
INPUT_TEST = "prepared_data/scaling_test.csv"
# O índice do CSV é a data real (ex: 2010-01-31), o TARGET é a inflação
TARGET = "Inflation Rate (%)"

OUTPUT_IMG_DIR = Path("images_preparation")
OUTPUT_DATA_DIR = Path("prepared_data")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_and_combine_data():
    """
    Carrega os dados escalados e combina-os para permitir a re-agregação contínua.
    """
    print("Loading data from scaling step...")
    try:
        # Carrega o CSV, onde a primeira coluna (indice) é a data
        train = pd.read_csv(INPUT_TRAIN, index_col=0, parse_dates=True)
        test = pd.read_csv(INPUT_TEST, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Erro: Ficheiros de scaling não encontrados. Corre o scaling.py primeiro.")
        exit()
    
    # Data de corte original (o último momento do treino)
    split_date = train.index.max()
    
    # Combinar
    full_data = pd.concat([train, test]).sort_index()
    return full_data, split_date

def ts_aggregation_by(data, gran_level="M", agg_func="mean"):
    """
    Faz o resampling da série.
    Agregação por MÉDIA para taxas como a Inflação.
    """
    df = data.copy()
    # Resample e aplica função (mean para taxas)
    df_agg = df.resample(gran_level).agg(agg_func)
    # Remove linhas vazias (períodos sem dados)
    df_agg = df_agg.dropna()
    return df_agg

def split_by_date(data, split_date):
    """Volta a separar em treino e teste baseado na data de corte original"""
    train = data.loc[data.index <= split_date]
    test = data.loc[data.index > split_date]
    return train, test

def evaluate_models(train, test):
    """Treina LR e Persistência e calcula RMSE"""
    
    # 1. Persistence (Baseline)
    if len(train) == 0: return {'RMSE': 9999, 'R2': -9999}
    
    last_train_val = train.iloc[[-1]]
    # Este pred_persist é importante para ter um valor base de comparação (Persistence Model)
    # pred_persist = pd.concat([last_train_val, test]).shift(1).iloc[1:]
    
    # 2. Linear Regression (Lags)
    def create_lags(df, lags=3):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags[TARGET].shift(i)
        return df_lags.dropna()
    
    # Ajustar lags para datasets pequenos (ex: Anual ou Trimestral)
    # Usar um lag de 3 períodos (Mês/Trimestre/Ano) é um bom ponto de partida.
    n_lags = 3
    if len(train) < 10: n_lags = 1 # Se tiver menos de 10 pontos de treino, usar 1 lag
        
    train_lags = create_lags(train, lags=n_lags)
    test_lags = create_lags(test, lags=n_lags)
    
    if len(test_lags) == 0:
        return {'RMSE': 9999, 'R2': -9999}

    X_train, y_train = train_lags.drop(TARGET, axis=1), train_lags[TARGET]
    X_test, y_test = test_lags.drop(TARGET, axis=1), test_lags[TARGET]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_test, pred_lr))
    r2 = r2_score(y_test, pred_lr)
    
    return {'RMSE': rmse, 'R2': r2}

def main():
    print(f"\n=== LAB 5: AGGREGATION PREPARATION (Target: {TARGET}) ===\n")
    
    full_data, split_date = load_and_combine_data()
    print(f"Base data shape (Daily): {full_data.shape}")
    
    # Parametrizações para dados macroeconómicos:
    granularities = ["M", "Q", "A"] # Mensal, Trimestral, Anual
    agg_func = "mean"               # Alterado para média
    
    results = {}
    best_rmse = float('inf')
    best_gran = "M" # Default
    best_datasets = (None, None)
    
    for gran in granularities:
        print(f"\nProcessing Granularity: {gran}...")
        
        # A. Agregar
        df_agg = ts_aggregation_by(full_data, gran_level=gran, agg_func=agg_func)
        
        # B. Dividir
        train_agg, test_agg = split_by_date(df_agg, split_date)
        print(f"  Shape after agg: Train={train_agg.shape}, Test={test_agg.shape}")
        
        if len(test_agg) < 2:
            print("  -> Not enough data for testing. Skipping.")
            continue
            
        # C. Visualizar
        plt.figure(figsize=(12, 4))
        plt.plot(train_agg.index, train_agg[TARGET], label="Train", linewidth=1.5)
        plt.plot(test_agg.index, test_agg[TARGET], label="Test", linewidth=1.5, alpha=0.7)
        plt.title(f"Inflation Rate Aggregation: {gran} (func={agg_func})")
        plt.xlabel("Time")
        plt.ylabel(TARGET)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"aggregation_inflation_{gran}.png")
        plt.close()
        
        # D. Avaliar
        metrics = evaluate_models(train_agg, test_agg)
        print(f"  -> RMSE (LinReg): {metrics['RMSE']:.4f}")
        
        results[gran] = metrics['RMSE']
        
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_gran = gran
            best_datasets = (train_agg, test_agg)

    # Comparação
    print("\n--- RESULTS ---")
    valid_results = {k: v for k, v in results.items() if v != 9999}
    
    plt.figure(figsize=(8, 5))
    if valid_results:
        plot_bar_chart(list(valid_results.keys()), list(valid_results.values()),
                       title="RMSE by Aggregation (Inflation Rate)",
                       xlabel="Granularity", ylabel="RMSE")
        plt.savefig(OUTPUT_IMG_DIR / "aggregation_comparison_inflation.png")
    plt.close()

    print(f"\nBEST AGGREGATION: {best_gran}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # Guardar
    if best_datasets[0] is not None:
        final_train, final_test = best_datasets
        final_train.to_csv(OUTPUT_DATA_DIR / "aggregation_train.csv")
        final_test.to_csv(OUTPUT_DATA_DIR / "aggregation_test.csv")
        print(f"Saved best aggregation to {OUTPUT_DATA_DIR}/aggregation_train.csv")
    else:
        print("Error: No valid aggregation found.")

if __name__ == "__main__":
    main()
