#!/usr/bin/env python3
"""
Lab 5 – DIFFERENTIATION PREPARATION (MACRO DATA)
Target: Inflation Rate (%)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dslabs_functions import plot_bar_chart

# --- CONFIGURAÇÃO ---
INPUT_TRAIN = "prepared_data/aggregation_train.csv"
INPUT_TEST = "prepared_data/aggregation_test.csv"
# Nome da coluna de tempo é o índice, TARGET é a Inflação
TARGET = "Inflation Rate (%)"

OUTPUT_IMG_DIR = Path("images_preparation") # AJUSTADO para consistência
OUTPUT_DATA_DIR = Path("prepared_data")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_and_combine_data():
    """
    Carrega os dados da fase de Agregação e combina para
    que a diferenciação na fronteira treino-teste seja contínua.
    """
    print("Loading data from aggregation step...")
    try:
        train = pd.read_csv(INPUT_TRAIN, index_col=0, parse_dates=True)
        test = pd.read_csv(INPUT_TEST, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Erro: Ficheiros de agregação não encontrados. Corre o aggregation.py primeiro.")
        exit()
    
    split_date = train.index.max()
    full_data = pd.concat([train, test]).sort_index()
    return full_data, split_date

def apply_differentiation(data, order=1):
    """
    Aplica diferenciação de ordem N.
    Order 0: Original
    Order 1: x(t) - x(t-1)
    Order 2: (x(t) - x(t-1)) - (x(t-1) - x(t-2))
    """
    df = data.copy()
    
    if order == 0:
        return df
    elif order == 1:
        # Primeira Diferença
        df[TARGET] = df[TARGET].diff()
    elif order == 2:
        # Segunda Diferença (Diferença da Diferença)
        df[TARGET] = df[TARGET].diff().diff()
            
    # Removemos os NaNs iniciais que resultam da diferenciação
    return df.dropna()

def split_by_date(data, split_date):
    """Volta a separar em treino e teste"""
    train = data.loc[data.index <= split_date]
    test = data.loc[data.index > split_date]
    return train, test

def evaluate_models(train, test):
    """Treina LR e Persistência e calcula RMSE"""
    
    # --- 1. Persistence ---
    if len(train) == 0: return {'RMSE': 9999, 'R2': -9999}
    
    last_train_val = train.iloc[[-1]]
    # pred_persist é necessário para calcular as métricas do Persistence Model,
    # embora o código atual só retorne métricas da Regressão Linear.
    # pred_persist = pd.concat([last_train_val, test]).shift(1).iloc[1:]
    
    # --- 2. Linear Regression ---
    def create_lags(df, lags=3):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags[TARGET].shift(i)
        return df_lags.dropna()
    
    # Ajuste de lags
    n_lags = 3
    if len(train) < 20: n_lags = 1
            
    train_lags = create_lags(train, lags=n_lags)
    test_lags = create_lags(test, lags=n_lags)
    
    if len(test_lags) == 0:
        return {'RMSE': 9999, 'R2': -9999}

    X_train, y_train = train_lags.drop(TARGET, axis=1), train_lags[TARGET]
    X_test, y_test = test_lags.drop(TARGET, axis=1), test_lags[TARGET]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    
    # Alinhar índices
    rmse = np.sqrt(mean_squared_error(y_test, pred_lr))
    r2 = r2_score(y_test, pred_lr)
    
    return {'RMSE': rmse, 'R2': r2}

def main():
    print(f"\n=== LAB 5: DIFFERENTIATION PREPARATION (Target: {TARGET}) ===\n")
    
    # 1. Carregar dados
    full_data, split_date = load_and_combine_data()
    print(f"Dados carregados. Shape: {full_data.shape}")
    
    # 2. Definir parâmetros (Ordens de diferenciação)
    diff_orders = [0, 1, 2] # Ordem 0 (Original), 1 e 2
    
    results = {}
    best_rmse = float('inf')
    best_order = 0
    best_datasets = (None, None)
    
    for order in diff_orders:
        name = f"Diff_Order_{order}"
        print(f"\nProcessing: {name}...")
        
        # A. Aplicar Diferenciação
        df_diff = apply_differentiation(full_data, order=order)
        
        # B. Dividir
        train_diff, test_diff = split_by_date(df_diff, split_date)
        print(f"  Shape: Train={train_diff.shape}, Test={test_diff.shape}")

        if len(test_diff) < 2:
            print("  -> Not enough data. Skipping.")
            continue
            
        # C. Visualizar
        plt.figure(figsize=(12, 4))
        plt.plot(train_diff.index, train_diff[TARGET], label="Train", linewidth=1.5)
        plt.plot(test_diff.index, test_diff[TARGET], label="Test", linewidth=1.5)
        plt.title(f"{TARGET} - Differentiation: Order {order}")
        plt.ylabel(f"Value (Order {order})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"differentiation_inflation_{name}.png")
        plt.close()
        
        # D. Avaliar
        metrics = evaluate_models(train_diff, test_diff)
        print(f"  -> RMSE (LinReg): {metrics['RMSE']:.4f}")
        
        results[name] = metrics['RMSE']
        
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_order = order
            best_datasets = (train_diff, test_diff)

    # 3. Comparação Final
    print("\n--- RESULTS ---")
    valid_results = {k: v for k, v in results.items() if v != 9999}
    
    plt.figure(figsize=(8, 5))
    if valid_results:
        plot_bar_chart(list(valid_results.keys()), list(valid_results.values()),
                       title="RMSE by Differentiation (Inflation Rate)",
                       xlabel="Diff Order", ylabel="RMSE")
        plt.savefig(OUTPUT_IMG_DIR / "differentiation_comparison_inflation.png")
    plt.close()
    
    print(f"BEST DIFFERENTIATION ORDER: {best_order}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # 4. Guardar
    if best_datasets[0] is not None:
        final_train, final_test = best_datasets
        # Guardamos com o nome differentiation_...
        final_train.to_csv(OUTPUT_DATA_DIR / "differentiation_train.csv")
        final_test.to_csv(OUTPUT_DATA_DIR / "differentiation_test.csv")
        print(f"\nSaved best datasets to:\n  {OUTPUT_DATA_DIR}/differentiation_train.csv\n  {OUTPUT_DATA_DIR}/differentiation_test.csv")
    else:
        print("Error: No valid differentiation found.")

if __name__ == "__main__":
    main()
