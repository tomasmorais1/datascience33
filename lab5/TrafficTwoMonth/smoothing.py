#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dslabs_functions import plot_bar_chart, plot_line_chart

# --- CONFIGURAÇÃO ---
INPUT_TRAIN = "prepared_data/differentiation_train.csv"
INPUT_TEST = "prepared_data/differentiation_test.csv"
TIMESTAMP = "Timestamp"  # Nome da coluna de tempo
TARGET = "Total"         # O teu target correto

OUTPUT_IMG_DIR = Path("images/preparation")
OUTPUT_DATA_DIR = Path("prepared_data")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_and_combine_data():
    """
    Carrega os dados da fase anterior (Differentiation) e combina
    para garantir continuidade na suavização entre treino e teste.
    """
    print("Loading data from differentiation step...")
    try:
        train = pd.read_csv(INPUT_TRAIN, index_col=0, parse_dates=True)
        test = pd.read_csv(INPUT_TEST, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_TRAIN}. Run differentiation.py first.")
        exit()
    
    # Data de corte para voltar a separar depois
    split_date = train.index.max()
    full_data = pd.concat([train, test]).sort_index()
    
    return full_data, split_date

def apply_smoothing(data, win_size):
    """
    Aplica Rolling Mean (Média Móvel) com o tamanho de janela especificado.
    O método rolling().mean() calcula a média dos últimos 'win_size' valores.
    """
    df = data.copy()
    if win_size > 1:
        # Aplica a suavização
        df[TARGET] = df[TARGET].rolling(window=win_size).mean()
        
    # Remove os NaNs gerados no início da série
    return df.dropna()

def split_by_date(data, split_date):
    """Volta a separar treino e teste"""
    train = data.loc[data.index <= split_date]
    test = data.loc[data.index > split_date]
    return train, test

def evaluate_models(train, test):
    """
    Treina e avalia modelos de Persistência e Regressão Linear.
    Retorna RMSE e R2.
    """
    # --- 1. Persistence ---
    if len(train) == 0: return {'RMSE': 9999, 'R2': -9999}
    
    last_train_val = train.iloc[[-1]]
    pred_persist = pd.concat([last_train_val, test]).shift(1).iloc[1:]
    
    # --- 2. Linear Regression (Lags) ---
    def create_lags(df, lags=3):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags[TARGET].shift(i)
        return df_lags.dropna()
    
    # Ajuste dinâmico de lags para datasets pequenos
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
    
    # Métricas
    y_true = y_test.values
    rmse = np.sqrt(mean_squared_error(y_true, pred_lr))
    r2 = r2_score(y_true, pred_lr)
    
    return {'RMSE': rmse, 'R2': r2}

def main():
    print(f"\n=== LAB 5: SMOOTHING PREPARATION (Target: {TARGET}) ===\n")
    
    # 1. Carregar dados
    full_data, split_date = load_and_combine_data()
    print(f"Data Loaded. Shape: {full_data.shape}")
    
    # 2. Definir parâmetros (Tamanhos de Janela)
    # win=1 (Original), win=3, win=5
    win_sizes = [1, 3, 5]
    
    results = {}
    best_rmse = float('inf')
    best_win = 1
    best_datasets = (None, None)
    
    for win in win_sizes:
        name = f"Smooth_Win_{win}"
        print(f"\nProcessing: {name}...")
        
        # A. Transformar
        df_smooth = apply_smoothing(full_data, win_size=win)
        
        # B. Dividir
        train_sm, test_sm = split_by_date(df_smooth, split_date)
        print(f"  Shape: Train={train_sm.shape}, Test={test_sm.shape}")
        
        if len(test_sm) < 2:
            print("  -> Not enough data for testing. Skipping.")
            continue

        # C. Visualizar (Plot como o prof faz)
        plt.figure(figsize=(12, 4))
        # Usamos plot_line_chart da biblioteca do prof se possível, ou plt direto
        try:
            plot_line_chart(
                train_sm.index, 
                train_sm[TARGET], 
                title=f"Smoothing window={win} (Train + Test)",
                xlabel=TIMESTAMP, 
                ylabel=TARGET
            )
            # Adicionar o teste ao plot para ver a continuidade
            plt.plot(test_sm.index, test_sm[TARGET], label="Test", color="orange")
        except:
             plt.plot(train_sm.index, train_sm[TARGET], label="Train")
             plt.plot(test_sm.index, test_sm[TARGET], label="Test")
             plt.title(f"Smoothing window={win}")

        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"smoothing_win_{win}.png")
        plt.close()
        
        # D. Avaliar
        metrics = evaluate_models(train_sm, test_sm)
        print(f"  -> RMSE (LinReg): {metrics['RMSE']:.4f}")
        
        results[name] = metrics['RMSE']
        
        # E. Selecionar o Melhor
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_win = win
            best_datasets = (train_sm, test_sm)

    # 3. Comparação Final
    print("\n--- RESULTS ---")
    valid_results = {k: v for k, v in results.items() if v != 9999}
    
    plt.figure(figsize=(8, 5))
    if valid_results:
        plot_bar_chart(list(valid_results.keys()), list(valid_results.values()), 
                       title="RMSE by Smoothing Window", 
                       xlabel="Window Size", ylabel="RMSE")
        plt.savefig(OUTPUT_IMG_DIR / "smoothing_comparison.png")
    plt.close()
    
    print(f"\nBEST SMOOTHING WINDOW: {best_win}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # 4. Guardar Ficheiros Finais
    if best_datasets[0] is not None:
        final_train, final_test = best_datasets
        # Guardamos como smoothing_train.csv para consistência e final_train para uso posterior
        final_train.to_csv(OUTPUT_DATA_DIR / "smoothing_train.csv")
        final_test.to_csv(OUTPUT_DATA_DIR / "smoothing_test.csv")
        
        # Opcional: Criar cópia com nome "final" se quiseres marcar o fim do pipeline
        final_train.to_csv(OUTPUT_DATA_DIR / "final_train.csv")
        final_test.to_csv(OUTPUT_DATA_DIR / "final_test.csv")

        print(f"\n=== PIPELINE COMPLETE ===")
        print(f"Final processed files saved to:\n  {OUTPUT_DATA_DIR}/smoothing_train.csv\n  {OUTPUT_DATA_DIR}/smoothing_test.csv")
    else:
        print("Error: No valid smoothing found.")

if __name__ == "__main__":
    main()