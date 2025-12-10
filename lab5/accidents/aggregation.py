#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dslabs_functions import plot_bar_chart, plot_line_chart

# Configuração
INPUT_TRAIN = "prepared_data/scaling_train.csv"
INPUT_TEST = "prepared_data/scaling_test.csv"
TIMESTAMP = "crash_date"
TARGET = "crashes"

OUTPUT_IMG_DIR = Path("images/preparation")
OUTPUT_DATA_DIR = Path("prepared_data")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_and_combine_data():
    """
    Carrega os dados escalados e combina-os para permitir a re-agregação correta.
    Mantemos o registo de onde terminava o treino original (split_date).
    """
    print("Loading data from scaling step...")
    train = pd.read_csv(INPUT_TRAIN, index_col=0, parse_dates=True)
    test = pd.read_csv(INPUT_TEST, index_col=0, parse_dates=True)
    
    # Data de corte original (o último momento do treino)
    split_date = train.index.max()
    
    # Combinar para agregação contínua
    full_data = pd.concat([train, test]).sort_index()
    return full_data, split_date

def ts_aggregation_by(data, gran_level="D", agg_func="sum"):
    """
    Função baseada na dslabs_functions.
    Faz o resampling da série.
    """
    df = data.copy()
    # Resample e aplica função (sum ou mean)
    df_agg = df.resample(gran_level).agg(agg_func)
    # Remove linhas vazias geradas (se houver buracos na data)
    df_agg = df_agg.dropna()
    return df_agg

def split_by_date(data, split_date):
    """Volta a separar em treino e teste baseado na data de corte original"""
    train = data.loc[data.index <= split_date]
    test = data.loc[data.index > split_date]
    return train, test

def evaluate_models(train, test):
    """Treina LR e Persistência e calcula RMSE"""
    
    # --- 1. Persistence (t = t-1) ---
    if len(train) == 0: return {'RMSE': 9999, 'R2': -9999}
    
    last_train_val = train.iloc[[-1]]
    pred_persist = pd.concat([last_train_val, test]).shift(1).iloc[1:]
    
    # --- 2. Linear Regression (Lags) ---
    def create_lags(df, lags=3):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags[TARGET].shift(i)
        return df_lags.dropna()
    
    # Ajustar lags dependendo do tamanho dos dados (se for Mensal, lags=3 pode ser muito)
    n_lags = 3
    if len(train) < 20: n_lags = 1 # Fallback para dados muito curtos
        
    train_lags = create_lags(train, lags=n_lags)
    test_lags = create_lags(test, lags=n_lags)
    
    if len(test_lags) == 0:
        return {'RMSE': 9999, 'R2': -9999}

    X_train, y_train = train_lags.drop(TARGET, axis=1), train_lags[TARGET]
    X_test, y_test = test_lags.drop(TARGET, axis=1), test_lags[TARGET]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    
    # Alinhar índices para avaliação
    common_idx = y_test.index
    y_true = y_test.values
    p_lr = pred_lr
    
    rmse = np.sqrt(mean_squared_error(y_true, p_lr))
    r2 = r2_score(y_true, p_lr)
    
    return {'RMSE': rmse, 'R2': r2}

def main():
    print("\n=== LAB 5: AGGREGATION PREPARATION ===\n")
    
    # 1. Carregar dados do passo anterior (Scaling)
    full_data, split_date = load_and_combine_data()
    print(f"Full dataset shape: {full_data.shape}")
    print(f"Original Split Date: {split_date}")
    
    # 2. Definir Agregações a testar
    # Como o original já é "H" (Horário), vamos testar Daily, Weekly, Monthly
    granularities = ["D", "W", "ME"] # ME = Month End (pandas moderno usa ME ou M)
    agg_func = "sum" # Professor sugere SUM para contagens
    
    results = {}
    best_rmse = float('inf')
    best_gran = "D"
    best_datasets = (None, None)
    
    for gran in granularities:
        print(f"\nProcessing Granularity: {gran}...")
        
        # A. Agregar
        df_agg = ts_aggregation_by(full_data, gran_level=gran, agg_func=agg_func)
        
        # B. Dividir
        train_agg, test_agg = split_by_date(df_agg, split_date)
        
        print(f"  Shape after agg: Train={train_agg.shape}, Test={test_agg.shape}")
        
        # Se a agregação for muito agressiva (ex: Mensal) e não sobrar teste suficiente, saltar
        if len(test_agg) < 2:
            print("  -> Not enough data for testing. Skipping.")
            continue
            
        # C. Plot Visual
        plt.figure(figsize=(12, 4))
        plt.plot(train_agg.index, train_agg[TARGET], label="Train")
        plt.plot(test_agg.index, test_agg[TARGET], label="Test")
        plt.title(f"Aggregation: {gran} (func={agg_func})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"aggregation_{gran}.png")
        plt.close()
        
        # D. Avaliar
        metrics = evaluate_models(train_agg, test_agg)
        print(f"  -> RMSE (LinReg): {metrics['RMSE']:.4f}")
        
        results[gran] = metrics['RMSE']
        
        # E. Selecionar o Melhor
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_gran = gran
            best_datasets = (train_agg, test_agg)

    # 3. Comparação Final
    print("\n--- RESULTS ---")
    valid_results = {k: v for k, v in results.items() if v != 9999}
    
    plt.figure(figsize=(8, 5))
    if valid_results:
        plot_bar_chart(list(valid_results.keys()), list(valid_results.values()), 
                       title="Linear Regression RMSE by Aggregation", 
                       xlabel="Granularity", ylabel="RMSE")
        plt.savefig(OUTPUT_IMG_DIR / "aggregation_comparison.png")
    plt.close()

    print(f"BEST AGGREGATION: {best_gran}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    # 4. Guardar
    if best_datasets[0] is not None:
        final_train, final_test = best_datasets
        final_train.to_csv(OUTPUT_DATA_DIR / "aggregation_train.csv")
        final_test.to_csv(OUTPUT_DATA_DIR / "aggregation_test.csv")
        print(f"\nSaved best datasets to:\n  {OUTPUT_DATA_DIR}/aggregation_train.csv\n  {OUTPUT_DATA_DIR}/aggregation_test.csv")
    else:
        print("Error: No valid aggregation found.")

if __name__ == "__main__":
    main()