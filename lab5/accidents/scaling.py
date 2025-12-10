#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dslabs_functions import plot_bar_chart, HEIGHT

# Configuração
DATAFILE = "traffic_accidents.csv"
TIMESTAMP = "crash_date"
TRN_PCT = 0.90
OUTPUT_IMG_DIR = Path("images/preparation")
OUTPUT_DATA_DIR = Path("prepared_data")

# Criar diretorias se não existirem
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """Lê o csv e cria a Time Series base (Hourly Counts)"""
    df = pd.read_csv(DATAFILE)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df = df.set_index(TIMESTAMP).sort_index()
    # Agregar por hora para ter uma série numérica inicial
    ts = df.resample("H").size()
    ts.name = "crashes"
    return pd.DataFrame(ts)

def split_temporal(data):
    """Divide treino e teste sequencialmente (sem baralhar)"""
    train_size = int(len(data) * TRN_PCT)
    return data.iloc[:train_size], data.iloc[train_size:]

def scale_datasets(train, test, method="Original"):
    """
    Aplica o scaling:
    1. Fit no Treino.
    2. Transform no Treino e no Teste.
    Retorna os dataframes escalados e o objeto scaler.
    """
    if method == "Original":
        return train.copy(), test.copy(), None
    
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        
    # Fit apenas no treino para evitar data leakage
    scaler.fit(train)
    
    # Transformar e manter índices/colunas
    train_scaled = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    
    return train_scaled, test_scaled, scaler

def evaluate_models(train, test, scaler=None):
    """
    Treina Persistência e Linear Regression.
    Devolve erro calculado na escala ORIGINAL (inverse transform).
    """
    # --- 1. Persistence (t = t-1) ---
    # Usamos o último valor do treino para prever o 1º do teste
    last_train_val = train.iloc[[-1]]
    # Shift: empurra os valores 1 para a frente
    pred_persist = pd.concat([last_train_val, test]).shift(1).iloc[1:]
    
    # --- 2. Linear Regression (Lagged) ---
    def create_lags(df, lags=5):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags['crashes'].shift(i)
        return df_lags.dropna()
        
    train_lags = create_lags(train)
    test_lags = create_lags(test)
    
    # Se o dataset for muito pequeno após lags (pode acontecer em testes pequenos)
    if len(test_lags) == 0:
        return {'RMSE': 9999, 'R2': -9999} # Valor penalizador

    X_train, y_train = train_lags.drop('crashes', axis=1), train_lags['crashes']
    X_test, y_test = test_lags.drop('crashes', axis=1), test_lags['crashes']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    
    # --- Comparação na Escala Original ---
    # Precisamos alinhar os índices (LR perde as primeiras N linhas devido aos lags)
    common_index = y_test.index
    
    # Filtrar valores reais e predição de persistência pelos índices comuns
    y_true_scaled = y_test.values
    pred_lr_scaled = pred_lr
    
    # Se houver scaler, inverter para valores originais (nº de acidentes)
    if scaler:
        y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        p_lr = scaler.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()
    else:
        y_true = y_true_scaled
        p_lr = pred_lr_scaled

    rmse = np.sqrt(mean_squared_error(y_true, p_lr))
    r2 = r2_score(y_true, p_lr)
    
    return {'RMSE': rmse, 'R2': r2}

def main():
    print("\n=== LAB 5: SCALING PREPARATION ===\n")
    
    # 1. Carregar e Dividir
    data = prepare_data()
    train, test = split_temporal(data)
    print(f"Data split: Train={len(train)}, Test={len(test)}")
    
    results = {}
    methods = ["Original", "StandardScaler", "MinMaxScaler"]
    best_rmse = float('inf')
    best_method = "Original"
    best_datasets = (train, test) # Default
    
    # 2. Avaliar cada método
    metrics_list = []
    
    for method in methods:
        print(f"Testing {method}...")
        trn_s, tst_s, scaler = scale_datasets(train, test, method)
        
        # Guardar gráfico visual da série
        plt.figure(figsize=(12, 4))
        plt.plot(trn_s.index, trn_s.values, label="Train")
        plt.plot(tst_s.index, tst_s.values, label="Test", alpha=0.6)
        plt.title(f"Scaling Series: {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"scaling_{method}_ts.png")
        plt.close()

        # Avaliar
        scores = evaluate_models(trn_s, tst_s, scaler)
        print(f"  -> RMSE (LinReg): {scores['RMSE']:.4f}")
        
        results[method] = scores['RMSE']
        
        # Verificar se é o melhor
        if scores['RMSE'] < best_rmse:
            best_rmse = scores['RMSE']
            best_method = method
            best_datasets = (trn_s, tst_s)

    # 3. Gráfico de Comparação
    plt.figure(figsize=(8, 5))
    plot_bar_chart(list(results.keys()), list(results.values()), title="Linear Regression RMSE by Scaling", xlabel="Scaler", ylabel="RMSE")
    plt.savefig(OUTPUT_IMG_DIR / "scaling_comparison.png")
    plt.close()
    
    print("\n===============================")
    print(f"BEST SCALING METHOD: {best_method}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("===============================\n")

    # 4. Guardar o Vencedor para o próximo script
    final_train, final_test = best_datasets
    final_train.to_csv(OUTPUT_DATA_DIR / "scaling_train.csv")
    final_test.to_csv(OUTPUT_DATA_DIR / "scaling_test.csv")
    
    print(f"Saved best datasets to:\n  {OUTPUT_DATA_DIR}/scaling_train.csv\n  {OUTPUT_DATA_DIR}/scaling_test.csv")

if __name__ == "__main__":
    main()