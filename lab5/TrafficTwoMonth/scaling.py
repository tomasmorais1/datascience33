#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dslabs_functions import plot_bar_chart

# --- CONFIGURAÇÃO ---
DATAFILE = "TrafficTwoMonth.csv"
TARGET = "Total"  # A coluna que queremos prever
TRN_PCT = 0.90    # Split de treino

OUTPUT_IMG_DIR = Path("images/preparation")
OUTPUT_DATA_DIR = Path("prepared_data")
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """
    Lê o csv e prepara a Série Temporal.
    Gera um índice temporal limpo para evitar erros de 'day out of range'.
    """
    print(f"Loading {DATAFILE}...")
    try:
        df = pd.read_csv(DATAFILE)
    except FileNotFoundError:
        print(f"ERRO: Ficheiro {DATAFILE} não encontrado.")
        exit()

    # 1. GERAR DATAS (Solução Robusta)
    # Vemos que os dados começam no dia 10 às 00:00 (12:00:00 AM) e têm freq 15 min.
    # Vamos gerar uma sequência contínua a partir de uma data segura (ex: 2023-01-10).
    print("Generating Clean Time Index (15min freq)...")
    
    # Criar range exato com o número de linhas do dataset
    clean_index = pd.date_range(
        start='2023-01-10 00:00:00', 
        periods=len(df), 
        freq='15T' # 'T' significa minutos no Pandas
    )
    
    df['Timestamp'] = clean_index
    df = df.set_index('Timestamp').sort_index()

    # 2. SELEÇÃO DO TARGET
    # O enunciado diz para aplicar transformação a todas as variáveis se for multivariado,
    # mas para este exercício de preparação foca-se na série univariada do TARGET.
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' não encontrado. Colunas disponíveis: {df.columns.tolist()}")

    # Selecionamos apenas a coluna alvo para a série temporal
    ts = df[[TARGET]].astype(float)
    return ts

def split_temporal(data):
    """
    Divide o dataset em treino e teste mantendo a ordem temporal.
    """
    train_size = int(len(data) * TRN_PCT)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test

def scale_datasets(train, test, method="Original"):
    """
    Aplica o scaling (Original, Standard, MinMax).
    Fit no Treino -> Transform no Treino e Teste.
    """
    if method == "Original":
        return train.copy(), test.copy(), None
    
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        
    # Fit apenas no treino
    scaler.fit(train)
    
    # Transformar mantendo a estrutura de DataFrame
    vars = train.columns
    train_scaled = pd.DataFrame(scaler.transform(train), index=train.index, columns=vars)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=vars)
    
    return train_scaled, test_scaled, scaler

def evaluate_models(train, test, scaler=None):
    """
    Treina Persistence e Linear Regression e devolve métricas.
    As métricas são calculadas na escala original.
    """
    # 1. Persistence (Previsão t = valor em t-1)
    # O shift(1) pega no valor anterior. O primeiro valor do teste precisa do último do treino.
    last_train = train.iloc[[-1]]
    pred_persist = pd.concat([last_train, test]).shift(1).iloc[1:]
    
    # 2. Linear Regression (com Lags)
    def create_lags(df, lags=5):
        df_lags = df.copy()
        for i in range(1, lags+1):
            df_lags[f'lag_{i}'] = df_lags[TARGET].shift(i)
        return df_lags.dropna()
        
    train_lags = create_lags(train)
    test_lags = create_lags(test)
    
    # Se o teste for pequeno demais
    if len(test_lags) == 0:
        return {'RMSE': 9999, 'MAE': 9999, 'R2': -9999}

    X_train = train_lags.drop(TARGET, axis=1)
    y_train = train_lags[TARGET]
    X_test = test_lags.drop(TARGET, axis=1)
    y_test = test_lags[TARGET]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    
    # --- AVALIAÇÃO (Converter para escala original se necessário) ---
    y_true = y_test.values.flatten()
    y_pred_lr = pred_lr.flatten()
    
    if scaler:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_lr = scaler.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
    
    # Calcular Métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_lr))
    mae = mean_absolute_error(y_true, y_pred_lr)
    r2 = r2_score(y_true, y_pred_lr)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main():
    print(f"\n=== LAB 5: SCALING PREPARATION (Target: {TARGET}) ===\n")
    
    # 1. Preparar Dados
    data = prepare_data()
    print(f"Range: {data.index.min()} -> {data.index.max()}")
    print(f"Total rows: {len(data)}")
    
    # 2. Split Treino/Teste
    train, test = split_temporal(data)
    print(f"Train size: {len(train)} | Test size: {len(test)}")
    
    # 3. Testar parametrizações
    methods = ["Original", "StandardScaler", "MinMaxScaler"]
    results = {}
    best_score = float('inf') # Minimizar RMSE
    best_method = "Original"
    best_datasets = (train, test)
    
    metric_to_compare = 'RMSE' #
    
    for method in methods:
        print(f"\nProcessing: {method}...")
        
        # A. Scaling
        trn_s, tst_s, scaler = scale_datasets(train, test, method)
        
        # B. Plot da Série Transformada (Visualização)
        plt.figure(figsize=(10, 4))
        plt.plot(trn_s.index, trn_s[TARGET], label="Train", linewidth=1)
        plt.plot(tst_s.index, tst_s[TARGET], label="Test", linewidth=1, alpha=0.7)
        plt.title(f"Scaling: {method}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG_DIR / f"scaling_{method}_ts.png")
        plt.close()
        
        # C. Treinar e Avaliar Modelos
        metrics = evaluate_models(trn_s, tst_s, scaler)
        print(f"  Linear Regression -> RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}")
        
        results[method] = metrics[metric_to_compare]
        
        # D. Selecionar o melhor
        # Escolhemos o que tiver menor RMSE
        if metrics[metric_to_compare] < best_score:
            best_score = metrics[metric_to_compare]
            best_method = method
            best_datasets = (trn_s, tst_s)

    # 4. Comparação Final e Seleção
    print("\n=== RESULTS COMPARISON ===")
    plt.figure(figsize=(8, 5))
    plot_bar_chart(list(results.keys()), list(results.values()), 
                   title=f"Model Performance by Scaling ({metric_to_compare})", 
                   xlabel="Method", ylabel=metric_to_compare)
    plt.savefig(OUTPUT_IMG_DIR / "scaling_comparison.png")
    plt.close()
    
    print(f"Best Method: {best_method} with {metric_to_compare}={best_score:.4f}")
    
    # 5. Guardar datasets para a próxima tarefa (Aggregation)
    final_train, final_test = best_datasets
    final_train.to_csv(OUTPUT_DATA_DIR / "scaling_train.csv")
    final_test.to_csv(OUTPUT_DATA_DIR / "scaling_test.csv")
    print(f"Datasets saved to {OUTPUT_DATA_DIR}/scaling_train.csv and scaling_test.csv")

if __name__ == "__main__":
    main()