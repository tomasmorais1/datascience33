import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Importa as funções do módulo que você forneceu
from dslabs_functions import (
    FORECAST_MEASURES,
    DELTA_IMPROVE,
    plot_line_chart,
    plot_forecasting_eval,
    plot_forecasting_series,
    dataframe_temporal_train_test_split,
    plot_multiline_chart # <--- CORREÇÃO AQUI
)

# ------------------------------------------------------------------------------
# 1. Funções Utilitárias para S.T.
# ------------------------------------------------------------------------------

def series_to_supervised(data: pd.Series, n_in: int = 1) -> pd.DataFrame:
    """
    Converte uma série temporal em um formato de aprendizagem supervisionada.
    
    data: a série temporal a ser convertida.
    n_in: número de passos anteriores (lags) a serem usados como features.
    """
    df = pd.DataFrame(data.copy())
    
    # Cria as colunas de lag (X)
    cols = []
    for i in range(n_in, 0, -1):
        df[f'Lag_{i}'] = df[data.name].shift(i)
        cols.append(f'Lag_{i}')
        
    # Remove as linhas com NaN resultantes do shift (primeiras 'n_in' linhas)
    df.dropna(inplace=True)
    
    # Coluna Target (Y)
    df.rename(columns={data.name: 'Target'}, inplace=True)
    
    # Reordena colunas para X seguido por Y
    return df[cols + ['Target']]


# ------------------------------------------------------------------------------
# 2. Configuração e Carregamento dos Dados
# ------------------------------------------------------------------------------

DATA_PATH = "prepared_data"
# Usamos os arquivos 'final' para ser o conjunto de dados base
TRAIN_FILENAME = os.path.join(DATA_PATH, "final_train.csv")
TEST_FILENAME = os.path.join(DATA_PATH, "final_test.csv")

FILE_TAG: str = "mlp_data"
# Assumindo que a coluna alvo no final_train.csv se chama "Total"
TARGET_COL: str = "Inflation Rate (%)"
MEASURE: str = "R2"
MAX_LAG: int = 12 # Número máximo de lags a serem testados

print(f"--- 🚀 Multi-layer Perceptrons (MLP) para {TARGET_COL} ---")

try:
    # Carregando dados de treino e teste
    train_df = pd.read_csv(TRAIN_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    test_df = pd.read_csv(TEST_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    
    # Extrair as Séries Temporais (univariada)
    train_series: pd.Series = train_df[TARGET_COL]
    test_series: pd.Series = test_df[TARGET_COL]

    print(f"Dados carregados: Treino ({len(train_series)} pontos), Teste ({len(test_series)} pontos).")
    
except Exception as e:
    print(f"ERRO ao carregar dados. Verifique se os arquivos final_train/test.csv e a coluna '{TARGET_COL}' existem.")
    print(f"Detalhes do erro: {e}")
    exit()

# ------------------------------------------------------------------------------
# 3. Função de Estudo de Hyperparâmetros (Lags e Neurónios)
# ------------------------------------------------------------------------------

def mlp_study(train_s: pd.Series, test_s: pd.Series, measure: str = "R2"):
    """
    Estuda o número de lags e o número de neurónios na primeira camada oculta.
    """
    lag_values = list(range(1, MAX_LAG + 1))
    neurons_values = [5, 10, 20] # Testando diferentes tamanhos de camada
    
    best_performance: float = -100000 if measure == "R2" else 100000
    is_maximizing = measure == "R2"
    best_model = None
    best_params: dict = {"name": "MLP", "metric": measure, "params": ()}
    
    study_results = {} # {num_neurons: [performance_lag1, performance_lag2, ...]}

    print("\nIniciando estudo de Hyperparâmetros (Lags x Neurónios)...")

    for n_neurons in neurons_values:
        perf_n = []
        for n_lag in lag_values:
            # 1. Preparar os dados (Treino e Teste)
            
            # Combinar treino e teste temporariamente para a criação de lags, garantindo continuidade
            full_series = pd.concat([train_s, test_s])
            
            # Criar a matriz supervisionada (X, Y)
            supervised_df = series_to_supervised(full_series, n_in=n_lag)
            
            # Dividir X e Y em treino e teste (considerando o tamanho da série original)
            # Como a criação de lags remove n_lag pontos, o split é ajustado:
            # Novo tamanho de treino: len(train_s) - n_lag
            train_size_adj = len(train_s) - n_lag
            
            train_X = supervised_df.iloc[:train_size_adj].drop('Target', axis=1)
            train_y = supervised_df.iloc[:train_size_adj]['Target']
            
            test_X = supervised_df.iloc[train_size_adj:].drop('Target', axis=1)
            test_y = supervised_df.iloc[train_size_adj:]['Target']

            # 2. Treinar o MLP
            model = MLPRegressor(
                hidden_layer_sizes=(n_neurons,), # Uma única camada oculta
                max_iter=500,
                random_state=42,
                activation='relu',
                solver='adam'
            )
            model.fit(train_X, train_y)
            
            # 3. Previsão e Avaliação
            prd_tst = model.predict(test_X)
            
            # Ajusta o tamanho das séries para a avaliação (após o lag shift)
            test_y_adjusted = test_y.iloc[-len(prd_tst):]
            test_index_adjusted = test_y_adjusted.index
            
            prd_tst_series = pd.Series(prd_tst, index=test_index_adjusted)
            
            eval_result: float = FORECAST_MEASURES[measure](test_y_adjusted, prd_tst_series)
            
            perf_n.append(eval_result)

            # 4. Atualizar o melhor modelo
            is_current_better = (is_maximizing and eval_result > best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE) or \
                                (not is_maximizing and eval_result < best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE)
            
            if is_current_better:
                best_performance = eval_result
                best_params["params"] = (n_lag, n_neurons)
                best_model = model
            
            # print(f"  Lags={n_lag}, Neurons={n_neurons} -> {measure}={eval_result:.4f}")

        study_results[f'{n_neurons} neurons'] = perf_n

    print(f"\n✅ Melhor Modelo: Lags={best_params['params'][0]}, Neurons={best_params['params'][1]} -> {measure}={best_performance:.4f}")
    
    # Geração do gráfico de estudo de hyperparameters (Multiline Chart)
    plot_line_chart(
        lag_values,
        study_results[f'{neurons_values[0]} neurons'], # Plota a primeira linha
        title=f"MLP Study: Lags vs {measure}",
        xlabel="Number of Lags",
        ylabel=measure,
        percentage=measure == 'R2' or measure == 'MAPE',
    )
    
    # Adicionar as outras linhas usando o plot_multiline_chart
    # (assumindo que plot_line_chart configura o eixo, e plot_multiline_chart é melhor para o estudo)
    # A função plot_line_chart no dslabs_functions não suporta múltiplas linhas diretamente
    # Vamos usar plot_multiline_chart para um melhor visual:
    
    plot_multiline_chart(
        lag_values,
        study_results,
        title=f"MLP Hyperparameter Study ({measure})",
        xlabel="Number of Lags",
        ylabel=measure,
        percentage=measure == 'R2' or measure == 'MAPE',
    )
    
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{FILE_TAG}_mlp_{measure}_study.png")
    plt.close()

    return best_model, best_params, train_series, test_series # Retorna as séries originais


# ------------------------------------------------------------------------------
# 4. Execução do Estudo e Geração dos Gráficos Finais
# ------------------------------------------------------------------------------

best_model_mlp, best_params_mlp, train_s, test_s = mlp_study(train_series, test_series, measure=MEASURE)
best_n_lag, best_n_neurons = best_params_mlp["params"]

# 5. Preparação dos dados finais para o melhor modelo
full_series = pd.concat([train_s, test_s])
supervised_df = series_to_supervised(full_series, n_in=best_n_lag)

train_size_adj = len(train_s) - best_n_lag

train_X_final = supervised_df.iloc[:train_size_adj].drop('Target', axis=1)
train_y_final = supervised_df.iloc[:train_size_adj]['Target']

test_X_final = supervised_df.iloc[train_size_adj:].drop('Target', axis=1)
test_y_final = supervised_df.iloc[train_size_adj:]['Target']


# 6. Geração de Previsões no Treino e Teste
prd_trn_array = best_model_mlp.predict(train_X_final)
prd_tst_array = best_model_mlp.predict(test_X_final)

# Reajustar índices e converter para Series para plotting
prd_trn = pd.Series(prd_trn_array, index=train_y_final.index)
prd_tst = pd.Series(prd_tst_array, index=test_y_final.index)

# Para o plotting do forecast, precisamos reajustar o treino e teste
# O treino original (train_s) deve ser truncado para coincidir com prd_trn
# O teste original (test_s) deve ser truncado para coincidir com prd_tst
train_s_adj = train_s.loc[prd_trn.index] # Trunca o treino (remove o lag)
test_s_adj = test_s.loc[prd_tst.index] # Trunca o teste (remove o lag)

# 7. Geração dos Gráficos

# Gráfico de Avaliação (Performance)
print("Gerando gráficos de avaliação e previsão...")
plot_forecasting_eval(
    train_s_adj, test_s_adj, prd_trn, prd_tst,
    title=f"{FILE_TAG} - MLP (Lags={best_n_lag}, Neurons={best_n_neurons}) Performance"
)
plt.savefig(f"images/{FILE_TAG}_mlp_{MEASURE}_eval.png")
plt.close()

# Gráfico de Previsão da Série (Forecast)
plot_forecasting_series(
    train_s_adj, test_s_adj, prd_tst,
    title=f"{FILE_TAG} - MLP (Lags={best_n_lag}, Neurons={best_n_neurons}) Forecast",
    xlabel="Timestamp",
    ylabel=TARGET_COL,
)
plt.savefig(f"images/{FILE_TAG}_mlp_{MEASURE}_forecast.png")
plt.close()

# 8. Cálculo e Impressão das Métricas Finais
print("\n--- 📊 Performance do Melhor Modelo (Conjunto de Teste) ---")

metrics = ["MSE", "MAE", "R2"]
performance_results = {}

for metric in metrics:
    # Usamos o test_y_final e prd_tst para as métricas
    score = FORECAST_MEASURES[metric](test_y_final, prd_tst)
    performance_results[metric] = score
    print(f"{metric}: {score:.6f}")

print("\nRelatório Resumo para PDF:")
print(f"Técnica: Multi-layer Perceptrons")
print(f"Melhor Hyperparâmetro: Lags={best_n_lag}, Neurons={best_n_neurons}")
print(f"Performance (Teste): R2={performance_results['R2']:.6f} | MSE={performance_results['MSE']:.6f} | MAE={performance_results['MAE']:.6f}")
print("\n3 gráficos (Study, Performance, Forecast) foram salvos na pasta 'images'.")
