import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

# Importa as funções do módulo que você forneceu
from dslabs_functions import (
    FORECAST_MEASURES,
    DELTA_IMPROVE,
    plot_forecasting_eval,
    plot_forecasting_series,
    plot_multiline_chart
)

# ------------------------------------------------------------------------------
# 1. Configuração e Carregamento dos Dados (igual)
# ------------------------------------------------------------------------------

DATA_PATH = "prepared_data"
TRAIN_FILENAME = os.path.join(DATA_PATH, "final_train.csv")
TEST_FILENAME = os.path.join(DATA_PATH, "final_test.csv")

FILE_TAG: str = "final_data"
TARGET_COL: str = "Inflation Rate (%)"
MEASURE: str = "R2"

try:
    train_df = pd.read_csv(TRAIN_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    test_df = pd.read_csv(TEST_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    train_series: pd.Series = train_df[TARGET_COL]
    test_series: pd.Series = test_df[TARGET_COL]
except Exception as e:
    print(f"Erro ao carregar dados: {e}")
    exit()

# ------------------------------------------------------------------------------
# 2. Função de Estudo do Hyperparâmetro ARIMA (p, d, q) - CORRIGIDA
# ------------------------------------------------------------------------------

def arima_study_corrected(train: pd.Series, test: pd.Series, measure: str = "R2"):
    
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7)
    q_params = (1, 3, 5, 7)

    percentage_flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    
    is_maximizing = measure == "R2"
    best_performance: float = -100000 if is_maximizing else 100000
    
    # Lista para armazenar TODOS os resultados de R2 para o cálculo do limite Y
    all_r2_values = []
    
    # Inicializa a figura para o estudo
    fig, axs = plt.subplots(1, len(d_values), figsize=(len(d_values) * 4.5, 4.5))
    
    print("\nIniciando pesquisa de grade (Grid Search) para ARIMA (p, d, q)...")
    
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        
        for q in q_params:
            yvalues = []
            for p in p_params:
                try:
                    # Tenta treinar o modelo
                    arima = ARIMA(train, order=(p, d, q))
                    model = arima.fit()
                    
                    # Previsão e Avaliação
                    prd_tst = model.forecast(steps=len(test))
                    eval_result: float = FORECAST_MEASURES[measure](test, prd_tst)
                    
                    # Armazena o R2 para o cálculo do limite Y
                    all_r2_values.append(eval_result)
                    
                    # Atualiza o Melhor Modelo
                    is_current_better = (is_maximizing and eval_result > best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE) or \
                                        (not is_maximizing and eval_result < best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE)
                    
                    if is_current_better:
                        best_performance = eval_result
                        best_params["params"] = (p, d, q)
                        best_model = model
                    
                    yvalues.append(eval_result)
                
                except Exception as e:
                    # Usa o pior desempenho em caso de falha do fit
                    yvalues.append(-10000 if is_maximizing else 10000)
            
            values[f'q={q}'] = yvalues
        
        # 5. Plotagem do Subgráfico para o valor 'd' atual
        plot_multiline_chart(
            p_params,
            values,
            ax=axs[i],
            title=f"ARIMA d={d} ({measure})",
            xlabel="p (Autoregressive order)",
            ylabel=measure,
            percentage=percentage_flag
        )
        
    # 6. CÁLCULO E AJUSTE GLOBAL DO LIMITE Y (CORREÇÃO)
    
    # Encontrar o R2 mínimo de todos os testes
    min_r2 = min(all_r2_values)
    # Define um limite inferior um pouco menor que o valor mínimo para dar margem
    y_min_limit = np.floor(min_r2) - 0.5
    
    # Ajusta o ylim de TODOS os subplots
    for ax in axs:
        ax.set_ylim(bottom=y_min_limit, top=1.0)
        
    print(f"\nAJUSTE Y-LIMIT: De [-1.0, 1.0] para [{y_min_limit:.2f}, 1.0].")
    print(
        f"\n✅ ARIMA best results achieved with (p,d,q)=({best_params['params'][0]}, {best_params['params'][1]}, {best_params['params'][2]}) ==> {measure}={best_performance:.4f}"
    )

    # 7. Salvar o Gráfico de Estudo
    os.makedirs("images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_arima_{measure}_study_CORRECTED.png")
    plt.close(fig)

    return best_model, best_params


# ------------------------------------------------------------------------------
# 3. Execução do Estudo e Geração dos Gráficos Finais
# ------------------------------------------------------------------------------

# 3.1 Execução
best_model_arima, best_params_arima = arima_study_corrected(train_series, test_series, measure=MEASURE)
best_p, best_d, best_q = best_params_arima["params"]

# 3.2 Geração de Previsões no Treino e Teste
print("\nGerando previsões e gráficos finais...")

prd_trn = best_model_arima.predict(start=0, end=len(train_series) - 1)
prd_tst = best_model_arima.forecast(steps=len(test_series))

# 3.3 Geração dos Gráficos

# Gráfico de Avaliação (Performance)
plot_forecasting_eval(
    train_series, test_series, prd_trn, prd_tst,
    title=f"{FILE_TAG} - ARIMA (p={best_p}, d={best_d}, q={best_q}) Performance"
)
plt.savefig(f"images/{FILE_TAG}_arima_{MEASURE}_eval.png")
plt.close()

# Gráfico de Previsão da Série (Forecast)
plot_forecasting_series(
    train_series, test_series, prd_tst,
    title=f"{FILE_TAG} - ARIMA (p={best_p}, d={best_d}, q={best_q}) Forecast",
    xlabel="Timestamp",
    ylabel=TARGET_COL,
)
plt.savefig(f"images/{FILE_TAG}_arima_{MEASURE}_forecast.png")
plt.close()

# 3.4 Cálculo e Impressão das Métricas Finais
print("\n--- 📊 Performance do Melhor Modelo (Conjunto de Teste) ---")

metrics = ["MSE", "MAE", "R2"]
performance_results = {}

for metric in metrics:
    score = FORECAST_MEASURES[metric](test_series, prd_tst)
    performance_results[metric] = score
    print(f"{metric}: {score:.6f}")

print("\nRelatório Resumo para PDF:")
print(f"Técnica: ARIMA")
print(f"Melhor Hyperparâmetro: (p, d, q) = ({best_p}, {best_d}, {best_q})")
print(f"Performance (Teste): R2={performance_results['R2']:.6f} | MSE={performance_results['MSE']:.6f} | MAE={performance_results['MAE']:.6f}")
print("\n3 gráficos (Study, Performance, Forecast) foram salvos na pasta 'images'. O gráfico de estudo corrigido é '...study_CORRECTED.png'.")
