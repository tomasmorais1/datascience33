import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os
import numpy as np
from matplotlib import pyplot as plt

# Importa as funções do módulo que você forneceu
from dslabs_functions import (
    FORECAST_MEASURES,
    DELTA_IMPROVE,
    plot_line_chart,
    plot_forecasting_eval,
    plot_forecasting_series,
)

# ------------------------------------------------------------------------------
# 1. Configuração e Carregamento dos Dados (igual)
# ------------------------------------------------------------------------------

DATA_PATH = "prepared_data"
TRAIN_FILENAME = os.path.join(DATA_PATH, "smoothing_train.csv")
TEST_FILENAME = os.path.join(DATA_PATH, "smoothing_test.csv")

FILE_TAG: str = "smoothing_data"
TARGET_COL: str = "Total"
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
# 2. Função de Estudo do Hyperparâmetro Alpha (Corrigida e Formatada)
# ------------------------------------------------------------------------------

def exponential_smoothing_study_formatted(train: pd.Series, test: pd.Series, measure: str = "R2"):
    
    # Granularidade aumentada (mantida para curva suave)
    alpha_values = [i / 50 for i in range(5, 50, 1)] # De 0.10 a 0.98
    
    percentage_flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Exponential Smoothing", "metric": measure, "params": ()}
    is_maximizing = measure == "R2"
    best_performance: float = -100000 if is_maximizing else 100000
    yvalues = []
    
    is_better = lambda new_eval, old_eval: (
        (is_maximizing and new_eval > old_eval and abs(new_eval - old_eval) > DELTA_IMPROVE) or
        (not is_maximizing and new_eval < old_eval and abs(new_eval - old_eval) > DELTA_IMPROVE)
    )

    print("\nIniciando estudo de Alpha...")
    for alpha in alpha_values:
        tool = SimpleExpSmoothing(train)
        model = tool.fit(smoothing_level=alpha, optimized=False)
        prd_tst = model.forecast(steps=len(test))
        eval_result: float = FORECAST_MEASURES[measure](test, prd_tst)
        
        if is_better(eval_result, best_performance):
            best_performance = eval_result
            best_params["params"] = (alpha,)
            best_model = model
        
        yvalues.append(eval_result)

    min_r2 = min(yvalues)
    y_min_limit = np.floor(min_r2) - 0.5
    if y_min_limit > -1.0:
        y_min_limit = -1.0

    print(f"\n✅ Melhor Alpha: {best_params['params'][0]:.2f} -> {measure}={best_performance:.4f}")
    
    # Geração do gráfico de estudo (Hyperparameters study)
    plot_line_chart(
        alpha_values,
        yvalues,
        title=f"Exponential Smoothing Study ({measure})",
        xlabel="Alpha (smoothing_level)",
        ylabel=measure,
        percentage=percentage_flag,
    )
    
    ax = plt.gca()

    # 🚨 CORREÇÃO 1: Ajuste forçado do Y-limit (para visibilidade da curva)
    ax.set_ylim(bottom=y_min_limit, top=1.0)
    
    # 🚨 CORREÇÃO 2: Formatação do Eixo X (para legibilidade)
    # Definir onde os ticks devem aparecer (ex: a cada 0.1)
    x_ticks = np.arange(0.1, 1.0, 0.1).tolist()
    x_labels = [f"{x:.1f}" for x in x_ticks]
    
    # Forçar a exibição apenas dos ticks definidos
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=0, fontsize='xx-small', ha='center') # Rotation=0 elimina a sobreposição
    
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{FILE_TAG}_exponential_smoothing_{measure}_study_FORMATTED.png")
    plt.close()

    return best_model, best_params

# ------------------------------------------------------------------------------
# 3. Execução do Estudo e Geração dos Gráficos Finais
# ------------------------------------------------------------------------------

# Chamando a nova função formatada
best_model, best_params = exponential_smoothing_study_formatted(train_series, test_series, measure=MEASURE)
best_alpha = best_params["params"][0]

# Previsão no conjunto de TREINO e TESTE
prd_trn = best_model.predict(start=0, end=len(train_series) - 1)
prd_tst = best_model.forecast(steps=len(test_series))

# Geração dos gráficos de Avaliação e Previsão (os dois restantes)
print("Gerando gráficos de avaliação e previsão (os dois restantes)...")

# Gráfico de Avaliação (Performance)
plot_forecasting_eval(
    train_series, test_series, prd_trn, prd_tst,
    title=f"{FILE_TAG} - Exponential Smoothing (Alpha={best_alpha:.2f}) Performance"
)
plt.savefig(f"images/{FILE_TAG}_exponential_smoothing_{MEASURE}_eval.png")
plt.close()

# Gráfico de Previsão da Série (Forecast)
plot_forecasting_series(
    train_series, test_series, prd_tst,
    title=f"{FILE_TAG} - Exponential Smoothing (Alpha={best_alpha:.2f}) Forecast",
    xlabel="Timestamp",
    ylabel=TARGET_COL,
)
plt.savefig(f"images/{FILE_TAG}_exponential_smoothing_{MEASURE}_forecast.png")
plt.close()

# 4. Cálculo e Impressão das Métricas Finais
print("\n--- 📊 Performance do Melhor Modelo (Conjunto de Teste) ---")

metrics = ["MSE", "MAE", "R2"]
performance_results = {}

for metric in metrics:
    score = FORECAST_MEASURES[metric](test_series, prd_tst)
    performance_results[metric] = score
    print(f"{metric}: {score:.6f}")

print("\nRelatório Resumo para PDF:")
print(f"Técnica: Exponential Smoothing")
print(f"Melhor Hyperparâmetro: Alpha={best_alpha:.2f}")
# A f-string corrigida é:
print(f"Performance (Teste): R2={performance_results['R2']:.6f} | MSE={performance_results['MSE']:.6f} | MAE={performance_results['MAE']:.6f}")

print("\nOs 3 gráficos foram salvos na pasta 'images'. O gráfico de estudo corrigido e formatado é '...study_FORMATTED.png'.")
