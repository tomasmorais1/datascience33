import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy
from math import sqrt
from torch import no_grad, tensor, float32
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Importa as funções do módulo que você forneceu
from dslabs_functions import (
    FORECAST_MEASURES,
    DELTA_IMPROVE,
    plot_forecasting_eval,
    plot_forecasting_series,
    plot_multiline_chart
)

# ------------------------------------------------------------------------------
# 1. Classes e Funções de LSTM
# ------------------------------------------------------------------------------

def prepare_dataset_for_lstm(series, seq_length: int = 4):
    setX: list = []
    setY: list = []
    
    if series.ndim == 2:
        series = series.ravel()
        
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length]
        future = series[i + seq_length]
        setX.append(past)
        setY.append(future)
        
    return tensor(setX).unsqueeze(-1).to(float32), tensor(setY).unsqueeze(-1).to(float32)

class DS_LSTM(Module):
    def __init__(self, train, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        # Verifica se o trnX está vazio (pode acontecer se o length for muito grande)
        if len(trnX) == 0:
             raise ValueError(f"O tamanho da sequência ({length}) é muito grande para o conjunto de treino.")
             
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=len(train) // 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        # Usamos o output do último passo da sequência para a previsão (1-step-ahead)
        x = self.linear(x[:, -1, :])
        return x

    def fit(self):
        self.train()
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def predict(self, X):
        self.eval()
        with no_grad():
            y_pred = self(X)
        return y_pred


# ------------------------------------------------------------------------------
# 2. Configuração e Carregamento dos Dados
# ------------------------------------------------------------------------------

DATA_PATH = "prepared_data"
TRAIN_FILENAME = os.path.join(DATA_PATH, "final_train.csv")
TEST_FILENAME = os.path.join(DATA_PATH, "final_test.csv")

FILE_TAG: str = "final_data"
TARGET_COL: str = "Total"
MEASURE: str = "R2"

print(f"--- 🚀 LSTMs ({MEASURE}) para {TARGET_COL} ---")

try:
    data_df = pd.read_csv(TRAIN_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    test_df = pd.read_csv(TEST_FILENAME, index_col=0, sep=",", decimal=".", parse_dates=True)
    
    data_full = pd.concat([data_df, test_df])
    series_full = data_full[TARGET_COL].copy()
    
    train_size = len(data_df)
    
    series_np = series_full.values.astype(np.float32).reshape(-1, 1)
    train_np, test_np = series_np[:train_size], series_np[train_size:]

    print(f"Dados carregados: Treino ({len(train_np)} pontos), Teste ({len(test_np)} pontos).")
    
except Exception as e:
    print(f"ERRO ao carregar dados. Detalhes: {e}")
    exit()

# ------------------------------------------------------------------------------
# 3. Função de Estudo do LSTM (seq_length, hidden_units, nr_episodes)
# ------------------------------------------------------------------------------

def lstm_study_corrected(train, test, nr_episodes: int = 3000, measure: str = "R2"):
    
    # CORREÇÃO CRUCIAL: Removemos o 8, pois len(test) = 6. O máximo deve ser 5.
    sequence_size = [2, 4]
    nr_hidden_units = [25, 50, 100]

    step: int = nr_episodes // 10
    episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure == "R2" or measure == "MAPE"
    
    best_model = None
    best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
    is_maximizing = measure == "R2"
    best_performance: float = -100000 if is_maximizing else 100000
    
    all_r2_values = []

    fig, axs = plt.subplots(1, len(sequence_size), figsize=(len(sequence_size) * 4.5, 4.5))

    print("\nIniciando estudo de Hyperparâmetros (Sequence Length x Hidden Units)...")

    for i in range(len(sequence_size)):
        length = sequence_size[i]
        
        tstX, tstY_tensor = prepare_dataset_for_lstm(test, seq_length=length)
        # Se tstX estiver vazio (o que não deve acontecer mais com length=2 ou 4)
        if len(tstX) == 0:
            print(f"Aviso: Seq. Length={length} é muito grande para os dados de teste ({len(test)}). Ignorando.")
            continue
            
        tstY_numpy = tstY_tensor.squeeze().numpy()
        
        values = {}
        for hidden in nr_hidden_units:
            yvalues = []
            
            # Re-inicializa o modelo para cada combinação de hidden_size
            try:
                model = DS_LSTM(train, hidden_size=hidden, length=length)
            except ValueError as e:
                print(f"Erro ao criar modelo LSTM para Seq={length}: {e}")
                continue # Pula para o próximo hidden_unit

            for n in range(0, nr_episodes + 1):
                model.fit()
                if n % step == 0:
                    prd_tst_tensor = model.predict(tstX)
                    prd_tst_numpy = prd_tst_tensor.squeeze().numpy()
                    
                    eval_result: float = FORECAST_MEASURES[measure](tstY_numpy, prd_tst_numpy)
                    all_r2_values.append(eval_result)
                    
                    is_current_better = (is_maximizing and eval_result > best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE) or \
                                        (not is_maximizing and eval_result < best_performance and abs(eval_result - best_performance) > DELTA_IMPROVE)

                    if is_current_better:
                        best_performance = eval_result
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)
                    
                    yvalues.append(eval_result)
            values[f'H={hidden}'] = yvalues
            
        # 4. Plotagem do Subgráfico para o Sequence Length atual
        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM Seq. Length={length} ({measure})",
            xlabel="Number of Episodes",
            ylabel=measure,
            percentage=flag,
        )

    # 5. CÁLCULO E AJUSTE GLOBAL DO LIMITE Y (CORREÇÃO para R2 negativo)
    # Se a lista de resultados estiver vazia (todos falharam), pulamos o ajuste
    if not all_r2_values:
        print("\nAviso: Não foram encontrados resultados válidos de R2 para plotar.")
        y_min_limit = -1.0
    else:
        min_r2 = min(all_r2_values)
        y_min_limit = np.floor(min_r2) - 0.5
    
    # Ajusta o ylim de TODOS os subplots
    for ax in axs:
        ax.set_ylim(bottom=y_min_limit, top=1.0)
        
    print(f"\nAJUSTE Y-LIMIT: De [-1.0, 1.0] para [{y_min_limit:.2f}, 1.0].")
    print(
        f"LSTM best results achieved with length={best_params['params'][0]} hidden_units={best_params['params'][1]} and nr_episodes={best_params['params'][2]}) ==> measure={best_performance:.4f}"
    )

    # 6. Salvar o Gráfico de Estudo
    os.makedirs("images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_lstms_{measure}_study_CORRECTED.png")
    plt.close(fig)

    return best_model, best_params


# ------------------------------------------------------------------------------
# 4. Execução do Estudo e Geração dos Gráficos Finais
# ------------------------------------------------------------------------------

best_model_lstm, best_params_lstm = lstm_study_corrected(train_np, test_np, nr_episodes=3000, measure=MEASURE)
best_length, best_hidden, best_episodes = best_params_lstm["params"]

# 4.2 Preparação dos dados finais para o melhor modelo
trnX, trnY_tensor = prepare_dataset_for_lstm(train_np, seq_length=best_length)
tstX, tstY_tensor = prepare_dataset_for_lstm(test_np, seq_length=best_length)

# Previsão
prd_trn_tensor = best_model_lstm.predict(trnX)
prd_tst_tensor = best_model_lstm.predict(tstX)

# Converter resultados e targets para Series pandas para plotagem/avaliação
index_train_adj = series_full.index[best_length : train_size]
index_test_adj = series_full.index[train_size + best_length :]

prd_trn_series = pd.Series(prd_trn_tensor.squeeze().numpy(), index=index_train_adj)
prd_tst_series = pd.Series(prd_tst_tensor.squeeze().numpy(), index=index_test_adj)

train_s_adj = series_full.loc[index_train_adj]
test_s_adj = series_full.loc[index_test_adj]


# 4.3 Geração dos Gráficos
print("\nGerando gráficos de avaliação e previsão...")

# Gráfico de Avaliação (Performance)
plot_forecasting_eval(
    train_s_adj, test_s_adj, prd_trn_series, prd_tst_series,
    title=f"{FILE_TAG} - LSTM (length={best_length}, hidden={best_hidden}, epochs={best_episodes}) Performance"
)
plt.savefig(f"images/{FILE_TAG}_lstms_{MEASURE}_eval.png")
plt.close()

# Gráfico de Previsão da Série (Forecast)
plot_forecasting_series(
    train_s_adj, test_s_adj, prd_tst_series,
    title=f"{FILE_TAG} - LSTM (length={best_length}, hidden={best_hidden}, epochs={best_episodes}) Forecast",
    xlabel="Timestamp",
    ylabel=TARGET_COL,
)
plt.savefig(f"images/{FILE_TAG}_lstms_{MEASURE}_forecast.png")
plt.close()

# 4.4 Cálculo e Impressão das Métricas Finais
print("\n--- 📊 Performance do Melhor Modelo (Conjunto de Teste) ---")

metrics = ["MSE", "MAE", "R2"]
performance_results = {}

for metric in metrics:
    score = FORECAST_MEASURES[metric](test_s_adj, prd_tst_series)
    performance_results[metric] = score
    print(f"{metric}: {score:.6f}")

print("\nRelatório Resumo Final:")
print(f"Técnica: LSTMs")
print(f"Melhor Hyperparâmetro: Length={best_length}, Hidden={best_hidden}, Epochs={best_episodes}")
print(f"Performance (Teste): R2={performance_results['R2']:.6f} | MSE={performance_results['MSE']:.6f} | MAE={performance_results['MAE']:.6f}")
print("\n4 modelos completos. O gráfico de estudo LSTM corrigido é '...study_CORRECTED.png'.")
