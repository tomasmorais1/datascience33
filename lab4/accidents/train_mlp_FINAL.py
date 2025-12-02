import os
import pandas as pd
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from dslabs_functions import plot_multiline_chart, plot_bar_chart
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import array, arange
from dslabs_functions import plot_evaluation_results # Manter se necessário para o gráfico de performance

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# Variável para controlar o incremento mínimo de melhoria para early stop (simulado)
DELTA_IMPROVE = 0.0005
# ------------------------------------

# --- Hyperparameter Search Space ---
NR_MAX_ITER = 2000
LAG = 500
learning_rates = [0.05, 0.005] # Ajustado para valores mais típicos para MLP
lr_types = ["constant", "adaptive"]
eval_metric = "accuracy"

os.makedirs("images", exist_ok=True)

# Load data (Dados já separados e escalados)
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Certifique-se de que 'train_scaled.csv' e 'test_scaled.csv' estão no diretório correto.")
    exit()

# Separar X e y
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())

# --- Hyperparameters study ---
best_model = None
best_params = {"name": "MLP", "metric": eval_metric, "params": ()}
best_performance = 0.0
# Iterações a testar, incluindo o primeiro LAG
nr_iterations = [LAG] + list(range(2 * LAG, NR_MAX_ITER + 1, LAG))
hidden_layers = (50,) # Camada oculta simples (pode ser tunado mais tarde)
solver_choice = "adam" # Adam é mais robusto que SGD para a maioria dos problemas
activation_choice = "relu" # ReLU é mais comum que logistic

for lr_type in lr_types:
    values_acc = {}
    print(f"\n--- Estudo de Hiperparâmetros MLP (LR Type: {lr_type}) ---")

    for lr in learning_rates:
        y_tst_values = []
        
        # Inicialização do modelo fora do loop de iteração para usar warm_start
        # Max_iter=LAG e warm_start=True permitem treinar em blocos de LAG iterações
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=LAG,
            warm_start=True, # NOVO: Essencial para treinar incrementalmente
            activation=activation_choice,
            solver=solver_choice,
            random_state=42,
        )

        current_n_iter = 0
        
        for n_iter in nr_iterations:
            # Treinar pelo número de iterações do LAG
            clf.fit(trnX, trnY)
            current_n_iter += LAG # Acumula o número de iterações reais

            prdY = clf.predict(tstX)
            eval_score = accuracy_score(tstY, prdY) # Avaliação pela Accuracy
            y_tst_values.append(eval_score)
            
            # Atualizar melhor modelo
            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                # Guarda o número total de iterações que o modelo levou até este ponto
                best_params["params"] = (lr_type, lr, current_n_iter, hidden_layers)
                # Cria uma cópia do classificador para o best_model
                best_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers, learning_rate=lr_type, learning_rate_init=lr,
                    max_iter=current_n_iter, activation=activation_choice, solver=solver_choice, random_state=42
                ).fit(trnX, trnY)

            print(f"  LR={lr}, Iterations={current_n_iter} -> Accuracy: {eval_score:.4f}")

        values_acc[f'LR={lr}'] = y_tst_values
        
    # Plotagem dos resultados da Accuracy para o tipo de LR
    figure(figsize=(10, 6))
    plot_multiline_chart(
        nr_iterations,
        values_acc,
        title=f"MLP Hyperparameters (Accuracy vs Iterations) - {lr_type}",
        xlabel="Número Total de Iterações",
        ylabel=eval_metric.capitalize(),
        percentage=True,
    )
    plt.tight_layout()
    savefig(f"images/mlp_{lr_type}_hyperparameters.png")
    close()


# --- Best model performance e Overfitting Study ---

# Re-avaliar o melhor modelo (para garantir que usamos a versão final)
# O best_model foi copiado e treinado com o número ideal de iterações
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

# Métricas
metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Descrição do Melhor Modelo MLP ---")
print(f"Hiperparâmetros encontrados: LR Type={best_params['params'][0]}, LR={best_params['params'][1]}, Iters={best_params['params'][2]}")
print(f"Estrutura: {best_params['params'][3]}, Solver: {solver_choice}, Activation: {activation_choice}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    trn_val = func(trnY, prd_trn)
    tst_val = func(tstY, prd_tst)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# Overfitting Study (Gráfico sugerido: Estudo de Overfitting)
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, prd_trn),
        accuracy_score(tstY, prd_tst)
    ]
}

figure(figsize=(6, 4))
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'],
               title=f"Overfitting Study - MLP (Iters={best_params['params'][2]})", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig("images/mlp_overfitting.png")
close()

print("\nGráficos do MLP concluídos e guardados na pasta 'images'.")
