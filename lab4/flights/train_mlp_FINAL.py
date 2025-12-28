import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dslabs_functions import plot_multiline_chart, plot_bar_chart

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
DELTA_IMPROVE = 0.0005

# --- AMOSTRAGEM (NOVO) ---
SAMPLE_SIZE = 20000  # Vamos usar apenas 20k linhas para treino
# -------------------------

# --- Hyperparameter Search Space ---
NR_MAX_ITER = 2000
LAG = 500
learning_rates = [0.05, 0.005] 
lr_types = ["constant", "adaptive"]
eval_metric = "accuracy"

os.makedirs("images", exist_ok=True)

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: train_scaled.csv ou test_scaled.csv não encontrados.")
    exit()

# --- BLOCO CRÍTICO: REMOVER DATA LEAKAGE ---
cols_to_drop = [
    'ArrDelay', 'DepDelay', 'ActualElapsedTime', 
    'AirTime', 'ArrTime', 'DepTime', 
    'WheelsOff', 'WheelsOn', 'TaxiIn', 'TaxiOut',
    'Diverted', 
    'ArrivalDelayGroups', 'DepartureDelayGroups',
    'ArrDel15', 'DepDel15',
    'ArrDelayMinutes', 'DepDelayMinutes'
]

cols_in_train = [c for c in cols_to_drop if c in trn_df.columns]
cols_in_test = [c for c in cols_to_drop if c in tst_df.columns]

if len(cols_in_train) > 0:
    print(f"--- A remover Data Leakage ({len(cols_in_train)} colunas)... ---")
    trn_df = trn_df.drop(columns=cols_in_train)
    tst_df = tst_df.drop(columns=cols_in_test)

# --- APLICAR SAMPLING (SOLUÇÃO PARA A LENTIDÃO) ---
if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"--- Dataset grande ({trn_df.shape[0]}). A reduzir para {SAMPLE_SIZE} linhas para MLP... ---")
    trn_df = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
# --------------------------------------------------

# --- Separar X e y ---
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

print(f"Dados prontos para MLP: Train={len(trnX)}, Test={len(tstX)}")

# --- Hyperparameters study ---
best_model = None
best_params = {"name": "MLP", "metric": eval_metric, "params": ()}
best_performance = 0.0

# Iterações a testar
nr_iterations = [LAG] + list(range(2 * LAG, NR_MAX_ITER + 1, LAG))
hidden_layers = (50,) # 50 neurónios numa camada oculta
solver_choice = "adam" 
activation_choice = "relu"

for lr_type in lr_types:
    values_acc = {}
    print(f"\n--- Estudo de Hiperparâmetros MLP (LR Type: {lr_type}) ---")

    for lr in learning_rates:
        y_tst_values = []
        
        # Inicialização do modelo com warm_start=True
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=LAG,
            warm_start=True, 
            activation=activation_choice,
            solver=solver_choice,
            random_state=42,
        )

        current_n_iter = 0
        
        for n_iter in nr_iterations:
            # Treinar pelo número de iterações do LAG
            clf.fit(trnX, trnY)
            current_n_iter += LAG 

            prdY = clf.predict(tstX)
            eval_score = accuracy_score(tstY, prdY)
            y_tst_values.append(eval_score)
            
            # Atualizar melhor modelo
            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (lr_type, lr, current_n_iter, hidden_layers)
                # Copiar e treinar um modelo novo limpo
                best_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layers, learning_rate=lr_type, learning_rate_init=lr,
                    max_iter=current_n_iter, activation=activation_choice, solver=solver_choice, random_state=42
                ).fit(trnX, trnY)

            print(f"   LR={lr}, Total Iterations={current_n_iter} -> Accuracy: {eval_score:.4f}")

        values_acc[f'LR={lr}'] = y_tst_values
        
    # Plotagem
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
    savefig(f"images/{FILE_TAG}_mlp_{lr_type}_study.png")
    close()

# --- Best model performance e Overfitting Study ---

if best_model is None:
    print("Não foi possível treinar nenhum modelo com sucesso.")
    exit()

# Recalcular previsões finais
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Descrição do Melhor Modelo MLP ---")
print(f"Hiperparâmetros: LR Type={best_params['params'][0]}, LR={best_params['params'][1]}, Iters={best_params['params'][2]}")
print(f"Estrutura: {best_params['params'][3]}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    print(f"{metric} - Train: {func(trnY, prd_trn):.4f}, Test: {func(tstY, prd_tst):.4f}")

# Overfitting Study
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, prd_trn),
        accuracy_score(tstY, prd_tst)
    ]
}

figure(figsize=(6, 4))
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'],
               title=f"Overfitting Study - MLP (Iters={best_params['params'][2]})", 
               ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_mlp_overfitting.png")
close()

print("\nGráficos do MLP concluídos e guardados na pasta 'images'.")