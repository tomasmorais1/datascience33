import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dslabs_functions import plot_multiline_chart, plot_line_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "f1" # Dataset 2 -> F1 Score
DELTA_IMPROVE = 0.0005

# --- PARÂMETROS MLP ---
NR_MAX_ITER = 3000
LAG = 500
LEARNING_RATES = [0.1, 0.01, 0.001]
LR_TYPES = ["constant", "invscaling", "adaptive"]
SAMPLE_SIZE = 7000 

# --------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

# --- REMOVER DATA LEAKAGE ---
cols_to_drop = [
    'ArrDelay', 'DepDelay', 'ActualElapsedTime', 'AirTime', 'ArrTime', 'DepTime', 
    'WheelsOff', 'WheelsOn', 'TaxiIn', 'TaxiOut', 'Diverted', 
    'ArrivalDelayGroups', 'DepartureDelayGroups', 'ArrDel15', 'DepDel15',
    'ArrDelayMinutes', 'DepDelayMinutes'
]
cols_in_train = [c for c in cols_to_drop if c in trn_df.columns]
if len(cols_in_train) > 0:
    print(f"--- A remover Data Leakage ({len(cols_in_train)} colunas)... ---")
    trn_df = trn_df.drop(columns=cols_in_train)
    tst_df = tst_df.drop(columns=[c for c in cols_to_drop if c in tst_df.columns])


# --- SAMPLING ---
if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"--- Dataset grande. A usar sample de {SAMPLE_SIZE} linhas para MLP... ---")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
os.makedirs("images", exist_ok=True)
print(f"Dados (Flights): Train={len(trnX)}, Test={len(tstX)}")


# --- 1. Hyperparameter Study (LR Types vs Rates) - F1 ---
nr_iterations = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

best_model = None
best_params = {"name": "MLP", "metric": EVAL_METRIC, "params": ()}
best_performance = 0.0

# Subplots: Um gráfico por LR Type
cols = len(LR_TYPES)
_, axs = plt.subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)

print(f"\n--- Estudo de Hiperparâmetros (MLP - {EVAL_METRIC}) ---")

for i, lr_type in enumerate(LR_TYPES):
    values = {}
    print(f"   > Testing LR Type: {lr_type}...")

    for lr in LEARNING_RATES:
        y_tst_values = []
        
        clf = MLPClassifier(
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=LAG,
            warm_start=True,
            activation="logistic", 
            solver="sgd",          
            verbose=False,
            random_state=42
        )
        
        for n in range(len(nr_iterations)):
            clf.fit(trnX, trnY)
            
            # Calcular F1 Score (Weighted)
            score = f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0)
            y_tst_values.append(score)

            if score - best_performance > DELTA_IMPROVE:
                best_performance = score
                best_params["params"] = (lr_type, lr, nr_iterations[n])
                best_params["f1"] = score
                best_model = clf
        
        values[f'LR={lr}'] = y_tst_values

    plot_multiline_chart(
        nr_iterations,
        values,
        ax=axs[0, i],
        title=f"MLP ({lr_type})",
        xlabel="Iterations",
        ylabel="F1-Score",
        percentage=True,
    )

plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_{EVAL_METRIC}_study.png")
plt.close()

print(f"Melhor MLP: Type={best_params['params'][0]}, LR={best_params['params'][1]}, Iter={best_params['params'][2]} (F1={best_performance:.4f})")


# --- 2. Best Model Results (Confusion Matrix) ---
print("\n--- Gerando Matrizes de Confusão ---")
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_best_model_eval.png")
plt.close()


# --- 3. Overfitting Study (F1 Evolution) ---
print("\n--- Gerando Gráfico de Overfitting (F1) ---")
best_lr_type = best_params['params'][0]
best_lr = best_params['params'][1]

clf_overfit = MLPClassifier(
    learning_rate=best_lr_type,
    learning_rate_init=best_lr,
    max_iter=LAG,
    warm_start=True,
    activation="logistic",
    solver="sgd",
    verbose=False,
    random_state=42
)

f1_train = []
f1_test = []
steps = []

for i in range(len(nr_iterations)):
    total_iter = nr_iterations[i]
    clf_overfit.set_params(max_iter=total_iter)
    clf_overfit.fit(trnX, trnY)
    
    # F1 Train
    s_trn = f1_score(trnY, clf_overfit.predict(trnX), average="weighted", zero_division=0)
    # F1 Test
    s_tst = f1_score(tstY, clf_overfit.predict(tstX), average="weighted", zero_division=0)
        
    f1_train.append(s_trn)
    f1_test.append(s_tst)
    steps.append(total_iter)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    steps,
    {"Train": f1_train, "Test": f1_test},
    title=f"MLP Overfitting (Type={best_lr_type}, LR={best_lr})",
    xlabel="Iterations",
    ylabel="F1-Score",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_overfitting.png")
plt.close()


# --- 4. Loss Curve Analysis ---
print("\n--- Gerando Loss Curve ---")
loss_values = best_model.loss_curve_

plt.figure(figsize=(10, 6))
plot_line_chart(
    np.arange(len(loss_values)),
    loss_values,
    title="Loss Curve (Best Model)",
    xlabel="Iterations",
    ylabel="Loss",
    percentage=False
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_loss_curve.png")
plt.close()

print("\nProcesso concluído (MLP Flights).")