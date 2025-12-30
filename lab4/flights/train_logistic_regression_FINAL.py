import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_horizontal_bar_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "f1" # Dataset 2 -> F1 Score
DELTA_IMPROVE = 0.0005
SAMPLE_SIZE = 20000 
# --------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
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
    print(f"--- Dataset grande. A usar sample de {SAMPLE_SIZE} linhas para LR... ---")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
vars_names = trnX.columns.tolist()
os.makedirs("images", exist_ok=True)

print(f"Dados (Flights): Train={len(trnX)}, Test={len(tstX)}")


# --- 1. Hyperparameter Study (Penalty & Iterations - F1 Score) ---
iter_range = [100, 300, 500, 1000, 1500, 2000]
penalties = ["l2", "l1"]

best_model = None
best_score = 0
best_params = {"name": "LR", "metric": EVAL_METRIC, "params": ()}

values_f1 = {}

print(f"\n--- Estudo de Hiperparâmetros (LR - F1) ---")

for pen in penalties:
    y_tst_values = []
    print(f"Testing Penalty={pen}...")
    
    for n in iter_range:
        clf = LogisticRegression(penalty=pen, max_iter=n, solver="liblinear", random_state=42)
        clf.fit(trnX, trnY)
        
        # Calcular F1 Score (Weighted)
        f1 = f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0)
            
        y_tst_values.append(f1)
        
        if f1 - best_score > DELTA_IMPROVE:
            best_score = f1
            best_model = clf
            best_params["params"] = (pen, n)
            best_params["f1"] = f1

    values_f1[pen] = y_tst_values

# Plot do Estudo
plt.figure(figsize=(10, 6))
plot_multiline_chart(iter_range, values_f1,
                     title=f"LR Hyperparameters (F1-Score)",
                     xlabel="Iterations", ylabel="F1-Score", percentage=True)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_lr_{EVAL_METRIC}_study.png")
plt.close()

print(f"Melhor LR: Penalty={best_params['params'][0]}, Iter={best_params['params'][1]} (F1={best_score:.4f})")


# --- 2. Best Model Results (Confusion Matrix) ---
print("\n--- Gerando Matrizes de Confusão ---")
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_lr_best_model_eval.png")
plt.close()


# --- 3. Overfitting Study (Evolution over Iterations) ---
print("\n--- Gerando Gráfico de Overfitting (F1) ---")
best_pen = best_params['params'][0]
eval_iters = [50, 100, 300, 500, 1000, 1500, 2000, 2500]

f1_train = []
f1_test = []

for n in eval_iters:
    clf = LogisticRegression(penalty=best_pen, max_iter=n, solver="liblinear", random_state=42)
    clf.fit(trnX, trnY)
    
    # F1 Train
    s_trn = f1_score(trnY, clf.predict(trnX), average="weighted", zero_division=0)
    # F1 Test
    s_tst = f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0)
        
    f1_train.append(s_trn)
    f1_test.append(s_tst)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    eval_iters,
    {"Train": f1_train, "Test": f1_test},
    title=f"LR Overfitting Study (Penalty={best_pen})",
    xlabel="Iterations",
    ylabel="F1-Score",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_lr_overfitting.png")
plt.close()


# --- 4. Feature Importance ---
if hasattr(best_model, 'coef_'):
    print("\n--- Feature Importance ---")
    # Para binário, coef_ é (1, n_features)
    importance = np.abs(best_model.coef_[0])
    
    indices = np.argsort(importance)[::-1]
    top_n = 10
    
    elems = []
    imp_values = []
    
    for i in range(top_n):
        idx = indices[i]
        elems.append(vars_names[idx])
        imp_values.append(importance[idx])

    plt.figure(figsize=(10, 8))
    # Usar plot_horizontal_bar_chart se disponível
    try:
        plot_horizontal_bar_chart(
            elems, imp_values, 
            title=f"LR Variable Importance (Abs Coef)", 
            xlabel="Abs Coefficient", ylabel="Variables", percentage=False
        )
    except:
        plt.barh(elems[::-1], imp_values[::-1], color='skyblue')
        plt.title("LR Variable Importance")
        plt.xlabel("Abs Coefficient")
    
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_lr_vars_ranking.png")
    plt.close()
    
    print("Top 3 Variáveis:", elems[:3])

print("\nProcesso concluído (LR Flights).")