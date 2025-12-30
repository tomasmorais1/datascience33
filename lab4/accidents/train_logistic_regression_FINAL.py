import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_horizontal_bar_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
EVAL_METRIC = "auc" # Métrica pedida para Dataset 1
DELTA_IMPROVE = 0.0005

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

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
vars_names = trnX.columns.tolist()
os.makedirs("images", exist_ok=True)

print(f"Dados (Accidents): Train={len(trnX)}, Test={len(tstX)}")
print(f"Labels: {labels}")

# --- 1. Hyperparameter Study (Penalty & Iterations) ---
# Iterações aumentadas para garantir convergência
iter_range = [100, 300, 500, 1000, 1500, 2000]
penalties = ["l2", "l1"]

best_model = None
best_score = 0
best_params = {"name": "LR", "metric": EVAL_METRIC, "params": ()}

values_auc = {}

print(f"\n--- Estudo de Hiperparâmetros (LR - {EVAL_METRIC.upper()}) ---")

for pen in penalties:
    y_tst_values = []
    print(f"Testing Penalty={pen}...")
    
    for n in iter_range:
        # Solver liblinear suporta l1 e l2
        clf = LogisticRegression(penalty=pen, max_iter=n, solver="liblinear", random_state=42)
        clf.fit(trnX, trnY)
        
        # Calcular AUC (precisa de probabilidades)
        try:
            y_probs = clf.predict_proba(tstX)
            if len(labels) > 2:
                # Multiclasse OvR
                score = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
            else:
                score = roc_auc_score(tstY, y_probs[:, 1])
        except:
            score = 0.5
            
        y_tst_values.append(score)
        
        if score - best_score > DELTA_IMPROVE:
            best_score = score
            best_model = clf
            best_params["params"] = (pen, n)
            best_params["auc"] = score

    values_auc[pen] = y_tst_values

# Plot do Estudo
plt.figure(figsize=(10, 6))
plot_multiline_chart(iter_range, values_auc,
                     title=f"LR Hyperparameters ({EVAL_METRIC})",
                     xlabel="Iterations", ylabel=EVAL_METRIC, percentage=True)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_lr_{EVAL_METRIC}_study.png")
plt.close()

print(f"Melhor LR: Penalty={best_params['params'][0]}, Iter={best_params['params'][1]} ({EVAL_METRIC.upper()}={best_score:.4f})")


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
print("\n--- Gerando Gráfico de Overfitting ---")
# Usamos o melhor penalty e variamos as iterações
best_pen = best_params['params'][0]
eval_iters = [50, 100, 300, 500, 1000, 1500, 2000, 2500]

auc_train = []
auc_test = []

for n in eval_iters:
    clf = LogisticRegression(penalty=best_pen, max_iter=n, solver="liblinear", random_state=42)
    clf.fit(trnX, trnY)
    
    # AUC Train
    y_probs_trn = clf.predict_proba(trnX)
    if len(labels) > 2:
        s_trn = roc_auc_score(trnY, y_probs_trn, multi_class='ovr', average='weighted')
    else:
        s_trn = roc_auc_score(trnY, y_probs_trn[:, 1])
    
    # AUC Test
    y_probs_tst = clf.predict_proba(tstX)
    if len(labels) > 2:
        s_tst = roc_auc_score(tstY, y_probs_tst, multi_class='ovr', average='weighted')
    else:
        s_tst = roc_auc_score(tstY, y_probs_tst[:, 1])
        
    auc_train.append(s_trn)
    auc_test.append(s_tst)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    eval_iters,
    {"Train": auc_train, "Test": auc_test},
    title=f"LR Overfitting Study (Penalty={best_pen})",
    xlabel="Iterations",
    ylabel="AUC",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_lr_overfitting.png")
plt.close()


# --- 4. Feature Importance ---
if hasattr(best_model, 'coef_'):
    print("\n--- Feature Importance ---")
    # Para multiclasse, coef_ é (n_classes, n_features).
    # Vamos pegar no valor absoluto máximo de cada feature entre todas as classes.
    # Isto diz-nos "quão importante é esta variável para distinguir PELO MENOS UMA classe".
    importance = np.max(np.abs(best_model.coef_), axis=0)
    
    # Ordenar
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
            title=f"LR Variable Importance (Max Abs Coef)", 
            xlabel="Coefficient Magnitude", ylabel="Variables", percentage=False
        )
    except:
        plt.barh(elems[::-1], imp_values[::-1], color='skyblue')
        plt.title("LR Variable Importance")
        plt.xlabel("Abs Coefficient")
    
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_lr_vars_ranking.png")
    plt.close()
    
    print("Top 3 Variáveis:", elems[:3])

print("\nProcesso concluído (LR Accidents).")