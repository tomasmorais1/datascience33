import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dslabs_functions import plot_multiline_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
EVAL_METRIC = "auc" # Métrica para Dataset 1
DELTA_IMPROVE = 0.0005

# --- AMOSTRAGEM PARA KNN (MUITO IMPORTANTE) ---
# KNN é lento com datasets grandes. Usamos amostragem.
SAMPLE_SIZE = 10000 
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

# --- SAMPLING ---
if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"--- Dataset grande. A usar sample de {SAMPLE_SIZE} linhas para KNN... ---")
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
print(f"Dados (Accidents): Train (Sample)={len(trnX)}, Test={len(tstX)}")


# --- 1. Hyperparameter Study (Distance & K) - AUC ---
k_max = 25
lag = 2
kvalues = [i for i in range(1, k_max + 1, lag)]
distances = ['manhattan', 'euclidean', 'chebyshev']

best_model = None
best_params = {'name': 'KNN', 'metric': EVAL_METRIC, 'params': ()}
best_performance = 0.0

values = {}

print(f"\n--- Estudo de Hiperparâmetros (KNN - {EVAL_METRIC}) ---")

for d in distances:
    y_tst_values = []
    print(f"Testing Distance={d}...")
    for k in kvalues:
        clf = KNeighborsClassifier(n_neighbors=k, metric=d)
        clf.fit(trnX, trnY)
        
        # Calcular AUC
        try:
            y_probs = clf.predict_proba(tstX)
            if len(labels) > 2:
                score = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
            else:
                score = roc_auc_score(tstY, y_probs[:, 1])
        except:
            score = 0.5
            
        y_tst_values.append(score)
        
        if score - best_performance > DELTA_IMPROVE:
            best_performance = score
            best_params['params'] = (k, d)
            best_params['auc'] = score
            best_model = clf
            
    values[d] = y_tst_values

plt.figure(figsize=(10, 6))
plot_multiline_chart(kvalues, values, title=f'KNN Models ({EVAL_METRIC})', xlabel='k', ylabel=EVAL_METRIC, percentage=True)
plt.tight_layout()
plt.savefig(f'images/{FILE_TAG}_knn_{EVAL_METRIC}_study.png')
plt.close()

print(f"Melhor KNN: k={best_params['params'][0]}, Distance={best_params['params'][1]} ({EVAL_METRIC.upper()}={best_performance:.4f})")


# --- 2. Best Model Results (Confusion Matrix) ---
print("\n--- Gerando Matrizes de Confusão ---")
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_knn_best_model_eval.png")
plt.close()


# --- 3. Overfitting Study (Train vs Test Evolution) ---
print("\n--- Gerando Gráfico de Overfitting (AUC) ---")
best_dist = best_params['params'][1]
kvalues_overfit = [i for i in range(1, k_max + 1, 2)]

auc_train = []
auc_test = []

for k in kvalues_overfit:
    clf = KNeighborsClassifier(n_neighbors=k, metric=best_dist)
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
    kvalues_overfit,
    {"Train": auc_train, "Test": auc_test},
    title=f"KNN Overfitting (Distance={best_dist})",
    xlabel="K",
    ylabel="AUC",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_knn_overfitting.png")
plt.close()

print("\nProcesso concluído (KNN Accidents).")