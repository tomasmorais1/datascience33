import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dslabs_functions import plot_multiline_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "f1" # Dataset 2 exige F1 Score
DELTA_IMPROVE = 0.0005

# --- AMOSTRAGEM ---
# KNN é extremamente lento com +50k linhas. 
# 10.000 é um bom compromisso para o estudo demorar menos de 5 min.
SAMPLE_SIZE = 5000 
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
print(f"Dados (Flights): Train (Sample)={len(trnX)}, Test={len(tstX)}")


# --- 1. Hyperparameter Study (Distance & K) - F1 ---
k_max = 25
lag = 2
kvalues = [i for i in range(1, k_max + 1, lag)]
distances = ['manhattan', 'euclidean', 'chebyshev']

best_model = None
best_params = {'name': 'KNN', 'metric': EVAL_METRIC, 'params': ()}
best_performance = 0.0

values = {}

print(f"\n--- Estudo de Hiperparâmetros (KNN - F1) ---")

for d in distances:
    y_tst_values = []
    print(f"Testing Distance={d}...")
    for k in kvalues:
        # n_jobs=-1 usa todos os cores do CPU para acelerar
        clf = KNeighborsClassifier(n_neighbors=k, metric=d, n_jobs=-1)
        clf.fit(trnX, trnY)
        
        # Calcular F1 Score (Weighted)
        y_pred = clf.predict(tstX)
        score = f1_score(tstY, y_pred, average="weighted", zero_division=0)
            
        y_tst_values.append(score)
        
        if score - best_performance > DELTA_IMPROVE:
            best_performance = score
            best_params['params'] = (k, d)
            best_params['f1'] = score
            best_model = clf
            
    values[d] = y_tst_values

plt.figure(figsize=(10, 6))
plot_multiline_chart(kvalues, values, title=f'KNN Models ({EVAL_METRIC})', xlabel='k', ylabel='F1-Score', percentage=True)
plt.tight_layout()
plt.savefig(f'images/{FILE_TAG}_knn_{EVAL_METRIC}_study.png')
plt.close()

print(f"Melhor KNN: k={best_params['params'][0]}, Distance={best_params['params'][1]} (F1={best_performance:.4f})")


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
print("\n--- Gerando Gráfico de Overfitting (F1) ---")
best_dist = best_params['params'][1]
kvalues_overfit = [i for i in range(1, k_max + 1, 2)]

f1_train = []
f1_test = []

for k in kvalues_overfit:
    clf = KNeighborsClassifier(n_neighbors=k, metric=best_dist, n_jobs=-1)
    clf.fit(trnX, trnY)
    
    # F1 Train
    s_trn = f1_score(trnY, clf.predict(trnX), average="weighted", zero_division=0)
    # F1 Test
    s_tst = f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0)
        
    f1_train.append(s_trn)
    f1_test.append(s_tst)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    kvalues_overfit,
    {"Train": f1_train, "Test": f1_test},
    title=f"KNN Overfitting (Distance={best_dist})",
    xlabel="K",
    ylabel="F1-Score",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_knn_overfitting.png")
plt.close()

print("\nProcesso concluído (KNN Flights).")