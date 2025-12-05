import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from dslabs_functions import plot_multiline_chart, plot_bar_chart

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# ------------------------------------

# Hiperparâmetros para estudo
K_MAX = 9
DISTANCES = ["manhattan", "euclidean", "chebyshev"]

os.makedirs("images", exist_ok=True)

# Load data
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Certifique-se de que 'train_scaled.csv' e 'test_scaled.csv' estão no diretório correto.")
    exit()

# --- OTIMIZAÇÃO CRÍTICA: SAMPLING ---
# O KNN é demasiado lento para o dataset completo (>50k linhas).
# Vamos usar uma amostra representativa para o Tuning.
SAMPLE_SIZE = 5000 

if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"[Aviso] Dataset grande ({trn_df.shape[0]} linhas).")
    print(f"-> A usar amostragem aleatória de {SAMPLE_SIZE} linhas para permitir a execução.")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

# --- Hyperparameters study (Tuning de K e Distância) ---
acc_values = {}
prec_values = {}
rec_values = {}

best_model, best_score, best_params = None, 0, None

k_values = list(range(1, K_MAX + 1))

print("--- Estudo de Hiperparâmetros (KNN - K-Nearest Neighbors) ---")

for dist in DISTANCES:
    acc_list = []
    prec_list = []
    rec_list = []

    print(f"A testar distância: {dist}...") 

    for k in k_values:
        # n_jobs=-1 usa todos os processadores do computador
        clf = KNeighborsClassifier(n_neighbors=k, metric=dist, n_jobs=-1)
        clf.fit(trnX, trnY)
        
        # Previsão APENAS no Teste dentro do loop
        y_tst_pred = clf.predict(tstX)

        acc = accuracy_score(tstY, y_tst_pred)
        prec = precision_score(tstY, y_tst_pred, average="weighted", zero_division=0)
        rec = recall_score(tstY, y_tst_pred, average="weighted", zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        
        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (k, dist)

        print(f"   k={k} -> Accuracy Teste: {acc:.4f}")

    acc_values[dist] = acc_list
    prec_values[dist] = prec_list
    rec_values[dist] = rec_list

# --- Plotagem dos Resultados ---

# Plot Accuracy
figure(figsize=(10, 6))
plot_multiline_chart(k_values, acc_values,
                     title="KNN Accuracy vs k",
                     xlabel="k (Número de Vizinhos)", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig("images/knn_accuracy.png")
close()

# Plot Precision
figure(figsize=(10, 6))
plot_multiline_chart(k_values, prec_values,
                     title="KNN Precision vs k",
                     xlabel="k (Número de Vizinhos)", ylabel="Precision", percentage=True)
plt.tight_layout()
savefig("images/knn_precision.png")
close()

# Plot Recall
figure(figsize=(10, 6))
plot_multiline_chart(k_values, rec_values,
                     title="KNN Recall vs k",
                     xlabel="k (Número de Vizinhos)", ylabel="Recall", percentage=True)
plt.tight_layout()
savefig("images/knn_recall.png")
close()

# --- Best model performance e Overfitting Study ---

print(f"\n--- Descrição do Melhor Modelo KNN ---")
print(f"Hiperparâmetros encontrados: k={best_params[0]}, metric={best_params[1]}")

# Recalcular previsões finais
# Nota: Usamos o trnX (sample) para treinar e prever, senão o predict de treino bloqueia novamente
y_trn_pred_best = best_model.predict(trnX)
y_tst_pred_best = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    if metric == "Accuracy":
        trn_val = func(trnY, y_trn_pred_best)
        tst_val = func(tstY, y_tst_pred_best)
    else:
        trn_val = func(trnY, y_trn_pred_best, average="weighted", zero_division=0)
        tst_val = func(tstY, y_tst_pred_best, average="weighted", zero_division=0)

    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# Overfitting Study
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred_best),
        accuracy_score(tstY, y_tst_pred_best)
    ]
}

figure(figsize=(6, 4))
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'],
               title=f"Overfitting Study - KNN (k={best_params[0]})", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig("images/knn_overfitting.png")
close()

print("\nGráficos do KNN guardados na pasta 'images'.")