import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from dslabs_functions import plot_bar_chart

# --- VARIÁVEIS A AJUSTAR ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# ----------------------------

os.makedirs("images", exist_ok=True)

# Load data
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Certifique-se de que 'train_scaled.csv' e 'test_scaled.csv' estão no diretório correto.")
    exit()

# Separar X e y (SEM Data Cleaning extra)
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())

# --- Hyperparameters study (Apenas GaussianNB) ---
estimators = {
    "GaussianNB": GaussianNB(),
}

xvalues = []
acc_values = []
prec_values = []
rec_values = []

best_model, best_score, best_name = None, 0, ""

print("--- Estudo de Hiperparâmetros (Naïve Bayes) ---")
for name, clf in estimators.items():
    clf.fit(trnX, trnY)
    y_pred = clf.predict(tstX)

    # Cálculo das métricas
    acc = accuracy_score(tstY, y_pred)
    prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
    rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

    xvalues.append(name)
    acc_values.append(acc)
    prec_values.append(prec)
    rec_values.append(rec)

    if acc > best_score:
        best_model = clf
        best_score = acc
        best_name = name

    print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# --- Plotagem dos Resultados ---

# Plot Accuracy
figure(figsize=(8, 5))
plot_bar_chart(xvalues, acc_values, title="Naive Bayes - Comparação de Accuracy", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig("images/nb_accuracy.png")
close()

# Plot Precision
figure(figsize=(8, 5))
plot_bar_chart(xvalues, prec_values, title="Naive Bayes - Comparação de Precision", ylabel="Precision", percentage=True)
plt.tight_layout()
savefig("images/nb_precision.png")
close()

# Plot Recall
figure(figsize=(8, 5))
plot_bar_chart(xvalues, rec_values, title="Naive Bayes - Comparação de Recall", ylabel="Recall", percentage=True)
plt.tight_layout()
savefig("images/nb_recall.png")
close()

# --- Best model performance ---
print(f"\n--- Desempenho do Melhor Modelo: {best_name} ---")

y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score
}

for metric, func in metrics.items():
    if metric == "Accuracy":
        trn_val = func(trnY, y_trn_pred)
        tst_val = func(tstY, y_tst_pred)
    else:
        trn_val = func(trnY, y_trn_pred, average="weighted", zero_division=0)
        tst_val = func(tstY, y_tst_pred, average="weighted", zero_division=0)

    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# --- Overfitting Study ---
data = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 4))
plot_bar_chart(data['Dataset'], data['Accuracy'], title=f"Overfitting Study - {best_name}", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig("images/nb_overfitting.png")
close()

# --- CONFUSION MATRIX (Adicionado) ---
print(f"\n--- A gerar Matriz de Confusão ({best_name}) ---")

# Gera a matriz
cm = confusion_matrix(tstY, y_tst_pred, labels=best_model.classes_)

# Prepara a visualização
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

# Desenha e guarda
figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
savefig("images/nb_confusion_matrix.png")
close()

print(f"\nMelhor Modelo Naïve Bayes: {best_name}")
print("\nGráficos (incluindo Confusion Matrix) guardados na pasta 'images'.")