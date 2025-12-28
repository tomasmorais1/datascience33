import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart

# --- CONFIGURAÇÃO DE FONTE GLOBAL ---
# Aumenta títulos e eixos
plt.rcParams.update({'font.size': 14}) 
# ------------------------------------

# --- CONFIGURAÇÃO ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"

os.makedirs("images", exist_ok=True)

# --- 1. CARREGAR DADOS ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

# --- 2. BLOCO CRÍTICO: REMOVER DATA LEAKAGE ---
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

# --- 3. SEPARAR X e Y ---
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
print(f"Dados prontos para Naive Bayes: Train={len(trnX)}, Test={len(tstX)}")

# --- 4. ESTUDO DE HIPERPARÂMETROS ---
estimators = {
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB() 
}

xvalues = []
acc_values = []
prec_values = []
rec_values = []

best_model, best_score, best_name = None, 0, ""

print("\n--- Estudo de Hiperparâmetros (Naïve Bayes) ---")

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

# --- 5. PLOTAGEM DOS RESULTADOS ---

def plot_nb_metric(metric_name, values):
    figure(figsize=(8, 6))
    ax = plt.gca()
    
    # 1. Desenha o gráfico com a função do professor
    plot_bar_chart(xvalues, values, title=f"Naive Bayes - Comparação de {metric_name}", ylabel=metric_name, percentage=True)
    
    # 2. HACK: Força o tamanho da fonte dos números nas barras
    # Percorre todos os textos adicionados ao gráfico (que são os valores) e aumenta a fonte
    for text in ax.texts:
        text.set_fontsize(14)
        
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_nb_{metric_name.lower()}.png")
    close()

# Gerar os 3 gráficos
plot_nb_metric("Accuracy", acc_values)
plot_nb_metric("Precision", prec_values)
plot_nb_metric("Recall", rec_values)

# --- 6. DESEMPENHO DO MELHOR MODELO ---
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

# --- 7. ESTUDO DE OVERFITTING ---
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 5))
ax = plt.gca()
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'], 
               title=f"Overfitting Study - {best_name}", ylabel="Accuracy", percentage=True)

# HACK: Força o tamanho da fonte dos números nas barras também aqui
for text in ax.texts:
    text.set_fontsize(14)

plt.tight_layout()
savefig(f"images/{FILE_TAG}_nb_overfitting.png")
close()

# --- 8. MATRIZ DE CONFUSÃO ---
print(f"\n--- A gerar Matriz de Confusão ({best_name}) ---")
# 

# Gera a matriz
cm = confusion_matrix(tstY, y_tst_pred, labels=best_model.classes_)

# Prepara a visualização
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

# Desenha e guarda
figure(figsize=(8, 8))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
savefig(f"images/{FILE_TAG}_nb_confusion_matrix.png")
close()

print(f"\nMelhor Modelo Naïve Bayes: {best_name}")
print("\nTodos os gráficos concluídos e guardados na pasta 'images'.")