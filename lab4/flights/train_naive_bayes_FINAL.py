import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# ADICIONADO: f1_score e roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart, plot_evaluation_results

# --- CONFIGURAÇÃO DE FONTE GLOBAL ---
plt.rcParams.update({'font.size': 12}) 
# ------------------------------------

# --- VARIÁVEIS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# Tag ajustada para não sobrepor as imagens do dataset accidents
FILE_TAG = "flights_nb"

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

# --- 4. ESTUDO DE HIPERPARÂMETROS (Todas as Métricas) ---
estimators = {
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB()
}

xvalues = []
acc_values = []
prec_values = []
rec_values = []
f1_values = []  # Novo
auc_values = [] # Novo

best_model = None
best_score = 0
best_name = ""
# Guardamos a métrica F1 no params para consistência com a escolha
best_params = {"name": "", "metric": "f1", "params": ()}

print("\n--- Estudo de Hiperparâmetros (Naïve Bayes - Flights) ---")

for name, clf in estimators.items():
    clf.fit(trnX, trnY)
    y_pred = clf.predict(tstX)

    # 1. Cálculo das Métricas
    acc = accuracy_score(tstY, y_pred)
    prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
    rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(tstY, y_pred, average="weighted", zero_division=0)

    # 2. Cálculo do AUC (Binário)
    try:
        y_probs = clf.predict_proba(tstX)
        # Para binário (0/1), usamos a probabilidade da classe positiva (coluna 1)
        auc = roc_auc_score(tstY, y_probs[:, 1])
    except Exception as e:
        print(f"Aviso: Não foi possível calcular AUC para {name}. Erro: {e}")
        auc = 0.0

    # Guardar valores
    xvalues.append(name)
    acc_values.append(acc)
    prec_values.append(prec)
    rec_values.append(rec)
    f1_values.append(f1)
    auc_values.append(auc)

    # SELEÇÃO DO MELHOR MODELO:
    # Para Flights (Dataset 2 - Desequilibrado), o F1 é geralmente o critério
    if f1 > best_score:
        best_model = clf
        best_score = f1
        best_name = name
        best_params["name"] = name
        best_params["metric"] = f1

    print(f"{name} -> Acc:{acc:.3f}, F1:{f1:.3f}, AUC:{auc:.3f}")

# --- 5. PLOTAGEM DE TODAS AS MÉTRICAS ---

def plot_nb_metric(metric_name, values):
    figure(figsize=(8, 6))
    ax = plt.gca()
    plot_bar_chart(xvalues, values, title=f"Naive Bayes - {metric_name}", ylabel=metric_name, percentage=True)
    
    for text in ax.texts:
        text.set_fontsize(12)
        
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_nb_{metric_name.lower()}.png")
    close()

# Gerar todos os gráficos
plot_nb_metric("Accuracy", acc_values)
plot_nb_metric("Precision", prec_values)
plot_nb_metric("Recall", rec_values)
plot_nb_metric("F1-Score", f1_values)
plot_nb_metric("AUC", auc_values)

# --- 6. BEST MODEL EVALUATION ---
print(f"\n--- Desempenho do Melhor Modelo ({best_name}) ---")

y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, y_trn_pred, tstY, y_tst_pred, labels)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_nb_best_model_eval.png")
close()

# --- 7. OVERFITTING STUDY ---
data = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 5))
ax = plt.gca()
plot_bar_chart(data['Dataset'], data['Accuracy'], title=f"Overfitting Study - {best_name}", ylabel="Accuracy", percentage=True)

for text in ax.texts:
    text.set_fontsize(12)

plt.tight_layout()
savefig(f"images/{FILE_TAG}_nb_overfitting.png")
close()

print(f"\nMelhor Modelo Naïve Bayes (F1): {best_name}")
print("\nGráficos gerados na pasta 'images':")
print(f"- {FILE_TAG}_nb_accuracy.png")
print(f"- {FILE_TAG}_nb_precision.png")
print(f"- {FILE_TAG}_nb_recall.png")
print(f"- {FILE_TAG}_nb_f1-score.png")
print(f"- {FILE_TAG}_nb_auc.png")