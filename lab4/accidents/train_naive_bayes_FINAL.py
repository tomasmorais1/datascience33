import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# ADICIONEI precision_score e roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart, plot_evaluation_results

# --- CONFIGURAÇÃO DE FONTE GLOBAL ---
plt.rcParams.update({'font.size': 12})
# ------------------------------------

# --- VARIÁVEIS A AJUSTAR ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# Mudei a tag para identificar que é o dataset accidents
FILE_TAG = "accidents_nb" 

os.makedirs("images", exist_ok=True)

# Load data
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Ficheiros não encontrados.")
    exit()

if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())

# --- Hyperparameters study ---
estimators = {
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB()
}

xvalues = []
acc_values = []
prec_values = [] # Precision
rec_values = []
f1_values = []   # F1
auc_values = []  # AUC

best_model = None
best_score = 0
best_name = ""
# Por defeito guardamos F1 no best_params, mas podes mudar
best_params = {"name": "", "metric": "f1", "params": ()}

print("--- Estudo de Hiperparâmetros (Naïve Bayes - Todas as Métricas) ---")

for name, clf in estimators.items():
    clf.fit(trnX, trnY)
    y_pred = clf.predict(tstX)

    # 1. Cálculo das Métricas Padrão
    acc = accuracy_score(tstY, y_pred)
    prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
    rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(tstY, y_pred, average="weighted", zero_division=0)
    
    # 2. Cálculo do AUC (Requer probabilidades e tratamento multiclasse)
    try:
        y_probs = clf.predict_proba(tstX)
        # Se for binário (2 classes) vs Multiclasse
        if len(labels) > 2:
            auc = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
        else:
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

    # Lógica de seleção do melhor modelo (Aqui mantive F1, mas podes mudar a métrica de decisão)
    # Se quiseres escolher pelo AUC, muda para: if auc > best_score:
    if f1 > best_score: 
        best_model = clf
        best_score = f1
        best_name = name
        best_params["name"] = name
        best_params["metric"] = f1

    print(f"{name} -> Acc:{acc:.3f}, Prec:{prec:.3f}, Rec:{rec:.3f}, F1:{f1:.3f}, AUC:{auc:.3f}")

# --- Plotagem dos Resultados (TODAS AS OPÇÕES) ---

def plot_nb_metric(metric_name, values):
    figure(figsize=(8, 6))
    ax = plt.gca()
    plot_bar_chart(xvalues, values, title=f"Naive Bayes - {metric_name}", ylabel=metric_name, percentage=True)
    
    for text in ax.texts:
        text.set_fontsize(12)
        
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_nb_{metric_name.lower()}.png")
    close()

# Gerar TODOS os gráficos
plot_nb_metric("Accuracy", acc_values)
plot_nb_metric("Precision", prec_values)
plot_nb_metric("Recall", rec_values)
plot_nb_metric("F1-Score", f1_values)
plot_nb_metric("AUC", auc_values)


# --- Best Model Results ---
print(f"\n--- Desempenho do Melhor Modelo ({best_name}) ---")

y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, y_trn_pred, tstY, y_tst_pred, labels)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_nb_best_model_eval.png")
close()

# --- Overfitting Study (Accuracy) ---
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

print(f"\nMelhor Modelo Naïve Bayes: {best_name}")
print("\nGráficos gerados na pasta 'images':")
print(f"- {FILE_TAG}_nb_accuracy.png")
print(f"- {FILE_TAG}_nb_precision.png")
print(f"- {FILE_TAG}_nb_recall.png")
print(f"- {FILE_TAG}_nb_f1-score.png")
print(f"- {FILE_TAG}_nb_auc.png")