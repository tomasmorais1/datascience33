import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from dslabs_functions import plot_multiline_chart, plot_bar_chart

# --- VARIÁVEIS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"

# Hiperparâmetros para estudo
MAX_DEPTH = 25
CRITERIA = ["gini", "entropy"]

os.makedirs("images", exist_ok=True)

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

# --- BLOCO CRÍTICO: REMOVER DATA LEAKAGE ---
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

# --- Separar X e Y ---
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
print(f"Dados prontos para DT: Train={len(trnX)}, Test={len(tstX)}")

# --- Hyperparameters study (Accuracy, Precision, Recall) ---
values_acc = {}
values_prec = {}
values_rec = {}

best_model = None
best_score = 0
best_params = None

depths = list(range(2, MAX_DEPTH + 1))

print("\n--- Estudo de Hiperparâmetros (DT) ---")

for crit in CRITERIA:
    acc_list = []
    prec_list = []
    rec_list = []
    
    print(f"Processing criterion: {crit}...")

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, random_state=42)
        clf.fit(trnX, trnY)
        
        y_pred = clf.predict(tstX)
        
        # Calcular as 3 métricas
        acc = accuracy_score(tstY, y_pred)
        prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
        rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)
        
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        # Escolher o melhor com base na Accuracy (ou podes mudar para Precision/Recall se preferires)
        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (crit, d)

    values_acc[crit] = acc_list
    values_prec[crit] = prec_list
    values_rec[crit] = rec_list

# --- PLOTS DOS HIPERPARÂMETROS ---

# 1. Accuracy
figure(figsize=(10, 6))
plot_multiline_chart(depths, values_acc,
    title="DT Hyperparameters - Accuracy",
    xlabel="Max Depth", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_study_accuracy.png")
close()

# 2. Precision
figure(figsize=(10, 6))
plot_multiline_chart(depths, values_prec,
    title="DT Hyperparameters - Precision",
    xlabel="Max Depth", ylabel="Precision", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_study_precision.png")
close()

# 3. Recall
figure(figsize=(10, 6))
plot_multiline_chart(depths, values_rec,
    title="DT Hyperparameters - Recall",
    xlabel="Max Depth", ylabel="Recall", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_study_recall.png")
close()

print("Gráficos de estudo (Acc, Prec, Rec) guardados.")

# --- Best model performance ---
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Melhor Modelo DT ---")
print(f"Params: Criterion={best_params[0]}, Max_Depth={best_params[1]}")

for metric, func in metrics.items():
    print(f"{metric} - Train: {func(trnY, y_trn_pred):.4f}, Test: {func(tstY, y_tst_pred):.4f}")

# Overfitting Study
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 4))
ax = plt.gca()
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'],
               title=f"Overfitting Study - DT (Depth={best_params[1]})", ylabel="Accuracy", percentage=True)
ax.bar_label(ax.containers[0], fmt='%.4f')
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_overfitting.png")
close()

# --- Feature Importance (CORRIGIDO) ---
if hasattr(best_model, 'feature_importances_'):
    importance = pd.Series(best_model.feature_importances_, index=trnX.columns)
    top_importance = importance.sort_values(ascending=False).head(10)

    figure(figsize=(12, 6))
    ax = plt.gca()
    bars = ax.bar(top_importance.index, top_importance.values)
    
    plt.title("Top 10 Variáveis (Decision Tree)", fontsize=14)
    plt.ylabel("Importância", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Formatação com 4 casas decimais para não aparecer 0
    ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
    
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_dt_vars_ranking.png")
    close()
    
    print("\nTop 10 Variáveis:")
    print(top_importance)

# --- Visualização da Árvore ---
figure(figsize=(20, 10))
plot_tree(best_model, feature_names=trnX.columns.tolist(), class_names=[str(c) for c in labels],
          max_depth=3, filled=True, rounded=True, fontsize=8)
plt.title(f"Visualização da Árvore (Topo) - Max Depth Real: {best_params[1]}")
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_tree_viz.png")
close()

print("\nTodos os gráficos concluídos e guardados na pasta 'images'.")