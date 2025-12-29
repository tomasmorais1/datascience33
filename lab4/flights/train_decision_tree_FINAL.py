import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Importar métricas necessárias, incluindo F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib.pyplot import figure, savefig, close
# Importar funções do dslabs
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_evaluation_results, plot_horizontal_bar_chart

# --- CONFIGURAÇÃO GLOBAL ---
plt.rcParams.update({'font.size': 12})
# ---------------------------

# --- VARIÁVEIS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"

# Hiperparâmetros
MAX_DEPTH = 25
CRITERIA = ["gini", "entropy"]

os.makedirs("images", exist_ok=True)

# --- 1. CARREGAR DADOS ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Ficheiros csv não encontrados.")
    exit()

# Remover Data Leakage
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

# Separar X e Y
if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
print(f"Dados prontos para DT: Train={len(trnX)}, Test={len(tstX)}")

# --- 2. ESTUDO DE HIPERPARÂMETROS (Focando no F1-Score) ---
values_f1 = {}  # Para o gráfico pedido
depths = list(range(2, MAX_DEPTH + 1))

best_model = None
best_score = 0
best_params = {'name': 'DT', 'metric': 'f1', 'params': ()}

print("\n--- Estudo de Hiperparâmetros (DT - F1 Score) ---")

for crit in CRITERIA:
    f1_list = []
    print(f"Testing criterion: {crit}...")

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, random_state=42)
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)
        
        # Calcular F1 (Weighted)
        f1 = f1_score(tstY, y_pred, average="weighted", zero_division=0)
        f1_list.append(f1)

        # Escolher melhor modelo (Maximizar F1)
        if f1 > best_score:
            best_model = clf
            best_score = f1
            best_params['params'] = (crit, d)
            best_params['f1'] = f1

    values_f1[crit] = f1_list

print(f"Melhor Modelo DT: {best_params['params'][0]} com depth={best_params['params'][1]} (F1={best_score:.4f})")

# --- 3. PLOT 1: Comparação de Modelos (F1-Score) ---
figure(figsize=(10, 6))
plot_multiline_chart(depths, values_f1,
    title="DT Models (F1-Score)",
    xlabel="max_depth", ylabel="F1-Score", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_f1_study.png")
close()
print("Gráfico 'dt_f1_study.png' guardado.")

# --- 4. PLOT 2: Best Model Results (Evaluation Matrix) ---
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, y_trn_pred, tstY, y_tst_pred, labels)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_best_model_eval.png")
close()
print("Gráfico 'dt_best_model_eval.png' guardado.")

# --- 5. PLOT 3: Overfitting Study (Curva Train vs Test) ---
# Usamos os parâmetros do melhor modelo (critério) e variamos a profundidade
best_crit = best_params['params'][0]
print(f"\n--- Estudo de Overfitting para {best_crit} ---")

eval_metric = "accuracy" # Overfitting vê-se bem com accuracy
y_tst_values = []
y_trn_values = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, criterion=best_crit, random_state=42)
    clf.fit(trnX, trnY)
    
    prd_tst_Y = clf.predict(tstX)
    prd_trn_Y = clf.predict(trnX)
    
    y_tst_values.append(accuracy_score(tstY, prd_tst_Y))
    y_trn_values.append(accuracy_score(trnY, prd_trn_Y))

figure(figsize=(10, 6))
plot_multiline_chart(
    depths,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"DT Overfitting Study ({best_crit})",
    xlabel="max_depth",
    ylabel="Accuracy",
    percentage=True
)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_overfitting_curve.png")
close()
print("Gráfico 'dt_overfitting_curve.png' guardado.")


# --- 6. PLOT 4: Feature Importance (Horizontal Bar Chart) ---
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1] # Ordenar descendente
    
    # Selecionar Top 10
    top_n = 10
    top_indices = indices[:top_n]
    
    elems = []
    imp_values = []
    
    print("\n--- Variáveis Mais Importantes ---")
    for f in range(top_n):
        idx = top_indices[f]
        name = trnX.columns[idx]
        val = importances[idx]
        elems.append(name)
        imp_values.append(val)
        print(f"{f+1}. {name} ({val:.4f})")
    
    # Gráfico Horizontal (Barh) para melhor leitura
    figure(figsize=(10, 8))
    # Usamos o plot_horizontal_bar_chart se disponível, ou barh nativo
    # Como o professor forneceu, vamos assumir que existe. 
    # Se der erro, usa-se o código nativo abaixo:
    try:
        plot_horizontal_bar_chart(
            elems[::-1], # Inverter para o maior ficar em cima
            imp_values[::-1],
            title="Decision Tree Variables Importance",
            xlabel="Importance",
            ylabel="Variables",
            percentage=True
        )
    except:
        # Fallback para matplotlib puro se a função falhar
        plt.barh(elems[::-1], imp_values[::-1], color='skyblue')
        plt.xlabel("Importance")
        plt.title("Decision Tree Variables Importance")
        # Labels nas barras
        ax = plt.gca()
        ax.bar_label(ax.containers[0], fmt='%.4f', padding=3)

    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_dt_vars_ranking.png")
    close()
    print("Gráfico 'dt_vars_ranking.png' guardado.")

# --- 7. PLOT 5: Árvore Visual (Topo) ---
figure(figsize=(14, 6))
plot_tree(
    best_model,
    max_depth=3,
    feature_names=trnX.columns.tolist(),
    class_names=[str(c) for c in labels],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_tree_viz.png")
close()
print("Gráfico 'dt_tree_viz.png' guardado.")

print("\nProcesso concluído.")