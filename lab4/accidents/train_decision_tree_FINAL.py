import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Importar roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_evaluation_results, plot_horizontal_bar_chart

# --- CONFIGURAÇÃO GLOBAL ---
plt.rcParams.update({'font.size': 12})
# ---------------------------

# --- VARIÁVEIS ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"

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

if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
print(f"Dados prontos para DT (Accidents): Train={len(trnX)}, Test={len(tstX)}")

# --- 2. ESTUDO DE HIPERPARÂMETROS (Focando no AUC) ---
values_auc = {}  # Métrica principal para Dataset 1
depths = list(range(2, MAX_DEPTH + 1))

best_model = None
best_score = 0
best_params = {'name': 'DT', 'metric': 'auc', 'params': ()}

print("\n--- Estudo de Hiperparâmetros (DT - AUC) ---")

for crit in CRITERIA:
    auc_list = []
    print(f"Testing criterion: {crit}...")

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, random_state=42)
        clf.fit(trnX, trnY)
        
        # Calcular AUC (Multiclasse OVR)
        try:
            y_probs = clf.predict_proba(tstX)
            # Para multiclasse ou binário, o roc_auc_score adapta-se se configurado
            if len(labels) > 2:
                auc = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(tstY, y_probs[:, 1])
        except Exception as e:
            print(f"Erro no AUC: {e}")
            auc = 0.5

        auc_list.append(auc)

        # Escolher melhor modelo (Maximizar AUC)
        if auc > best_score:
            best_model = clf
            best_score = auc
            best_params['params'] = (crit, d)
            best_params['auc'] = auc

    values_auc[crit] = auc_list

print(f"Melhor Modelo DT: {best_params['params'][0]} com depth={best_params['params'][1]} (AUC={best_score:.4f})")

# --- 3. PLOT 1: Comparação de Modelos (AUC) ---
figure(figsize=(10, 6))
plot_multiline_chart(depths, values_auc,
    title="DT Models (AUC)",
    xlabel="max_depth", ylabel="AUC", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_auc_study.png")
close()
print("Gráfico 'dt_auc_study.png' guardado.")

# --- 4. PLOT 2: Best Model Results (Evaluation Matrix) ---
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

figure(figsize=(12, 8))
# Nota: A matriz de confusão não depende do AUC, mostra os acertos brutos
plot_evaluation_results(best_params, trnY, y_trn_pred, tstY, y_tst_pred, labels)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_best_model_eval.png")
close()
print("Gráfico 'dt_best_model_eval.png' guardado.")

# --- 5. PLOT 3: Overfitting Study (Curva Train vs Test - AUC) ---
# Aqui usamos AUC também para ser consistente com a métrica de escolha
best_crit = best_params['params'][0]
print(f"\n--- Estudo de Overfitting para {best_crit} (Metric: AUC) ---")

y_tst_values = []
y_trn_values = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, criterion=best_crit, random_state=42)
    clf.fit(trnX, trnY)
    
    # Probabilidades para AUC
    try:
        y_probs_trn = clf.predict_proba(trnX)
        y_probs_tst = clf.predict_proba(tstX)
        
        if len(labels) > 2:
            auc_trn = roc_auc_score(trnY, y_probs_trn, multi_class='ovr', average='weighted')
            auc_tst = roc_auc_score(tstY, y_probs_tst, multi_class='ovr', average='weighted')
        else:
            auc_trn = roc_auc_score(trnY, y_probs_trn[:, 1])
            auc_tst = roc_auc_score(tstY, y_probs_tst[:, 1])
    except:
        auc_trn = 0.5
        auc_tst = 0.5

    y_trn_values.append(auc_trn)
    y_tst_values.append(auc_tst)

figure(figsize=(10, 6))
plot_multiline_chart(
    depths,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"DT Overfitting Study ({best_crit}) - AUC",
    xlabel="max_depth",
    ylabel="AUC",
    percentage=True
)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_dt_overfitting_curve.png")
close()
print("Gráfico 'dt_overfitting_curve.png' guardado.")


# --- 6. PLOT 4: Feature Importance (Horizontal) ---
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
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
    
    figure(figsize=(10, 8))
    try:
        plot_horizontal_bar_chart(
            elems[::-1],
            imp_values[::-1],
            title="Decision Tree Variables Importance",
            xlabel="Importance",
            ylabel="Variables",
            percentage=True
        )
    except:
        plt.barh(elems[::-1], imp_values[::-1], color='skyblue')
        plt.xlabel("Importance")
        plt.title("Decision Tree Variables Importance")
        ax = plt.gca()
        ax.bar_label(ax.containers[0], fmt='%.4f', padding=3)

    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_dt_vars_ranking.png")
    close()
    print("Gráfico 'dt_vars_ranking.png' guardado.")

# --- 7. PLOT 5: Árvore Visual ---
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

print("\nProcesso concluído (Accidents).")