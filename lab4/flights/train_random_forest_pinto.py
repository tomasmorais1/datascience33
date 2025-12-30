import matplotlib
matplotlib.use("Agg") # Previne erros de GUI no terminal

import os
import pandas as pd
from numpy import array, argsort, std
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# Importar F1 e outras métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
from dslabs_functions import plot_multiline_chart, plot_horizontal_bar_chart, plot_evaluation_results

# ----------------------------- Config -----------------------------
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
# Métrica principal para Dataset 2
EVAL_METRIC = "f1" 
DELTA_IMPROVE = 0.0005

# Sampling para Random Forest (como é pesado, usamos sample se for enorme)
SAMPLE_SIZE = 10000 
# ------------------------------------------------------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: train_scaled.csv ou test_scaled.csv não encontrados.")
    exit()

if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
    exit()

# --- REMOVER DATA LEAKAGE ---
# (Mesma lista que usámos nos outros modelos)
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
    print(f"--- Dataset grande. A usar sample de {SAMPLE_SIZE} linhas para RF... ---")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

VARS = trnX.columns.tolist()
labels = sorted(trnY.unique())
os.makedirs("images", exist_ok=True)
print(f"Dados prontos (Flights): Train (Sample)={len(trnX)}, Test={len(tstX)}")


# ----------------- Hyperparameter Study (Optimization by F1) -----------------
def random_forests_study(trnX, trnY, tstX, tstY,
                         nr_max_trees=500, lag=100, metric=EVAL_METRIC):
    
    n_estimators = [10] + [i for i in range(50, nr_max_trees + 1, lag)]
    # Profundidades pedidas
    max_depths = [2, 5, 7]
    # Features (percentagem)
    max_features = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model = None
    best_params = {"name": "RF", "metric": metric, "params": ()}
    best_perf = 0.0

    cols = len(max_depths)
    _, axs = plt.subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)

    print("\n--- Hyperparameter Study: Random Forest (F1) ---")
    
    for i, d in enumerate(max_depths):
        values = {}
        print(f"   > Testing Depth={d}...")
        for f in max_features:
            y_test_vals = []
            for n in n_estimators:
                clf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, 
                                             random_state=42, n_jobs=-1)
                clf.fit(trnX, trnY)
                
                # Previsão
                y_pred = clf.predict(tstX)
                
                # Métrica F1 Score (Weighted)
                score = f1_score(tstY, y_pred, average="weighted", zero_division=0)
                y_test_vals.append(score)

                if score - best_perf > DELTA_IMPROVE:
                    best_perf = score
                    best_params["params"] = (d, f, n)
                    best_params["f1"] = score
                    best_model = clf
            
            values[f'F={f}'] = y_test_vals

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"RF F1-Score (Depth={d})",
            xlabel="Nr Estimators",
            ylabel="F1-Score",
            percentage=True
        )
        axs[0, i].legend(title="Max Features", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_rf_{EVAL_METRIC}_study.png")
    plt.close()

    print(f"Melhor RF encontrado: Depth={best_params['params'][0]}, MaxFeatures={best_params['params'][1]}, Estimators={best_params['params'][2]} (F1={best_perf:.4f})")
    return best_model, best_params

# Executar estudo
best_model, params = random_forests_study(trnX, trnY, tstX, tstY, nr_max_trees=500, lag=100)


# ----------------- Best Model Results (Confusion Matrix) -----------------
print("\n--- Gerando Matrizes de Confusão (Best Model) ---")
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_rf_best_model_eval.png")
plt.close()
print("Gráfico 'rf_best_model_eval.png' guardado.")


# ----------------- Overfitting Study (F1) -----------------
d_max, feat_best = params['params'][0], params['params'][1]
nr_test_estimators = [10, 50, 100, 150, 200, 300, 400, 500]

y_trn_vals, y_tst_vals = [], []
print("\n--- Gerando Gráfico de Overfitting (F1)... ---")

for n in nr_test_estimators:
    clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat_best, random_state=42, n_jobs=-1)
    clf.fit(trnX, trnY)
    
    # Calcular F1 Score
    f1_trn = f1_score(trnY, clf.predict(trnX), average="weighted", zero_division=0)
    f1_tst = f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0)

    y_trn_vals.append(f1_trn)
    y_tst_vals.append(f1_tst)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    nr_test_estimators,
    {"Train": y_trn_vals, "Test": y_tst_vals},
    title=f"RF Overfitting - F1 (Depth={d_max}, Feat={feat_best})",
    xlabel="Nr Estimators",
    ylabel="F1-Score",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_rf_overfitting.png")
plt.close()
print("Gráfico 'rf_overfitting.png' guardado.")


# ----------------- Feature Importance -----------------
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Feature Importance ---")
    importances = best_model.feature_importances_
    
    # Calcular desvio padrão entre as árvores
    std_dev = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    
    indices = argsort(importances)[::-1]
    
    elems = []
    imp_values = []
    stdevs = []
    
    for i in range(10): # Top 10
        idx = indices[i]
        elems.append(VARS[idx])
        imp_values.append(importances[idx])
        stdevs.append(std_dev[idx])

    plt.figure(figsize=(10, 8))
    # Usamos o plot_horizontal_bar_chart se disponível, senão matplotlib puro
    try:
        plot_horizontal_bar_chart(
            elems, imp_values, error=stdevs, 
            title="RF Variable Importance", xlabel="Importance", ylabel="Variables", percentage=True
        )
    except:
        plt.barh(elems[::-1], imp_values[::-1], xerr=stdevs[::-1], color='skyblue')
        plt.xlabel("Importance")
        plt.title("RF Variable Importance")

    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_rf_vars_ranking.png")
    plt.close()
    print("Gráfico 'rf_vars_ranking.png' guardado.")

# ----------------- Model Learnt (Tree Visualization) -----------------
print("\n--- Visualização de uma Árvore ---")
try:
    # Vamos buscar a primeira árvore do Random Forest
    tree_to_plot = best_model.estimators_[0]
    
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, feature_names=VARS, class_names=[str(c) for c in labels],
              max_depth=3, filled=True, rounded=True, fontsize=9)
    plt.title(f"Random Forest - Example Tree (Depth={d_max})")
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_rf_tree_viz.png")
    plt.close()
    print("Gráfico 'rf_tree_viz.png' guardado.")
except Exception as e:
    print(f"Não foi possível gerar a árvore: {e}")

print("\nProcesso concluído (Flights).")