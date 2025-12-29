import matplotlib
matplotlib.use("Agg") # Previne erros de GUI no terminal

import os
import pandas as pd
from numpy import array, argsort
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
# Importar F1-Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dslabs_functions import plot_multiline_chart, plot_horizontal_bar_chart, plot_evaluation_results

# ----------------------------- Config -----------------------------
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "f1" # Dataset 2 -> F1 Score
DELTA_IMPROVE = 0.0005

# Uniformizado com os outros (20k é um bom compromisso para GB)
SAMPLE_SIZE = 10000 
# ------------------------------------------------------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: train_scaled.csv ou test_scaled.csv não encontrados.")
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

# --- SAMPLING ---
if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"--- Dataset grande. A usar sample de {SAMPLE_SIZE} linhas para GB... ---")
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
print(f"Dados prontos: Train (Sample)={len(trnX)}, Test={len(tstX)}")

# ----------------- Hyperparameter Study (Best Model Search) -----------------
def gradient_boosting_study(trnX, trnY, tstX, tstY,
                            nr_max_trees=200, lag=50, metric=EVAL_METRIC):
    
    n_estimators = [10] + [i for i in range(50, nr_max_trees + 1, lag)]
    # As profundidades pedidas:
    max_depths = [2, 5, 7]        
    learning_rates = [0.1, 0.3, 0.5] # Adicionei mais opções para o gráfico ficar mais rico

    best_model = None
    best_params = {"name": "GB", "metric": metric, "params": ()}
    best_perf = 0.0

    # Layout para os gráficos de comparação
    cols = len(max_depths)
    _, axs = plt.subplots(1, cols, figsize=(cols * 6, 5), squeeze=False)

    print("\n--- Hyperparameter Study: Gradient Boosting ---")
    
    for i, d in enumerate(max_depths):
        values = {}
        print(f"   > Testing Depth={d}...")
        for lr in learning_rates:
            y_test_vals = []
            
            for n in n_estimators:
                clf = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
                clf.fit(trnX, trnY)
                
                y_pred = clf.predict(tstX)
                
                # Métrica de avaliação (F1)
                score = f1_score(tstY, y_pred, average="weighted", zero_division=0)
                y_test_vals.append(score)

                if score - best_perf > DELTA_IMPROVE:
                    best_perf = score
                    best_params["params"] = (d, lr, n)
                    best_params["f1"] = score
                    best_model = clf
            
            values[f'LR={lr}'] = y_test_vals

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"GB F1-Score (Depth={d})",
            xlabel="Nr Estimators",
            ylabel="F1-Score",
            percentage=True
        )
        axs[0, i].legend(title="Learning Rate", fontsize=8)

    plt.tight_layout()
    # Este é o gráfico "Gradient boosting different parameterisations comparison"
    plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_study.png")
    plt.close()

    print(f"Melhor GB encontrado: Depth={best_params['params'][0]}, LR={best_params['params'][1]}, Estimators={best_params['params'][2]} (F1={best_perf:.4f})")
    return best_model, best_params

# Executar estudo
best_model, params = gradient_boosting_study(trnX, trnY, tstX, tstY, nr_max_trees=200, lag=50)


# ----------------- Best Model Results (Confusion Matrix) -----------------
print("\n--- Gerando Matrizes de Confusão (Best Model) ---")
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_gb_best_model_eval.png")
plt.close()
print("Gráfico 'gb_best_model_eval.png' guardado.")


# ----------------- Overfitting Study -----------------
d_max, lr_best = params['params'][0], params['params'][1]
nr_test_estimators = [10, 50, 100, 150, 200, 250, 300] # Range para o gráfico

y_trn_vals, y_tst_vals = [], []
print("\n--- Gerando Gráfico de Overfitting... ---")

for n in nr_test_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr_best, random_state=42)
    clf.fit(trnX, trnY)
    
    # Avaliar com F1-Score
    y_trn_vals.append(f1_score(trnY, clf.predict(trnX), average="weighted", zero_division=0))
    y_tst_vals.append(f1_score(tstY, clf.predict(tstX), average="weighted", zero_division=0))

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    nr_test_estimators,
    {"Train": y_trn_vals, "Test": y_tst_vals},
    title=f"GB Overfitting (Depth={d_max}, LR={lr_best})",
    xlabel="Nr Estimators",
    ylabel="F1-Score",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_gb_overfitting.png")
plt.close()
print("Gráfico 'gb_overfitting.png' guardado.")


# ----------------- Feature Importance -----------------
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    
    # Top 10
    elems = []
    imp_values = []
    for i in range(10):
        idx = indices[i]
        elems.append(VARS[idx])
        imp_values.append(importances[idx])

    plt.figure(figsize=(10, 8))
    # Horizontal Bar Chart
    plt.barh(elems[::-1], imp_values[::-1], color='skyblue')
    plt.xlabel("Importance")
    plt.title("GB Variable Importance")
    
    # Labels
    ax = plt.gca()
    ax.bar_label(ax.containers[0], fmt='%.4f', padding=3)
    
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_vars_ranking.png")
    plt.close()
    print("Gráfico 'gb_vars_ranking.png' guardado.")

print("\nProcesso concluído.")