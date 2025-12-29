import matplotlib
matplotlib.use("Agg") # Previne erros de GUI no terminal

import os
import pandas as pd
from numpy import array, argsort
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
# Importar AUC e outras métricas
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dslabs_functions import plot_multiline_chart, plot_horizontal_bar_chart, plot_evaluation_results

# ----------------------------- Config -----------------------------
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
# Métrica principal para Dataset 1
EVAL_METRIC = "auc" 
DELTA_IMPROVE = 0.0005

# Uniformizado com os outros
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

# --- SAMPLING (Se necessário) ---
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
print(f"Dados prontos (Accidents): Train (Sample)={len(trnX)}, Test={len(tstX)}")

# ----------------- Hyperparameter Study (Optimization by AUC) -----------------
def gradient_boosting_study(trnX, trnY, tstX, tstY,
                            nr_max_trees=200, lag=50, metric=EVAL_METRIC):
    
    n_estimators = [10] + [i for i in range(50, nr_max_trees + 1, lag)]
    # As profundidades pedidas explicitamente
    max_depths = [2, 5, 7]        
    learning_rates = [0.1, 0.3, 0.5] 

    best_model = None
    best_params = {"name": "GB", "metric": metric, "params": ()}
    best_perf = 0.0

    cols = len(max_depths)
    _, axs = plt.subplots(1, cols, figsize=(cols * 6, 5), squeeze=False)

    print("\n--- Hyperparameter Study: Gradient Boosting (AUC) ---")
    
    for i, d in enumerate(max_depths):
        values = {}
        print(f"   > Testing Depth={d}...")
        for lr in learning_rates:
            y_test_vals = []
            
            for n in n_estimators:
                clf = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
                clf.fit(trnX, trnY)
                
                # Para AUC, precisamos das probabilidades
                try:
                    y_probs = clf.predict_proba(tstX)
                    if len(labels) > 2:
                        score = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
                    else:
                        score = roc_auc_score(tstY, y_probs[:, 1])
                except:
                    score = 0.5

                y_test_vals.append(score)

                if score - best_perf > DELTA_IMPROVE:
                    best_perf = score
                    best_params["params"] = (d, lr, n)
                    best_params["auc"] = score
                    best_model = clf
            
            values[f'LR={lr}'] = y_test_vals

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"GB AUC (Depth={d})",
            xlabel="Nr Estimators",
            ylabel="AUC",
            percentage=True
        )
        axs[0, i].legend(title="Learning Rate", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_study.png")
    plt.close()

    print(f"Melhor GB encontrado: Depth={best_params['params'][0]}, LR={best_params['params'][1]}, Estimators={best_params['params'][2]} (AUC={best_perf:.4f})")
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


# ----------------- Overfitting Study (AUC) -----------------
d_max, lr_best = params['params'][0], params['params'][1]
nr_test_estimators = [10, 50, 100, 150, 200, 250, 300] 

y_trn_vals, y_tst_vals = [], []
print("\n--- Gerando Gráfico de Overfitting (AUC)... ---")

for n in nr_test_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr_best, random_state=42)
    clf.fit(trnX, trnY)
    
    # Calcular AUC para Treino e Teste
    y_probs_trn = clf.predict_proba(trnX)
    y_probs_tst = clf.predict_proba(tstX)
    
    if len(labels) > 2:
        auc_trn = roc_auc_score(trnY, y_probs_trn, multi_class='ovr', average='weighted')
        auc_tst = roc_auc_score(tstY, y_probs_tst, multi_class='ovr', average='weighted')
    else:
        auc_trn = roc_auc_score(trnY, y_probs_trn[:, 1])
        auc_tst = roc_auc_score(tstY, y_probs_tst[:, 1])

    y_trn_vals.append(auc_trn)
    y_tst_vals.append(auc_tst)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    nr_test_estimators,
    {"Train": y_trn_vals, "Test": y_tst_vals},
    title=f"GB Overfitting - AUC (Depth={d_max}, LR={lr_best})",
    xlabel="Nr Estimators",
    ylabel="AUC",
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

print("\nProcesso concluído (Accidents).")