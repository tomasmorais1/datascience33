import matplotlib
matplotlib.use("Agg") # Previne erros de GUI no terminal

import os
import pandas as pd
from numpy import array, argsort
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dslabs_functions import plot_multiline_chart, plot_horizontal_bar_chart

# ----------------------------- Config -----------------------------
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "accuracy"
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
# Removemos variáveis que contêm a resposta (ex: Atrasos, Tempos de voo reais)
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
os.makedirs("images", exist_ok=True)
print(f"Dados prontos: Train (Sample)={len(trnX)}, Test={len(tstX)}")

# ----------------- Hyperparameter Study -----------------
def gradient_boosting_study(trnX, trnY, tstX, tstY,
                            nr_max_trees=200, lag=50, metric=EVAL_METRIC):
    # Ajustei o lag para 50 para ter mais pontos no gráfico
    n_estimators = [10] + [i for i in range(50, nr_max_trees + 1, lag)]
    max_depths = [3, 5]        
    learning_rates = [0.1, 0.05]

    best_model = None
    best_params = {"params": ()}
    best_perf = 0.0

    cols = len(max_depths)
    _, axs = plt.subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)

    print("\n--- Hyperparameter Study: Gradient Boosting ---")
    
    for i, d in enumerate(max_depths):
        values = {}
        for lr in learning_rates:
            y_test_vals = []
            print(f"   > A testar: Depth={d}, LR={lr}...")
            
            for n in n_estimators:
                # GradientBoosting é sequencial (sem n_jobs=-1)
                clf = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
                clf.fit(trnX, trnY)
                
                y_pred = clf.predict(tstX)
                score = accuracy_score(tstY, y_pred)
                y_test_vals.append(score)

                if score - best_perf > DELTA_IMPROVE:
                    best_perf = score
                    best_params["params"] = (d, lr, n)
                    best_model = clf
            
            values[f'LR={lr}'] = y_test_vals

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"GB Accuracy vs Trees (Depth={d})",
            xlabel="Número de Árvores",
            ylabel="Accuracy",
            percentage=True
        )
        axs[0, i].legend(title="Learning Rate", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_study.png")
    plt.close()

    print(f"Melhor GB encontrado: Depth={best_params['params'][0]}, LR={best_params['params'][1]}, Estimators={best_params['params'][2]}")
    return best_model, best_params

# Executar estudo
best_model, params = gradient_boosting_study(trnX, trnY, tstX, tstY, nr_max_trees=200, lag=50)

# ----------------- Performance Metrics -----------------
# Recalcular previsões finais
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0)
}

print("\n--- Best Model Performance ---")
print(f"Hiperparâmetros: Depth={params['params'][0]}, LR={params['params'][1]}, Estimators={params['params'][2]}")
for metric, func in metrics.items():
    print(f"{metric} - Train: {func(trnY, prd_trn):.4f}, Test: {func(tstY, prd_tst):.4f}")

# ----------------- Overfitting Study -----------------
d_max, lr_best = params['params'][0], params['params'][1]
# Reduzi o range do overfitting para ser mais rápido
nr_test_estimators = [10, 50, 100, 150, 200]

y_trn_vals, y_tst_vals = [], []
print("\n--- A gerar gráfico de Overfitting... ---")

for n in nr_test_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr_best, random_state=42)
    clf.fit(trnX, trnY)
    y_trn_vals.append(accuracy_score(trnY, clf.predict(trnX)))
    y_tst_vals.append(accuracy_score(tstY, clf.predict(tstX)))

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    nr_test_estimators,
    {"Train": y_trn_vals, "Test": y_tst_vals},
    title=f"GB Overfitting Study (Depth={d_max}, LR={lr_best})",
    xlabel="Número de Árvores",
    ylabel="Accuracy",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_overfitting.png")
plt.close()

# ----------------- Feature Importance -----------------
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = [VARS[i] for i in indices[:10]]
    imp_values = [importances[i] for i in indices[:10]]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title=f"Top 10 Variáveis (Gradient Boosting)",
        xlabel="Importância",
        ylabel="Variáveis",
        percentage=False
    )
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_vars_ranking.png")
    plt.close()

print("\nGráficos do Gradient Boosting concluídos e guardados na pasta 'images'.")