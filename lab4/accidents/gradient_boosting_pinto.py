import os
import pandas as pd
from numpy import array, argsort
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dslabs_functions import plot_multiline_chart, plot_horizontal_bar_chart

# ----------------------------- Config -----------------------------
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
EVAL_METRIC = "accuracy"
DELTA_IMPROVE = 0.0005
# ------------------------------------------------------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: train_scaled.csv ou test_scaled.csv não encontrados.")
    exit()

if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada.")
    exit()

trnX, trnY = trn_df.drop(columns=[TARGET]), trn_df[TARGET]
tstX, tstY = tst_df.drop(columns=[TARGET]), tst_df[TARGET]
VARS = trnX.columns.tolist()

os.makedirs("images", exist_ok=True)
print(f"Train#={len(trnX)} Test#={len(tstX)} Labels={sorted(trnY.unique())}")

# ----------------- Hyperparameter Study -----------------
def gradient_boosting_study(trnX, trnY, tstX, tstY,
                            nr_max_trees=200, lag=100, metric=EVAL_METRIC):
    n_estimators = [10] + [i for i in range(50, nr_max_trees + 1, lag)]
    max_depths = [3, 5]        # Trees shallower for faster training
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
            for n in n_estimators:
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
            xlabel="Número de Árvores (n_estimators)",
            ylabel="Accuracy",
            percentage=True
        )
        axs[0, i].legend(title="Learning Rate", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_study.png")
    plt.close()

    print(f"Melhor GB encontrado: Depth={best_params['params'][0]}, LR={best_params['params'][1]}, Estimators={best_params['params'][2]}")
    return best_model, best_params

best_model, params = gradient_boosting_study(trnX, trnY, tstX, tstY, nr_max_trees=200, lag=100)

# ----------------- Performance Metrics -----------------
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
nr_test_estimators = [10, 50, 100, 150, 200]

y_trn_vals, y_tst_vals = [], []
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
    xlabel="Número de Árvores (n_estimators)",
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
