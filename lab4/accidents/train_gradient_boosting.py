import os
import pandas as pd
from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import subplots, figure, savefig, close
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_horizontal_bar_chart
# Definições auxiliares (necessárias para o estudo)
HEIGHT = 6
CLASS_EVAL_METRICS = {"accuracy": accuracy_score}
DELTA_IMPROVE = 0.0005

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
EVAL_METRIC = "accuracy"
# ------------------------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Certifique-se de que 'train_scaled.csv' e 'test_scaled.csv' estão no diretório correto.")
    exit()

# Separar X e y
if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

LABELS = sorted(trnY.unique())
VARS = trnX.columns.tolist()

os.makedirs("images", exist_ok=True)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={LABELS}")


# --- Hyperparameters study function (OTIMIZADO) ---
def gradient_boosting_study(
    trnX: pd.DataFrame,
    trnY: pd.Series,
    tstX: pd.DataFrame,
    tstY: pd.Series,
    # REDUÇÃO: Menos árvores são necessárias no boosting
    nr_max_trees: int = 500,
    # REDUÇÃO: Intervalo maior para menor tempo de treino
    lag: int = 150,
    metric: str = EVAL_METRIC,
) -> tuple[GradientBoostingClassifier | None, dict]:

    # REDUÇÃO: Intervalo de n_estimators
    n_estimators: list[int] = [10] + [i for i in range(150, nr_max_trees + 1, lag)]
    # REDUÇÃO: Profundidades típicas para GB (árvores rasas)
    max_depths: list[int] = [3, 6]
    # REDUÇÃO: Learning rates essenciais
    learning_rates: list[float] = [0.1, 0.01]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(max_depths)
    # Criamos subplots para plotar várias curvas de profundidade
    _, axs = plt.subplots(1, cols, figsize=(cols * 6, 6), squeeze=False)
    
    print("\n--- Estudo de Hiperparâmetros (Gradient Boosting) ---")
    
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        
        for lr in learning_rates:
            y_tst_values: list[float] = []
            
            for n in n_estimators:
                # Usa n_estimators=n, max_depth=d, learning_rate=lr
                clf = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
                
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                
                eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval_score)
                
                if eval_score - best_performance > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["params"] = (d, lr, n)
                    best_model = clf
            
            values[f'LR={lr}'] = y_tst_values

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"GB Accuracy vs Trees (Depth={d})",
            xlabel="Número de Árvores (n_estimators)",
            ylabel=metric.capitalize(),
            percentage=True,
        )
        axs[0, i].legend(title="Learning Rate", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_gb_{metric}_study.png")
    plt.close()

    print(f'Melhor GB encontrado: {best_params["params"][2]} árvores (Depth={best_params["params"][0]} e LR={best_params["params"][1]})')
    return best_model, best_params

# --- Execução do Estudo ---
best_model, params = gradient_boosting_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=500,
    lag=150,
    metric=EVAL_METRIC,
)


# --- Best model performance e Overfitting Study (Métricas Finais) ---
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Descrição do Melhor Modelo GB ---")
print(f"Hiperparâmetros encontrados: Max_Depth={params['params'][0]}, Learning Rate={params['params'][1]}, Estimators={params['params'][2]}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    trn_val = func(trnY, prd_trn)
    tst_val = func(tstY, prd_tst)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# --- Overfitting Study (Usando o número de árvores) ---
d_max: int = params["params"][0]
lr_best: float = params["params"][1]
nr_estimators_test: list[int] = [i for i in range(10, 501, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators_test:
    # Usar os melhores hiperparâmetros
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr_best, random_state=42)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(accuracy_score(tstY, prd_tst_Y))
    y_trn_values.append(accuracy_score(trnY, prd_trn_Y))

figure(figsize=(10, 6))
plot_multiline_chart(
    nr_estimators_test,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"GB Overfitting Study (Depth={d_max}, LR={lr_best})",
    xlabel="Número de Árvores (n_estimators)",
    ylabel=str(EVAL_METRIC).capitalize(),
    percentage=True,)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_overfitting.png")
close()


# --- Variáveis' Importance ---
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Variáveis Mais Importantes ---")
    
    # Calcular o desvio padrão da importância das features
    trees_importances: list[float] = []
    # Iterar sobre as árvores na lista de estimadores
    for lst_trees in best_model.estimators_:
        # No GradientBoostingClassifier, a cada estágio (árvore), há um classificador para cada classe
        if isinstance(lst_trees, GradientBoostingClassifier): # Para garantir que é uma lista de árvores
            for tree in lst_trees:
                trees_importances.append(tree.feature_importances_)
        else: # Se o modelo for multi-classe, estimators_ é uma matriz de arrays de árvores
             for tree in lst_trees:
                trees_importances.append(tree.feature_importances_)
    
    # Usar apenas a importância média se não houver complexidade para calcular o std
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    
    elems: list[str] = []
    imp_values: list[float] = []
    
    top_n = 10
    
    for f in range(min(top_n, len(VARS))):
        elems += [VARS[indices[f]]]
        imp_values.append(importances[indices[f]])
    
    # Plotar a importância das variáveis (usando plot_horizontal_bar_chart)
    figure(figsize=(10, 6))
    ax = plt.gca()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title=f"Importância das {top_n} Variáveis (Gradient Boosting)",
        xlabel="Importância",
        ylabel="Variáveis",
        percentage=False,)
    
    # Rótulos nas barras
    for container in ax.containers:
        plt.bar_label(container, fmt='%.4f', padding=5, fontsize=8)
    
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_gb_{EVAL_METRIC}_vars_ranking.png")
    close()
    
    # Imprimir os resultados no console
    print(pd.Series(imp_values, index=elems))
    print(f"\nGráfico 'gb_vars_ranking.png' (Importância) guardado na pasta 'images'.")


print("\nGráficos do Gradient Boosting concluídos e guardados na pasta 'images'.")
