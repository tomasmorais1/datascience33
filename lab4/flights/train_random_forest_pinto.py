import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import subplots, figure, savefig, close
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import plot_tree
from dslabs_functions import plot_multiline_chart, plot_bar_chart, plot_horizontal_bar_chart

# Definições auxiliares
HEIGHT = 6
CLASS_EVAL_METRICS = {"accuracy": accuracy_score}
DELTA_IMPROVE = 0.0005

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"
EVAL_METRIC = "accuracy"
# ------------------------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: Certifique-se de que 'train_scaled.csv' e 'test_scaled.csv' estão no diretório correto.")
    exit()

# --- REMOVER DATA LEAKAGE (CORREÇÃO) ---
# Estas colunas contêm o resultado futuro. Temos de as apagar.
cols_to_drop = [
    'ArrDelay', 'DepDelay', 'ActualElapsedTime', 
    'AirTime', 'ArrTime', 'DepTime', 
    'WheelsOff', 'WheelsOn', 'TaxiIn', 'TaxiOut',
    'Diverted', 
    'ArrivalDelayGroups', 'DepDelayGroups',
    'ArrDel15', 'DepDel15',
    'ArrDelayMinutes', 'DepDelayMinutes'
]

# Verifica quais destas colunas existem realmente no dataset e apaga
cols_in_train = [c for c in cols_to_drop if c in trn_df.columns]
cols_in_test = [c for c in cols_to_drop if c in tst_df.columns]

if len(cols_in_train) > 0:
    print(f"A remover Data Leakage: {cols_in_train}")
    trn_df = trn_df.drop(columns=cols_in_train)
    tst_df = tst_df.drop(columns=cols_in_test)

# Atualizar X e Y
trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

# Atualizar lista de variáveis
VARS = trnX.columns.tolist()
# ----------------------------------------
# --- OTIMIZAÇÃO: SAMPLING ---
# Random Forest é mais rápido que KNN, mas para Tuning usamos um sample seguro.
# 30.000 linhas é um bom compromisso entre velocidade e representatividade.
SAMPLE_SIZE = 20000 

if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada nos ficheiros.")
    exit()

if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"[Aviso] Dataset grande ({trn_df.shape[0]} linhas).")
    print(f"-> A usar amostragem de {SAMPLE_SIZE} para o Random Forest.")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

LABELS = sorted(trnY.unique())
VARS = trnX.columns.tolist()

os.makedirs("images", exist_ok=True)
print(f"Train (Sampled)#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={LABELS}")

# --- Hyperparameters study function (OTIMIZADO) ---
def random_forests_study(
    trnX: pd.DataFrame,
    trnY: pd.Series,
    tstX: pd.DataFrame,
    tstY: pd.Series,
    nr_max_trees: int = 500,
    lag: int = 200,
    metric: str = EVAL_METRIC,
) -> tuple[RandomForestClassifier | None, dict]:
    
    # Define range de árvores e profundidades
    n_estimators: list[int] = [10] + [i for i in range(100, nr_max_trees + 1, lag)]
    max_depths: list[int] = [10, 20] # Podes adicionar mais se tiveres tempo, ex: [5, 10, 20]
    max_features: list[float] = [0.3, 0.9]
    
    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    cols: int = len(max_depths)
    _, axs = plt.subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)
    
    print("\n--- Estudo de Hiperparâmetros (Random Forest) ---")

    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            print(f"Testando Depth={d}, Max_Features={f}...") # Feedback visual
            for n in n_estimators:
                # n_jobs=-1 usa todos os cores do CPU
                clf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f, random_state=42, n_jobs=-1)
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                
                eval_score: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval_score)
                
                if eval_score - best_performance > DELTA_IMPROVE:
                    best_performance = eval_score
                    best_params["params"] = (d, f, n)
                    best_model = clf
            values[f'{f*100:.0f}%'] = y_tst_values

        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"RF Accuracy vs Trees (Depth={d})",
            xlabel="Número de Árvores (n_estimators)",
            ylabel=metric.capitalize(),
            percentage=True,
        )
        axs[0, i].legend(title="Max Features", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"images/{FILE_TAG}_rf_{metric}_study_otimizado.png")
    plt.close()

    print(f'Melhor RF encontrado: {best_params["params"][2]} árvores (Depth={best_params["params"][0]} e Max Features={best_params["params"][1]})')
    return best_model, best_params

# --- Execução do Estudo OTIMIZADO ---
best_model, params = random_forests_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_trees=500,
    lag=200,
    metric=EVAL_METRIC,
)


# --- Best model performance e Overfitting Study ---
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Descrição do Melhor Modelo RF ---")
print(f"Hiperparâmetros encontrados: Max_Depth={params['params'][0]}, Max_Features={params['params'][1]}, Estimators={params['params'][2]}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    trn_val = func(trnY, prd_trn)
    tst_val = func(tstY, prd_tst)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# --- Overfitting Study (Ajustado) ---
d_max: int = params["params"][0]
feat: float = params["params"][1]
nr_estimators_test: list[int] = [i for i in range(10, 501, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []

print("\n--- A gerar gráfico de Overfitting... ---")
for n in nr_estimators_test:
    # Adicionado n_jobs=-1 aqui também para ser rápido
    clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat, random_state=42, n_jobs=-1)
    clf.fit(trnX, trnY)
    y_tst_values.append(accuracy_score(tstY, clf.predict(tstX)))
    y_trn_values.append(accuracy_score(trnY, clf.predict(trnX)))

figure(figsize=(10, 6))
plot_multiline_chart(
    nr_estimators_test,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"RF Overfitting Study (Depth={d_max}, Features={feat})",
    xlabel="Número de Árvores (n_estimators)",
    ylabel=str(EVAL_METRIC).capitalize(),
    percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_rf_{EVAL_METRIC}_overfitting.png")
close()


# --- Variáveis' Importance ---
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Variáveis Mais Importantes ---")
    
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    
    elems = [VARS[idx] for idx in indices[:10]]
    imp_values = [importances[idx] for idx in indices[:10]]
    
    figure(figsize=(10, 6))
    ax = plt.gca()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Importância das 10 Melhores Variáveis (Random Forest)",
        xlabel="Importância",
        ylabel="Variáveis",
        percentage=False,
    )
    
    for container in ax.containers:
        try:
            plt.bar_label(container, fmt='%.4f', padding=5, fontsize=8)
        except:
            pass
    
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_rf_{EVAL_METRIC}_vars_ranking.png")
    close()
    
    print(pd.Series(imp_values, index=elems))
    print("\nGráfico de importância guardado.")


# ----------------------------------------------------
# --- Model Learnt (Decision Tree Visualization) ---
# ----------------------------------------------------
print("\n--- RF Tree Visualization (Model Learnt) ---")

try:
    best_tree = best_model.estimators_[0]   # pick the first tree
    plt.figure(figsize=(18, 10))
    plot_tree(
        best_tree,
        feature_names=VARS,
        class_names=[str(c) for c in LABELS],
        filled=True,
        max_depth=3,     # Reduzido para 3 para caber melhor na imagem
        fontsize=8
    )
    savefig(f"images/{FILE_TAG}_rf_tree.png", bbox_inches="tight")
    close()
    print(f"Gráfico '{FILE_TAG}_rf_tree.png' criado com sucesso.")

except Exception as e:
    print(f"Falha ao gerar modelo aprendido (árvore). Erro: {e}")

print("\nGráficos do Random Forest concluídos e guardados na pasta 'images'.")