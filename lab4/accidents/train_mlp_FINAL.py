import matplotlib
matplotlib.use("Agg") # Previne erros de GUI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dslabs_functions import plot_multiline_chart, plot_line_chart, plot_evaluation_results

# --- CONFIGURAÇÃO ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "accidents"
EVAL_METRIC = "auc" 
DELTA_IMPROVE = 0.0005

# --- PARÂMETROS MLP ---
NR_MAX_ITER = 3000
LAG = 500
LEARNING_RATES = [0.1, 0.01, 0.001] 
LR_TYPES = ["constant", "invscaling", "adaptive"]

# --------------------

# --- Load data ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

if TARGET not in trn_df.columns:
    print(f"Erro: Target '{TARGET}' não encontrado.")
    exit()

trnX = trn_df.drop(columns=[TARGET])
trnY = trn_df[TARGET]
tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

labels = sorted(trnY.unique())
os.makedirs("images", exist_ok=True)
print(f"Dados (Accidents): Train={len(trnX)}, Test={len(tstX)}")


# --- 1. Hyperparameter Study (LR Types vs Rates) - AUC ---
nr_iterations = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

best_model = None
best_params = {"name": "MLP", "metric": EVAL_METRIC, "params": ()}
best_performance = 0.0

# Subplots: Um gráfico por LR Type
cols = len(LR_TYPES)
_, axs = plt.subplots(1, cols, figsize=(cols * 5, 5), squeeze=False)

print(f"\n--- Estudo de Hiperparâmetros (MLP - {EVAL_METRIC}) ---")

for i, lr_type in enumerate(LR_TYPES):
    values = {}
    print(f"   > Testing LR Type: {lr_type}...")

    for lr in LEARNING_RATES:
        y_tst_values = []
        
        # Warm_start permite continuar o treino
        clf = MLPClassifier(
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=LAG,
            warm_start=True,
            activation="logistic", # Como no código do stor
            solver="sgd",          # Necessário para 'learning_rate' ter efeito
            verbose=False,
            random_state=42
        )
        
        for n in range(len(nr_iterations)):
            clf.fit(trnX, trnY)
            
            # Calcular AUC
            try:
                y_probs = clf.predict_proba(tstX)
                if len(labels) > 2:
                    score = roc_auc_score(tstY, y_probs, multi_class='ovr', average='weighted')
                else:
                    score = roc_auc_score(tstY, y_probs[:, 1])
            except:
                score = 0.5
            
            y_tst_values.append(score)

            if score - best_performance > DELTA_IMPROVE:
                best_performance = score
                best_params["params"] = (lr_type, lr, nr_iterations[n])
                best_params["auc"] = score
                best_model = clf
        
        values[f'LR={lr}'] = y_tst_values

    plot_multiline_chart(
        nr_iterations,
        values,
        ax=axs[0, i],
        title=f"MLP ({lr_type})",
        xlabel="Iterations",
        ylabel=EVAL_METRIC.upper(),
        percentage=True,
    )

plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_{EVAL_METRIC}_study.png")
plt.close()

print(f"Melhor MLP: Type={best_params['params'][0]}, LR={best_params['params'][1]}, Iter={best_params['params'][2]} ({EVAL_METRIC}={best_performance:.4f})")


# --- 2. Best Model Results (Confusion Matrix) ---
print("\n--- Gerando Matrizes de Confusão ---")
# Resetar warm_start para False para garantir avaliação justa final
# best_model já está treinado até ao ponto ótimo pelo loop anterior
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

plt.figure(figsize=(12, 8))
plot_evaluation_results(best_params, trnY, prd_trn, tstY, prd_tst, labels)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_best_model_eval.png")
plt.close()


# --- 3. Overfitting Study (Train vs Test Evolution) ---
print("\n--- Gerando Gráfico de Overfitting (AUC) ---")
best_lr_type = best_params['params'][0]
best_lr = best_params['params'][1]

# Usamos warm_start novamente para ver a evolução
clf_overfit = MLPClassifier(
    learning_rate=best_lr_type,
    learning_rate_init=best_lr,
    max_iter=LAG,
    warm_start=True,
    activation="logistic",
    solver="sgd",
    verbose=False,
    random_state=42
)

auc_train = []
auc_test = []
steps = []

current_iter = 0
for i in range(len(nr_iterations)):
    # O loop do warm_start acumula iterações. 
    # Precisamos de chamar fit() várias vezes, incrementando o max_iter se não usarmos partial_fit.
    # Mas com warm_start=True e max_iter fixo em LAG, ele treina MAIS LAG iterações a cada fit?
    # Não, max_iter é o total. Temos de incrementar.
    
    total_iter = nr_iterations[i]
    clf_overfit.set_params(max_iter=total_iter)
    clf_overfit.fit(trnX, trnY)
    
    # AUC Train
    y_probs_trn = clf_overfit.predict_proba(trnX)
    if len(labels) > 2:
        s_trn = roc_auc_score(trnY, y_probs_trn, multi_class='ovr', average='weighted')
    else:
        s_trn = roc_auc_score(trnY, y_probs_trn[:, 1])
    
    # AUC Test
    y_probs_tst = clf_overfit.predict_proba(tstX)
    if len(labels) > 2:
        s_tst = roc_auc_score(tstY, y_probs_tst, multi_class='ovr', average='weighted')
    else:
        s_tst = roc_auc_score(tstY, y_probs_tst[:, 1])
        
    auc_train.append(s_trn)
    auc_test.append(s_tst)
    steps.append(total_iter)

plt.figure(figsize=(10, 6))
plot_multiline_chart(
    steps,
    {"Train": auc_train, "Test": auc_test},
    title=f"MLP Overfitting (Type={best_lr_type}, LR={best_lr})",
    xlabel="Iterations",
    ylabel="AUC",
    percentage=True
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_overfitting.png")
plt.close()


# --- 4. Loss Curve Analysis ---
print("\n--- Gerando Loss Curve ---")
# A loss curve está guardada no modelo
loss_values = best_model.loss_curve_

plt.figure(figsize=(10, 6))
plot_line_chart(
    np.arange(len(loss_values)),
    loss_values,
    title="Loss Curve (Best Model)",
    xlabel="Iterations",
    ylabel="Loss",
    percentage=False
)
plt.tight_layout()
plt.savefig(f"images/{FILE_TAG}_mlp_loss_curve.png")
plt.close()

print("\nProcesso concluído (MLP Accidents).")