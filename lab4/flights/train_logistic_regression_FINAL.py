import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
# Importamos matplotlib.pyplot como plt para o ajuste de layout e ticks
import matplotlib.pyplot as plt
# Assumo que 'dslabs_functions.py', 'plot_multiline_chart' e 'plot_bar_chart' estão disponíveis
from dslabs_functions import plot_multiline_chart, plot_bar_chart

# --- VARIÁVEIS A AJUSTAR ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# ----------------------------

# Hiperparâmetros para estudo
ITERATIONS = [100, 300, 500, 700, 1000]
PENALTIES = ["l2", "l1"]
os.makedirs("images", exist_ok=True)

# Load data
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

# --- Hyperparameters study (Tunning de max_iter e penalty) ---
acc_values = {}
prec_values = {}
rec_values = {}

best_model, best_score, best_params = None, 0, None

print("--- Estudo de Hiperparâmetros (Regressão Logística) ---")

for pen in PENALTIES:
    solver_choice = "liblinear"
    
    acc_list = []
    prec_list = []
    rec_list = []
    
    for n_iter in ITERATIONS:
        clf = LogisticRegression(penalty=pen, max_iter=n_iter, solver=solver_choice, random_state=42)
        
        try:
            clf.fit(trnX, trnY)
            y_pred = clf.predict(tstX)

            acc = accuracy_score(tstY, y_pred)
            prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
            rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            
            if acc > best_score:
                best_model = clf
                best_score = acc
                best_params = (pen, n_iter)

            print(f"Penalty={pen}, Iter={n_iter} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
            
        except Exception as e:
            print(f"Erro ao treinar com Penalty={pen}, Iter={n_iter}: {e}")
            acc_list.append(0)
            prec_list.append(0)
            rec_list.append(0)


    acc_values[pen] = acc_list
    prec_values[pen] = prec_list
    rec_values[pen] = rec_list

# --- Plotagem dos Resultados (Gráfico sugerido: Estudo de Hiperparâmetros) ---

# Plot Accuracy
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(ITERATIONS, acc_values, title="Logistic Regression Accuracy",
                     xlabel="Número de Iterações (max_iter)", ylabel="Accuracy", percentage=True)
plt.tight_layout() # NOVO: Ajusta o layout
savefig("images/lr_accuracy.png")
close()

# Plot Precision
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(ITERATIONS, prec_values, title="Logistic Regression Precision",
                     xlabel="Número de Iterações (max_iter)", ylabel="Precision", percentage=True)
plt.tight_layout() # NOVO: Ajusta o layout
savefig("images/lr_precision.png")
close()

# Plot Recall
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(ITERATIONS, rec_values, title="Logistic Regression Recall",
                     xlabel="Número de Iterações (max_iter)", ylabel="Recall", percentage=True)
plt.tight_layout() # NOVO: Ajusta o layout
savefig("images/lr_recall.png")
close()

# --- Best model performance e Overfitting Study ---
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}

print(f"\n--- Descrição do Melhor Modelo LR ---")
print(f"Hiperparâmetros encontrados: Penalty={best_params[0]}, Max_Iter={best_params[1]}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    if metric == "Accuracy":
        trn_val = func(trnY, y_trn_pred)
        tst_val = func(tstY, y_tst_pred)
    else:
        trn_val = func(trnY, y_trn_pred, average="weighted", zero_division=0)
        tst_val = func(tstY, y_tst_pred, average="weighted", zero_division=0)

    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")


# [cite_start]Overfitting Study (Gráfico sugerido: Estudo de Overfitting) [cite: 25]
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(8, 5)) # Tamanho razoável
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'], title=f"Overfitting Study - Regressão Logística", ylabel="Accuracy", percentage=True)
plt.tight_layout() # NOVO: Ajusta o layout
savefig("images/lr_overfitting.png")
close()


# --- Variáveis' Importance (Gráfico sugerido: Variáveis' importance) ---
if hasattr(best_model, 'coef_'):
    print("\n--- Variáveis Mais Importantes ---")
    
    # Calcular a magnitude (valor absoluto) do coeficiente máximo para cada feature
    # Isso funciona mesmo se a importância for 0 para a maioria das classes
    importance = pd.Series(best_model.coef_.T.max(axis=1), index=trnX.columns)
    
    # Selecionar as top 10 features por importância absoluta
    top_n = 10
    top_importance = importance.abs().sort_values(ascending=False).head(top_n)

    # Plotar a importância das variáveis
    figure(figsize=(12, 6)) # NOVO: Tamanho maior para acomodar labels no eixo X
    plot_bar_chart(top_importance.index.to_list(), top_importance.values.tolist(),
                   title=f"Importância das {top_n} Melhores Variáveis (Coeficientes Máx Absoluto)",
                   ylabel="Importância (Coeficiente Absoluto)", percentage=False)
    
    # NOVO: Rotacionar as labels do eixo X para caberem
    plt.xticks(rotation=45, ha='right', fontsize=7)
    
    plt.tight_layout() # NOVO: Ajuste final do layout
    savefig("images/lr_feature_importance.png")
    close()
    
    print(top_importance)
    print(f"\nGráfico 'lr_feature_importance.png' guardado na pasta 'images'.")


print("\nGráficos da Regressão Logística guardados na pasta 'images'.")
