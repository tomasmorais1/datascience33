import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
import matplotlib.pyplot as plt
from dslabs_functions import plot_multiline_chart, plot_bar_chart
# Necessário para a visualização da árvore
from sklearn.tree import plot_tree

# --- VARIÁVEIS DE DADOS PREPARADOS ---
TARGET = "crash_type"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
# ------------------------------------

# Hiperparâmetros para estudo
MAX_DEPTH = 25
CRITERIA = ["gini", "entropy"]

os.makedirs("images", exist_ok=True)

# Load data (Dados já separados e escalados)
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

labels = sorted(trnY.unique())

# --- Hyperparameters study (Tuning de Max Depth e Criterion) ---
values_accuracy = {}
values_precision = {}
values_recall = {}

best_model = None
best_score = 0
best_params = None

depths = list(range(2, MAX_DEPTH + 1))

print("--- Estudo de Hiperparâmetros (Árvores de Decisão) ---")

for crit in CRITERIA:
    acc_list = []
    prec_list = []
    rec_list = []

    for d in depths:
        # Nota: random_state=42 garante resultados reproduzíveis
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, random_state=42)
        clf.fit(trnX, trnY)
        
        # Previsão no Teste
        y_pred = clf.predict(tstX)

        acc = accuracy_score(tstY, y_pred)
        # Atenção: Usamos average="weighted" como nos scripts anteriores para maior consistência
        prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
        rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (crit, d)

        print(f"max_depth={d}, criterion={crit} -> "
              f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    values_accuracy[crit] = acc_list
    values_precision[crit] = prec_list
    values_recall[crit] = rec_list

# --- Plotagem dos Resultados (Gráfico sugerido: Estudo de Hiperparâmetros) ---

# Plot Accuracy
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(depths, values_accuracy,
    title="Decision Tree Hyperparameters - Accuracy",
    xlabel="Profundidade Máxima (max_depth)", ylabel="Accuracy", percentage=True)
plt.tight_layout() # Ajusta o layout
savefig("images/dt_hyperparameters_accuracy.png")
close()

# Plot Precision
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(depths, values_precision,
    title="Decision Tree Hyperparameters - Precision",
    xlabel="Profundidade Máxima (max_depth)", ylabel="Precision", percentage=True)
plt.tight_layout() # Ajusta o layout
savefig("images/dt_hyperparameters_precision.png")
close()

# Plot Recall
figure(figsize=(10, 6)) # Aumenta o tamanho da figura
plot_multiline_chart(depths, values_recall,
    title="Decision Tree Hyperparameters - Recall",
    xlabel="Profundidade Máxima (max_depth)", ylabel="Recall", percentage=True)
plt.tight_layout() # Ajusta o layout
savefig("images/dt_hyperparameters_recall.png")
close()

# --- Best model performance e Overfitting Study ---
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print(f"\n--- Descrição do Melhor Modelo DT ---")
print(f"Hiperparâmetros encontrados: Max_Depth={best_params[1]}, Criterion={best_params[0]}")

print("\n--- Performance do Melhor Modelo ---")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# Overfitting Study
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 4)) # Tamanho razoável para o Overfitting
# Guardamos a referência do Axes (ax) para usar plt.bar_label
ax = plt.gca()
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'],
               title=f"Overfitting Study - DT (Max Depth={best_params[1]})", ylabel="Accuracy", percentage=True)
plt.bar_label(ax.containers[0], fmt='%.4f', fontsize=8) # NOVO: Adiciona rótulos no topo das barras
plt.tight_layout() # Ajusta o layout
savefig("images/dt_overfitting.png")
close()

# --- Variáveis' Importance ---
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Variáveis Mais Importantes ---")

    importance = pd.Series(best_model.feature_importances_, index=trnX.columns)
    
    top_n = 10
    top_importance = importance.sort_values(ascending=False).head(top_n)

    figure(figsize=(12, 6)) # Tamanho grande para acomodar labels longas
    # Guardamos a referência do Axes (ax) para usar plt.bar_label
    ax = plt.gca()
    plot_bar_chart(top_importance.index.to_list(), top_importance.values.tolist(),
                   title=f"Importância das {top_n} Melhores Variáveis (DT)",
                   ylabel="Importância", percentage=False)
    
    # Rotação e redução de fonte para labels do eixo X
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.bar_label(ax.containers[0], fmt='%.4f', fontsize=8) # NOVO: Adiciona rótulos no topo das barras
    plt.tight_layout()
    savefig("images/dt_feature_importance.png")
    close()
    
    print(top_importance)
    print(f"\nGráfico 'dt_feature_importance.png' (Importância) guardado na pasta 'images'.")

# --- Model Learnt (Visualização da Árvore) ---
# [cite_start]A visualização de árvores é a sugestão "Model learnt, when possible" [cite: 27]
# Limite a profundidade para que o gráfico seja legível.
viz_depth = 3
figure(figsize=(20, 10))
# Obter os nomes das classes como strings para visualização
class_names_str = [str(c) for c in labels]
plot_tree(best_model, feature_names=trnX.columns.tolist(), class_names=class_names_str,
          max_depth=viz_depth, filled=True, rounded=True, fontsize=8)
plt.title(f"Visualização da Decision Tree (Depth {viz_depth}) - Max Depth Real: {best_params[1]}")
plt.tight_layout()
savefig("images/dt_model_learnt.png")
close()
print(f"\nGráfico 'dt_model_learnt.png' (visualização da árvore) guardado na pasta 'images'.")


print("\nGráficos da Árvore de Decisão concluídos e guardados na pasta 'images'.")
