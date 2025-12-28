import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart, plot_bar_chart

# --- CONFIGURAÇÃO DE FONTE GLOBAL ---
plt.rcParams.update({'font.size': 14}) 
# ------------------------------------

# --- VARIÁVEIS ---
TARGET = "Cancelled"
TRAIN_FILE = "train_scaled.csv"
TEST_FILE = "test_scaled.csv"
FILE_TAG = "flights"

# Hiperparâmetros para estudo
ITERATIONS = [100, 300, 500, 700, 1000]
PENALTIES = ["l2", "l1"]

os.makedirs("images", exist_ok=True)

# --- 1. CARREGAR DADOS ---
try:
    trn_df = pd.read_csv(TRAIN_FILE)
    tst_df = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print("Erro: 'train_scaled.csv' e 'test_scaled.csv' não encontrados.")
    exit()

# --- 2. BLOCO CRÍTICO: REMOVER DATA LEAKAGE ---
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

# --- 3. SAMPLING (Importante para LR) ---
SAMPLE_SIZE = 20000 
if trn_df.shape[0] > SAMPLE_SIZE:
    print(f"[Aviso] Dataset grande ({trn_df.shape[0]}). A reduzir para {SAMPLE_SIZE}.")
    trn_sample = trn_df.sample(n=SAMPLE_SIZE, random_state=42)
    trnX = trn_sample.drop(columns=[TARGET])
    trnY = trn_sample[TARGET]
else:
    trnX = trn_df.drop(columns=[TARGET])
    trnY = trn_df[TARGET]

tstX = tst_df.drop(columns=[TARGET])
tstY = tst_df[TARGET]

if TARGET not in trn_df.columns or TARGET not in tst_df.columns:
    print(f"Erro: Coluna alvo '{TARGET}' não encontrada.")
    exit()

print(f"Dados prontos para LR: Train={len(trnX)}, Test={len(tstX)}")

# --- 4. ESTUDO DE HIPERPARÂMETROS ---
acc_values = {}
prec_values = {}
rec_values = {}

best_model, best_score, best_params = None, 0, None

print("\n--- Estudo de Hiperparâmetros (Regressão Logística) ---")

for pen in PENALTIES:
    solver_choice = "liblinear"
    
    acc_list = []
    prec_list = []
    rec_list = []
    
    print(f"Testing Penalty: {pen}...")
    
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

        except Exception as e:
            print(f"Erro ao treinar com Penalty={pen}, Iter={n_iter}: {e}")
            acc_list.append(0)
            prec_list.append(0)
            rec_list.append(0)


    acc_values[pen] = acc_list
    prec_values[pen] = prec_list
    rec_values[pen] = rec_list

# --- 5. PLOTAGEM DOS RESULTADOS ---

# Plot Accuracy
figure(figsize=(10, 6))
plot_multiline_chart(ITERATIONS, acc_values, title="Logistic Regression Accuracy",
                     xlabel="Número de Iterações", ylabel="Accuracy", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_lr_accuracy.png")
close()

# Plot Precision
figure(figsize=(10, 6))
plot_multiline_chart(ITERATIONS, prec_values, title="Logistic Regression Precision",
                     xlabel="Número de Iterações", ylabel="Precision", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_lr_precision.png")
close()

# Plot Recall
figure(figsize=(10, 6))
plot_multiline_chart(ITERATIONS, rec_values, title="Logistic Regression Recall",
                     xlabel="Número de Iterações", ylabel="Recall", percentage=True)
plt.tight_layout()
savefig(f"images/{FILE_TAG}_lr_recall.png")
close()

# --- 6. BEST MODEL PERFORMANCE ---
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


# --- 7. OVERFITTING STUDY ---
data_overfit = {
    'Dataset': ['Train', 'Test'],
    'Accuracy': [
        accuracy_score(trnY, y_trn_pred),
        accuracy_score(tstY, y_tst_pred)
    ]
}

figure(figsize=(6, 5))
ax = plt.gca()
plot_bar_chart(data_overfit['Dataset'], data_overfit['Accuracy'], title=f"Overfitting Study - LR", ylabel="Accuracy", percentage=True)

# Aumentar tamanho dos textos nas barras
for text in ax.texts:
    text.set_fontsize(14)

plt.tight_layout()
savefig(f"images/{FILE_TAG}_lr_overfitting.png")
close()


# --- 8. VARIÁVEIS IMPORTANCE (FIXED) ---
if hasattr(best_model, 'coef_'):
    print("\n--- Variáveis Mais Importantes ---")
    
    # Calcular magnitude (absoluta) para ordenar
    importance = pd.Series(best_model.coef_[0], index=trnX.columns)
    
    # Selecionar as top 10 features
    top_n = 10
    top_importance = importance.abs().sort_values(ascending=False).head(top_n)
    
    # Buscar os valores originais (+/-)
    original_values = importance.loc[top_importance.index]

    figure(figsize=(12, 6))
    ax = plt.gca()
    
    bars = ax.bar(original_values.index, original_values.values)
    
    plt.title(f"Importância das {top_n} Melhores Variáveis (Coeficientes)")
    plt.ylabel("Importância")
    plt.xlabel("Variáveis")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # --- CORREÇÃO DE FORMATAÇÃO MANUAL ---
    # Criamos uma lista de strings explicitamente com 4 casas decimais
    # Isto garante que o gráfico mostra "0.1234" e não arredonda para "0.12"
    labels_formatted = [f"{val:.4f}" for val in original_values.values]
    
    ax.bar_label(bars, labels=labels_formatted, padding=3, fontsize=10)
    
    plt.tight_layout()
    savefig(f"images/{FILE_TAG}_lr_feature_importance.png")
    close()
    
    print(top_importance)
    print(f"\nGráfico 'lr_feature_importance.png' guardado na pasta 'images'.")

print("\nGráficos da Regressão Logística concluídos.")