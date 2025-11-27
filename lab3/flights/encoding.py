from pandas import read_csv, DataFrame, concat
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dslabs_functions import get_variable_types
from scipy.sparse import csr_matrix
import os
from matplotlib.pyplot import figure, savefig, show, text, title
from dslabs_functions import plot_bar_chart, plot_confusion_matrix

print("=== STARTING SCRIPT ===\n")

# ================================================================
# 1. LOAD TRAIN/TEST FROM PREVIOUS SPLIT
# ================================================================
print("[1/7] Loading train/test datasets from previous split...")
train = read_csv("prepared_data/train_imputed.csv")
test  = read_csv("prepared_data/test_imputed.csv")
print(f"Train: {train.shape} | Test: {test.shape}\n")

target = "Cancelled"

# Encode target
print("Encoding target variable...")

# Como os dados são Booleanos (False/True), basta converter para int.
# False torna-se 0
# True torna-se 1
train[target] = train[target].astype(int)
test[target]  = test[target].astype(int)

print("Target encoded.\n")

# ================================================================
# 2. CONVERT DATETIME COLUMNS
# ================================================================
print("[2/7] Converting datetime columns...")
vars_types = get_variable_types(train)

if len(vars_types["date"]) > 0:
    for col in vars_types["date"]:
        print(f"  → Converting {col} into numeric components...")
        for df_ in [train, test]:
            df_[col] = pd.to_datetime(df_[col], errors="coerce")
            df_[f"{col}_year"] = df_[col].dt.year
            df_[f"{col}_month"] = df_[col].dt.month
            df_[f"{col}_day"] = df_[col].dt.day
            df_[f"{col}_weekday"] = df_[col].dt.weekday
            df_[f"{col}_hour"] = df_[col].dt.hour
    print("Dropping original datetime columns:", vars_types["date"])
    train = train.drop(columns=vars_types["date"])
    test  = test.drop(columns=vars_types["date"])
else:
    print("No datetime columns found.")
print("Datetime conversion complete.\n")

vars_types = get_variable_types(train)
print("Variable types detected:", vars_types, "\n")

# ================================================================
# 3. APPROACH A — ORDINAL ENCODING
# ================================================================
print("[3/7] Applying ORDINAL encoding (fully numeric for Naive Bayes)...")

train_A = train.copy()
test_A  = test.copy()
ordinal_encoding = {}

for var in vars_types["symbolic"]:
    if var == target:
        continue
    unique_vals = sorted(train_A[var].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    ordinal_encoding[var] = mapping
    train_A[var] = train_A[var].map(mapping)
    test_A[var]  = test_A[var].map(mapping).fillna(len(mapping))

object_cols = train_A.select_dtypes(include='object').columns.tolist()
for col in object_cols:
    unique_vals = sorted(train_A[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    train_A[col] = train_A[col].map(mapping)
    test_A[col]  = test_A[col].map(mapping)

print("Ordinal dataset ready for Naive Bayes.\n")

# ================================================================
# 4. APPROACH B — ONE-HOT ENCODING
# ================================================================
print("[4/7] Applying ONE-HOT encoding (train/test consistent)...")
vars_to_dummies = [v for v in vars_types["symbolic"] if v != target]

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float")
X_train_dummies = enc.fit_transform(train[vars_to_dummies])
X_test_dummies  = enc.transform(test[vars_to_dummies])

train_B = concat([train.drop(vars_to_dummies, axis=1),
                  DataFrame(X_train_dummies, columns=enc.get_feature_names_out(vars_to_dummies), index=train.index)],
                 axis=1)
test_B = concat([test.drop(vars_to_dummies, axis=1),
                 DataFrame(X_test_dummies, columns=enc.get_feature_names_out(vars_to_dummies), index=test.index)],
                axis=1)

object_cols_B = train_B.select_dtypes(include='object').columns.tolist()
for col in object_cols_B:
    unique_vals = sorted(train_B[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    train_B[col] = train_B[col].map(mapping)
    test_B[col]  = test_B[col].map(mapping)


# --- SUBSTITUI O BLOCO FINAL DA SECÇÃO 4 POR ISTO ---

# 1. Converter tudo para numérico no Pandas e preencher NaNs com 0
# (Isto garante que textos viram NaN e depois 0)
train_B_numeric = train_B.drop(target, axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
test_B_numeric = test_B.drop(target, axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)

# 2. Debug: Verificar se sobrou alguma coluna teimosa (Opcional, mas útil)
non_numeric_cols = train_B_numeric.select_dtypes(include=['object']).columns
if len(non_numeric_cols) > 0:
    print(f"⚠️ AVISO: Colunas ainda detetadas como objeto: {non_numeric_cols}")

# 3. Criação da matriz esparsa FORÇANDO o tipo float
# O .astype(float) aqui é o segredo para corrigir o erro 'dtype O'
X_train_B_sparse = csr_matrix(train_B_numeric.values.astype(float))
X_test_B_sparse  = csr_matrix(test_B_numeric.values.astype(float))

print("One-hot encoding completed.\n")

print("One-hot encoding completed.\n")

# ================================================================
# 5. TRAIN MODELS
# ================================================================
print("[5/7] Training Naive Bayes + KNN on both datasets...\n")

def evaluate(train_df, test_df, label_col, knn_sparse=None):
    X_train = train_df.drop(label_col, axis=1)
    y_train = train_df[label_col]
    X_test = test_df.drop(label_col, axis=1)
    y_test = test_df[label_col]

    results = {}
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)
    results["NB_accuracy"] = accuracy_score(y_test, pred_nb)
    results["NB_predictions"] = pred_nb

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if knn_sparse is None:
        knn.fit(X_train, y_train)
        pred_knn = knn.predict(X_test)
    else:
        knn.fit(knn_sparse[0], y_train)
        pred_knn = knn.predict(knn_sparse[1])
    results["KNN_accuracy"] = accuracy_score(y_test, pred_knn)
    results["KNN_predictions"] = pred_knn

    return results, y_test

resA, y_testA = evaluate(train_A, test_A, target)
resB, y_testB = evaluate(train_B, test_B, target, knn_sparse=(X_train_B_sparse, X_test_B_sparse))

# ================================================================
# 6. SELECT BEST APPROACH AND SAVE DATASETS
# ================================================================
best = "A" if max(resA["NB_accuracy"], resA["KNN_accuracy"]) > \
               max(resB["NB_accuracy"], resB["KNN_accuracy"]) else "B"
print(f"\n>>> BEST APPROACH IS: {best}\n")

if best == "A":
    train_best, test_best = train_A, test_A
else:
    train_best, test_best = train_B, test_B

os.makedirs("prepared_data", exist_ok=True)
train_best.to_csv("prepared_data/train_encoded.csv", index=False)
test_best.to_csv("prepared_data/test_encoded.csv", index=False)
print("Saved best encoded datasets for next step.\n")

# ================================================================
# 7. CONFUSION MATRICES
# ================================================================
if best == "A":
    y_true = y_testA
    pred_nb = resA["NB_predictions"]
    pred_knn = resA["KNN_predictions"]
else:
    y_true = y_testB
    pred_nb = resB["NB_predictions"]
    pred_knn = resB["KNN_predictions"]

cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

print("Naive Bayes Confusion Matrix:\n", cm_nb)
print("KNN Confusion Matrix:\n", cm_knn)

# ================================================================
# 8. PLOTS
# ================================================================
os.makedirs("images", exist_ok=True)

labels = ["NB (Ordinal)", "NB (One-Hot)", "KNN (Ordinal)", "KNN (One-Hot)"]
scores = [resA["NB_accuracy"], resB["NB_accuracy"], resA["KNN_accuracy"], resB["KNN_accuracy"]]

figure(figsize=(7,4))
plot_bar_chart(labels, scores)
for i, v in enumerate(scores):
    text(i, v + 0.01, f"{v:.3f}", ha='center')
title("Model Accuracy per Algorithm and Encoding Approach")
savefig("images/encoding_performance.png")
show()

best_scores = [max(resA["NB_accuracy"], resA["KNN_accuracy"]),
               max(resB["NB_accuracy"], resB["KNN_accuracy"])]
figure(figsize=(5,3))
plot_bar_chart(["Ordinal", "One-Hot"], best_scores)
for i, v in enumerate(best_scores):
    text(i, v + 0.01, f"{v:.3f}", ha='center')
title("Best Approach by Maximum Accuracy")
savefig("images/encoding_best_approach.png")
show()

figure()
plot_confusion_matrix(cm_nb, ["0","1"])
title(f"Naive Bayes Confusion Matrix — Best Approach ({best})")
savefig("images/encoding_cm_nb.png")
show()

figure()
plot_confusion_matrix(cm_knn, ["0","1"])
title(f"KNN Confusion Matrix — Best Approach ({best})")
savefig("images/encoding_cm_knn.png")
show()
