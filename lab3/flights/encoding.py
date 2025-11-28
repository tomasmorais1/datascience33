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
# 1. LOAD TRAIN/TEST
# ================================================================
train = read_csv("prepared_data/train_imputed.csv")
test  = read_csv("prepared_data/test_imputed.csv")

target = "Cancelled"
print("[1] Loaded:", train.shape, test.shape)

# ================================================================
# 2. TARGET ENCODING
# ================================================================
print("[2] Encoding target...")
train[target] = train[target].astype(int)
test[target]  = test[target].astype(int)
print("Target encoded.\n")

# ================================================================
# 3. CONVERT DATETIME COLUMNS
# ================================================================
print("[3] Converting datetime columns...")
vars_types = get_variable_types(train)

if vars_types["date"]:
    for col in vars_types["date"]:
        print(f"  → Expanding {col}...")
        for df in (train, test):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_hour"] = df[col].dt.hour

    train = train.drop(columns=vars_types["date"])
    test  = test.drop(columns=vars_types["date"])

print("Datetime processed.\n")

vars_types = get_variable_types(train)
print("Variable types detected:", vars_types, "\n")

# ================================================================
# 4. APPROACH A — ORDINAL ENCODING
# ================================================================
print("[4] Performing ORDINAL encoding...")

train_A = train.copy()
test_A  = test.copy()

for col in vars_types["symbolic"]:
    if col == target:
        continue
    unique_vals = sorted(train_A[col].dropna().unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}

    train_A[col] = train_A[col].map(mapping)
    test_A[col]  = test_A[col].map(mapping).fillna(len(mapping))

print("Ordinal encoding complete.\n")

# ================================================================
# 5. APPROACH B — ONE-HOT ENCODING
# ================================================================
print("[5] Performing ONE-HOT encoding...")

symbolic_cols = [v for v in vars_types["symbolic"] if v != target]

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
train_dums = enc.fit_transform(train[symbolic_cols])
test_dums  = enc.transform(test[symbolic_cols])

train_B = concat([
    train.drop(columns=symbolic_cols),
    DataFrame(train_dums, columns=enc.get_feature_names_out(symbolic_cols), index=train.index)
], axis=1)

test_B = concat([
    test.drop(columns=symbolic_cols),
    DataFrame(test_dums, columns=enc.get_feature_names_out(symbolic_cols), index=test.index)
], axis=1)

# Convert any remaining object columns to category codes
for df in (train_B, test_B):
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("category").cat.codes

# SciPy sparse requires numeric types
if "Diverted" in train_B.columns:
    train_B["Diverted"] = train_B["Diverted"].astype(int)
    test_B["Diverted"]  = test_B["Diverted"].astype(int)

# Build sparse matrices
X_train_B_sparse = csr_matrix(train_B.drop(columns=[target]).values)
X_test_B_sparse  = csr_matrix(test_B.drop(columns=[target]).values)

print("One-hot encoding complete.\n")

# ================================================================
# 6. MODEL EVALUATION
# ================================================================
print("[6] Evaluating Naive Bayes & KNN...\n")

def evaluate(df_train, df_test, knn_sparse=None):
    X_train, y_train = df_train.drop(columns=[target]), df_train[target]
    X_test,  y_test  = df_test.drop(columns=[target]),  df_test[target]

    results = {}

    # NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, pred_nb)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if knn_sparse is None:
        knn.fit(X_train, y_train)
        pred_knn = knn.predict(X_test)
    else:
        knn.fit(knn_sparse[0], y_train)
        pred_knn = knn.predict(knn_sparse[1])

    acc_knn = accuracy_score(y_test, pred_knn)

    return acc_nb, acc_knn, pred_nb, pred_knn, y_test


# Evaluate both encoding strategies
accA_nb, accA_knn, predA_nb, predA_knn, yA_test = evaluate(train_A, test_A)
accB_nb, accB_knn, predB_nb, predB_knn, yB_test = evaluate(
    train_B, test_B, knn_sparse=(X_train_B_sparse, X_test_B_sparse)
)

print(f"Ordinal Accuracies  →  NB={accA_nb:.4f}, KNN={accA_knn:.4f}")
print(f"One-Hot Accuracies →  NB={accB_nb:.4f}, KNN={accB_knn:.4f}\n")

# ================================================================
# 7. SELECT BEST APPROACH
# ================================================================
best = "A" if max(accA_nb, accA_knn) > max(accB_nb, accB_knn) else "B"
print(f">>> BEST APPROACH = {best}\n")

if best == "A":
    train_best, test_best = train_A, test_A
    pred_nb, pred_knn, y_true = predA_nb, predA_knn, yA_test
else:
    train_best, test_best = train_B, test_B
    pred_nb, pred_knn, y_true = predB_nb, predB_knn, yB_test

train_best.to_csv("prepared_data/train_encoded.csv", index=False)
test_best.to_csv("prepared_data/test_encoded.csv", index=False)

print("Saved encoded datasets.\n")

# ================================================================
# 8. PLOTS
# ================================================================
os.makedirs("images", exist_ok=True)

# Performance plot
labels = ["NB (Ordinal)", "NB (One-Hot)", "KNN (Ordinal)", "KNN (One-Hot)"]
scores = [accA_nb, accB_nb, accA_knn, accB_knn]

figure(figsize=(7,4))
plot_bar_chart(labels, scores)
for i, v in enumerate(scores):
    text(i, v + 0.01, f"{v:.3f}", ha='center')
title("Model Accuracy by Encoding")
savefig("images/encoding_performance.png")
show()

# Best approach comparison
best_scores = [
    max(accA_nb, accA_knn),
    max(accB_nb, accB_knn)
]

figure(figsize=(5,3))
plot_bar_chart(["Ordinal", "One-Hot"], best_scores)
for i, v in enumerate(best_scores):
    text(i, v + 0.01, f"{v:.3f}", ha='center')
title("Best Encoding Approach")
savefig("images/encoding_best_approach.png")
show()

# Confusion matrices
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

figure()
plot_confusion_matrix(cm_nb, ["0", "1"])
title(f"NB Confusion Matrix — Best: {best}")
savefig("images/encoding_cm_nb.png")
show()

figure()
plot_confusion_matrix(cm_knn, ["0","1"])
title(f"KNN Confusion Matrix — Best: {best}")
savefig("images/encoding_cm_knn.png")
show()

print("=== ENCODING COMPLETE ===")
