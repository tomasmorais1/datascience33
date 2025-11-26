#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from dslabs_functions import plot_bar_chart, plot_confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING MISSING VALUE IMPUTATION SCRIPT ===\n")

# ----------------------------------------------------
# 1. Load split datasets
# ----------------------------------------------------
train_path = "prepared_data/train_raw.csv"
test_path = "prepared_data/test_raw.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train: {train.shape} | Test: {test.shape}")

# ----------------------------------------------------
# 2. Convert UNKNOWN → NaN
# ----------------------------------------------------
print("\nReplacing 'UNKNOWN' with NaN ...")
train = train.replace("UNKNOWN", np.nan)
test = test.replace("UNKNOWN", np.nan)

# Identify types
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()

# Target must NOT be imputed
TARGET = "crash_type"
if TARGET in categorical_cols:
    categorical_cols.remove(TARGET)
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

# ----------------------------------------------------
# Helper: evaluate NB + KNN for an imputed dataset
# ----------------------------------------------------
def evaluate_dataset(train_df, test_df):
    """Returns accuracies and predictions for NB and KNN."""
    
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Ordinal encode categorical
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    if len(categorical_cols) > 0:
        X_train_enc[categorical_cols] = enc.fit_transform(X_train[categorical_cols])
        X_test_enc[categorical_cols] = enc.transform(X_test[categorical_cols])

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_enc, y_train)
    pred_nb = nb.predict(X_test_enc)
    acc_nb = accuracy_score(y_test, pred_nb)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_enc, y_train)
    pred_knn = knn.predict(X_test_enc)
    acc_knn = accuracy_score(y_test, pred_knn)

    return acc_nb, acc_knn, pred_nb, pred_knn, y_test


# ----------------------------------------------------
# 3. Approach A — Most Frequent + Median
# ----------------------------------------------------
print("\n[1] Applying Most Frequent / Median Imputation...")

train_A = train.copy()
test_A = test.copy()

# Categorical: most frequent
if categorical_cols:
    imp_cat = SimpleImputer(strategy="most_frequent")
    train_A[categorical_cols] = imp_cat.fit_transform(train_A[categorical_cols])
    test_A[categorical_cols] = imp_cat.transform(test_A[categorical_cols])

# Numeric: median
if numeric_cols:
    imp_num = SimpleImputer(strategy="median")
    train_A[numeric_cols] = imp_num.fit_transform(train_A[numeric_cols])
    test_A[numeric_cols] = imp_num.transform(test_A[numeric_cols])

accA_nb, accA_knn, predA_nb, predA_knn, yA_test = evaluate_dataset(train_A, test_A)


# ----------------------------------------------------
# 4. Approach B — KNN Imputation (numeric only)
# ----------------------------------------------------
print("[2] Applying KNN Imputation...")

train_B = train.copy()
test_B = test.copy()

# KNN for numeric
if numeric_cols:
    imputer_knn = KNNImputer(n_neighbors=5)
    train_B[numeric_cols] = imputer_knn.fit_transform(train_B[numeric_cols])
    test_B[numeric_cols] = imputer_knn.transform(test_B[numeric_cols])

# Categorical still require most frequent
if categorical_cols:
    imp_cat2 = SimpleImputer(strategy="most_frequent")
    train_B[categorical_cols] = imp_cat2.fit_transform(train_B[categorical_cols])
    test_B[categorical_cols] = imp_cat2.transform(test_B[categorical_cols])

accB_nb, accB_knn, predB_nb, predB_knn, yB_test = evaluate_dataset(train_B, test_B)


# ----------------------------------------------------
# 5. Select Best Approach
# ----------------------------------------------------
print("\n=== RESULTS ===")
print(f"Approach A – Most Frequent:\n  NB Accuracy : {accA_nb}\n  KNN Accuracy: {accA_knn}\n")
print(f"Approach B – KNN:\n  NB Accuracy : {accB_nb}\n  KNN Accuracy: {accB_knn}\n")

best_approach = "A" if accA_nb >= accB_nb else "B"
print(f">>> BEST APPROACH IS: {best_approach}\n")

if best_approach == "A":
    best_train, best_test = train_A, test_A
    y_true = yA_test
    pred_nb = predA_nb
    pred_knn = predA_knn
else:
    best_train, best_test = train_B, test_B
    y_true = yB_test
    pred_nb = predB_nb
    pred_knn = predB_knn


# ----------------------------------------------------
# 6. PLOTS — consistent with encoding.py, updated labels
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# Human-readable labels for approaches
label_A = "Most Frequent Imputation"
label_B = "KNN Imputation"

# 6A — MODEL PERFORMANCE COMPARISON (NB & KNN per approach)
labels = [f"NB ({label_A})", f"NB ({label_B})", f"KNN ({label_A})", f"KNN ({label_B})"]
scores = [accA_nb, accB_nb, accA_knn, accB_knn]

plt.figure(figsize=(8,5))
plot_bar_chart(labels, scores, title="Missing Value Imputation — Model Performance")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/mvi_performance.png")
plt.close()

# 6B — BEST APPROACH (NB metric)
best_scores = [accA_nb, accB_nb]

plt.figure(figsize=(6,4))
plot_bar_chart([label_A, label_B], best_scores,
               title="Missing Values — Best Approach (NB Accuracy)")
for i, v in enumerate(best_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/mvi_best_approach.png")
plt.close()

# 6C — CONFUSION MATRICES FOR BEST APPROACH
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

plt.figure()
plot_confusion_matrix(cm_nb, ["0","1"])
plt.title(f"Naive Bayes Confusion Matrix — Best Approach ({label_A if best_approach=='A' else label_B})")
plt.savefig("images/mvi_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(cm_knn, ["0","1"])
plt.title(f"KNN Confusion Matrix — Best Approach ({label_A if best_approach=='A' else label_B})")
plt.savefig("images/mvi_cm_knn.png")
plt.close()

print("\nSaved charts into /images")



# ----------------------------------------------------
# 7. Save final imputed datasets
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_imputed.csv", index=False)
best_test.to_csv("prepared_data/test_imputed.csv", index=False)

print("\n=== MISSING VALUE IMPUTATION COMPLETE ===")
print("Saved:")
print("  prepared_data/train_imputed.csv")
print("  prepared_data/test_imputed.csv\n")
