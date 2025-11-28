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
train = pd.read_csv("prepared_data/train_raw.csv")
test  = pd.read_csv("prepared_data/test_raw.csv")

print(f"Train: {train.shape} | Test: {test.shape}")

TARGET = "Cancelled"

# ----------------------------------------------------
# SAFETY: Ensure target has both classes
# ----------------------------------------------------
def check_target(df):
    print("\nTarget distribution:")
    print(df[TARGET].value_counts())
    if df[TARGET].nunique() < 2:
        print("\n❌ ERROR: Only one class in target! Fix sampling.")
        exit(1)

check_target(train)

# ----------------------------------------------------
# 2. Replace blanks with NaN
# ----------------------------------------------------
print("\nReplacing empty cells and UNKNOWN with NaN ...")

def clean(df):
    df = df.replace(["UNKNOWN", "", " ", "  ", "   "], np.nan)
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip()=="" else x)
    return df

train = clean(train)
test  = clean(test)

# Ensure target never becomes NaN
train[TARGET] = train[TARGET].fillna(train[TARGET].mode()[0])
test[TARGET]  = test[TARGET].fillna(test[TARGET].mode()[0])

# ----------------------------------------------------
# Identify variable types
# ----------------------------------------------------
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()

if TARGET in categorical_cols: categorical_cols.remove(TARGET)
if TARGET in numeric_cols: numeric_cols.remove(TARGET)

# ----------------------------------------------------
# Helper: evaluate model performance
# ----------------------------------------------------
def evaluate_dataset(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()

    if categorical_cols:
        X_train_enc[categorical_cols] = enc.fit_transform(X_train[categorical_cols])
        X_test_enc[categorical_cols]  = enc.transform(X_test[categorical_cols])

    # NB
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
# 3. Approach A — Median + Most Frequent
# ----------------------------------------------------
print("\n[1] Applying Most Frequent / Median Imputation...")

train_A = train.copy()
test_A  = test.copy()

if categorical_cols:
    imp_cat = SimpleImputer(strategy="most_frequent")
    train_A[categorical_cols] = imp_cat.fit_transform(train_A[categorical_cols])
    test_A[categorical_cols]  = imp_cat.transform(test_A[categorical_cols])

if numeric_cols:
    imp_num = SimpleImputer(strategy="median")
    train_A[numeric_cols] = imp_num.fit_transform(train_A[numeric_cols])
    test_A[numeric_cols]  = imp_num.transform(test_A[numeric_cols])

accA_nb, accA_knn, predA_nb, predA_knn, yA_test = evaluate_dataset(train_A, test_A)

# ----------------------------------------------------
# 4. Approach B — KNN Imputation (subset fitting)
# ----------------------------------------------------
print("[2] Applying KNN Imputation (subset-based)...")

train_B = train.copy()
test_B  = test.copy()

subset_size = min(12000, len(train_B))   # safe for all datasets
train_subset = train_B[numeric_cols].sample(subset_size, random_state=42)

imputer_knn = KNNImputer(n_neighbors=5)
imputer_knn.fit(train_subset)   # fast fitting

train_B[numeric_cols] = imputer_knn.transform(train_B[numeric_cols])
test_B[numeric_cols]  = imputer_knn.transform(test_B[numeric_cols])

if categorical_cols:
    imp_cat2 = SimpleImputer(strategy="most_frequent")
    train_B[categorical_cols] = imp_cat2.fit_transform(train_B[categorical_cols])
    test_B[categorical_cols]  = imp_cat2.transform(test_B[categorical_cols])

accB_nb, accB_knn, predB_nb, predB_knn, yB_test = evaluate_dataset(train_B, test_B)

# ----------------------------------------------------
# 5. Select Best Approach
# ----------------------------------------------------
print("\n=== RESULTS ===")
print(f"A: NB={accA_nb:.4f}, KNN={accA_knn:.4f}")
print(f"B: NB={accB_nb:.4f}, KNN={accB_knn:.4f}")

best = "A" if accA_nb >= accB_nb else "B"
print(f"\n>>> BEST APPROACH = {best}\n")

if best == "A":
    best_train, best_test = train_A, test_A
    y_true, pred_nb, pred_knn = yA_test, predA_nb, predA_knn
else:
    best_train, best_test = train_B, test_B
    y_true, pred_nb, pred_knn = yB_test, predB_nb, predB_knn

# ----------------------------------------------------
# 6. PLOTS — FIXED LABELS
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

label_A = "Most Frequent + Median"
label_B = "KNN Imputation"

# Accuracy comparison
labels = [f"NB ({label_A})", f"NB ({label_B})",
          f"KNN ({label_A})", f"KNN ({label_B})"]
scores = [accA_nb, accB_nb, accA_knn, accB_knn]

plt.figure(figsize=(8,5))
plot_bar_chart(labels, scores, title="Missing Value Imputation — Model Performance")

# FIXED labels – correct values shown above bars
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("images/mvi_performance.png")
plt.close()

# Best approach (NB only)
plt.figure(figsize=(6,4))
plot_bar_chart([label_A, label_B], [accA_nb, accB_nb],
               title="Best Approach (NB Accuracy)")
plt.tight_layout()
plt.savefig("images/mvi_best_approach.png")
plt.close()

# Confusion matrices
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

plt.figure()
plot_confusion_matrix(cm_nb, ["0","1"])
plt.title(f"Naive Bayes — Best Approach ({best})")
plt.savefig("images/mvi_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(cm_knn, ["0","1"])
plt.title(f"KNN — Best Approach ({best})")
plt.savefig("images/mvi_cm_knn.png")
plt.close()

# ----------------------------------------------------
# 7. Save final outputs
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_imputed.csv", index=False)
best_test.to_csv("prepared_data/test_imputed.csv", index=False)

print("\n=== MISSING VALUE IMPUTATION COMPLETE ===")
