#!/usr/bin/env python3
from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
import os
from math import ceil
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dslabs_functions import plot_bar_chart, plot_confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING FEATURE SELECTION SCRIPT ===\n")

# ----------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------
TARGET = "crash_type"
# thresholds you can tweak if desired
LOW_VAR_THRESHOLD = 1.0     # variance threshold (changeable)
REDUNDANCY_THRESHOLD = 0.5  # correlation threshold (changeable)

# ----------------------------------------------------
# 1. LOAD balanced datasets (output of balancing step)
# ----------------------------------------------------
train = read_csv("prepared_data/train_balanced.csv")
test  = read_csv("prepared_data/test_balanced.csv")
print(f"Train: {train.shape} | Test: {test.shape}\n")

# ----------------------------------------------------
# 2. helper functions (based on DSLabs examples)
# ----------------------------------------------------
def select_low_variance_variables(data: DataFrame, max_threshold: float, target: str = TARGET) -> list:
    """Return list of variables with variance < max_threshold (excluding target)."""
    summary5 = data.describe()
    # variance ≈ std^2
    std_series = summary5.loc["std"]
    vars2drop = std_series[std_series * std_series < max_threshold].index.tolist()
    if target in vars2drop:
        vars2drop.remove(target)
    return vars2drop


def select_redundant_variables(data: DataFrame, min_threshold: float = 0.90, target: str = TARGET) -> list:
    """Return list of redundant variables (one from each correlated pair >= min_threshold)."""
    df = data.drop(columns=[target], errors="ignore")
    corr_matrix = abs(df.corr())
    variables = corr_matrix.columns
    vars2drop = []
    for v1 in variables:
        vars_corr = corr_matrix[v1].loc[corr_matrix[v1] >= min_threshold].copy()
        if v1 in vars_corr.index:
            vars_corr.drop(v1, inplace=True)
        if len(vars_corr) > 0:
            for v2 in vars_corr.index:
                if v2 not in vars2drop:
                    vars2drop.append(v2)
    return vars2drop


def evaluate_dataset(train_df: DataFrame, test_df: DataFrame, target: str = TARGET):
    """Train NB + KNN and return accuracies + predictions + y_true."""
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target]

    # Ensure numeric arrays (should already be, but just in case)
    # If categorical columns remain, sklearn will raise errors. This script assumes
    # previous encoding has been applied (pipeline order: encode -> impute -> outliers -> scale -> balance -> feature selection).
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, pred_nb)

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, pred_knn)

    return {"NB_accuracy": acc_nb, "NB_predictions": pred_nb, "KNN_accuracy": acc_knn, "KNN_predictions": pred_knn, "y_test": y_test}


# ----------------------------------------------------
# 3. Approach A: Low-variance removal
# ----------------------------------------------------
print("[A] Low-variance feature removal")
vars_lowvar = select_low_variance_variables(train, max_threshold=LOW_VAR_THRESHOLD, target=TARGET)
print("  Variables to drop (low variance):", vars_lowvar)

train_A = train.drop(columns=vars_lowvar, errors="ignore")
test_A  = test.drop(columns=vars_lowvar, errors="ignore")
print("  Shapes after low-var removal:", train_A.shape, test_A.shape)

resA = evaluate_dataset(train_A, test_A, target=TARGET)
print("  NB acc (low-var):", resA["NB_accuracy"])
print("  KNN acc(low-var):", resA["KNN_accuracy"], "\n")


# ----------------------------------------------------
# 4. Approach B: Redundancy removal (correlated features)
# ----------------------------------------------------
print("[B] Redundancy (correlation) based removal")
vars_redundant = select_redundant_variables(train, min_threshold=REDUNDANCY_THRESHOLD, target=TARGET)
print("  Variables to drop (redundant):", vars_redundant)

train_B = train.drop(columns=vars_redundant, errors="ignore")
test_B  = test.drop(columns=vars_redundant, errors="ignore")
print("  Shapes after redundancy removal:", train_B.shape, test_B.shape)

resB = evaluate_dataset(train_B, test_B, target=TARGET)
print("  NB acc (redundant):", resB["NB_accuracy"])
print("  KNN acc(redundant):", resB["KNN_accuracy"], "\n")


# ----------------------------------------------------
# 5. Select best approach
# ----------------------------------------------------
# Use maximum of NB and KNN accuracies for each approach to select (same logic as before)
bestA_score = max(resA["NB_accuracy"], resA["KNN_accuracy"])
bestB_score = max(resB["NB_accuracy"], resB["KNN_accuracy"])

best = "A" if bestA_score >= bestB_score else "B"
print(f">>> BEST FEATURE SELECTION APPROACH: {best} (A_score={bestA_score:.4f} vs B_score={bestB_score:.4f})\n")

if best == "A":
    train_best, test_best = train_A, test_A
    res_best = resA
else:
    train_best, test_best = train_B, test_B
    res_best = resB

# Save selected datasets for the next pipeline step
os.makedirs("prepared_data", exist_ok=True)
train_best.to_csv("prepared_data/train_selected.csv", index=False)
test_best.to_csv("prepared_data/test_selected.csv", index=False)
print("Saved selected datasets to prepared_data/train_selected.csv and test_selected.csv\n")

# ----------------------------------------------------
# 6. PLOTS — consistent with your previous scripts
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# 6A — Model Performance Comparison (NB & KNN for both approaches)
labels = ["NB (LowVar)", "NB (Redun)", "KNN (LowVar)", "KNN (Redun)"]
scores = [resA["NB_accuracy"], resB["NB_accuracy"], resA["KNN_accuracy"], resB["KNN_accuracy"]]

plt.figure(figsize=(7,4))
plot_bar_chart(labels, scores, title="Feature Selection — Model Performance")
ax = plt.gca()
# rotate x labels for readability and align right
ax.set_xticklabels(labels, rotation=45, ha="right")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/feature_selection_performance.png")
plt.close()
print("Saved: images/feature_selection_performance.png")

# 6B — Best approach bar chart (max accuracy per approach)
best_scores = [max(resA["NB_accuracy"], resA["KNN_accuracy"]), max(resB["NB_accuracy"], resB["KNN_accuracy"])]
plt.figure(figsize=(6,4))
plot_bar_chart(["LowVariance", "Redundancy"], best_scores, title="Feature Selection — Best Approach (Max Accuracy)")
ax = plt.gca()
ax.set_xticklabels(["LowVariance", "Redundancy"], rotation=45, ha="right")
for i, v in enumerate(best_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/feature_selection_best.png")
plt.close()
print("Saved: images/feature_selection_best.png")

# ----------------------------------------------------
# 6C — Confusion matrices for best approach
# ----------------------------------------------------
y_true = res_best["y_test"]
pred_nb = res_best["NB_predictions"]
pred_knn = res_best["KNN_predictions"]

# NB CM
plt.figure()
plot_confusion_matrix(confusion_matrix(y_true, pred_nb), sorted(y_true.unique()))
plt.title(f"Naive Bayes Confusion Matrix — Best FS ({best})")
plt.savefig("images/feature_selection_cm_nb.png")
plt.close()
print("Saved: images/feature_selection_cm_nb.png")

# KNN CM
plt.figure()
plot_confusion_matrix(confusion_matrix(y_true, pred_knn), sorted(y_true.unique()))
plt.title(f"KNN Confusion Matrix — Best FS ({best})")
plt.savefig("images/feature_selection_cm_knn.png")
plt.close()
print("Saved: images/feature_selection_cm_knn.png")


print("\n=== FEATURE SELECTION COMPLETE ===")
