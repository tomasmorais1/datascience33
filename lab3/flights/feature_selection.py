#!/usr/bin/env python3
from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dslabs_functions import plot_bar_chart, plot_confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING FEATURE SELECTION SCRIPT ===\n")

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
TARGET = "Cancelled"
LOW_VAR_THRESHOLD = 0.001
REDUNDANCY_THRESHOLD = 0.90

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
train = read_csv("prepared_data/train_balanced.csv")
test  = read_csv("prepared_data/test_balanced.csv")
print(f"Train: {train.shape} | Test: {test.shape}\n")

# Ensure all columns except target are numeric
for col in train.columns:
    if col != TARGET:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col]  = pd.to_numeric(test[col], errors="coerce")

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def select_low_variance_variables(data, threshold, target):
    std_series = data.describe().loc["std"]
    vars2drop = std_series[(std_series ** 2) < threshold].index.tolist()
    return [v for v in vars2drop if v != target]

def select_redundant_variables(data, threshold, target):
    df = data.drop(columns=[target])
    corr = abs(df.corr())
    vars2drop = []
    for v1 in corr.columns:
        for v2 in corr.columns:
            if v1 != v2 and corr.loc[v1, v2] >= threshold:
                if v2 not in vars2drop:
                    vars2drop.append(v2)
    return vars2drop

def evaluate_dataset(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)

    return {
        "NB_acc": accuracy_score(y_test, pred_nb),
        "KNN_acc": accuracy_score(y_test, pred_knn),
        "NB_pred": pred_nb,
        "KNN_pred": pred_knn,
        "y_test": y_test
    }

# ----------------------------------------------------
# APPROACH A — Low Variance
# ----------------------------------------------------
vars_lowvar = select_low_variance_variables(train, LOW_VAR_THRESHOLD, TARGET)
train_A = train.drop(columns=vars_lowvar)
test_A  = test.drop(columns=vars_lowvar)
resA = evaluate_dataset(train_A, test_A)

# ----------------------------------------------------
# APPROACH B — Redundancy
# ----------------------------------------------------
vars_redundant = select_redundant_variables(train, REDUNDANCY_THRESHOLD, TARGET)
train_B = train.drop(columns=vars_redundant)
test_B  = test.drop(columns=vars_redundant)
resB = evaluate_dataset(train_B, test_B)

# ----------------------------------------------------
# SELECT BEST
# ----------------------------------------------------
bestA = max(resA["NB_acc"], resA["KNN_acc"])
bestB = max(resB["NB_acc"], resB["KNN_acc"])
best = "A" if bestA >= bestB else "B"

train_best = train_A if best == "A" else train_B
test_best  = test_A  if best == "A" else test_B
res_best   = resA if best == "A" else resB

train_best.to_csv("prepared_data/train_selected.csv", index=False)
test_best.to_csv("prepared_data/test_selected.csv", index=False)

print(f">>> BEST FEATURE SELECTION APPROACH = {best}\n")

# ----------------------------------------------------
# PLOTS (fixed text positioning)
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

scores = [resA["NB_acc"], resB["NB_acc"], resA["KNN_acc"], resB["KNN_acc"]]
labels = ["NB (LowVar)", "NB (Redun)", "KNN (LowVar)", "KNN (Redun)"]

plt.figure(figsize=(7,4))
plot_bar_chart(labels, scores, title="Feature Selection Performance")
for i, v in enumerate(scores):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.savefig("images/feature_selection_performance.png")
plt.close()

# Confusion matrices
plt.figure()
plot_confusion_matrix(confusion_matrix(res_best["y_test"], res_best["NB_pred"]), ["0","1"])
plt.title(f"NB Confusion Matrix — FS ({best})")
plt.savefig("images/feature_selection_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(confusion_matrix(res_best["y_test"], res_best["KNN_pred"]), ["0","1"])
plt.title(f"KNN Confusion Matrix — FS ({best})")
plt.savefig("images/feature_selection_cm_knn.png")
plt.close()

print("\n=== FEATURE SELECTION COMPLETE ===")
