#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from dslabs_functions import get_variable_types, plot_bar_chart, plot_confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING SCALING SCRIPT ===\n")

# ----------------------------------------------------
# 1. Load encoded datasets
# ----------------------------------------------------
train = pd.read_csv("prepared_data/train_outliers.csv")
test  = pd.read_csv("prepared_data/test_outliers.csv")

TARGET = "Cancelled"

print(f"Train: {train.shape} | Test: {test.shape}")

# ----------------------------------------------------
# 2. Identify variables
# ----------------------------------------------------
vars_types = get_variable_types(train)

numeric_cols     = vars_types["numeric"].copy()
symbolic_cols    = vars_types["symbolic"].copy()
binary_cols      = vars_types["binary"].copy()

# Remove target
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)
if TARGET in symbolic_cols:
    symbolic_cols.remove(TARGET)
if TARGET in binary_cols:
    binary_cols.remove(TARGET)

# IMPORTANT: Do NOT scale binary variables
scale_cols = [col for col in numeric_cols if col not in binary_cols]

print("\nNumeric columns to scale:", scale_cols)
print("Categorical columns:", symbolic_cols)
print("Binary columns (not scaled):", binary_cols)

# ----------------------------------------------------
# 3. Scaling helper
# ----------------------------------------------------
def scale_dataset(train_df, test_df, scaler_type="standard"):
    train_copy = train_df.copy()
    test_copy  = test_df.copy()

    X_train_num = train_copy[scale_cols]
    X_test_num  = test_copy[scale_cols]

    # Select scaler
    if scaler_type == "standard":
        scaler = StandardScaler().fit(X_train_num)
    else:
        scaler = MinMaxScaler().fit(X_train_num)

    # Apply
    train_copy[scale_cols] = scaler.transform(X_train_num)
    test_copy[scale_cols]  = scaler.transform(X_test_num)

    return train_copy, test_copy

# ----------------------------------------------------
# 4. Evaluation helper (NB + KNN)
# ----------------------------------------------------
def evaluate(df_train, df_test):
    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]
    X_test  = df_test.drop(columns=[TARGET])
    y_test  = df_test[TARGET]

    # Encode categorical (ordinal)
    if len(symbolic_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[symbolic_cols] = enc.fit_transform(X_train[symbolic_cols])
        X_test[symbolic_cols]  = enc.transform(X_test[symbolic_cols])

    # NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, pred_nb)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, pred_knn)

    return acc_nb, acc_knn, pred_nb, pred_knn, y_test

# ----------------------------------------------------
# 5. Try StandardScaler
# ----------------------------------------------------
print("\n[1] Applying StandardScaler...")
train_std, test_std = scale_dataset(train, test, "standard")
acc_std_nb, acc_std_knn, pred_std_nb, pred_std_knn, y_std = evaluate(train_std, test_std)

# ----------------------------------------------------
# 6. Try MinMaxScaler
# ----------------------------------------------------
print("[2] Applying MinMaxScaler...")
train_mm, test_mm = scale_dataset(train, test, "minmax")
acc_mm_nb, acc_mm_knn, pred_mm_nb, pred_mm_knn, y_mm = evaluate(train_mm, test_mm)

# ----------------------------------------------------
# 7. Compare approaches
# ----------------------------------------------------
print("\n=== RESULTS ===")
print(f"StandardScaler  — NB: {acc_std_nb:.4f}, KNN: {acc_std_knn:.4f}")
print(f"MinMaxScaler    — NB: {acc_mm_nb:.4f}, KNN: {acc_mm_knn:.4f}")

best_scaler = (
    "standard"
    if max(acc_std_nb, acc_std_knn) >= max(acc_mm_nb, acc_mm_knn)
    else "minmax"
)

print(f">>> BEST SCALER: {best_scaler}\n")

if best_scaler == "standard":
    best_train, best_test = train_std, test_std
    y_true, pred_nb, pred_knn = y_std, pred_std_nb, pred_std_knn
else:
    best_train, best_test = train_mm, test_mm
    y_true, pred_nb, pred_knn = y_mm, pred_mm_nb, pred_mm_knn

# ----------------------------------------------------
# 8. Save scaled datasets
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_scaled.csv", index=False)
best_test.to_csv("prepared_data/test_scaled.csv", index=False)

print("Saved scaled datasets.")

# ----------------------------------------------------
# 9. Plots (with FIXED bar labels)
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# === 9A — Model Performance ===
labels = ["NB (Standard)", "NB (MinMax)", "KNN (Standard)", "KNN (MinMax)"]
scores = [acc_std_nb, acc_mm_nb, acc_std_knn, acc_mm_knn]

plt.figure(figsize=(7,4))
plot_bar_chart(labels, scores, title="Scaling — Model Performance")

# FIX: correct numeric labels on bars
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

plt.savefig("images/scaling_performance.png")
plt.close()

# === 9B — Best Scaler ===
best_scores = [max(acc_std_nb, acc_std_knn), max(acc_mm_nb, acc_mm_knn)]

plt.figure(figsize=(6,4))
plot_bar_chart(["Standard", "MinMax"], best_scores,
               title="Scaling — Best Scaler")

# FIX: correct numeric labels
for i, v in enumerate(best_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

plt.savefig("images/scaling_best.png")
plt.close()

# === 9C — Confusion Matrices ===
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

plt.figure()
plot_confusion_matrix(cm_nb, ["0", "1"])
plt.title(f"NB Confusion Matrix — Best Scaler ({best_scaler})")
plt.savefig("images/scaling_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(cm_knn, ["0", "1"])
plt.title(f"KNN Confusion Matrix — Best Scaler ({best_scaler})")
plt.savefig("images/scaling_cm_knn.png")
plt.close()


print("\n=== SCALING COMPLETE ===\n")
