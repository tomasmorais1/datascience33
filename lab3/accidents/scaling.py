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
train_path = "prepared_data/train_outliers.csv"
test_path  = "prepared_data/test_outliers.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train: {train.shape} | Test: {test.shape}")

TARGET = "crash_type"

# ----------------------------------------------------
# 2. Identify numeric and categorical variables
# ----------------------------------------------------
vars_types = get_variable_types(train)
numeric_cols = vars_types["numeric"]
categorical_cols = vars_types["symbolic"] + vars_types["binary"]

if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)
if TARGET in categorical_cols:
    categorical_cols.remove(TARGET)

# ----------------------------------------------------
# 3. Scaling helper
# ----------------------------------------------------
def scale_dataset(train_df, test_df, scaler_type="standard"):
    train_copy = train_df.copy()
    test_copy  = test_df.copy()

    # Separate numeric and categorical
    X_train_num = train_copy[numeric_cols]
    X_test_num  = test_copy[numeric_cols]

    if scaler_type == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(X_train_num)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(X_train_num)
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    # Transform numeric
    train_copy[numeric_cols] = scaler.transform(X_train_num)
    test_copy[numeric_cols]  = scaler.transform(X_test_num)

    return train_copy, test_copy

# ----------------------------------------------------
# 4. Evaluate NB and KNN
# ----------------------------------------------------
def evaluate(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # Encode categorical
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if categorical_cols:
        X_train[categorical_cols] = enc.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols]  = enc.transform(X_test[categorical_cols])

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
# 5. Apply StandardScaler
# ----------------------------------------------------
print("\n[1] Applying StandardScaler...")
train_std, test_std = scale_dataset(train, test, scaler_type="standard")
acc_std_nb, acc_std_knn, pred_std_nb, pred_std_knn, y_std = evaluate(train_std, test_std)

# ----------------------------------------------------
# 6. Apply MinMaxScaler
# ----------------------------------------------------
print("[2] Applying MinMaxScaler...")
train_mm, test_mm = scale_dataset(train, test, scaler_type="minmax")
acc_mm_nb, acc_mm_knn, pred_mm_nb, pred_mm_knn, y_mm = evaluate(train_mm, test_mm)

# ----------------------------------------------------
# 7. Compare results and select best scaler
# ----------------------------------------------------
print("\n=== RESULTS ===")
print(f"StandardScaler  — NB: {acc_std_nb:.4f}, KNN: {acc_std_knn:.4f}")
print(f"MinMaxScaler    — NB: {acc_mm_nb:.4f}, KNN: {acc_mm_knn:.4f}")

best_scaler = "standard" if max(acc_std_nb, acc_std_knn) >= max(acc_mm_nb, acc_mm_knn) else "minmax"
print(f">>> BEST SCALER: {best_scaler}\n")

if best_scaler == "standard":
    best_train, best_test = train_std, test_std
    y_true, pred_nb, pred_knn = y_std, pred_std_nb, pred_std_knn
else:
    best_train, best_test = train_mm, test_mm
    y_true, pred_nb, pred_knn = y_mm, pred_mm_nb, pred_mm_knn

# ----------------------------------------------------
# 8. PLOTS
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# 8A — Model performance
labels = ["NB (Standard)", "NB (MinMax)", "KNN (Standard)", "KNN (MinMax)"]
scores = [acc_std_nb, acc_mm_nb, acc_std_knn, acc_mm_knn]
plt.figure(figsize=(7,4))
plot_bar_chart(labels, scores, title="Scaling — Model Performance")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.savefig("images/scaling_performance.png")
plt.close()

# 8B — Best scaler
best_scores = [max(acc_std_nb, acc_std_knn), max(acc_mm_nb, acc_mm_knn)]
plt.figure(figsize=(6,4))
plot_bar_chart(["StandardScaler", "MinMaxScaler"], best_scores,
               title="Scaling — Best Scaler (Max Accuracy)")
for i, v in enumerate(best_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.savefig("images/scaling_best.png")
plt.close()

# 8C — Confusion matrices
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

plt.figure()
plot_confusion_matrix(cm_nb, sorted(y_true.unique()))
plt.title(f"Naive Bayes Confusion Matrix — Best Scaler ({best_scaler})")
plt.savefig("images/scaling_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(cm_knn, sorted(y_true.unique()))
plt.title(f"KNN Confusion Matrix — Best Scaler ({best_scaler})")
plt.savefig("images/scaling_cm_knn.png")
plt.close()

# ----------------------------------------------------
# 9. Save scaled datasets
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_scaled.csv", index=False)
best_test.to_csv("prepared_data/test_scaled.csv", index=False)

print("\n=== SCALING COMPLETE ===")
print("Saved:")
print("  prepared_data/train_scaled.csv")
print("  prepared_data/test_scaled.csv\n")
