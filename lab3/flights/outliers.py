#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from dslabs_functions import get_variable_types, determine_outlier_thresholds_for_var, plot_bar_chart, plot_confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING OUTLIERS HANDLING SCRIPT ===\n")

# ----------------------------------------------------
# 1. Load encoded datasets
# ----------------------------------------------------
train_path = "prepared_data/train_encoded.csv"
test_path  = "prepared_data/test_encoded.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print(f"Train: {train.shape} | Test: {test.shape}")

TARGET = "crash_type"

# Identify numeric variables
numeric_cols = get_variable_types(train)["numeric"]

# ----------------------------------------------------
# Helper: evaluate NB + KNN
# ----------------------------------------------------
def evaluate_dataset(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X_train_enc[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test_enc[cat_cols]  = enc.transform(X_test[cat_cols])

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
# 2. Define outlier handling functions
# ----------------------------------------------------
def drop_outliers(df):
    df_copy = df.copy(deep=True)
    summary = df_copy[numeric_cols].describe()
    for var in numeric_cols:
        top, bottom = determine_outlier_thresholds_for_var(summary[var])
        df_copy = df_copy[(df_copy[var] <= top) & (df_copy[var] >= bottom)]
    return df_copy

def replace_outliers(df):
    df_copy = df.copy(deep=True)
    summary = df_copy[numeric_cols].describe()
    for var in numeric_cols:
        top, bottom = determine_outlier_thresholds_for_var(summary[var])
        median = df_copy[var].median()
        df_copy[var] = df_copy[var].apply(lambda x: median if x > top or x < bottom else x)
    return df_copy

def truncate_outliers(df):
    df_copy = df.copy(deep=True)
    summary = df_copy[numeric_cols].describe()
    for var in numeric_cols:
        top, bottom = determine_outlier_thresholds_for_var(summary[var])
        df_copy[var] = df_copy[var].apply(lambda x: top if x > top else bottom if x < bottom else x)
    return df_copy

# ----------------------------------------------------
# 3. Apply each approach
# ----------------------------------------------------
print("[1] Dropping outliers...")
train_drop = drop_outliers(train)
test_drop  = drop_outliers(test)
acc_drop_nb, acc_drop_knn, pred_drop_nb, pred_drop_knn, y_drop = evaluate_dataset(train_drop, test_drop)

print("[2] Replacing outliers with median...")
train_replace = replace_outliers(train)
test_replace  = replace_outliers(test)
acc_replace_nb, acc_replace_knn, pred_replace_nb, pred_replace_knn, y_replace = evaluate_dataset(train_replace, test_replace)

print("[3] Truncating outliers...")
train_trunc = truncate_outliers(train)
test_trunc  = truncate_outliers(test)
acc_trunc_nb, acc_trunc_knn, pred_trunc_nb, pred_trunc_knn, y_trunc = evaluate_dataset(train_trunc, test_trunc)

# ----------------------------------------------------
# 4. Compare results
# ----------------------------------------------------
approaches = ["Drop", "Replace", "Truncate"]
nb_scores = [acc_drop_nb, acc_replace_nb, acc_trunc_nb]
knn_scores = [acc_drop_knn, acc_replace_knn, acc_trunc_knn]

# Determine best approach (NB primary metric)
best_idx = np.argmax(nb_scores)
best_approach = approaches[best_idx]
print("\n=== RESULTS ===")
for i, app in enumerate(approaches):
    print(f"{app}: NB={nb_scores[i]:.4f}, KNN={knn_scores[i]:.4f}")
print(f">>> BEST APPROACH: {best_approach}\n")

# Map best datasets/predictions
best_train, best_test = [train_drop, train_replace, train_trunc][best_idx], [test_drop, test_replace, test_trunc][best_idx]
best_pred_nb, best_pred_knn, y_true = [pred_drop_nb, pred_replace_nb, pred_trunc_nb][best_idx], [pred_drop_knn, pred_replace_knn, pred_trunc_knn][best_idx], [y_drop, y_replace, y_trunc][best_idx]

# ----------------------------------------------------
# 5. Plotting
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# Performance comparison
plt.figure(figsize=(8,5))
labels_plot = [f"NB ({a})" for a in approaches] + [f"KNN ({a})" for a in approaches]
scores_plot = nb_scores + knn_scores
plot_bar_chart(labels_plot, scores_plot, title="Outliers Handling — Model Performance")
for i, v in enumerate(scores_plot):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/outliers_performance.png")
plt.close()

# Best approach chart
plt.figure(figsize=(6,4))
plot_bar_chart(approaches, nb_scores, title="Outliers — Best Approach (NB Accuracy)")
for i, v in enumerate(nb_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig("images/outliers_best_approach.png")
plt.close()

# Confusion matrices
plt.figure()
plot_confusion_matrix(confusion_matrix(y_true, best_pred_nb), ["0","1"])
plt.title(f"Naive Bayes Confusion Matrix — Best Approach ({best_approach})")
plt.savefig("images/outliers_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(confusion_matrix(y_true, best_pred_knn), ["0","1"])
plt.title(f"KNN Confusion Matrix — Best Approach ({best_approach})")
plt.savefig("images/outliers_cm_knn.png")
plt.close()

print("Saved charts into /images")

# ----------------------------------------------------
# 6. Save final datasets
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_outliers.csv", index=False)
best_test.to_csv("prepared_data/test_outliers.csv", index=False)

print("\n=== OUTLIERS HANDLING COMPLETE ===")
print("Saved:")
print("  prepared_data/train_outliers.csv")
print("  prepared_data/test_outliers.csv\n")
