#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
    plot_bar_chart,
    plot_confusion_matrix
)
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n=== STARTING OUTLIERS HANDLING SCRIPT ===\n")

# ----------------------------------------------------
# 1. Load encoded datasets (before scaling)
# ----------------------------------------------------
train = pd.read_csv("prepared_data/train_encoded.csv")
test  = pd.read_csv("prepared_data/test_encoded.csv")

TARGET = "Cancelled"

print(f"Train: {train.shape} | Test: {test.shape}")

numeric_cols = get_variable_types(train)["numeric"]

# ----------------------------------------------------
# 2. Compute outlier thresholds ON TRAIN ONLY
# ----------------------------------------------------
print("\nComputing outlier thresholds (TRAIN ONLY)...")

summary_train = train[numeric_cols].describe()

thresholds = {}
for var in numeric_cols:
    top, bottom = determine_outlier_thresholds_for_var(summary_train[var])
    thresholds[var] = (top, bottom)

print("Thresholds computed.\n")

# ----------------------------------------------------
# 3. Outlier handling methods
# ----------------------------------------------------
def apply_drop(df):
    df2 = df.copy()
    for var in numeric_cols:
        top, bottom = thresholds[var]
        df2 = df2[(df2[var] <= top) & (df2[var] >= bottom)]
    return df2

def apply_replace(df):
    df2 = df.copy()
    for var in numeric_cols:
        top, bottom = thresholds[var]
        median = df2[var].median()
        df2[var] = df2[var].apply(
            lambda x: median if (x > top or x < bottom) else x
        )
    return df2

def apply_truncate(df):
    df2 = df.copy()
    for var in numeric_cols:
        top, bottom = thresholds[var]
        df2[var] = df2[var].clip(lower=bottom, upper=top)
    return df2

# ----------------------------------------------------
# 4. Evaluation: NB + KNN
# ----------------------------------------------------
def evaluate_dataset(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # Ordinal encode symbolic columns
    cat_cols = X_train.select_dtypes(include="object").columns
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if len(cat_cols) > 0:
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols]  = enc.transform(X_test[cat_cols])

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
# 5. Apply approaches
# ----------------------------------------------------
print("[1] Dropping outliers...")
train_drop = apply_drop(train)
test_drop  = apply_drop(test)
acc_drop_nb, acc_drop_knn, pred_drop_nb, pred_drop_knn, y_drop = evaluate_dataset(train_drop, test_drop)

print("[2] Replacing outliers with median...")
train_replace = apply_replace(train)
test_replace  = apply_replace(test)
acc_replace_nb, acc_replace_knn, pred_replace_nb, pred_replace_knn, y_replace = evaluate_dataset(train_replace, test_replace)

print("[3] Truncating outliers...")
train_trunc = apply_truncate(train)
test_trunc  = apply_truncate(test)
acc_trunc_nb, acc_trunc_knn, pred_trunc_nb, pred_trunc_knn, y_trunc = evaluate_dataset(train_trunc, test_trunc)

# ----------------------------------------------------
# 6. Compare results
# ----------------------------------------------------
approaches = ["Drop", "Replace", "Truncate"]

nb_scores  = [acc_drop_nb,  acc_replace_nb,  acc_trunc_nb]
knn_scores = [acc_drop_knn, acc_replace_knn, acc_trunc_knn]

best_idx = np.argmax(nb_scores)
best_name = approaches[best_idx]

print("\n=== RESULTS ===")
for i, a in enumerate(approaches):
    print(f"{a}: NB={nb_scores[i]:.4f}, KNN={knn_scores[i]:.4f}")
print(f"\n>>> BEST APPROACH = {best_name}\n")

best_train = [train_drop, train_replace, train_trunc][best_idx]
best_test  = [test_drop, test_replace, test_trunc][best_idx]
best_y          = [y_drop, y_replace, y_trunc][best_idx]
best_pred_nb    = [pred_drop_nb, pred_replace_nb, pred_trunc_nb][best_idx]
best_pred_knn   = [pred_drop_knn, pred_replace_knn, pred_trunc_knn][best_idx]

# ----------------------------------------------------
# 7. PLOTS — FIXED LABELS (no more 1.0 bug)
# ----------------------------------------------------
os.makedirs("images", exist_ok=True)

# --- Performance chart ---
plt.figure(figsize=(9,5))
labels_plot = (
    [f"NB ({a})" for a in approaches] +
    [f"KNN ({a})" for a in approaches]
)
scores_plot = nb_scores + knn_scores

plot_bar_chart(labels_plot, scores_plot,
               title="Outliers Handling — Model Performance")

# FIXED LABELS (correct values)
for i, v in enumerate(scores_plot):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.savefig("images/outliers_performance.png")
plt.close()

# --- Best approach chart ---
plt.figure(figsize=(6,4))
plot_bar_chart(approaches, nb_scores,
               title="Outliers — NB Accuracy")

# FIX labels
for i, v in enumerate(nb_scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.savefig("images/outliers_best.png")
plt.close()

# --- Confusion matrices ---
plt.figure()
plot_confusion_matrix(confusion_matrix(best_y, best_pred_nb), ["0","1"])
plt.title(f"NB Confusion Matrix — {best_name}")
plt.savefig("images/outliers_cm_nb.png")
plt.close()

plt.figure()
plot_confusion_matrix(confusion_matrix(best_y, best_pred_knn), ["0","1"])
plt.title(f"KNN Confusion Matrix — {best_name}")
plt.savefig("images/outliers_cm_knn.png")
plt.close()

print("Charts saved.")

# ----------------------------------------------------
# 8. Save output
# ----------------------------------------------------
best_train.to_csv("prepared_data/train_outliers.csv", index=False)
best_test.to_csv("prepared_data/test_outliers.csv", index=False)

print("\n=== OUTLIERS HANDLING COMPLETE ===")
print("Saved:")
print("  prepared_data/train_outliers.csv")
print("  prepared_data/test_outliers.csv\n")
