#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from dslabs_functions import plot_bar_chart, plot_confusion_matrix
import warnings
from matplotlib.pyplot import figure, savefig, text, title

warnings.filterwarnings("ignore")
print("\n=== STARTING DATA BALANCING SCRIPT ===\n")

# ---------------------------------------------------
# 1. Load encoded datasets
# ---------------------------------------------------
train = pd.read_csv("prepared_data/train_scaled.csv")
test  = pd.read_csv("prepared_data/test_scaled.csv")
TARGET = "crash_type"

print(f"Train: {train.shape} | Test: {test.shape}\n")

# ---------------------------------------------------
# 2. Identify minority and majority class
# ---------------------------------------------------
target_count = train[TARGET].value_counts()
minority_class = target_count.idxmin()
majority_class = target_count.idxmax()

print("Minority class =", minority_class, ":", target_count[minority_class])
print("Majority class =", majority_class, ":", target_count[majority_class])
print("Proportion:", round(target_count[minority_class] / target_count[majority_class], 2), ": 1\n")

# ---------------------------------------------------
# 3. Helper: evaluate NB + KNN
# ---------------------------------------------------
def evaluate_dataset(train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # Naive Bayes
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

# ---------------------------------------------------
# 4. Approach 1: Undersampling
# ---------------------------------------------------
df_min = train[train[TARGET] == minority_class]
df_maj = train[train[TARGET] == majority_class].sample(len(df_min), random_state=42)
train_under = pd.concat([df_min, df_maj], axis=0).sample(frac=1, random_state=42)

acc_under_nb, acc_under_knn, pred_under_nb, pred_under_knn, y_under_test = evaluate_dataset(train_under, test)
print("Undersampling done.\n")

# ---------------------------------------------------
# 5. Approach 2: SMOTE
# ---------------------------------------------------
train_smote = train.copy()

X = train_smote.drop(columns=[TARGET]).values
y = train_smote[TARGET].values

smote = SMOTE(sampling_strategy="minority", random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

train_smote_final = pd.DataFrame(X_sm, columns=train_smote.drop(columns=[TARGET]).columns)
train_smote_final[TARGET] = y_sm

acc_smote_nb, acc_smote_knn, pred_smote_nb, pred_smote_knn, y_smote_test = evaluate_dataset(train_smote_final, test)
print("SMOTE done.\n")

# ---------------------------------------------------
# 6. Compare and select best approach
# ---------------------------------------------------
labels = ["Undersample", "SMOTE"]
scores = [acc_under_nb, acc_smote_nb, acc_under_knn, acc_smote_knn]

figure(figsize=(7,4))
plot_bar_chart(["NB (Und)", "NB (SMOTE)", "KNN (Und)", "KNN (SMOTE)"], scores)
for i, v in enumerate(scores):
    text(i, v + 0.01, f"{v:.3f}", ha='center')
title("Model Accuracy per Algorithm and Balancing Approach")
savefig("images/balancing_performance.png")

best_index = np.argmax([acc_under_nb, acc_smote_nb])
best_name = labels[best_index]
print(f">>> BEST BALANCING APPROACH: {best_name}\n")

if best_name == "Undersample":
    y_true, pred_nb, pred_knn = y_under_test, pred_under_nb, pred_under_knn
    best_train = train_under
else:
    y_true, pred_nb, pred_knn = y_smote_test, pred_smote_nb, pred_smote_knn
    best_train = train_smote_final

# ---------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------
cm_nb = confusion_matrix(y_true, pred_nb)
cm_knn = confusion_matrix(y_true, pred_knn)

figure()
plot_confusion_matrix(cm_nb, ["0","1"])
title(f"Naive Bayes Confusion Matrix — Best Balancing ({best_name})")
savefig("images/balancing_cm_nb.png")

figure()
plot_confusion_matrix(cm_knn, ["0","1"])
title(f"KNN Confusion Matrix — Best Balancing ({best_name})")
savefig("images/balancing_cm_knn.png")

# ---------------------------------------------------
# 8. Save final datasets
# ---------------------------------------------------
os.makedirs("prepared_data", exist_ok=True)
best_train.to_csv("prepared_data/train_balanced.csv", index=False)
test.to_csv("prepared_data/test_balanced.csv", index=False)

print("\n=== DATA BALANCING COMPLETE ===")
print("Saved:")
print("  prepared_data/train_balanced.csv")
print("  prepared_data/test_balanced.csv\n")
