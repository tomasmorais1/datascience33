import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score
)
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "Combined_Flights_2022_cleaned.csv"
TARGET = "Cancelled"
TEST_SIZE = 0.3
SAMPLE_SIZE = 50_000
K_MAX = 25
DISTANCES = ["manhattan", "euclidean", "chebyshev"]

os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Stratified sampling
if SAMPLE_SIZE < len(df):
    df = df.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(int(SAMPLE_SIZE * len(x) / len(df)), random_state=42)
    )

# Encode target if needed
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes

X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

# Hyperparameter metrics storage
values_accuracy = {}
values_precision = {}
values_recall = {}

best_model = None
best_score = 0
best_params = None

k_values = list(range(1, K_MAX + 1))

for dist in DISTANCES:
    acc_list = []
    prec_list = []
    rec_list = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=dist)
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)

        acc = accuracy_score(tstY, y_pred)
        prec = precision_score(tstY, y_pred, average="macro", zero_division=0)
        rec = recall_score(tstY, y_pred, average="macro", zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (k, dist)

        print(f"k={k}, metric={dist} → "
              f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    values_accuracy[dist] = acc_list
    values_precision[dist] = prec_list
    values_recall[dist] = rec_list

# Plot Accuracy
figure()
plot_multiline_chart(
    k_values, values_accuracy,
    title="KNN Hyperparameters – Accuracy",
    xlabel="k", ylabel="Accuracy", percentage=True
)
savefig("images/knn_hyperparameters_accuracy.png")
close()

# Plot Precision
figure()
plot_multiline_chart(
    k_values, values_precision,
    title="KNN Hyperparameters – Precision",
    xlabel="k", ylabel="Precision", percentage=True
)
savefig("images/knn_hyperparameters_precision.png")
close()

# Plot Recall
figure()
plot_multiline_chart(
    k_values, values_recall,
    title="KNN Hyperparameters – Recall",
    xlabel="k", ylabel="Recall", percentage=True
)
savefig("images/knn_hyperparameters_recall.png")
close()

# Best model performance summary
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="macro", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="macro", zero_division=0),
}

print(f"\nBest model: k={best_params[0]}, metric={best_params[1]}")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} – Train: {trn_val:.4f}, Test: {tst_val:.4f}")
