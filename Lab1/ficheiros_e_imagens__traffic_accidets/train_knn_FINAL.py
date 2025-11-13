import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "traffic_accidents_cleaned.csv"
TARGET = "crash_type"
TEST_SIZE = 0.3
K_MAX = 25  # Test all k from 1 to K_MAX
DISTANCES = ["manhattan", "euclidean", "chebyshev"]

os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes
X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

# Hyperparameters study
values = {}
best_model, best_score, best_params = None, 0, None

k_values = list(range(1, K_MAX + 1))

for dist in DISTANCES:
    acc_list = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=dist)
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)
        acc = accuracy_score(tstY, y_pred)
        acc_list.append(acc)
        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (k, dist)
        print(f"k={k}, metric={dist} -> Accuracy: {acc:.4f}")
    values[dist] = acc_list

# Plot hyperparameters
figure()
plot_multiline_chart(k_values, values, title="KNN Hyperparameters", xlabel="k", ylabel="Accuracy", percentage=True)
savefig("images/knn_hyperparameters.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}
print(f"\nBest model: k={best_params[0]}, metric={best_params[1]}")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
