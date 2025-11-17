import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "Combined_Flights_2022_cleaned.csv"
TARGET = "Cancelled"
TEST_SIZE = 0.3
MAX_DEPTH = 25
CRITERIA = ["gini", "entropy"]

os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes
X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

# Hyperparameters study
values_accuracy = {}
values_precision = {}
values_recall = {}

best_model = None
best_score = 0
best_params = None

depths = list(range(2, MAX_DEPTH + 1))

for crit in CRITERIA:
    acc_list = []
    prec_list = []
    rec_list = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit)
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
            best_params = (crit, d)

        print(f"max_depth={d}, criterion={crit} -> "
              f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    values_accuracy[crit] = acc_list
    values_precision[crit] = prec_list
    values_recall[crit] = rec_list

# Plot Accuracy
figure()
plot_multiline_chart(
    depths, values_accuracy,
    title="Decision Tree Hyperparameters - Accuracy",
    xlabel="max_depth", ylabel="Accuracy", percentage=True
)
savefig("images/dt_hyperparameters_accuracy.png")
close()

# Plot Precision
figure()
plot_multiline_chart(
    depths, values_precision,
    title="Decision Tree Hyperparameters - Precision",
    xlabel="max_depth", ylabel="Precision", percentage=True
)
savefig("images/dt_hyperparameters_precision.png")
close()

# Plot Recall
figure()
plot_multiline_chart(
    depths, values_recall,
    title="Decision Tree Hyperparameters - Recall",
    xlabel="max_depth", ylabel="Recall", percentage=True
)
savefig("images/dt_hyperparameters_recall.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="macro", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="macro", zero_division=0),
}

print(f"\nBest Decision Tree: max_depth={best_params[1]}, criterion={best_params[0]}")

for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
