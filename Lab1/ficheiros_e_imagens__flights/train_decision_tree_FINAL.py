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

trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

# Hyperparameters study
values = {}
best_model, best_score, best_params = None, 0, None

depths = list(range(2, MAX_DEPTH + 1))
for crit in CRITERIA:
    acc_list = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit)
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)
        acc = accuracy_score(tstY, y_pred)
        acc_list.append(acc)
        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (crit, d)
        print(f"max_depth={d}, criterion={crit} -> Accuracy: {acc:.4f}")
    values[crit] = acc_list

# Plot hyperparameters
figure()
plot_multiline_chart(depths, values, title="Decision Tree Hyperparameters", xlabel="max_depth", ylabel="Accuracy", percentage=True)
savefig("images/dt_hyperparameters.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}
print(f"\nBest Decision Tree: max_depth={best_params[1]}, criterion={best_params[0]}")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
