import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "traffic_accidents_cleaned.csv"
TARGET = "crash_type"
TEST_SIZE = 0.3
ITERATIONS = [100, 300, 500, 700, 1000]
PENALTIES = ["l1", "l2"]
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes
X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

# Hyperparameters study
acc_values = {}
prec_values = {}
rec_values = {}

best_model, best_score, best_params = None, 0, None

for pen in PENALTIES:
    acc_list = []
    prec_list = []
    rec_list = []
    for n_iter in ITERATIONS:
        clf = LogisticRegression(penalty=pen, max_iter=n_iter, solver="liblinear")
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)

        acc = accuracy_score(tstY, y_pred)
        prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
        rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)

        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (pen, n_iter)

        print(f"Penalty={pen}, Iter={n_iter} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    acc_values[pen] = acc_list
    prec_values[pen] = prec_list
    rec_values[pen] = rec_list

# Plot Accuracy
figure()
plot_multiline_chart(ITERATIONS, acc_values, title="Logistic Regression Accuracy", xlabel="Iterations", ylabel="Accuracy", percentage=True)
savefig("images/lr_accuracy.png")
close()

# Plot Precision
figure()
plot_multiline_chart(ITERATIONS, prec_values, title="Logistic Regression Precision", xlabel="Iterations", ylabel="Precision", percentage=True)
savefig("images/lr_precision.png")
close()

# Plot Recall
figure()
plot_multiline_chart(ITERATIONS, rec_values, title="Logistic Regression Recall", xlabel="Iterations", ylabel="Recall", percentage=True)
savefig("images/lr_recall.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}

print(f"\nBest Logistic Regression: Penalty={best_params[0]}, Iter={best_params[1]}")
for metric, func in metrics.items():
    if metric == "Accuracy":
        trn_val = func(trnY, y_trn_pred)
        tst_val = func(tstY, y_tst_pred)
    else:
        trn_val = func(trnY, y_trn_pred, average="weighted", zero_division=0)
        tst_val = func(tstY, y_tst_pred, average="weighted", zero_division=0)

    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
