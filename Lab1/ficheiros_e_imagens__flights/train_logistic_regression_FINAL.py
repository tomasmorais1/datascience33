import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "Combined_Flights_2022_cleaned.csv"
TARGET = "Cancelled"
TEST_SIZE = 0.3
SAMPLE_SIZE = 300_000
ITERATIONS = [100, 300, 500, 700, 1000]
PENALTIES = ["l1", "l2"]

os.makedirs("images", exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Stratified sampling
if SAMPLE_SIZE < len(df):
    df = df.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(int(SAMPLE_SIZE * len(x) / len(df)), random_state=42)
    )

# Convert target to numeric if necessary
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes

X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

# Hyperparameter study
values_accuracy = {}
values_precision = {}
values_recall = {}

best_model, best_score, best_params = None, 0, None

for pen in PENALTIES:
    acc_list = []
    prec_list = []
    rec_list = []

    for n_iter in ITERATIONS:
        clf = LogisticRegression(
            penalty=pen, max_iter=n_iter, solver="liblinear"
        )
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
            best_params = (pen, n_iter)

        print(f"Penalty={pen}, Iter={n_iter} -> "
              f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    values_accuracy[pen] = acc_list
    values_precision[pen] = prec_list
    values_recall[pen] = rec_list

# Plot Accuracy
figure()
plot_multiline_chart(
    ITERATIONS, values_accuracy,
    title="Logistic Regression Hyperparameters – Accuracy",
    xlabel="Iterations", ylabel="Accuracy", percentage=True
)
savefig("images/lr_hyperparameters_accuracy.png")
close()

# Plot Precision
figure()
plot_multiline_chart(
    ITERATIONS, values_precision,
    title="Logistic Regression Hyperparameters – Precision",
    xlabel="Iterations", ylabel="Precision", percentage=True
)
savefig("images/lr_hyperparameters_precision.png")
close()

# Plot Recall
figure()
plot_multiline_chart(
    ITERATIONS, values_recall,
    title="Logistic Regression Hyperparameters – Recall",
    xlabel="Iterations", ylabel="Recall", percentage=True
)
savefig("images/lr_hyperparameters_recall.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="macro", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="macro", zero_division=0),
}

print(f"\nBest Logistic Regression Model: Penalty={best_params[0]}, Iter={best_params[1]}")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
