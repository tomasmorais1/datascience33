import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart

DATA_FILE = "traffic_accidents_cleaned.csv"
TARGET = "crash_type"
TEST_SIZE = 0.3
os.makedirs("images", exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes
X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
labels = sorted(y.unique())

# Hyperparameters study
estimators = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
}

xvalues = []
acc_values = []
prec_values = []
rec_values = []

best_model, best_score, best_name = None, 0, ""

for name, clf in estimators.items():
    clf.fit(trnX, trnY)
    y_pred = clf.predict(tstX)

    acc = accuracy_score(tstY, y_pred)
    prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
    rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

    xvalues.append(name)
    acc_values.append(acc)
    prec_values.append(prec)
    rec_values.append(rec)

    if acc > best_score:
        best_model = clf
        best_score = acc
        best_name = name

    print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# Plot Accuracy
figure()
plot_bar_chart(xvalues, acc_values, title="Naive Bayes Accuracy", ylabel="Accuracy", percentage=True)
savefig("images/nb_accuracy.png")
close()

# Plot Precision
figure()
plot_bar_chart(xvalues, prec_values, title="Naive Bayes Precision", ylabel="Precision", percentage=True)
savefig("images/nb_precision.png")
close()

# Plot Recall
figure()
plot_bar_chart(xvalues, rec_values, title="Naive Bayes Recall", ylabel="Recall", percentage=True)
savefig("images/nb_recall.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score
}

print(f"\nBest model: {best_name}")
for metric, func in metrics.items():
    if metric == "Accuracy":
        trn_val = func(trnY, y_trn_pred)
        tst_val = func(tstY, y_tst_pred)
    else:
        trn_val = func(trnY, y_trn_pred, average="weighted", zero_division=0)
        tst_val = func(tstY, y_tst_pred, average="weighted", zero_division=0)

    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
