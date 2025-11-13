import os
import pandas as pd
from numpy import array
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

xvalues, yvalues = [], []
best_model, best_score, best_name = None, 0, ""

for name, clf in estimators.items():
    clf.fit(trnX, trnY)
    y_pred = clf.predict(tstX)
    acc = accuracy_score(tstY, y_pred)
    xvalues.append(name)
    yvalues.append(acc)
    if acc > best_score:
        best_model = clf
        best_score = acc
        best_name = name
    print(f"{name} -> Accuracy: {acc:.4f}")

# Plot hyperparameter study
figure()
plot_bar_chart(xvalues, yvalues, title="Naive Bayes Models", ylabel="Accuracy", percentage=True)
savefig("images/nb_hyperparameters.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {
    "Accuracy": (accuracy_score, y_trn_pred, y_tst_pred),
    "Precision": (precision_score, y_trn_pred, y_tst_pred),
    "Recall": (recall_score, y_trn_pred, y_tst_pred)
}

print(f"\nBest model: {best_name}")
for metric, (func, trn_pred, tst_pred) in metrics.items():
    trn_val = func(trnY, trn_pred)
    tst_val = func(tstY, tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")
