import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart

DATA_FILE = "Combined_Flights_2022_cleaned.csv"
TARGET = "Cancelled"
TEST_SIZE = 0.3

os.makedirs("images", exist_ok=True)

print(">>> USING SIMPLE GAUSSIANNB VERSION <<<")

# Load data
df = pd.read_csv(DATA_FILE)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train-test split
trnX, tstX, trnY, tstY = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

# Only GaussianNB (other NB variants don't fit this dataset)
clf = GaussianNB()
clf.fit(trnX, trnY)
y_pred = clf.predict(tstX)

acc = accuracy_score(tstY, y_pred)
prec = precision_score(tstY, y_pred, average="weighted", zero_division=0)
rec = recall_score(tstY, y_pred, average="weighted", zero_division=0)

print(f"GaussianNB -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

# For plotting, we keep the same structure (one model)
xvalues = ["GaussianNB"]
acc_values = [acc]
prec_values = [prec]
rec_values = [rec]

# Plot Accuracy
figure()
plot_bar_chart(xvalues, acc_values,
               title="Naive Bayes Accuracy",
               ylabel="Accuracy",
               percentage=True)
savefig("images/nb_accuracy.png")
close()

# Plot Precision
figure()
plot_bar_chart(xvalues, prec_values,
               title="Naive Bayes Precision",
               ylabel="Precision",
               percentage=True)
savefig("images/nb_precision.png")
close()

# Plot Recall
figure()
plot_bar_chart(xvalues, rec_values,
               title="Naive Bayes Recall",
               ylabel="Recall",
               percentage=True)
savefig("images/nb_recall.png")
close()

# Best model performance (only one model here)
y_trn_pred = clf.predict(trnX)
y_tst_pred = y_pred

metrics = {
    "Accuracy": accuracy_score,
    "Precision": lambda t, p: precision_score(t, p, average="weighted", zero_division=0),
    "Recall": lambda t, p: recall_score(t, p, average="weighted", zero_division=0),
}

print("\nBest model: GaussianNB")
for metric, func in metrics.items():
    print(f"{metric} - Train: {func(trnY, y_trn_pred):.4f}, Test: {func(tstY, y_tst_pred):.4f}")
