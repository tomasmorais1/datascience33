import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_multiline_chart

DATA_FILE = "Combined_Flights_2022_cleaned.csv"
TARGET = "Cancelled"
TEST_SIZE = 0.3
SAMPLE_SIZE = 300_000  # usar apenas 100k linhas para treinar rápido
ITERATIONS = [100, 300, 500, 700, 1000]
PENALTIES = ["l1", "l2"]
os.makedirs("images", exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Amostragem estratificada para preservar proporção da classe
if SAMPLE_SIZE < len(df):
    df = df.groupby(TARGET, group_keys=False).apply(lambda x: x.sample(int(SAMPLE_SIZE * len(x)/len(df)), random_state=42))

# Converter target se necessário
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].astype("category").cat.codes

X = df.drop(columns=[TARGET])
y = df[TARGET]

trnX, tstX, trnY, tstY = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

# Hyperparameters study
values = {}
best_model, best_score, best_params = None, 0, None

for pen in PENALTIES:
    acc_list = []
    for n_iter in ITERATIONS:
        clf = LogisticRegression(penalty=pen, max_iter=n_iter, solver="liblinear")
        clf.fit(trnX, trnY)
        y_pred = clf.predict(tstX)
        acc = accuracy_score(tstY, y_pred)
        acc_list.append(acc)
        if acc > best_score:
            best_model = clf
            best_score = acc
            best_params = (pen, n_iter)
        print(f"Penalty={pen}, Iter={n_iter} -> Accuracy: {acc:.4f}")
    values[pen] = acc_list

# Plot hyperparameters
figure()
plot_multiline_chart(ITERATIONS, values, title="Logistic Regression Hyperparameters", xlabel="Iterations", ylabel="Accuracy", percentage=True)
savefig("images/lr_hyperparameters.png")
close()

# Best model performance
y_trn_pred = best_model.predict(trnX)
y_tst_pred = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}
print(f"\nBest model: Penalty={best_params[0]}, Iter={best_params[1]}")
for metric, func in metrics.items():
    trn_val = func(trnY, y_trn_pred)
    tst_val = func(tstY, y_tst_pred)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

# Classification report para avaliação completa
print("\nClassification Report (Test Set):")
print(classification_report(tstY, y_tst_pred, digits=4))
