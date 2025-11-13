import os
from numpy import array, arange
from matplotlib.pyplot import figure, savefig, show
from sklearn.neural_network import MLPClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
    plot_multiline_chart,
    plot_evaluation_results,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- Configuration ---
file_tag = "traffic_accidents"
train_filename = "traffic_accidents_cleaned.csv"
test_filename = "traffic_accidents_cleaned.csv"  # if no separate test file, split internally
target = "crash_type"
eval_metric = "accuracy"

NR_MAX_ITER = 2000  # reduced for faster testing
LAG = 500
learning_rates = [0.5, 0.05]  # reduced number of learning rates
lr_types = ["constant", "adaptive"]  # two lr types for faster testing

os.makedirs("images", exist_ok=True)

# --- Load dataset ---
trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

# --- Hyperparameters study ---
best_model = None
best_params = {"name": "MLP", "metric": eval_metric, "params": ()}
best_performance = 0.0
nr_iterations = [LAG] + [i for i in range(2 * LAG, NR_MAX_ITER + 1, LAG)]

for lr_type in lr_types:
    values = {}
    print(f"\nStarting hyperparameter study for learning rate type: {lr_type}")
    for lr in learning_rates:
        warm_start = False
        y_tst_values = []
        print(f"  Testing learning rate: {lr}")
        for n_iter in nr_iterations:
            print(f"    Iteration step: {n_iter}")
            clf = MLPClassifier(
                learning_rate=lr_type,
                learning_rate_init=lr,
                max_iter=LAG,
                warm_start=warm_start,
                activation="logistic",
                solver="sgd",
                verbose=False,
                random_state=42,
            )
            clf.fit(trnX, trnY)
            prdY = clf.predict(tstX)
            eval_score = CLASS_EVAL_METRICS[eval_metric](tstY, prdY)
            y_tst_values.append(eval_score)
            warm_start = True
            if eval_score - best_performance > DELTA_IMPROVE:
                best_performance = eval_score
                best_params["params"] = (lr_type, lr, n_iter)
                best_model = clf
        values[lr] = y_tst_values
    figure()
    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"MLP Hyperparameters ({lr_type})",
        xlabel="nr iterations",
        ylabel=eval_metric,
        percentage=True,
    )
    savefig(f"images/{file_tag}_mlp_{lr_type}_hyperparameters.png")
    show()

# --- Best model performance ---
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)

metrics = {"Accuracy": accuracy_score, "Precision": precision_score, "Recall": recall_score}
print(f"\nBest MLP model found: lr_type={best_params['params'][0]}, lr={best_params['params'][1]}, iterations={best_params['params'][2]}")
for metric, func in metrics.items():
    trn_val = func(trnY, prd_trn)
    tst_val = func(tstY, prd_tst)
    print(f"{metric} - Train: {trn_val:.4f}, Test: {tst_val:.4f}")

figure()
plot_evaluation_results(best_params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f"images/{file_tag}_mlp_best_model_performance.png")
show()
