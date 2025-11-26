from __future__ import annotations

"""
Lab 3 - Data Preparation (Encoding step) for the traffic accidents dataset.

This script:
 1. Loads the dataset from traffic_accidents.csv.
 2. Splits it into train and test sets.
 3. Applies TWO different encoding approaches to the input variables:
       - Approach A: Ordinal encoding of all categorical predictors.
       - Approach B: One-hot encoding (dummification) of all categorical predictors.
 4. For each encoded dataset, trains:
       - K-Nearest Neighbors classifier
       - Gaussian Naive Bayes classifier
 5. Evaluates performance on the test set and compares approaches.
 6. Selects the best approach (by accuracy) and saves:
       - Metrics table for all models/approaches.
       - Confusion matrix plots for the best-approach models.
       - Encoded train/test datasets for the best approach (for use in later lab steps).

Run from the command line:
    python lab3_encoding.py
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from dslabs_functions import get_variable_types, dummify


DATA_PATH = "/home/daniel/Desktop/DataScience/Lab3/traffic_accidents.csv/traffic_accidents.csv"
OUTPUT_DIR = "/home/daniel/Desktop/DataScience/Lab3/encoding_outputs"


# You can change this if you want to predict a different target variable.
TARGET_COLUMN = "crash_type"


@dataclass
class EncodedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    name: str  # name of the encoding approach


def load_data(path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X, y."""
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop the raw date column (temporal), since we already have derived time features
    if "crash_date" in df.columns:
        df = df.drop(columns=["crash_date"])

    # Separate target
    y = df[target_col].astype("category")
    X = df.drop(columns=[target_col])

    # Drop rows with missing target values (if any)
    mask_valid = ~y.isna()
    X = X.loc[mask_valid].reset_index(drop=True)
    y = y.loc[mask_valid].reset_index(drop=True)

    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def encode_ordinal(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> EncodedDataset:
    """
    Approach A: Ordinal encode all categorical predictors.

    Categorical columns -> manual integer codes (via category codes).
    Numeric columns -> left unchanged.
    """
    vars_types = get_variable_types(X_train.copy())
    numeric_cols = vars_types["numeric"]
    symbolic_cols = vars_types["symbolic"] + vars_types["binary"]

    X_train_ord = X_train.copy()
    X_test_ord = X_test.copy()

    # Encode each symbolic/binary column using pandas category codes (train-based)
    for col in symbolic_cols:
        # Fit categories on train
        X_train_ord[col] = X_train_ord[col].astype("category")
        categories = X_train_ord[col].cat.categories

        # Map train
        X_train_ord[col] = X_train_ord[col].cat.codes

        # Map test using the same categories; unseen values get code -1
        X_test_ord[col] = pd.Categorical(X_test_ord[col], categories=categories)
        X_test_ord[col] = X_test_ord[col].cat.codes

    X_train_encoded = X_train_ord[numeric_cols + symbolic_cols]
    X_test_encoded = X_test_ord[numeric_cols + symbolic_cols]

    return EncodedDataset(
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=None,  # to be filled later
        y_test=None,
        name="ordinal",
    )


def encode_onehot(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> EncodedDataset:
    """
    Approach B: One-hot encode all categorical predictors (dummification).

    Categorical columns -> dummify (OneHotEncoder, as in dslabs).
    Numeric columns -> left unchanged.
    """
    vars_types = get_variable_types(X_train.copy())
    # Do not dummify numeric or date variables
    vars_to_dummify = vars_types["symbolic"] + vars_types["binary"]

    # Use the course helper to fit on train and transform both train and test
    X_train_encoded = dummify(X_train.copy(), vars_to_dummify)

    # To apply the same transformation to test, we need to ensure it has the same columns
    X_test_encoded = dummify(X_test.copy(), vars_to_dummify)
    # Align test to train's columns (missing columns -> 0, extra columns dropped)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    return EncodedDataset(
        X_train=X_train_encoded,
        X_test=X_test_encoded,
        y_train=None,  # to be filled later
        y_test=None,
        name="onehot",
    )


def train_models(
    dataset: EncodedDataset,
) -> Dict[str, Dict[str, float]]:
    """
    Train KNN and Naive Bayes on the given encoded dataset and return metrics.

    Returns a dict:
        {
            "KNN": {metric_name: value, ...},
            "NaiveBayes": {metric_name: value, ...},
        }
    """
    X_train = dataset.X_train.values
    X_test = dataset.X_test.values
    y_train = dataset.y_train.values
    y_test = dataset.y_test.values

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "NaiveBayes": GaussianNB(),
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

    return metrics


def plot_confusion_matrices(
    dataset: EncodedDataset,
    models: Dict[str, object],
    output_dir: str,
    prefix: str,
) -> None:
    """Save confusion matrix plots for the given models and dataset."""
    X_test = dataset.X_test.values
    y_test = dataset.y_test.values
    class_labels = dataset.y_test.cat.categories

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=class_labels,
            cmap="Blues",
            xticks_rotation=45,
        )
        plt.tight_layout()
        filename = os.path.join(output_dir, f"{prefix}_{dataset.name}_{model_name}_cm.png")
        plt.savefig(filename, dpi=150)
        plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Lab 3 Encoding step ===")
    print(f"Reading data from: {DATA_PATH}")

    # 1. Load and split data
    X, y = load_data(DATA_PATH, TARGET_COLUMN)
    print(f"Loaded dataset with {X.shape[0]} rows and {X.shape[1]} predictors.")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Apply encoding approaches
    print("Encoding with Approach A: ordinal...")
    ordinal_ds = encode_ordinal(X_train, X_test)
    ordinal_ds.y_train = y_train
    ordinal_ds.y_test = y_test
    print(f"Ordinal encoded train shape: {ordinal_ds.X_train.shape}")

    print("Encoding with Approach B: one-hot (dummify)...")
    onehot_ds = encode_onehot(X_train, X_test)
    onehot_ds.y_train = y_train
    onehot_ds.y_test = y_test
    print(f"One-hot encoded train shape: {onehot_ds.X_train.shape}")

    datasets = [ordinal_ds, onehot_ds]

    # 3. Train models and collect metrics
    from typing import List  # local import to avoid cluttering top

    all_results: List[Dict[str, object]] = []
    best_accuracy = -np.inf
    best_dataset: EncodedDataset | None = None
    best_models: Dict[str, object] | None = None

    for ds in datasets:
        print(f"\nTraining models for encoding approach: {ds.name}")
        ds_metrics = train_models(ds)

        # Train models again to keep fitted instances for confusion matrices
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "NaiveBayes": GaussianNB(),
        }
        models["KNN"].fit(ds.X_train.values, ds.y_train.values)
        models["NaiveBayes"].fit(ds.X_train.values, ds.y_train.values)

        for model_name, m in ds_metrics.items():
            print(
                f"  {model_name} -> "
                f"accuracy={m['accuracy']:.4f}, "
                f"precision_macro={m['precision_macro']:.4f}, "
                f"recall_macro={m['recall_macro']:.4f}, "
                f"f1_macro={m['f1_macro']:.4f}"
            )
            result_row = {
                "encoding_approach": ds.name,
                "model": model_name,
                **m,
            }
            all_results.append(result_row)

            # Use accuracy to select best dataset/approach
            if m["accuracy"] > best_accuracy:
                best_accuracy = m["accuracy"]
                best_dataset = ds
                best_models = models

    # 4. Save metrics to CSV (for later inclusion in the report)
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, "encoding_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved metrics table to: {results_path}")

    # 5. Plot confusion matrices for the best approach
    if best_dataset is not None and best_models is not None:
        print(
            f"Best approach: {best_dataset.name} "
            f"(accuracy={best_accuracy:.4f})"
        )

        plot_confusion_matrices(
            best_dataset,
            best_models,
            OUTPUT_DIR,
            prefix="best_encoding",
        )

        # 6. Save encoded datasets for the best approach
        best_train_path = os.path.join(OUTPUT_DIR, f"{best_dataset.name}_train.csv")
        best_test_path = os.path.join(OUTPUT_DIR, f"{best_dataset.name}_test.csv")

        # Include target column in saved files
        train_df_out = best_dataset.X_train.copy()
        train_df_out[TARGET_COLUMN] = best_dataset.y_train.values

        test_df_out = best_dataset.X_test.copy()
        test_df_out[TARGET_COLUMN] = best_dataset.y_test.values

        train_df_out.to_csv(best_train_path, index=False)
        test_df_out.to_csv(best_test_path, index=False)

        print(f"Saved best-approach train set to: {best_train_path}")
        print(f"Saved best-approach test set to:  {best_test_path}")

    else:
        print("No best dataset identified. Check if training/evaluation ran correctly.")

    print("\n=== Encoding step finished ===")


if __name__ == "__main__":
    main()


