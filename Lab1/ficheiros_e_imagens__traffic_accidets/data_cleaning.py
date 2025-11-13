# data_cleaning.py
# Lab 1 - Baseline Models
# Goal: Clean dataset according to lab requirements
# (keep target variable even if it’s non-numeric)

import pandas as pd

def clean_data(input_file="traffic_accidents.csv",
               output_file="traffic_accidents_cleaned.csv",
               target_col="crash_type"):
    """
    Cleans dataset while keeping the target column (even if non-numeric).
    Steps:
      1. Drop fully empty columns.
      2. Drop rows with any missing values.
      3. Keep only numeric columns + the target column.
    """

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    # Drop rows with missing values
    df = df.dropna(axis=0, how='any')

    # Keep numeric features + target column
    numeric_df = df.select_dtypes(include=['number'])
    if target_col in df.columns:
        numeric_df[target_col] = df[target_col]
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Encode target if it's categorical
    if numeric_df[target_col].dtype == 'object':
        numeric_df[target_col] = numeric_df[target_col].astype('category').cat.codes

    print(f"Cleaned shape: {numeric_df.shape}")
    numeric_df.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved to: {output_file}")

    return numeric_df


if __name__ == "__main__":
    clean_data()
