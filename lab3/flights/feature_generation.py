#!/usr/bin/env python3
# ============================================================
# FEATURE GENERATION FOR FLIGHTS DATASET (NO LEAKAGE)
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.labelsize"] = 12

# Ensure output directory exists
os.makedirs("prepared_data/feature_plots", exist_ok=True)


# ------------------------------------------------------------
# Helper: convert HHMM numeric time to minutes since midnight
# ------------------------------------------------------------
def parse_hhmm(value):
    """
    Convert HHMM (e.g., 2135) to minutes since midnight.
    Returns NaN if value is invalid.
    """
    if pd.isna(value):
        return np.nan
    try:
        iv = int(value)
    except (ValueError, TypeError):
        return np.nan

    hh = iv // 100
    mm = iv % 100
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return np.nan
    return hh * 60 + mm


# ------------------------------------------------------------
# FEATURE GENERATION FUNCTION (applied to both train & test)
# ------------------------------------------------------------
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------- DATE FEATURES -----------
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

    df["FlightYear_from_date"] = df["FlightDate"].dt.year
    df["FlightMonth_from_date"] = df["FlightDate"].dt.month
    df["FlightDay_from_date"] = df["FlightDate"].dt.day
    df["FlightWeekday_from_date"] = df["FlightDate"].dt.weekday  # 0=Mon, 6=Sun
    df["IsWeekend"] = (df["FlightWeekday_from_date"] >= 5).astype(int)

    # ----------- TIME FEATURES -----------
    # Scheduled / actual departure & arrival in minutes since midnight
    df["CRSDepMinutes"] = df["CRSDepTime"].apply(parse_hhmm)
    df["DepMinutes"] = df["DepTime"].apply(parse_hhmm)
    df["ArrMinutes"] = df["ArrTime"].apply(parse_hhmm)

    # Derive hours from minutes
    df["CRSDepHour"] = (df["CRSDepMinutes"] // 60).astype("Int64")
    df["DepHour"] = (df["DepMinutes"] // 60).astype("Int64")
    df["ArrHour"] = (df["ArrMinutes"] // 60).astype("Int64")

    # Peak hours (rush windows)
    df["MorningPeak"] = df["CRSDepMinutes"].between(6 * 60, 10 * 60, inclusive="both").astype(int)
    df["EveningPeak"] = df["CRSDepMinutes"].between(16 * 60, 20 * 60, inclusive="both").astype(int)

    # Night / redeye
    df["NightFlight"] = (
        (df["CRSDepHour"] < 6) | (df["CRSDepHour"] >= 22)
    ).astype(int)

    df["RedEye"] = (
        df["CRSDepHour"].between(0, 5, inclusive="both")
    ).astype(int)

    # ----------- ROUTE & DISTANCE FEATURES -----------
    df["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)

    df["IsShortHaul"] = (df["Distance"] < 800).astype(int)
    df["IsMediumHaul"] = ((df["Distance"] >= 800) & (df["Distance"] <= 2500)).astype(int)
    df["IsLongHaul"] = (df["Distance"] > 2500).astype(int)

    # ----------- DELAY FLAGS -----------
    # Already have DepDel15, but we keep it and add further detail
    df["IsDelayedDep"] = (df["DepDelayMinutes"] > 15).astype(int)
    df["IsDelayedArr"] = (df["ArrDelayMinutes"] > 15).astype(int)

    df["BigDepDelay"] = (df["DepDelayMinutes"] >= 60).astype(int)
    df["BigArrDelay"] = (df["ArrDelayMinutes"] >= 60).astype(int)

    # ----------- DURATIONS -----------
    df["TotalTaxiTime"] = df["TaxiOut"].fillna(0) + df["TaxiIn"].fillna(0)

    df["FlightDurationDiff"] = df["ActualElapsedTime"] - df["CRSElapsedTime"]

    # Avoid division by zero in ratio
    df["TaxiRatio"] = np.where(
        df["ActualElapsedTime"] > 0,
        df["TotalTaxiTime"] / df["ActualElapsedTime"],
        np.nan,
    )

    # ----------- INTERACTIONS -----------
    df["NightAndLongHaul"] = (df["NightFlight"] & df["IsLongHaul"]).astype(int)
    df["PeakAndShortHaul"] = (
        (df["MorningPeak"] | df["EveningPeak"]) & df["IsShortHaul"]
    ).astype(int)

    return df


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def plot_feature_distribution(df, column, title):
    plt.figure(figsize=(9, 4))
    if df[column].nunique() <= 10 and df[column].dtype != float:
        # treat as categorical / binary
        sns.countplot(data=df, x=column)
    else:
        sns.histplot(data=df, x=column, kde=False)
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"prepared_data/feature_plots/{column}.png")
    plt.close()


# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------
if __name__ == "__main__":
    print("▶ Loading RAW datasets...")

    train_raw = pd.read_csv("prepared_data/train_raw.csv")
    test_raw  = pd.read_csv("prepared_data/test_raw.csv")

    print(f"Train shape BEFORE features: {train_raw.shape}")
    print(f"Test shape BEFORE features:  {test_raw.shape}")

    print("▶ Generating features for TRAIN...")
    train_feat = generate_features(train_raw)
    print(f"Train shape AFTER features:  {train_feat.shape}")

    print("▶ Generating features for TEST...")
    test_feat = generate_features(test_raw)
    print(f"Test shape AFTER features:   {test_feat.shape}")

    # Save
    train_feat.to_csv("prepared_data/train_features.csv", index=False)
    test_feat.to_csv("prepared_data/test_features.csv", index=False)

    print("\n✔ Features saved to:")
    print("  prepared_data/train_features.csv")
    print("  prepared_data/test_features.csv")

    # Diagnostic plots for selected engineered features
    plot_columns = [
        "CRSDepHour",
        "DepHour",
        "ArrHour",
        "IsWeekend",
        "MorningPeak",
        "EveningPeak",
        "NightFlight",
        "RedEye",
        "IsShortHaul",
        "IsMediumHaul",
        "IsLongHaul",
        "IsDelayedDep",
        "IsDelayedArr",
        "BigDepDelay",
        "BigArrDelay",
        "TotalTaxiTime",
        "FlightDurationDiff",
        "TaxiRatio",
        "NightAndLongHaul",
        "PeakAndShortHaul",
    ]

    print("\n▶ Creating plots for engineered features...")
    for col in plot_columns:
        if col in train_feat.columns:
            print(f"   Plotting {col} ...")
            plot_feature_distribution(train_feat, col, f"Distribution of {col}")

    print("\n✔ DONE – flights feature generation complete.\n")
