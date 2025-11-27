# ============================================================
# FEATURE GENERATION PIPELINE – RAW DATA ONLY
# Creates meaningful features for train & test without leakage
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.labelsize"] = 12

# ------------------------------------------------------------
# Ensure output directory exists
# ------------------------------------------------------------
os.makedirs("prepared_data/feature_plots", exist_ok=True)

RAW_COLUMNS = [
    "crash_date",
    "traffic_control_device",
    "weather_condition",
    "lighting_condition",
    "first_crash_type",
    "trafficway_type",
    "alignment",
    "roadway_surface_cond",
    "road_defect",
    "crash_type",
    "intersection_related_i",
    "damage",
    "prim_contributory_cause",
    "num_units",
    "most_severe_injury",
    "injuries_total",
    "injuries_fatal",
    "injuries_incapacitating",
    "injuries_non_incapacitating",
    "injuries_reported_not_evident",
    "injuries_no_indication",
    "crash_hour",
    "crash_day_of_week",
    "crash_month"
]

# ------------------------------------------------------------
# Feature Generation Function (applied to both train & test)
# ------------------------------------------------------------
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Parse Date ---
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")

    df["crash_year"] = df["crash_date"].dt.year
    df["crash_day"] = df["crash_date"].dt.day
    df["crash_weekday"] = df["crash_date"].dt.weekday
    df["crash_hour_from_date"] = df["crash_date"].dt.hour

    # Add safety fallback: prefer existing crash_hour if valid
    df["crash_hour"] = df["crash_hour"].fillna(df["crash_hour_from_date"])

    # --- Weather-based Features ---
    df["is_rain"] = df["weather_condition"].astype(str).str.contains("RAIN|WET", case=False, na=False).astype(int)
    df["is_snow"] = df["weather_condition"].astype(str).str.contains("SNOW", case=False, na=False).astype(int)
    df["is_fog"]  = df["weather_condition"].astype(str).str.contains("FOG|MIST", case=False, na=False).astype(int)
    df["is_clear"] = df["weather_condition"].astype(str).str.contains("CLEAR", case=False, na=False).astype(int)

    # --- Lighting Conditions ---
    df["is_dark"] = df["lighting_condition"].astype(str).str.contains("DARK", case=False, na=False).astype(int)
    df["is_daylight"] = df["lighting_condition"].astype(str).str.contains("DAYLIGHT", case=False, na=False).astype(int)

    # --- Road Conditions ---
    df["is_wet_surface"]  = df["roadway_surface_cond"].astype(str).str.contains("WET", case=False, na=False).astype(int)
    df["is_snow_surface"] = df["roadway_surface_cond"].astype(str).str.contains("SNOW|SLUSH", case=False, na=False).astype(int)
    df["is_dry_surface"]  = df["roadway_surface_cond"].astype(str).str.contains("DRY", case=False, na=False).astype(int)

    # --- Interaction Features ---
    df["night_and_rain"] = (df["is_dark"] & df["is_rain"]).astype(int)
    df["snow_and_dark"] = (df["is_snow"] & df["is_dark"]).astype(int)
    df["rain_and_turning"] = (
        df["is_rain"] & df["first_crash_type"].astype(str).str.contains("TURN", case=False, na=False)
    ).astype(int)

    # Remove intermediate date column (optional)
    df.drop(columns=["crash_date"], inplace=True)

    return df


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def plot_feature_distribution(df, column, title):
    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, x=column)
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

    train_raw = pd.read_csv("prepared_data/train_raw.csv", names=RAW_COLUMNS, header=None)
    test_raw  = pd.read_csv("prepared_data/test_raw.csv",  names=RAW_COLUMNS, header=None)

    print(f"Train shape: {train_raw.shape}")
    print(f"Test shape: {test_raw.shape}")

    print("▶ Generating features for TRAIN...")
    train_feat = generate_features(train_raw)

    print("▶ Generating features for TEST...")
    test_feat = generate_features(test_raw)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    train_feat.to_csv("prepared_data/train_features.csv", index=False)
    test_feat.to_csv("prepared_data/test_features.csv", index=False)

    print("▶ Saved:")
    print(" - prepared_data/train_features.csv")
    print(" - prepared_data/test_features.csv")

    # --------------------------------------------------------
    # Generate diagnostic plots
    # --------------------------------------------------------
    print("▶ Creating plots...")

    binary_features = [
        "is_rain", "is_snow", "is_fog", "is_clear",
        "is_dark", "is_daylight",
        "is_wet_surface", "is_snow_surface", "is_dry_surface",
        "night_and_rain", "snow_and_dark", "rain_and_turning"
    ]

    for col in binary_features:
        print(f"   Plotting {col} ...")
        plot_feature_distribution(train_feat, col, f"Distribution of {col}")

    print("✔ DONE – All features generated and plots exported.")
