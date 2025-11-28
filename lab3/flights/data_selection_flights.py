# ============================================================
# DATA SELECTION & REDUCTION SCRIPT (ROBUST FOR CANCELLED)
# ============================================================

import pandas as pd
import numpy as np
import os

FILE = "Combined_Flights_2022.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/flights_reduced_5pct.csv"

# 1. LOAD DATA
df = pd.read_csv(FILE)
print("\n=== DATA LOADED ===")
print("Shape:", df.shape)

# 2. KEEP ONLY RELEVANT COLUMNS
important_columns = [
    "FlightDate",
    "Airline",
    "Origin", "Dest",
    "Cancelled", "Diverted",

    "CRSDepTime", "DepTime", "DepDelayMinutes", "DepDelay",
    "ArrTime", "ArrDelayMinutes", "ArrDelay",

    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "AirTime", "CRSElapsedTime", "ActualElapsedTime",

    "Distance", "DistanceGroup",

    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",

    "DepDel15", "ArrivalDelayGroups"
]

df = df[important_columns]
print("After column reduction:", df.shape)

print("\n=== ORIGINAL CANCELLED DISTRIBUTION (RAW) ===")
print(df["Cancelled"].value_counts(dropna=False))

# ------------------------------------------------------------
# 3. BUILD A ROBUST MASK FOR CANCELLED FLIGHTS
#    (handles 0/1, True/False, "0"/"1"/"True"/"False")
# ------------------------------------------------------------
col = df["Cancelled"]

if col.dtype == bool:
    cancelled_mask = col
elif np.issubdtype(col.dtype, np.number):
    cancelled_mask = (col == 1)
else:
    cancelled_mask = col.astype(str).isin(["1", "True", "true", "TRUE"])

df_cancelled = df[cancelled_mask].copy()
df_non_cancelled = df[~cancelled_mask].copy()

print("\nCancelled flights BEFORE cleaning:", df_cancelled.shape[0])
print("Non-cancelled flights BEFORE cleaning:", df_non_cancelled.shape[0])

# ------------------------------------------------------------
# 4. BASIC CLEANING **ONLY ON NON-CANCELLED FLIGHTS**
#    Cancelled flights often have missing/invalid times -> keep them!
# ------------------------------------------------------------
df_nc = df_non_cancelled.copy()

df_nc = df_nc.dropna(subset=["CRSDepTime", "DepTime", "ArrTime"])

numeric_cols = [
    "DepDelayMinutes", "ArrDelayMinutes",
    "TaxiOut", "TaxiIn", "AirTime",
    "CRSElapsedTime", "ActualElapsedTime"
]

for c in numeric_cols:
    if c in df_nc.columns:
        df_nc = df_nc[df_nc[c] >= 0]

print("\n=== AFTER CLEANING NON-CANCELLED ONLY ===")
print("Non-cancelled kept:", df_nc.shape[0])
print("Cancelled kept (unchanged):", df_cancelled.shape[0])

# Recombine
cleaned = pd.concat([df_cancelled, df_nc], ignore_index=True)
print("\nCombined cleaned dataset:", cleaned.shape)

print("\n=== CANCELLED DISTRIBUTION AFTER CLEANING ===")
print(cleaned["Cancelled"].value_counts(dropna=False))

# ------------------------------------------------------------
# 5. SMART SAMPLING:
#    KEEP ALL CANCELLED + SAMPLE 5% OF NON-CANCELLED
# ------------------------------------------------------------
# Recompute mask on cleaned
col_c = cleaned["Cancelled"]
if col_c.dtype == bool:
    cancelled_mask_clean = col_c
elif np.issubdtype(col_c.dtype, np.number):
    cancelled_mask_clean = (col_c == 1)
else:
    cancelled_mask_clean = col_c.astype(str).isin(["1", "True", "true", "TRUE"])

clean_cancelled = cleaned[cancelled_mask_clean].copy()
clean_non_cancelled = cleaned[~cancelled_mask_clean].copy()

print("\nCancelled flights AFTER cleaning:", clean_cancelled.shape[0])
print("Non-cancelled flights AFTER cleaning:", clean_non_cancelled.shape[0])

sample_fraction = 0.05

sampled_non_cancelled = clean_non_cancelled.sample(frac=sample_fraction,
                                                   random_state=42)

sampled = pd.concat([clean_cancelled, sampled_non_cancelled])\
            .sample(frac=1, random_state=42)

print("\n=== FINAL SAMPLED DATASET ===")
print("Shape:", sampled.shape)
print("Cancelled distribution:")
print(sampled["Cancelled"].value_counts(dropna=False))
print("Cancelled rate (%):", round(100 * (
    sampled["Cancelled"].astype(str).isin(["1", "True", "true", "TRUE"]).mean()
), 2))

# ------------------------------------------------------------
# 6. SAVE RESULT
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
sampled.to_csv(OUTPUT_FILE, index=False)

print(f"\n=== SAVED TO: {OUTPUT_FILE} ===\n")
