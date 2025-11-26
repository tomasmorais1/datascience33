# split_data.py
from pandas import read_csv
from sklearn.model_selection import train_test_split
import os

# ================================================================
# CONFIG
# ================================================================
DATA_PATH = "traffic_accidents.csv/traffic_accidents.csv"
OUTPUT_DIR = "prepared_data"
TARGET = "crash_type"
TEST_SIZE = 0.3
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("Loading raw dataset...")
df = read_csv(DATA_PATH, na_values="")
print("Data loaded. Shape:", df.shape)

# ================================================================
# 2. SPLIT TRAIN / TEST
# ================================================================
print(f"Splitting dataset into train/test (test_size={TEST_SIZE})...")
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET])

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ================================================================
# 3. SAVE SPLIT DATASETS
# ================================================================
train_file = f"{OUTPUT_DIR}/train_raw.csv"
test_file  = f"{OUTPUT_DIR}/test_raw.csv"

train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

print(f"Train and test datasets saved in '{OUTPUT_DIR}' folder.")
