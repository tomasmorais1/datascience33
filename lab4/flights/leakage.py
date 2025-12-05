import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO ---
TRAIN_FILE = "train_scaled.csv"
TARGET = "Cancelled"

print("--- A carregar dados... ---")
try:
    df = pd.read_csv(TRAIN_FILE)
except FileNotFoundError:
    print("Erro: train_scaled.csv não encontrado.")
    exit()

# Usar apenas uma amostra para ser instantâneo
if len(df) > 10000:
    df = df.sample(10000, random_state=42)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("--- A treinar modelo rápido para detetar 'batota'... ---")
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
clf.fit(X, y)

# Calcular Importâncias
importances = clf.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

print("\n" + "="*40)
print("TOP 10 VARIÁVEIS MAIS IMPORTANTES")
print("(Se a primeira for > 0.5, é Data Leakage)")
print("="*40)

for i in range(10):
    if i < len(features):
        print(f"{i+1}. {features[indices[i]]:<25} -> {importances[indices[i]]:.4f}")

print("="*40)