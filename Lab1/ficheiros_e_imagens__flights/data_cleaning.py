import pandas as pd

def clean_data_ml(
    input_file="Combined_Flights_2022.csv",
    output_file="Combined_Flights_2022_cleaned.csv",
    target_col="Cancelled"
):
    """
    Cleaning para treinar modelos preditivos de cancelamento.
    Mantém apenas features pre-voo e remove qualquer coluna que indique
    cancelamento ou aconteça após o voo.
    """

    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")

    # 1️⃣ Remover colunas completamente vazias
    df = df.dropna(axis=1, how='all')

    # 2️⃣ Remover colunas pós-voo / leakage conhecido
    leakage_cols = [
        "WheelsOff", "WheelsOn", "AirTime", "TaxiOut", "TaxiIn",
        "ActualElapsedTime", "ArrivalDelay", "DepartureDelay",
        "ArrTime", "DepTime", "TailNum", "FlightNum"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    # 3️⃣ Manter target e preencher NaNs restantes com 0
    numeric_df = df.select_dtypes(include=['number', 'bool']).copy()
    if target_col not in numeric_df.columns:
        numeric_df[target_col] = df[target_col]
    
    numeric_df = numeric_df.fillna(0)

    # 4️⃣ Converter target boolean para int
    if numeric_df[target_col].dtype == "bool":
        numeric_df[target_col] = numeric_df[target_col].astype(int)

    print(f"Cleaned shape: {numeric_df.shape}")

    # 5️⃣ Salvar CSV pronto para ML
    numeric_df.to_csv(output_file, index=False)
    print(f"✅ Saved cleaned dataset to: {output_file}")

    # 6️⃣ Mostrar contagem de classes
    print("\nTarget value counts:")
    print(numeric_df[target_col].value_counts())

    return numeric_df

if __name__ == "__main__":
    clean_data_ml()
