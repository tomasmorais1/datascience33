# ==========================================================
#  Data Profiling - Data Dimensionality
#  Script pronto a correr COM dslabs_functions
#  Implementado exatamente como no exemplo fornecido
# ==========================================================

import os
from pandas import read_csv, DataFrame, Series, to_numeric, to_datetime
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart

# ==========================================================
# CONFIGURAÇÕES — ALTERAR AQUI
# ==========================================================
filename = "/Users/tomasmorais/Documents/MEIC/DS/datascience33/Lab1/data/Combined_Flights_2022.csv"
target_var = "Cancelled"                 # <-- ALTERA
index_col = None                      # id se existir, senão None

# Criar pasta das imagens se não existir
os.makedirs("images", exist_ok=True)

# ==========================================================
# 1) CARREGAR O DATASET (FIEL AO EXEMPLO)
# ==========================================================
data: DataFrame = read_csv(
    filename,
    na_values="",           # <--- EXATAMENTE COMO NO EXEMPLO
    index_col=index_col
)

# Limitar a 200000 linhas aleatórias
sample_size = 200000
if data.shape[0] > sample_size:
    data = data.sample(n=sample_size, random_state=42)

print("\nDataset carregado.")
print("Dimensões:", data.shape)



# ==========================================================
# 2) DIMENSIONALIDADE — Nº REGISTOS VS Nº VARIÁVEIS
# ==========================================================
figure(figsize=(4, 2))
values: dict[str, int] = {
    "nr records": data.shape[0],
    "nr variables": data.shape[1]
}

plot_bar_chart(
    list(values.keys()),
    list(values.values()),
    title="Nr of records vs nr variables"
)

savefig(f"images/records_variables.png")
show()


# ==========================================================
# 3) MISSING VALUES (EXATAMENTE COMO NO EXEMPLO)
# ==========================================================
mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)

savefig(f"images/_mv.png")
show()


# ==========================================================
# 4) FUNÇÃO PARA IDENTIFICAR TIPOS DE VARIÁVEIS (COPIADA DO EXEMPLO)
# ==========================================================
def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:

        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types


# ==========================================================
# 5) OBTER TIPOS E FAZER O GRÁFICO (FIEL AO EXEMPLO)
# ==========================================================
variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)

counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()),
    list(counts.values()),
    title="Nr of variables per type"
)

savefig(f"images/variable_types.png")
show()


# ==========================================================
# 6) CONVERTER SIMBÓLICAS EM CATEGORY (FIEL AO EXEMPLO)
# ==========================================================
symbolic: list[str] = variable_types["symbolic"]
data[symbolic] = data[symbolic].apply(lambda x: x.astype("category"))

print("\nDtypes após conversão:")
print(data.dtypes)


print("\n✔ Data Dimensionality completo!")
print("Imagens guardadas na pasta images/")
