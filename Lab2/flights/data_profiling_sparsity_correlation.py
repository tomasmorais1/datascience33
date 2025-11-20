# ==========================================================
#  Data Profiling - Sparsity & Correlation (SAFE VERSION)
# ==========================================================

from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from seaborn import heatmap

from dslabs_functions import (
    HEIGHT,
    plot_multi_scatters_chart,
    get_variable_types,
)

# ==========================================================
# CONFIG
# ==========================================================
filename = "/Users/tomasmorais/Documents/MEIC/DS/datascience33/Lab1/data/Combined_Flights_2022.csv"
file_tag = "flights"              # <-- podes alterar se quiseres
target = "Cancelled"
index_col = None

# ==========================================================
# LOAD DATA
# ==========================================================
data: DataFrame = read_csv(filename, index_col=index_col, na_values="")
data = data.dropna()

# ==========================================================
# REDUCE DATASET SIZE (CRITICAL!)
# ==========================================================
if len(data) > 5000:
    data = data.sample(5000, random_state=42)

# ==========================================================
# ONLY USE NUMERIC VARIABLES FOR SPARSITY
# ==========================================================
variables_types = get_variable_types(data)
numeric_vars = variables_types["numeric"]

if len(numeric_vars) < 2:
    print("Not enough numeric variables for sparsity analysis.")
else:
    n = len(numeric_vars) - 1

    # ==========================================================
    # SPARSITY STUDY (WITHOUT CLASS)
    # ==========================================================
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    fig.suptitle("Sparsity Study (Numeric Variables)")

    for i in range(len(numeric_vars)):
        for j in range(i + 1, len(numeric_vars)):
            plot_multi_scatters_chart(
                data, numeric_vars[i], numeric_vars[j], ax=axs[i, j - 1]
            )

    savefig(f"images/{file_tag}_sparsity_study.png")
    show()

    # ==========================================================
    # SPARSITY STUDY PER CLASS
    # ==========================================================
    fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    fig.suptitle("Sparsity Per Class (Numeric Variables)")

    for i in range(len(numeric_vars)):
        for j in range(i + 1, len(numeric_vars)):
            plot_multi_scatters_chart(
                data, numeric_vars[i], numeric_vars[j], target, ax=axs[i, j - 1]
            )

    savefig(f"images/{file_tag}_sparsity_per_class_study.png")
    show()

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================
numeric = numeric_vars

if len(numeric) > 1:
    corr_mtx: DataFrame = data[numeric].corr().abs()

    figure()
    heatmap(
        corr_mtx,
        xticklabels=numeric,
        yticklabels=numeric,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    savefig(f"images/{file_tag}_correlation_analysis.png")
    show()
else:
    print("Not enough numeric variables for correlation analysis.")
