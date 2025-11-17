# ==========================================================
#  Data Profiling - Data Distribution
#  Script pronto a correr COM dslabs_functions
#  100% alinhado com o exemplo fornecido
# ==========================================================

from pandas import DataFrame, read_csv, Series
from numpy import ndarray, log
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, savefig, show, subplots
from scipy.stats import norm, expon, lognorm

from dslabs_functions import (
    plot_bar_chart,
    plot_multibar_chart,
    plot_multiline_chart,
    set_chart_labels,
    define_grid,
    get_variable_types,
    HEIGHT,
)

# ==========================================================
# CONFIGURAÇÕES — ALTERAR AQUI
# ==========================================================
filename = "/Users/guilhermeribeiro/Desktop/Mestrado/Data Science/pratico/traffic_accidents.csv"     # <-- ALTERA
file_tag = "stroke"              # <-- ALTERA
target = "crash_type"                     # <-- ALTERA (nome da variável target)
index_col = None                      # coluna ID se existir

# ==========================================================
# CARREGAR DATASET
# ==========================================================
data: DataFrame = read_csv(filename, index_col=index_col, na_values="")
variables_types = get_variable_types(data)

numeric: list[str] = variables_types["numeric"]
symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]

# ==========================================================
# SUMMARY (describe)
# ==========================================================
summary5: DataFrame = data.describe(include="all")
summary5.to_csv(f"images/{file_tag}_summary.csv")  # opcional

print(summary5)

# ==========================================================
# GLOBAL BOXPLOT PARA VARIÁVEIS NUMÉRICAS
# ==========================================================
if [] != numeric:
    data[numeric].boxplot(rot=45)
    savefig(f"images/{file_tag}_global_boxplot.png")
    show()
else:
    print("There are no numeric variables.")

# ==========================================================
# BOXPLOTS INDIVIDUAIS
# ==========================================================
if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title(f"Boxplot for {numeric[n]}")
        axs[i, j].boxplot(data[numeric[n]].dropna().values)

        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"images/{file_tag}_single_boxplots.png")
    show()
else:
    print("There are no numeric variables.")

# ==========================================================
# OUTLIERS (IQR e stdev)
# ==========================================================
NR_STDEV = 2
IQR_FACTOR = 1.5


def determine_outlier_thresholds_for_var(summary5: Series, std_based=True, threshold=NR_STDEV):
    if std_based:
        std = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom


def count_outliers(data: DataFrame, numeric: list[str], nrstdev=NR_STDEV, iqrfactor=IQR_FACTOR):
    outliers_iqr = []
    outliers_stdev = []
    summary = data[numeric].describe()

    for var in numeric:
        # STDEV
        top, bottom = determine_outlier_thresholds_for_var(summary[var], std_based=True, threshold=nrstdev)
        outliers_stdev.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

        # IQR
        top, bottom = determine_outlier_thresholds_for_var(summary[var], std_based=False, threshold=iqrfactor)
        outliers_iqr.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}


if [] != numeric:
    outliers = count_outliers(data, numeric)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers,
        title="Nr of standard outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"images/{file_tag}_outliers_standard.png")
    show()

    # versão ajustada (4 stdev e 4.5 IQR)
    outliers_adjusted = count_outliers(data, numeric, nrstdev=4, iqrfactor=4.5)
    figure(figsize=(12, HEIGHT))
    plot_multibar_chart(
        numeric,
        outliers_adjusted,
        title="Nr of outliers per variable",
        xlabel="variables",
        ylabel="nr outliers",
        percentage=False,
    )
    savefig(f"images/{file_tag}_outliers.png")
    show()
else:
    print("There are no numeric variables.")

# ==========================================================
# HISTOGRAMAS NUMÉRICOS (simples)
# ==========================================================
if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="nr records",
        )
        axs[i, j].hist(data[numeric[n]].dropna().values, bins="auto")

        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"images/{file_tag}_single_histograms_numeric.png")
    show()
else:
    print("There are no numeric variables.")

# ==========================================================
# HISTOGRAMS + DISTRIBUTIONS (SAFE VERSION)
# ==========================================================

from numpy import linspace

def compute_known_distributions_safe(values: list) -> dict:
    distributions = {}

    # Use reduced range for smoother curves
    x = linspace(min(values), max(values), 200)

    # --- Normal ---
    try:
        mean, sigma = norm.fit(values)
        distributions[f"Normal({mean:.1f},{sigma:.2f})"] = norm.pdf(x, mean, sigma)
    except Exception:
        pass

    # --- Exponential ---
    try:
        loc, scale = expon.fit(values)
        distributions[f"Exp({1/scale:.2f})"] = expon.pdf(x, loc, scale)
    except Exception:
        pass

    # --- Lognormal ---
    try:
        # lognorm fails if there are zeros or negatives
        if all(v > 0 for v in values):
            sigma, loc, scale = lognorm.fit(values)
            distributions[f"LogNor({log(scale):.1f},{sigma:.2f})"] = lognorm.pdf(
                x, sigma, loc, scale
            )
    except Exception:
        pass

    return x, distributions


def histogram_with_distributions(ax, series: Series, var: str):
    values = series.dropna().sort_values().to_list()

    # reduce sample if dataset is huge
    if len(values) > 5000:
        values = values[::10]  # take every 10th element

    ax.hist(values, 20, density=True)

    x, distributions = compute_known_distributions_safe(values)

    if len(distributions) > 0:
        plot_multiline_chart(
            x,
            distributions,
            ax=ax,
            title=f"Best fit for {var}",
            xlabel=var,
            ylabel="density",
        )
    else:
        ax.set_title(f"Histogram (no distribution fits) – {var}")


# ----- Run the charts -----
if [] != numeric:
    rows, cols = define_grid(len(numeric))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    i, j = 0, 0
    for n in range(len(numeric)):
        print(f"Processing distribution fit for: {numeric[n]}")
        histogram_with_distributions(axs[i, j], data[numeric[n]], numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"images/{file_tag}_histogram_numeric_distribution.png")
    show()
else:
    print("There are no numeric variables.")

# ==========================================================
# HISTOGRAMAS PARA VARIÁVEIS SIMBÓLICAS & BINÁRIAS
# ==========================================================
if [] != symbolic:
    rows, cols = define_grid(len(symbolic))
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    i, j = 0, 0
    for n in range(len(symbolic)):
        counts: Series = data[symbolic[n]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[i, j],
            title=f"Histogram for {symbolic[n]}",
            xlabel=symbolic[n],
            ylabel="nr records",
            percentage=False,
        )

        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"images/{file_tag}_histograms_symbolic.png")
    show()
else:
    print("There are no symbolic variables.")

# ==========================================================
# DISTRIBUIÇÃO DA CLASSE (TARGET)
# ==========================================================
values: Series = data[target].value_counts()

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)

savefig(f"images/{file_tag}_class_distribution.png")
show()
