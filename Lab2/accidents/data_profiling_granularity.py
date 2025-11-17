# ==========================================================
#  Data Profiling - Data Granularity
#  Script aligned with IST DS Lab specification
#  Uses dslabs_functions and matches the template exactly
# ==========================================================

from pandas import DataFrame, read_csv, Series
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show

from dslabs_functions import (
    get_variable_types,
    plot_bar_chart,
    define_grid,
    HEIGHT,
)

# ==========================================================
# CONFIG — EDIT HERE
# ==========================================================
filename = "/Users/guilhermeribeiro/Desktop/Mestrado/Data Science/pratico/traffic_accidents.csv"
file_tag = "stroke"
index_col = None

# ==========================================================
# LOAD DATA
# ==========================================================
data: DataFrame = read_csv(filename, index_col=index_col, na_values="", parse_dates=True)
variables_types = get_variable_types(data)

# ==========================================================
# FUNCTION: derive_date_variables
# ==========================================================
def derive_date_variables(df: DataFrame, date_vars: list[str]) -> DataFrame:
    for date in date_vars:
        df[date + "_year"] = df[date].dt.year
        df[date + "_quarter"] = df[date].dt.quarter
        df[date + "_month"] = df[date].dt.month
        df[date + "_day"] = df[date].dt.day
    return df

# ==========================================================
# FUNCTION: analyse_date_granularity
# ==========================================================
def analyse_date_granularity(data: DataFrame, var: str, levels: list[str]) -> ndarray:
    cols: int = len(levels)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {var}")

    for i in range(cols):
        counts: Series[int] = data[var + "_" + levels[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=levels[i],
            xlabel=levels[i],
            ylabel="nr records",
            percentage=False,
        )
    return axs

# ==========================================================
# APPLY GRANULARITY TO DATE VARIABLES
# ==========================================================
date_vars = variables_types["date"]

if len(date_vars) > 0:
    data_ext: DataFrame = derive_date_variables(data, date_vars)

    for v_date in date_vars:
        analyse_date_granularity(data_ext, v_date, ["year", "quarter", "month", "day"])
        savefig(f"images/{file_tag}_granularity_{v_date}.png")
        show()
else:
    print("There are no date variables.")

# ==========================================================
# EXTRA: GRANULARITY FOR EXISTING TEMPORAL VARIABLES
# (crash_hour, crash_day_of_week, crash_month)
# ==========================================================
temporal_vars = ["crash_hour", "crash_day_of_week", "crash_month"]
temporal_vars = [v for v in temporal_vars if v in data.columns]

if len(temporal_vars) > 0:
    analyse_property_granularity = lambda df, prop, vars: (
        __import__("matplotlib.pyplot").subplots(1, len(vars), figsize=(len(vars) * HEIGHT, HEIGHT), squeeze=False)[1]
    )  # placeholder override removed later

    # Proper definition using the standard lab function:
    def analyse_property_granularity(data: DataFrame, prop: str, vars: list[str]) -> ndarray:
        cols: int = len(vars)
        fig: Figure
        axs: ndarray
        fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
        fig.suptitle(f"Granularity study for {prop}")

        for i in range(cols):
            counts: Series[int] = data[vars[i]].value_counts()
            plot_bar_chart(
                counts.index.to_list(),
                counts.to_list(),
                ax=axs[0, i],
                title=vars[i],
                xlabel=vars[i],
                ylabel="nr records",
                percentage=False,
            )
        return axs

    analyse_property_granularity(data, "temporal_components", temporal_vars)
    savefig(f"images/{file_tag}_granularity_temporal_components.png")
    show()
else:
    print("No temporal component variables (hour/day_of_week/month) found.")

# ==========================================================
# SYMBOLIC LOCATION-LIKE GRANULARITY (empty for your dataset)
# ==========================================================
location_vars = []  # no location hierarchy in this dataset

if len(location_vars) > 0:
    analyse_property_granularity(data, "location", location_vars)
    savefig(f"images/{file_tag}_granularity_location.png")
    show()
else:
    print("No location-like variables defined.")

print("\n✔ Granularity analysis completed.")
