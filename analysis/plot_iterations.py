from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from analysis.plot_parameters import plot_parameters

plt.rcParams.update(plot_parameters)


def plot_iterations(
    df: pd.DataFrame, attribute_name: str, y_axis_label: str, ax: Axes, title: Optional[str] = None, y_range: Optional[tuple[int, int]] = None
):
    markers = ["s", "o", "D", "v", "^"]
    lambdas = df.lambda_.unique()
    colors = ["C{}".format(i) for i in range(10)]
    if len(markers) < len(lambdas):
        markers = ["o" for _ in lambdas]

    # Add data for each lambda_ in
    ordered_lambdas = lambdas[::-1]
    for index, lambda_ in enumerate(ordered_lambdas):
        df_to_plot = df[df.lambda_ == lambda_]
        ax.plot(
            df_to_plot.iteration,
            df_to_plot[attribute_name],
            label=str(lambda_),
            marker=markers[index],
            color=colors[index],
        )
    ax.tick_params(direction="in")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(ncol=4)

    # Clean up title and x/y-axis
    attribute_lb, attribute_ub = y_range or (np.floor(df[attribute_name].min()), np.ceil(df[attribute_name].max()))
    if 0 <= attribute_lb and attribute_ub <= 1:
        ylim = (0, 1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
    else:
        spacing = (attribute_ub - attribute_lb) * 0.1
        steps = max(1, int((attribute_ub - attribute_lb) / 4))
        ylim = (attribute_lb - spacing, attribute_ub + spacing)
        yticks = list(np.arange(attribute_lb, attribute_ub, steps))

    xlim_ub = df.iteration.max() + 1
    ax.plot((0, xlim_ub), [0 for _ in (0, xlim_ub)], label="", marker="", linestyle="--", color="black")
    plot_settings = {
        "title": title,
        "xlabel": "Iteration",
        "xticks": list(range(0, xlim_ub, 2)),
        "xlim": (0, xlim_ub),
        "ylabel": y_axis_label,
        "yticks": yticks,
        "ylim": ylim,
    }

    ax.set(**plot_settings)
