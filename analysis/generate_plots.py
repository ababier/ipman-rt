import os
import re
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from algorithms.evaluation.summary import Summary

from analysis.plot_iterations import plot_iterations


def plot_all_from_summary_csv(experiment_summary: Summary, save_dir: Path):
    column_to_plot_with_axis_label = {"objective": "Objective function value", "feasibility": "Fraction of Feasible Plans"}
    experiment_summary.values["feasibility"] = experiment_summary.check_feasibility(experiment_summary.relaxed_constraints)
    relevant_columns = ["lambda_", "iteration", *column_to_plot_with_axis_label]
    df_means = experiment_summary.values[relevant_columns].groupby(["lambda_", "iteration"], as_index=False).mean()
    df_ipman_means = df_means[df_means.iteration > 0]
    os.makedirs(save_dir, exist_ok=True)
    for column, axis_label in column_to_plot_with_axis_label.items():
        fig, ax = plt.subplots()
        plot_iterations(df_ipman_means, column, axis_label, ax)
        fig.savefig(save_dir / f"{column}.png", bbox_inches="tight")


def plot_criteria_satisfaction(summary: Summary, save_dir: Path):
    column_to_plot_with_filename = {
        tuple(summary.hidden_constraints): "hidden_constraints",
        tuple(summary.known_lb_constraints): "known_lb_constraints",
        tuple(summary.known_ub_constraints): "known_ub_constraints",
    }
    relevant_columns = ["lambda_", "iteration", *chain.from_iterable(column_to_plot_with_filename)]
    df_means = summary.values[relevant_columns].groupby(["lambda_", "iteration"], as_index=False).mean()
    df_ipman_means = df_means[df_means.iteration > 0]
    for attributes, filename in column_to_plot_with_filename.items():
        fig, axs = plt.subplots(nrows=1, ncols=len(attributes), sharey="all", figsize=(6 * len(attributes), 6))
        plt.subplots_adjust(wspace=0.08)
        y_range = (np.floor(np.nanmin(df_ipman_means[[*attributes]].values)), np.ceil(np.nanmax(df_ipman_means[[*attributes]].values)))
        is_first_attribute = True
        for attribute, ax in zip(attributes, axs):
            structure = attribute.split("_")[0]
            title = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", structure)
            plot_iterations(df_ipman_means, attribute, "Mean difference (Gy)" if is_first_attribute else None, ax, title, y_range)
            is_first_attribute = False
        fig.savefig(save_dir / f"{filename}.png", bbox_inches="tight")
