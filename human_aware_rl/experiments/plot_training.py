#!/usr/bin/env python3

'''
Utility script for generating plots from data stored in RLLib-generated CSV files
'''

import argparse
from collections import namedtuple
import os
import pandas  # NOTE: Pandas may not be an option, as it seems to throw a security error - even if this script is not itself a vulnerability.

import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np
import scipy
import scipy.stats

def parse_args():
    parser = argparse.ArgumentParser("Generates a plot of a set of RLLib experiments for the specified metrics.")
# experiments "custom_metrics/sparse_reward_mean ~/ray_results/PPO_cramped_room_False_nw=2_mini=2k_lr=0.001_gamma=0.9_lambda=0.95_vf=0.5_kl=0.2_clip=0.2_sgd=8"
    parser.add_argument("experiments", type=str, nargs="*",
                        help="labels and directories of experiments to plot (label1 dir1 label2 dir2 ...)")
    parser.add_argument("--output", default="mean_return", type=str,
                        help="path to the image file where the plot will be saved")
    parser.add_argument("--x-axis", default="timesteps_total", type=str,
                        help="column name for x-axis values")
    parser.add_argument("--y-axis", default="custom_metrics/sparse_reward_mean", type=str,
                        help="column name for y-axis values")
    parser.add_argument("--x-label", default="Environment Timesteps", type=str,
                        help="label for the x-axis")
    parser.add_argument("--y-label", default="Mean episode reward", type=str,
                        help="label for the y-axis")
    parser.add_argument("--title", default="Mean Episode Reward during Training", type=str,
                        help="title for the plot to be generated")
    parser.add_argument("--errors", default="error", type=str,
                        help="error values to plot as shaded regions \{'range', 'deviation', 'error', 'None'\}")
    parser.add_argument("--incomplete", default="truncate", type=str,
                        help="how to handle incomplete runs \{'ignore', 'truncate'\}")

    return parser.parse_args()


def load_experiments(args):
    if len(args) % 2 != 0:
        print(args)
        raise Exception("Must provide a label for each experiment")

    experiments = dict()

    for index in range(0, len(args), 4):
        directory = args[index + 1]
        directory2 = args[index + 2]
        directory3 = args[index + 3]
        runs = []

        if not os.path.isdir(directory):
            raise Exception(f"Experiment directory {directory} does not exist")

        data = pandas.read_csv(os.path.join(directory, "progress.csv"))
        data2 = pandas.read_csv(os.path.join(directory2, "progress.csv"))
        data3 = pandas.read_csv(os.path.join(directory3, "progress.csv"))

        data_inter = data.add(data2, fill_value=0)
        data_final = data_inter.add(data3, fill_value=0)
        for col in data_final.columns:
            if type(data_final[col].values[0]) == np.int64 or type(data_final[col].values[0]) == np.float64:
                data_final[col] = data_final[col] / 3
        # Filter out empy data series
        if data_final.shape[0] > 0:
            runs.append(data_final)
        alg_name = args[index]

        col = 'custom_metrics/sparse_reward_mean'
        mse = scipy.stats.sem([data[col], data2[col], data3[col]], axis=0, ddof=1)

        experiments[alg_name + str(index) if alg_name in experiments else alg_name] = (runs, mse)

    return experiments


if __name__ == "__main__":
    args = parse_args()

    # Load experiment data
    experiments = load_experiments(args.experiments)

    # Plot results
    color_map = colors.get_cmap("tab20").colors
    legend_entries = []
    y_min = np.infty
    y_max = -np.infty

    plot.clf()
    fig, axs = plot.subplots(1, 3, figsize=(16,4))

    for index, (label, (runs, mse)) in enumerate(experiments.items()):
        i = int(index/2)
        if len(runs) > 0:
            lengths = [len(run) for run in runs]

            if "ignore" == args.incomplete:
                # Compute the maximum length over runs
                max_length = min(lengths)

                # Remove incomplete sequences
                runs = [run for run in runs if len(run) == max_length]

                # Define x-axis
                x_axis = runs[0][args.x_axis]

                # Construct data series and compute means
                series = [run[args.y_axis] for run in runs]
            else:
                # Compute the minimum length over runs
                min_length = min(lengths)

                # Define x-axis
                x_axis = runs[0][args.x_axis][0:min_length]

                # Print run information
                print(f"\n\nExperiment: {label}")
                print(f"    data keys: {runs[0].keys()}")

                # Construct data series and compute means
                series = [run[args.y_axis][0:min_length] for run in runs]

            # Convert series data to a single numpy array
            series = np.asarray(series, dtype=np.float32)
            means = np.mean(series, axis=0)

            # Update ranges
            y_min = min(y_min, np.min(series))
            y_max = max(y_max, np.max(series))

            # Compute error bars
            if "range" == args.errors:
                upper = np.max(series, axis=0)
                lower = np.min(series, axis=0)
            elif "deviation" == args.errors:
                std = np.std(series, axis=0, ddof=1)
                upper = means + std
                lower = means - std
            elif "error" == args.errors:
                upper = means + mse
                lower = means - mse
            else:
                upper = means
                lower = means
            axs[i].plot(x_axis, means, color=color_map[2 * index - i * 4], alpha=1.0)
            axs[i].fill_between(x_axis, lower, upper, color=color_map[2 * index - i * 4 + 1], alpha=0.3)

        # Add a legend entry even if there were no non-empty data series
        if i < 1:
            legend_entries.append(patches.Patch(color=color_map[2 * index], label=label))

    # Set ranges
    if y_min > y_max:  # No data, set an arbitrary range
        y_min = 0.0
        y_max = 100.0
    elif 0.0 == y_min and 0.0 == y_max:  # All data is zero, set and arbitrary range
        y_min = -100.0
        y_max = 100.0
    elif y_min > 0.0:  # All values positive, set range from 0 to 120% of max
        y_min = 0.0
        y_max *= 1.2
    elif y_max < 0.0:  # All values negative, set range from 120% of min to 0
        y_min *= 1.2
        y_max = 0.0
    else:  # Both positive and negative values, expand range by 20%
        y_min *= 1.2
        y_max *= 1.2

    # Create plot
    for ax in axs:
        ax.legend(handles=legend_entries)
        ax.set_xlabel(args.x_label)
        ax.set_ylabel(args.y_label)
        ax.set_ylim(bottom=y_min, top=y_max)
    axs[0].set_title("Cramped Room")
    axs[1].set_title("Asymmetric Advantages")
    axs[2].set_title("Coordination Ring")
    fig.suptitle(args.title)
    fig.savefig(args.output, bbox_inches="tight")