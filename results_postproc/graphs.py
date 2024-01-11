import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("TkAgg")

def exec_times():
    df_par_256, df_par_320, df_par_512, df_par_768, df_par_1024, df_seq = get_df()

    width = 1.3  # the width of the bars
    multiplier = -0.5

    sizes = len(df_seq)
    x = np.arange(sizes) * 9

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(12, 6)

    ax.set_prop_cycle(color=plt.colormaps['Set2'].colors)
    plt.yscale("log")

    multiplier = plot_bar(ax, df_seq["execution_time"], multiplier, width, x, "sequential", 3)
    multiplier = plot_bar(ax, df_par_256["execution_time"], multiplier, width, x, "parallel (block 16x16)", 3)
    multiplier = plot_bar(ax, df_par_320["execution_time"], multiplier, width, x, "parallel (block 20x16)", 9)
    multiplier = plot_bar(ax, df_par_512["execution_time"], multiplier, width, x, "parallel (block 32x16)", 3)
    multiplier = plot_bar(ax, df_par_768["execution_time"], multiplier, width, x, "parallel (block 32x24)", 9)
    multiplier = plot_bar(ax, df_par_1024["execution_time"], multiplier, width, x, "parallel (block 32x32)", 3)

    ax.set_ylabel('Execution time')
    ax.set_title('Execution time')
    ax.set_xticks(x + width, df_seq["image_size"])
    ax.legend(loc='upper left')
    plt.savefig("execution_times")
    plt.close()

def speedup():
    df_par_256, df_par_320, df_par_512, df_par_768, df_par_1024, df_seq = get_df()

    width = 1.5  # the width of the bars
    multiplier = -1

    sizes = len(df_seq)
    x = np.arange(sizes) * 8

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(12, 6)
    ax.set_prop_cycle(color=plt.colormaps['Set2'].colors)
    ax.bar([], [])
    multiplier = plot_bar(ax, df_seq["execution_time"] / df_par_256["execution_time"], multiplier, width, x, "block 16x16", 3, True)
    multiplier = plot_bar(ax, df_seq["execution_time"] / df_par_320["execution_time"], multiplier, width, x, "block 20x16", 3, True)
    multiplier = plot_bar(ax, df_seq["execution_time"] / df_par_512["execution_time"], multiplier, width, x, "block 32x16", 3, True)
    multiplier = plot_bar(ax, df_seq["execution_time"] / df_par_768["execution_time"], multiplier, width, x, "block 32x24", 3, True)
    multiplier = plot_bar(ax, df_seq["execution_time"] / df_par_1024["execution_time"], multiplier, width, x, "block 32x32", 3, True)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup')
    ax.set_xticks(x + width, df_seq["image_size"])
    ax.legend(loc='upper left')

    plt.savefig("speedup")
    plt.close()


def get_df():
    df_par_256 = pd.read_csv("results_par_16x16.csv", sep="; ")
    df_par_320 = pd.read_csv("results_par_20x16.csv", sep="; ")
    df_par_512 = pd.read_csv("results_par_32x16.csv", sep="; ")
    df_par_768 = pd.read_csv("results_par_32x24.csv", sep="; ")
    df_par_1024 = pd.read_csv("results_par_32x32.csv", sep="; ")
    df_seq = pd.read_csv("results_seq.csv", sep="; ")
    df_par_256.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_320.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_512.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_768.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_1024.sort_values(by=["image_size_pixels"], inplace=True)
    df_seq.sort_values(by=["image_size_pixels"], inplace=True)
    return df_par_256, df_par_320, df_par_512, df_par_768, df_par_1024, df_seq


def plot_bar(ax, df, multiplier, width, x, label, padding, bar_label=False):
    offset = width * multiplier
    rects = ax.bar(x + offset, df, width,
                   label=label)
    if bar_label:
        labels_seq = [f"{t:.3}" for t in df]
        ax.bar_label(rects, padding=padding, labels=labels_seq)
    multiplier += 1
    return multiplier


if __name__ == "__main__":
    speedup()
    exec_times()
    df_par_256, df_par_320, df_par_512, df_par_768, df_par_1024, df_seq = get_df()
    df_seq.rename(columns={"execution_time": "execution_time_seq"})
    df_seq["execution_time_256"] = df_par_256["execution_time"]
    df_seq["execution_time_320"] = df_par_320["execution_time"]
    df_seq["execution_time_512"] = df_par_512["execution_time"]
    df_seq["execution_time_768"] = df_par_768["execution_time"]
    df_seq["execution_time_1024"] = df_par_1024["execution_time"]
    df_seq.to_csv("results_all.csv", index=False, sep=";")