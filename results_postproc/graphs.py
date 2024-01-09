import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("TkAgg")

def exec_times():
    df_par_8 = pd.read_csv("results_par_8.csv", sep="; ")
    df_par_16 = pd.read_csv("results_par_16.csv", sep="; ")
    df_par_32 = pd.read_csv("results_par_32.csv", sep="; ")
    df_seq = pd.read_csv("results_seq.csv", sep="; ")
    df_par_8.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_16.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_32.sort_values(by=["image_size_pixels"], inplace=True)
    df_seq.sort_values(by=["image_size_pixels"], inplace=True)

    width = 1.8  # the width of the bars
    multiplier = -0.5

    sizes = len(df_seq)
    x = np.arange(sizes) * 8

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(15, 8)

    plt.yscale("log")
    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"], width=width,
                   label="sequential")
    labels_seq = [f"{t:.3}" for t in df_seq["execution_time"]]
    ax.bar_label(rects, padding=3, labels=labels_seq)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_par_8["execution_time"], width=width,
                   label="parallel (BLOCK_DIM=8)")
    labels_seq = [f"{t:.3}" for t in df_par_8["execution_time"]]
    ax.bar_label(rects, padding=7, labels=labels_seq)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_par_16["execution_time"], width=width,
                   label="parallel (BLOCK_DIM=16)")
    labels_seq = [f"{t:.3}" for t in df_par_16["execution_time"]]
    ax.bar_label(rects, padding=2, labels=labels_seq)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_par_32["execution_time"], width=width,
                   label="parallel (BLOCK_DIM=32)")
    labels_seq = [f"{t:.3}" for t in df_par_32["execution_time"]]
    ax.bar_label(rects, padding=12, labels=labels_seq)
    multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution time')
    ax.set_title('Execution time')
    ax.set_xticks(x + width, df_seq["image_size"])
    ax.legend(loc='upper left')

    plt.savefig("execution_times")
    plt.close()

def speedup():
    df_par_8 = pd.read_csv("results_par_8.csv", sep="; ")
    df_par_16 = pd.read_csv("results_par_16.csv", sep="; ")
    df_par_32 = pd.read_csv("results_par_32.csv", sep="; ")
    df_seq = pd.read_csv("results_seq.csv", sep="; ")
    df_par_8.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_16.sort_values(by=["image_size_pixels"], inplace=True)
    df_par_32.sort_values(by=["image_size_pixels"], inplace=True)
    df_seq.sort_values(by=["image_size_pixels"], inplace=True)

    width = 0.27  # the width of the bars
    multiplier = 1

    sizes = len(df_seq)
    x = np.arange(sizes)

    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(10, 8)

    ax.bar([], [])
    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"] / df_par_8["execution_time"], width,
                   label="parallel (BLOCK_DIM=8)")
    labels_seq = [f"{t:.3}" for t in df_seq["execution_time"] / df_par_8["execution_time"]]
    ax.bar_label(rects, padding=3, labels=labels_seq)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"] / df_par_16["execution_time"], width,
                   label="parallel (BLOCK_DIM=16)")
    labels_seq = [f"{t:.3}" for t in df_seq["execution_time"] / df_par_16["execution_time"]]
    ax.bar_label(rects, padding=3, labels=labels_seq)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"] / df_par_32["execution_time"], width,
                   label="parallel (BLOCK_DIM=32)")
    labels_seq = [f"{t:.3}" for t in df_seq["execution_time"] / df_par_32["execution_time"]]
    ax.bar_label(rects, padding=3, labels=labels_seq)
    multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup')
    ax.set_xticks(x + width, df_seq["image_size"])
    ax.legend(loc='upper left')

    plt.savefig("speedup")
    plt.close()

if __name__ == "__main__":
    speedup()
    exec_times()