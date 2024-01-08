import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("TkAgg")

def exec_times():
    df_par = pd.read_csv("results_par.csv", sep="; ")
    df_seq = pd.read_csv("results_seq.csv", sep="; ")
    df_par.sort_values(by=["image_size_pixels"], inplace=True)
    df_seq.sort_values(by=["image_size_pixels"], inplace=True)

    width = 0.25  # the width of the bars
    multiplier = 0.5

    sizes = len(df_par)
    x = np.arange(sizes)

    fig, ax = plt.subplots(layout='constrained')
    plt.yscale("log")
    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"], width,
                   label="sequential")
    # ax.bar_label(rects, padding=3)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, df_par["execution_time"], width,
                   label="parallel")
    # ax.bar_label(rects, padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution time')
    ax.set_title('Execution time')
    ax.set_xticks(x + width, df_par["image_size"])
    ax.legend(loc='upper left', ncols=3)

    plt.savefig("execution_times")
    plt.close()

def speedup():
    df_par = pd.read_csv("results_par.csv", sep="; ")
    df_seq = pd.read_csv("results_seq.csv", sep="; ")
    df_par.sort_values(by=["image_size_pixels"], inplace=True)
    df_seq.sort_values(by=["image_size_pixels"], inplace=True)

    width = 0.25  # the width of the bars
    multiplier = 1

    sizes = len(df_par)
    x = np.arange(sizes)

    fig, ax = plt.subplots(layout='constrained')

    offset = width * multiplier
    rects = ax.bar(x + offset, df_seq["execution_time"] / df_par["execution_time"], width, color="orange")
    ax.bar_label(rects, padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup')
    ax.set_xticks(x + width, df_par["image_size"])

    plt.savefig("speedup")
    plt.close()

if __name__ == "__main__":
    speedup()
    exec_times()