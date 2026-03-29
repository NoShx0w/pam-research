import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INDEX_CSV = "outputs/index.csv"

def summarize_by_alpha(df, value_col):
    g = df.groupby("alpha")[value_col]
    xs = np.array(sorted(df["alpha"].unique()), dtype=float)
    means = np.array([g.get_group(a).mean() for a in xs], dtype=float)
    stds = np.array([g.get_group(a).std(ddof=0) for a in xs], dtype=float)
    return xs, means, stds

def plot_band(ax, xs, means, stds, ylabel, title):
    ax.plot(xs, means)
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2)
    ax.set_xlabel("alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def main():
    df = pd.read_csv(INDEX_CSV)

    # keep one r slice
    r_value = 0.20
    df = df[np.isclose(df["r"], r_value)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, title in [
        (axes[0], "corr0", "corr0 vs alpha"),
        (axes[1], "delta_r2_freeze", "ΔR²_freeze vs alpha"),
        (axes[2], "var_H_joint", "var(H_joint) vs alpha"),
    ]:
        xs, means, stds = summarize_by_alpha(df, col)
        plot_band(ax, xs, means, stds, col, title)

    plt.suptitle(f"PAM phase slice at r = {r_value:.2f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
