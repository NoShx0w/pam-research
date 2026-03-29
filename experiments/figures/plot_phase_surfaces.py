import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INDEX_CSV = "outputs/index.csv"


def pivot(df, value):
    p = df.pivot_table(
        index="r",
        columns="alpha",
        values=value,
        aggfunc="mean"
    )
    return p.sort_index().sort_index(axis=1)


def draw_heatmap(ax, data, title):
    im = ax.imshow(data.values, aspect="auto", origin="lower")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in data.columns], rotation=45)

    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in data.index])

    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)

    return im


def main():

    df = pd.read_csv(INDEX_CSV)

    # drop incomplete rows
    df = df.dropna()

    corr = pivot(df, "corr0")
    freeze = pivot(df, "delta_r2_freeze")

    # K_max may not exist yet in your index
    if "K_max" in df.columns:
        K = pivot(df, "K_max")
    else:
        K = None

    fig, axes = plt.subplots(1, 3 if K is not None else 2, figsize=(14, 4))

    im0 = draw_heatmap(axes[0], corr, "corr0 (Freeze–Entropy coupling)")
    plt.colorbar(im0, ax=axes[0])

    im1 = draw_heatmap(axes[1], freeze, "ΔR²_freeze (Entropy → Freeze)")
    plt.colorbar(im1, ax=axes[1])

    if K is not None:
        im2 = draw_heatmap(axes[2], K, "K_max (Microstructure complexity)")
        plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Phase surfaces of the system")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

