import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INDEX_CSV = "outputs/index.csv"


def build_grid(df, value_col):
    alphas = np.array(sorted(df["alpha"].unique()), dtype=float)
    rs = np.array(sorted(df["r"].unique()), dtype=float)

    Z = np.full((len(rs), len(alphas)), np.nan, dtype=float)

    for i, r in enumerate(rs):
        for j, a in enumerate(alphas):
            sub = df[(df["r"] == r) & (df["alpha"] == a)]
            if len(sub):
                Z[i, j] = sub[value_col].mean()

    return alphas, rs, Z


def plot_surface(alphas, rs, Z, value_col, title):
    A, R = np.meshgrid(alphas, rs)

    plt.figure(figsize=(7, 5))

    # If the grid is dense enough, contours look great.
    # If not, fall back to pcolormesh-like appearance via imshow.
    valid = np.isfinite(Z)

    if np.sum(valid) >= 4 and len(alphas) >= 2 and len(rs) >= 2:
        # fill NaNs with nearest-neighbor-ish fallback using column/row means
        Z_plot = Z.copy()

        # simple, robust fill for sparse missing cells
        if np.any(~np.isfinite(Z_plot)):
            overall = np.nanmean(Z_plot)
            for i in range(Z_plot.shape[0]):
                row_mean = np.nanmean(Z_plot[i, :])
                for j in range(Z_plot.shape[1]):
                    if not np.isfinite(Z_plot[i, j]):
                        col_mean = np.nanmean(Z_plot[:, j])
                        vals = [v for v in [row_mean, col_mean, overall] if np.isfinite(v)]
                        Z_plot[i, j] = np.mean(vals) if vals else 0.0

        levels = 12
        cf = plt.contourf(A, R, Z_plot, levels=levels)
        cs = plt.contour(A, R, Z_plot, levels=levels, linewidths=0.6)
        plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        plt.colorbar(cf, label=value_col)
    else:
        plt.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[alphas.min(), alphas.max(), rs.min(), rs.max()],
        )
        plt.colorbar(label=value_col)

    # show actual sampled points
    pts = df[["alpha", "r"]].drop_duplicates()
    plt.scatter(pts["alpha"], pts["r"], marker="x", s=25)

    plt.xlabel("anchor injection α")
    plt.ylabel("replacement fraction r")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(INDEX_CSV)

    # choose the first surface you want to inspect
    value_col = "delta_r2_freeze"
    title = "PAM Phase Diagram (ΔR² freeze)"

    df = df.dropna(subset=["alpha", "r", value_col])

    alphas, rs, Z = build_grid(df, value_col)
    plot_surface(alphas, rs, Z, value_col, title)


if __name__ == "__main__":
    main()

