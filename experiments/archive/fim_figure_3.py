import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_mechanism_plane(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, n_bins: int = 12):
    work = df[[x_col, y_col, z_col]].copy()
    work[x_col] = safe_numeric(work[x_col])
    work[y_col] = safe_numeric(work[y_col])
    work[z_col] = safe_numeric(work[z_col])
    work = work.dropna()

    if len(work) == 0:
        return None, None, None

    xbins = np.linspace(work[x_col].min(), work[x_col].max(), n_bins + 1)
    ybins = np.linspace(work[y_col].min(), work[y_col].max(), n_bins + 1)

    work["x_bin"] = pd.cut(work[x_col], bins=xbins, include_lowest=True)
    work["y_bin"] = pd.cut(work[y_col], bins=ybins, include_lowest=True)

    grid = (
        work.groupby(["y_bin", "x_bin"], observed=False)[z_col]
        .mean()
        .unstack()
    )

    return grid, xbins, ybins


def render_figure_3(df: pd.DataFrame, outpath: Path, within_k: int):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))

    # Derived fields
    plot_df = df.copy()
    plot_df["distance_to_seam"] = safe_numeric(plot_df["distance_to_seam"])
    plot_df["lazarus_score"] = safe_numeric(plot_df["lazarus_score"])
    plot_df["scalar_curvature"] = safe_numeric(plot_df["scalar_curvature"])
    plot_df["transition_within_k"] = safe_numeric(plot_df["transition_within_k"])
    plot_df["log_curvature"] = np.log10(1.0 + plot_df["scalar_curvature"].clip(lower=0))

    # A. Lazarus vs distance to seam
    ax = axes[0, 0]
    a = plot_df[["distance_to_seam", "lazarus_score"]].dropna()
    ax.scatter(a["distance_to_seam"], a["lazarus_score"], alpha=0.18, s=18)
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("lazarus_score")
    ax.set_title("A. Compression localizes near the seam")

    # B. Lazarus vs log curvature
    ax = axes[0, 1]
    b = plot_df[["lazarus_score", "log_curvature"]].dropna()
    ax.scatter(b["lazarus_score"], b["log_curvature"], alpha=0.18, s=18)
    ax.set_xlabel("lazarus_score")
    ax.set_ylabel("log10(1 + curvature)")
    ax.set_title("B. Compression tracks geometric stress")

    # C. Transition probability in mechanism plane
    ax = axes[1, 0]
    grid, xbins, ybins = make_mechanism_plane(
        plot_df,
        x_col="lazarus_score",
        y_col="distance_to_seam",
        z_col="transition_within_k",
        n_bins=12,
    )
    if grid is not None and grid.size > 0:
        arr = grid.to_numpy(dtype=float)
        im = ax.imshow(arr, origin="lower", aspect="auto")
        ax.set_title(f"C. P(transition within {within_k} steps)")
        ax.set_xlabel("lazarus_score bins")
        ax.set_ylabel("distance_to_seam bins")
        fig.colorbar(im, ax=ax, shrink=0.84, label="transition probability")
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title(f"C. P(transition within {within_k} steps)")

    # D. Curvature-conditioned Lazarus effect
    ax = axes[1, 1]
    d = plot_df[["scalar_curvature", "lazarus_score", "transition_within_k"]].dropna().copy()
    if len(d) > 0:
        curv_med = d["scalar_curvature"].median()
        laz_med = d["lazarus_score"].median()

        d["curvature_group"] = np.where(d["scalar_curvature"] >= curv_med, "high_curvature", "low_curvature")
        d["lazarus_group"] = np.where(d["lazarus_score"] >= laz_med, "high_lazarus", "low_lazarus")

        summary = (
            d.groupby(["curvature_group", "lazarus_group"], observed=False)["transition_within_k"]
            .mean()
            .reset_index()
        )

        order = [("low_curvature", "low_lazarus"), ("low_curvature", "high_lazarus"),
                 ("high_curvature", "low_lazarus"), ("high_curvature", "high_lazarus")]
        labels = [f"{c}\n{l}" for c, l in order]
        vals = []
        for c, l in order:
            row = summary[(summary["curvature_group"] == c) & (summary["lazarus_group"] == l)]
            vals.append(float(row["transition_within_k"].iloc[0]) if len(row) else np.nan)

        ax.bar(labels, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"P(transition within {within_k} steps)")
        ax.set_title("D. Lazarus effect strengthens under curvature")
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title("D. Lazarus effect strengthens under curvature")

    fig.suptitle("Figure 3 — Transition mechanism", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render Figure 3 — Transition mechanism.")
    parser.add_argument("--transition-labeled-csv", default="outputs/fim_transition_rate/transition_rate_labeled.csv")
    parser.add_argument("--outdir", default="outputs/fim_figure_3")
    parser.add_argument("--within-k", type=int, default=2)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    outpath = outdir / "figure_3_transition_mechanism.png"
    render_figure_3(df, outpath, within_k=args.within_k)

    print(outpath)


if __name__ == "__main__":
    main()
