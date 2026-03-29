import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_mechanism_plane(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, n_bins: int = 12):
    work = df[[x_col, y_col, z_col]].copy()
    for c in [x_col, y_col, z_col]:
        work[c] = safe_numeric(work[c])
    work = work.dropna()
    if len(work) == 0:
        return None

    xbins = np.linspace(work[x_col].min(), work[x_col].max(), n_bins + 1)
    ybins = np.linspace(work[y_col].min(), work[y_col].max(), n_bins + 1)

    work["x_bin"] = pd.cut(work[x_col], bins=xbins, include_lowest=True)
    work["y_bin"] = pd.cut(work[y_col], bins=ybins, include_lowest=True)

    grid = work.groupby(["y_bin", "x_bin"], observed=False)[z_col].mean().unstack()
    return grid


def conditional_distance_bin_summary(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    work = df[["distance_to_seam", "lazarus_score", "transition_within_k"]].copy()
    work["distance_to_seam"] = safe_numeric(work["distance_to_seam"])
    work["lazarus_score"] = safe_numeric(work["lazarus_score"])
    work["transition_within_k"] = safe_numeric(work["transition_within_k"])
    work = work.dropna()

    work["distance_bin"] = pd.qcut(work["distance_to_seam"], q=n_bins, duplicates="drop")

    rows = []
    for i, (b, g) in enumerate(work.groupby("distance_bin", observed=False)):
        if len(g) < 20:
            continue
        thr = float(g["lazarus_score"].median())
        high = g[g["lazarus_score"] >= thr]
        low = g[g["lazarus_score"] < thr]

        def rate_and_se(x: pd.DataFrame):
            n = len(x)
            if n == 0:
                return np.nan, np.nan
            p = float(x["transition_within_k"].mean())
            se = float(np.sqrt(max(p * (1.0 - p), 0.0) / n))
            return p, se

        high_rate, high_se = rate_and_se(high)
        low_rate, low_se = rate_and_se(low)
        diff = high_rate - low_rate if np.isfinite(high_rate) and np.isfinite(low_rate) else np.nan
        diff_se = np.sqrt((high_se ** 2 if np.isfinite(high_se) else 0.0) + (low_se ** 2 if np.isfinite(low_se) else 0.0))

        rows.append({
            "distance_bin": str(b),
            "bin_label": f"bin {i+1}",
            "n": int(len(g)),
            "high_n": int(len(high)),
            "low_n": int(len(low)),
            "high_rate": high_rate,
            "low_rate": low_rate,
            "high_se": high_se,
            "low_se": low_se,
            "diff": diff,
            "diff_se": diff_se,
            "bin_mid": float(g["distance_to_seam"].median()),
        })
    return pd.DataFrame(rows)


def render_figure(df: pd.DataFrame, outpath: Path, within_k: int, n_bins: int):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))

    plot_df = df.copy()
    plot_df["distance_to_seam"] = safe_numeric(plot_df["distance_to_seam"])
    plot_df["lazarus_score"] = safe_numeric(plot_df["lazarus_score"])
    plot_df["scalar_curvature"] = safe_numeric(plot_df["scalar_curvature"])
    plot_df["transition_within_k"] = safe_numeric(plot_df["transition_within_k"])
    plot_df["log_curvature"] = np.log10(1.0 + plot_df["scalar_curvature"].clip(lower=0))

    # A
    ax = axes[0, 0]
    a = plot_df[["distance_to_seam", "lazarus_score"]].dropna()
    ax.scatter(a["distance_to_seam"], a["lazarus_score"], alpha=0.12, s=12)
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("lazarus_score")
    ax.set_title("A. Compression localizes near the seam")

    # B
    ax = axes[0, 1]
    b = plot_df[["lazarus_score", "log_curvature"]].dropna()
    ax.scatter(b["lazarus_score"], b["log_curvature"], alpha=0.12, s=12)
    ax.set_xlabel("lazarus_score")
    ax.set_ylabel("log10(1 + curvature)")
    ax.set_title("B. Compression tracks geometric stress")

    # C
    ax = axes[1, 0]
    grid = make_mechanism_plane(plot_df, "lazarus_score", "distance_to_seam", "transition_within_k", n_bins=12)
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

    # D upgraded conditional separation with error bars + diff panel overlay
    ax = axes[1, 1]
    cond = conditional_distance_bin_summary(plot_df, n_bins=n_bins)
    if len(cond):
        x = np.arange(len(cond))
        w = 0.34
        ax.bar(x - w/2, cond["low_rate"], width=w, yerr=cond["low_se"], capsize=4, label="low Lazarus")
        ax.bar(x + w/2, cond["high_rate"], width=w, yerr=cond["high_se"], capsize=4, label="high Lazarus")
        ax.set_xticks(x)
        ax.set_xticklabels(cond["bin_label"], rotation=0)
        ax.set_ylabel(f"P(transition within {within_k} steps)")
        ax.set_title("D. Conditional separation within distance bins")
        ax.legend(loc="upper left", fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(x, cond["diff"], marker="o", linestyle="--", label="high - low")
        ax2.fill_between(
            x,
            cond["diff"] - cond["diff_se"],
            cond["diff"] + cond["diff_se"],
            alpha=0.18,
        )
        ax2.set_ylabel("difference")
        ax2.axhline(0.0, linewidth=1.0, linestyle=":")
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
        ax.set_title("D. Conditional separation within distance bins")

    fig.suptitle("Figure 3 — Transition mechanism", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render upgraded Figure 3 with conditional separation.")
    parser.add_argument("--transition-labeled-csv", default="outputs/fim_transition_rate/transition_rate_labeled.csv")
    parser.add_argument("--outdir", default="outputs/fim_figure_3")
    parser.add_argument("--within-k", type=int, default=2)
    parser.add_argument("--distance-bins", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.transition_labeled_csv)
    outpath = outdir / "figure_3_transition_mechanism_conditional_v2.png"
    render_figure(df, outpath, within_k=args.within_k, n_bins=args.distance_bins)

    cond = conditional_distance_bin_summary(df, n_bins=args.distance_bins)
    cond.to_csv(outdir / "figure_3_conditional_distance_bins_v2.csv", index=False)

    print(outpath)
    print(outdir / "figure_3_conditional_distance_bins_v2.csv")


if __name__ == "__main__":
    main()
