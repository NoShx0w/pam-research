from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/phase_summary.csv")
DEFAULT_OUTDIR = Path("outputs/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PAM phase heatmaps and boundary contours from phase_summary.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input summary CSV from analyze_index.py (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory for output figures (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.5,
        help="Contour threshold for piF_tail_mean boundary overlay.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI for saved PNGs.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    p = df.pivot(index="r", columns="alpha", values=value_col)
    return p.sort_index().sort_index(axis=1)


def draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap: str = "viridis",
) -> plt.Axes:
    im = ax.imshow(
        data.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
    )
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in data.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    return im


def overlay_boundary(
    ax: plt.Axes,
    boundary_source: pd.DataFrame,
    threshold: float,
) -> None:
    z = boundary_source.values.astype(float)
    if np.all(np.isnan(z)):
        return

    x = np.arange(len(boundary_source.columns))
    y = np.arange(len(boundary_source.index))
    X, Y = np.meshgrid(x, y)

    try:
        cs = ax.contour(
            X,
            Y,
            z,
            levels=[threshold],
            linewidths=1.5,
            colors="white",
            origin="lower",
        )
        ax.clabel(cs, inline=True, fmt={threshold: f"πF={threshold:.2f}"}, fontsize=8)
    except Exception:
        # Contour can fail if the threshold is outside the observed range.
        pass


def plot_single_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    boundary_col: Optional[str],
    boundary_threshold: float,
    outpath: Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    heat = pivot_metric(df, metric_col)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = draw_heatmap(ax, heat, title=title, cmap=cmap)
    plt.colorbar(im, ax=ax)

    if boundary_col and boundary_col in df.columns:
        boundary = pivot_metric(df, boundary_col)
        overlay_boundary(ax, boundary, boundary_threshold)

    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_main_panel(
    df: pd.DataFrame,
    boundary_threshold: float,
    outpath: Path,
) -> None:
    panels = [
        ("piF_tail_mean", "Freeze occupancy πF (tail mean)", "viridis"),
        ("H_joint_mean_mean", "Joint signature entropy H (mean)", "viridis"),
        ("best_corr_mean", "Best lag correlation", "viridis"),
        ("delta_r2_freeze_mean", "ΔR²_freeze", "viridis"),
    ]

    available = [(c, t, cmap) for (c, t, cmap) in panels if c in df.columns]
    if not available:
        raise ValueError("None of the expected panel columns were found in phase_summary.csv")

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    boundary = None
    if "piF_tail_mean" in df.columns:
        boundary = pivot_metric(df, "piF_tail_mean")

    for ax, (col, title, cmap) in zip(axes, available):
        heat = pivot_metric(df, col)
        im = draw_heatmap(ax, heat, title=title, cmap=cmap)
        plt.colorbar(im, ax=ax)
        if boundary is not None:
            overlay_boundary(ax, boundary, boundary_threshold)

    plt.suptitle("PAM phase surfaces")
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_panel(
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    candidates = [
        ("piF_tail_std", "πF std across seeds"),
        ("H_joint_mean_std", "H std across seeds"),
        ("best_corr_std", "Best correlation std"),
        ("delta_r2_freeze_std", "ΔR²_freeze std"),
    ]
    available = [(c, t) for (c, t) in candidates if c in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        heat = pivot_metric(df, col)
        im = draw_heatmap(ax, heat, title=title, cmap="magma")
        plt.colorbar(im, ax=ax)

    plt.suptitle("PAM uncertainty / variance surfaces")
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_boundary_curve(
    df: pd.DataFrame,
    threshold: float,
    outpath: Path,
) -> None:
    if "piF_tail_mean" not in df.columns:
        return

    rows = []
    for r, sub in df.sort_values(["r", "alpha"]).groupby("r"):
        s = sub[["alpha", "piF_tail_mean"]].dropna().sort_values("alpha")
        if len(s) < 2:
            continue

        boundary_alpha = np.nan
        alphas = s["alpha"].to_numpy(dtype=float)
        vals = s["piF_tail_mean"].to_numpy(dtype=float)

        for i in range(len(vals) - 1):
            y0, y1 = vals[i], vals[i + 1]
            if (y0 - threshold) == 0:
                boundary_alpha = alphas[i]
                break
            if (y0 - threshold) * (y1 - threshold) <= 0:
                # Linear interpolation for alpha* at fixed r
                x0, x1 = alphas[i], alphas[i + 1]
                if y1 == y0:
                    boundary_alpha = x0
                else:
                    boundary_alpha = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
                break

        rows.append({"r": r, "alpha_star": boundary_alpha})

    boundary_df = pd.DataFrame(rows).dropna()
    if boundary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(boundary_df["r"], boundary_df["alpha_star"], marker="o")
    ax.set_xlabel("r")
    ax.set_ylabel("alpha*")
    ax.set_title(f"Estimated phase boundary from πF={threshold:.2f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    ensure_columns(df, ["r", "alpha"])

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Main multi-panel figure
    plot_main_panel(
        df=df,
        boundary_threshold=args.boundary_threshold,
        outpath=args.outdir / "phase_surfaces.png",
    )

    # Uncertainty / variance figure
    plot_uncertainty_panel(
        df=df,
        outpath=args.outdir / "phase_uncertainty.png",
    )

    # Individual figures that are convenient for papers / notes
    individual_targets = [
        ("piF_tail_mean", "Freeze occupancy πF (tail mean)", "piF_surface.png", "viridis"),
        ("H_joint_mean_mean", "Joint signature entropy H (mean)", "entropy_surface.png", "viridis"),
        ("best_corr_mean", "Best lag correlation", "corr_surface.png", "viridis"),
        ("delta_r2_freeze_mean", "ΔR²_freeze", "delta_r2_freeze_surface.png", "viridis"),
    ]

    for col, title, filename, cmap in individual_targets:
        if col in df.columns:
            plot_single_heatmap(
                df=df,
                metric_col=col,
                boundary_col="piF_tail_mean" if "piF_tail_mean" in df.columns else None,
                boundary_threshold=args.boundary_threshold,
                outpath=args.outdir / filename,
                title=title,
                cmap=cmap,
            )

    # Estimated alpha*(r) boundary curve
    plot_phase_boundary_curve(
        df=df,
        threshold=args.boundary_threshold,
        outpath=args.outdir / "phase_boundary_curve.png",
    )

    print(f"Loaded summary rows: {len(df)}")
    print(f"Wrote figures to: {args.outdir}")


if __name__ == "__main__":
    main()
