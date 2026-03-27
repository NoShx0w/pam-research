"""Canonical seam extraction stage for the PAM phase pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ["run_seam_extraction"]


def render_phase_boundary(df: pd.DataFrame, ridge: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots()

    ax.scatter(df.alpha, df.r, s=20, alpha=0.3)
    ax.scatter(
        ridge.alpha,
        ridge.r,
        s=80,
        color="red",
    )
    ax.plot(
        ridge.alpha,
        ridge.r,
        linewidth=2,
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title("PAM phase boundary (curvature ridge)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_seam_extraction(
    curvature_csv,
    outdir,
    threshold: float = 10,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(curvature_csv)

    df["absK"] = np.abs(df["scalar_curvature"])
    ridge = df[df["absK"] > threshold].copy()
    ridge = ridge.sort_values("r")

    ridge.to_csv(outdir / "phase_boundary_points.csv", index=False)
    render_phase_boundary(df, ridge, outdir / "phase_boundary.png")

    print("boundary extracted")
    return ridge