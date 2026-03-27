"""
Diagnostic curvature-proxy study for the PAM geometry pipeline.

This script is not the canonical scalar-curvature stage.
The canonical curvature implementation lives in:
    pam.geometry.curvature

This study computes derived curvature-like diagnostics from FIM summary fields
such as det(G) and condition structure.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate curvature-like diagnostics and ridge candidates on the PAM FIM surface."
    )
    parser.add_argument(
        "--fim-csv",
        default="outputs/fim/fim_surface.csv",
        help="Path to fim_surface.csv produced by experiments/fim.py",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_curvature",
        help="Directory for curvature outputs",
    )
    parser.add_argument(
        "--ridge-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold on det(G) used to mark ridge candidates.",
    )
    return parser.parse_args()


def pivot(df, col, r_vals, a_vals):
    p = df.pivot_table(index="r", columns="alpha", values=col, aggfunc="mean")
    return p.reindex(index=r_vals, columns=a_vals).to_numpy(dtype=float)


def finite_laplacian(Z, dr, da):
    """
    Simple finite-difference Laplacian approximation for scalar field Z(r, α).
    """
    nr, na = Z.shape
    L = np.full_like(Z, np.nan, dtype=float)

    for i in range(1, nr - 1):
        for j in range(1, na - 1):
            if not np.isfinite(Z[i, j]):
                continue

            zr = (Z[i + 1, j] - 2 * Z[i, j] + Z[i - 1, j]) / (dr ** 2)
            za = (Z[i, j + 1] - 2 * Z[i, j] + Z[i, j - 1]) / (da ** 2)
            L[i, j] = zr + za

    return L


def render_heatmap(data, r_vals, a_vals, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45)
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.fim_csv)

    r_vals = np.sort(df["r"].unique())
    a_vals = np.sort(df["alpha"].unique())

    detG = pivot(df, "fim_det", r_vals, a_vals)
    condG = pivot(df, "fim_cond", r_vals, a_vals)

    dr = float(np.mean(np.diff(r_vals))) if len(r_vals) > 1 else 1.0
    da = float(np.mean(np.diff(a_vals))) if len(a_vals) > 1 else 1.0

    curvature = finite_laplacian(detG, dr, da)

    ridge_thresh = np.nanpercentile(detG, args.ridge_percentile)
    ridge_mask = detG >= ridge_thresh

    rows = []
    for i, r in enumerate(r_vals):
        for j, a in enumerate(a_vals):
            rows.append({
                "r": r,
                "alpha": a,
                "detG": float(detG[i, j]) if np.isfinite(detG[i, j]) else np.nan,
                "condG": float(condG[i, j]) if np.isfinite(condG[i, j]) else np.nan,
                "curvature": float(curvature[i, j]) if np.isfinite(curvature[i, j]) else np.nan,
                "ridge_candidate": int(ridge_mask[i, j])
            })

    surf_df = pd.DataFrame(rows)

    csv_out = outdir / "curvature_surface.csv"
    surf_df.to_csv(csv_out, index=False)

    render_heatmap(detG, r_vals, a_vals, "det(G)", outdir / "detG_heatmap.png")
    render_heatmap(np.log10(np.clip(detG, 1e-16, None)), r_vals, a_vals, "log10 det(G)", outdir / "log10_detG_heatmap.png")
    render_heatmap(condG, r_vals, a_vals, "condition number", outdir / "condG_heatmap.png")
    render_heatmap(curvature, r_vals, a_vals, "curvature proxy", outdir / "curvature_heatmap.png")
    render_heatmap(ridge_mask.astype(float), r_vals, a_vals, "ridge candidates", outdir / "ridge_mask.png")

    meta = outdir / "curvature_metadata.txt"
    with meta.open("w") as f:
        f.write("PAM curvature diagnostics\n")
        f.write(f"fim_csv={args.fim_csv}\n")
        f.write(f"ridge_percentile={args.ridge_percentile}\n")
        f.write(f"dr={dr}\n")
        f.write(f"da={da}\n")

    print(csv_out)


if __name__ == "__main__":
    main()
