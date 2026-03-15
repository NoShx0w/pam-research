
"""
fim_curvature.py
Estimate scalar curvature of the 2D Fisher information metric field
over the (r, alpha) parameter grid.

Inputs
------
outputs/fim/fim_surface.csv

Expected columns
----------------
r, alpha, fim_rr, fim_ra, fim_aa

Outputs
-------
outputs/fim_curvature/
    curvature_surface.csv
    scalar_curvature.png
    log_abs_scalar_curvature.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    p.add_argument("--outdir", default="outputs/fim_curvature")
    return p.parse_args()


def pivot(df, col, r_vals, a_vals):
    grid = df.pivot_table(index="r", columns="alpha", values=col, aggfunc="mean")
    return grid.reindex(index=r_vals, columns=a_vals).to_numpy(dtype=float)


def central_diff(Z, axis, h):
    """
    Central finite difference along axis
    """
    d = np.full_like(Z, np.nan)
    if axis == 0:
        for i in range(1, Z.shape[0]-1):
            d[i] = (Z[i+1] - Z[i-1])/(2*h)
    else:
        for j in range(1, Z.shape[1]-1):
            d[:,j] = (Z[:,j+1] - Z[:,j-1])/(2*h)
    return d


def compute_scalar_curvature(E,F,G, dr, da):
    """
    Approximate scalar curvature of 2D metric
    """
    Er = central_diff(E,0,dr)
    Ea = central_diff(E,1,da)

    Fr = central_diff(F,0,dr)
    Fa = central_diff(F,1,da)

    Gr = central_diff(G,0,dr)
    Ga = central_diff(G,1,da)

    nr, na = E.shape
    K = np.full_like(E, np.nan)

    for i in range(1,nr-1):
        for j in range(1,na-1):

            g = np.array([[E[i,j],F[i,j]],[F[i,j],G[i,j]]])

            if np.linalg.det(g) <= 0:
                continue

            ginv = np.linalg.inv(g)

            # approximate Christoffel contraction term
            term1 = Er[i,j]*Ga[i,j] - Ea[i,j]*Gr[i,j]
            term2 = Fr[i,j]**2

            K[i,j] = (term1 - term2)/(np.linalg.det(g)**2 + 1e-12)

    return K


def heatmap(data, r_vals, a_vals, title, path):
    fig, ax = plt.subplots()
    im = ax.imshow(data, origin="lower", aspect="auto")
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45)
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.fim_csv)

    r_vals = np.sort(df["r"].unique())
    a_vals = np.sort(df["alpha"].unique())

    E = pivot(df,"fim_rr",r_vals,a_vals)
    F = pivot(df,"fim_ra",r_vals,a_vals)
    G = pivot(df,"fim_aa",r_vals,a_vals)

    dr = np.mean(np.diff(r_vals))
    da = np.mean(np.diff(a_vals))

    K = compute_scalar_curvature(E,F,G,dr,da)

    rows=[]
    for i,r in enumerate(r_vals):
        for j,a in enumerate(a_vals):
            rows.append({
                "r":r,
                "alpha":a,
                "scalar_curvature":K[i,j]
            })

    pd.DataFrame(rows).to_csv(outdir/"curvature_surface.csv",index=False)

    heatmap(K,r_vals,a_vals,"scalar curvature",outdir/"scalar_curvature.png")
    heatmap(np.log10(np.abs(K)+1e-12),r_vals,a_vals,"log10 |curvature|",outdir/"log_abs_scalar_curvature.png")

    print("curvature written to:", outdir)


if __name__ == "__main__":
    main()
