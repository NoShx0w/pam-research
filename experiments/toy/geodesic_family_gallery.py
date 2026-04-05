#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_family_gallery.py

Plot representative geodesic paths in MDS space.

Expected inputs
---------------
- representative path-node CSV:
  rep_id, path_id, path_family, rep_kind, step, mds1, mds2
- optional full node cloud CSV:
  node_id, mds1, mds2
"""

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot representative geodesic path gallery.")
    parser.add_argument("--rep-nodes-csv", required=True)
    parser.add_argument("--all-nodes-csv", default="")
    parser.add_argument("--outpath", default="outputs/toy_geodesic_family_representatives/geodesic_family_gallery.png")
    args = parser.parse_args()

    rep_nodes = pd.read_csv(args.rep_nodes_csv).copy()
    all_nodes = pd.read_csv(args.all_nodes_csv).copy() if args.all_nodes_csv else pd.DataFrame()

    reps = (
        rep_nodes[["rep_id", "path_id", "path_family", "rep_kind"]]
        .drop_duplicates()
        .sort_values(["path_family", "rep_kind", "rep_id"])
        .reset_index(drop=True)
    )

    n = len(reps)
    if n == 0:
        raise ValueError("No representative paths found.")

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, rep) in zip(axes, reps.iterrows()):
        sub = (
            rep_nodes[rep_nodes["rep_id"] == rep["rep_id"]]
            .sort_values("step")
            .copy()
        )

        if not all_nodes.empty and {"mds1", "mds2"}.issubset(all_nodes.columns):
            ax.scatter(all_nodes["mds1"], all_nodes["mds2"], s=15, alpha=0.25)

        ax.plot(sub["mds1"], sub["mds2"], linewidth=2)
        ax.scatter(sub["mds1"], sub["mds2"], s=24)

        if len(sub):
            ax.scatter([sub.iloc[0]["mds1"]], [sub.iloc[0]["mds2"]], s=60, marker="s")
            ax.scatter([sub.iloc[-1]["mds1"]], [sub.iloc[-1]["mds2"]], s=60, marker="X")

        ax.set_title(
            f"{rep['path_family']}\n{rep['rep_kind']} / {rep['path_id']}",
            fontsize=10,
        )
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    outpath = Path(args.outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(outpath)


if __name__ == "__main__":
    main()
