
# fim_mds_edges_patch.py
# MDS embedding with Fisher geodesic graph overlay

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def classical_mds(D, n_components=2):
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = np.clip(evals[:n_components], 0, None)
    X = evecs[:, :n_components] @ np.diag(np.sqrt(pos))

    return X, evals


def render_embedding(coords_df, edges_df, title, outpath, color=None, label=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    lookup = coords_df.set_index("node_id")[["mds1", "mds2"]]

    # draw geodesic edges
    if edges_df is not None:
        for _, e in edges_df.iterrows():
            if e.src_id not in lookup.index or e.dst_id not in lookup.index:
                continue

            p1 = lookup.loc[e.src_id]
            p2 = lookup.loc[e.dst_id]

            ax.plot(
                [p1.mds1, p2.mds1],
                [p1.mds2, p2.mds2],
                color="gray",
                alpha=0.25,
                linewidth=0.6,
                zorder=1,
            )

    x = coords_df["mds1"].to_numpy()
    y = coords_df["mds2"].to_numpy()

    if color is None:
        ax.scatter(x, y, s=55, zorder=2)
    else:
        sc = ax.scatter(x, y, c=color, s=60, zorder=2)
        fig.colorbar(sc, ax=ax, label=label)

    for _, r in coords_df.iterrows():
        ax.text(r.mds1, r.mds2, f"({r.r:.2f},{r.alpha:.3f})", fontsize=7, alpha=0.7)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-csv", default="outputs/fim_distance/fisher_distance_matrix.csv")
    parser.add_argument("--nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--outdir", default="outputs/fim_mds")
    parser.add_argument("--color-by", default="fim_det")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    D = pd.read_csv(args.distance_csv, index_col=0).to_numpy()
    nodes = pd.read_csv(args.nodes_csv)

    edges = None
    if Path(args.edges_csv).exists():
        edges = pd.read_csv(args.edges_csv)

    X, evals = classical_mds(D)

    coords = nodes.copy()
    coords["mds1"] = X[:, 0]
    coords["mds2"] = X[:, 1]

    if Path(args.fim_csv).exists():
        fim = pd.read_csv(args.fim_csv)
        keep = [c for c in ["r", "alpha", "fim_det", "fim_cond"] if c in fim.columns]
        coords = coords.merge(fim[keep].drop_duplicates(["r", "alpha"]), on=["r", "alpha"], how="left")

    coords.to_csv(outdir / "mds_coords.csv", index=False)

    render_embedding(
        coords,
        edges,
        "PAM Fisher-distance manifold (geodesic graph)",
        outdir / "mds_embedding_edges.png",
    )

    if args.color_by in coords.columns:
        render_embedding(
            coords,
            edges,
            f"MDS embedding colored by {args.color_by}",
            outdir / f"mds_embedding_colored_by_{args.color_by}.png",
            coords[args.color_by].to_numpy(),
            args.color_by,
        )

    print(outdir / "mds_coords.csv")


if __name__ == "__main__":
    main()
