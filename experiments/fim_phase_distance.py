
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Fisher-geodesic distance from each grid node to the extracted phase seam."
    )
    p.add_argument("--distance-csv", default="outputs/fim_distance/fisher_distance_matrix.csv")
    p.add_argument("--nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    p.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    p.add_argument("--outdir", default="outputs/fim_phase")
    return p.parse_args()


def render_heatmap(grid, r_vals, a_vals, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fisher distance to seam")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    D = pd.read_csv(args.distance_csv, index_col=0)
    D.columns = [int(c) for c in D.columns]
    D.index = D.index.astype(int)

    nodes = pd.read_csv(args.nodes_csv)
    seam = pd.read_csv(args.seam_csv)

    seam_clean = seam.drop(columns=["node_id"], errors="ignore")

    matched = seam_clean.merge(
        nodes[["node_id", "r", "alpha"]],
        on=["r", "alpha"],
        how="left",
    )

    if "node_id" not in matched.columns:
        raise ValueError("Failed to recover node_id after seam/node merge.")

    seam_nodes = matched["node_id"].dropna().astype(int).unique().tolist()

    if not seam_nodes:
        raise ValueError("No seam nodes could be matched to fisher_nodes.csv")

    rows = []
    for _, row in nodes.iterrows():
        nid = int(row["node_id"])
        dmin = min(float(D.loc[nid, sid]) for sid in seam_nodes if sid in D.columns)
        rows.append({
            "node_id": nid,
            "r": float(row["r"]),
            "alpha": float(row["alpha"]),
            "distance_to_seam": dmin,
        })

    basin = pd.DataFrame(rows)
    basin_csv = outdir / "phase_distance_to_seam.csv"
    basin.to_csv(basin_csv, index=False)

    r_vals = np.sort(basin["r"].unique())
    a_vals = np.sort(basin["alpha"].unique())
    grid = (
        basin.pivot_table(index="r", columns="alpha", values="distance_to_seam", aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )

    render_heatmap(
        grid,
        r_vals,
        a_vals,
        "Fisher distance to phase seam",
        outdir / "phase_distance_to_seam.png",
    )
    render_heatmap(
        np.log10(np.clip(grid, 1e-12, None)),
        r_vals,
        a_vals,
        "log10 Fisher distance to phase seam",
        outdir / "log10_phase_distance_to_seam.png",
    )

    print(basin_csv)


if __name__ == "__main__":
    main()
