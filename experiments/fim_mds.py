
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed the PAM Fisher-distance manifold with classical MDS."
    )
    parser.add_argument(
        "--distance-csv",
        default="outputs/fim_distance/fisher_distance_matrix.csv",
        help="Path to fisher_distance_matrix.csv produced by experiments/fim_distance.py",
    )
    parser.add_argument(
        "--nodes-csv",
        default="outputs/fim_distance/fisher_nodes.csv",
        help="Path to fisher_nodes.csv produced by experiments/fim_distance.py",
    )
    parser.add_argument(
        "--fim-csv",
        default="outputs/fim/fim_surface.csv",
        help="Optional FIM surface CSV for coloring/annotations.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_mds",
        help="Directory for MDS outputs",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of embedding dimensions.",
    )
    parser.add_argument(
        "--color-by",
        default="fim_det",
        help="Column used for the primary colored embedding plot.",
    )
    return parser.parse_args()


def classical_mds(D: np.ndarray, n_components: int = 2):
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = np.clip(evals[:n_components], 0.0, None)
    X = evecs[:, :n_components] @ np.diag(np.sqrt(pos))

    diff = X[:, None, :] - X[None, :, :]
    D_hat = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.isfinite(D)
    num = np.sum((D[mask] - D_hat[mask]) ** 2)
    den = np.sum(D[mask] ** 2)
    stress = np.sqrt(num / den) if den > 0 else np.nan

    return X, evals, stress


def render_embedding(coords_df: pd.DataFrame, title: str, outpath: Path, color_values=None, color_label=None):
    fig, ax = plt.subplots(figsize=(6.6, 5.5))

    x = coords_df["mds1"].to_numpy(dtype=float)
    y = coords_df["mds2"].to_numpy(dtype=float)

    if color_values is None:
        ax.scatter(x, y, s=52)
    else:
        sc = ax.scatter(x, y, c=np.asarray(color_values, dtype=float), s=60)
        fig.colorbar(sc, ax=ax, label=color_label or "value")

    for _, row in coords_df.iterrows():
        ax.text(row["mds1"], row["mds2"], f"({row['r']:.2f},{row['alpha']:.3f})", fontsize=7, alpha=0.7)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dist_df = pd.read_csv(args.distance_csv, index_col=0)
    node_df = pd.read_csv(args.nodes_csv)

    D = dist_df.to_numpy(dtype=float)
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if len(node_df) != D.shape[0]:
        raise ValueError("Node count does not match distance matrix size.")
    if not np.all(np.isfinite(D)):
        raise ValueError("Distance matrix contains non-finite values; graph may be disconnected.")

    X, evals, stress = classical_mds(D, n_components=args.n_components)

    coords_df = node_df.copy()
    coords_df["mds1"] = X[:, 0]
    coords_df["mds2"] = X[:, 1]
    if args.n_components == 3:
        coords_df["mds3"] = X[:, 2]

    fim_path = Path(args.fim_csv)
    if fim_path.exists():
        fim_df = pd.read_csv(fim_path)
        keep = [c for c in ["r", "alpha", "fim_det", "fim_trace", "fim_cond", "fim_eig1", "fim_eig2"] if c in fim_df.columns]
        if keep:
            coords_df = coords_df.merge(
                fim_df[keep].drop_duplicates(subset=["r", "alpha"]),
                on=["r", "alpha"],
                how="left",
            )

    coords_out = outdir / "mds_coords.csv"
    stress_out = outdir / "mds_stress.txt"
    base_png = outdir / "mds_embedding.png"

    coords_df.to_csv(coords_out, index=False)

    total_pos = float(np.sum(np.clip(evals, 0.0, None)))
    with stress_out.open("w", encoding="utf-8") as f:
        f.write("PAM Fisher-distance MDS\n")
        f.write(f"distance_csv={args.distance_csv}\n")
        f.write(f"nodes_csv={args.nodes_csv}\n")
        f.write(f"n_components={args.n_components}\n")
        f.write(f"stress={stress}\n")
        f.write("eigenvalues=\n")
        f.write(np.array2string(evals[:10], precision=6, suppress_small=False))
        f.write("\n")
        if total_pos > 0:
            ratios = np.clip(evals, 0.0, None) / total_pos
            f.write("variance_ratios=\n")
            f.write(np.array2string(ratios[:10], precision=6, suppress_small=False))
            f.write("\n")

    render_embedding(coords_df, "PAM Fisher-distance MDS embedding", base_png)

    if args.color_by in coords_df.columns:
        render_embedding(
            coords_df,
            f"MDS embedding colored by {args.color_by}",
            outdir / f"mds_embedding_colored_by_{args.color_by}.png",
            color_values=coords_df[args.color_by].to_numpy(dtype=float),
            color_label=args.color_by,
        )

    for col in ["fim_det", "fim_cond", "r", "alpha"]:
        if col in coords_df.columns:
            render_embedding(
                coords_df,
                f"MDS embedding colored by {col}",
                outdir / f"mds_embedding_colored_by_{col}.png",
                color_values=coords_df[col].to_numpy(dtype=float),
                color_label=col,
            )

    print(coords_out)


if __name__ == "__main__":
    main()
