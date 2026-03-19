import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def build_graph(edges):
    import networkx as nx

    src_candidates = ["src", "u", "src_id"]
    dst_candidates = ["dst", "v", "dst_id"]
    weight_candidates = ["distance", "weight", "edge_cost", "fisher_distance", "dist", "length"]

    src_col = next((c for c in src_candidates if c in edges.columns), None)
    dst_col = next((c for c in dst_candidates if c in edges.columns), None)
    w_col = next((c for c in weight_candidates if c in edges.columns), None)

    if src_col is None or dst_col is None:
        raise ValueError(
            f"fisher_edges.csv must contain a source and destination column. "
            f"Found columns: {list(edges.columns)}"
        )

    if w_col is None:
        raise ValueError(
            f"fisher_edges.csv must contain a distance/weight column. "
            f"Found columns: {list(edges.columns)}"
        )

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(
            int(row[src_col]),
            int(row[dst_col]),
            weight=float(row[w_col]),
        )

    return G


def infer_node_column(df: pd.DataFrame) -> str:
    for c in ["node_id", "node", "id"]:
        if c in df.columns:
            return c
    raise ValueError("Could not infer node id column")


def select_representative_pairs(nodes_df: pd.DataFrame, n_pairs: int = 5):
    df = nodes_df.dropna(subset=["signed_phase"]).copy()
    if df.empty:
        return []

    pos = df.sort_values("signed_phase", ascending=False).reset_index(drop=True)
    neg = df.sort_values("signed_phase", ascending=True).reset_index(drop=True)
    neutral = df.iloc[(df["signed_phase"].abs()).argsort()].reset_index(drop=True)

    node_col = infer_node_column(df)
    pairs = []

    pairs.append((int(neg.iloc[0][node_col]), int(pos.iloc[0][node_col]), "basin_to_basin_extreme"))
    if len(pos) > 5 and len(neg) > 5:
        pairs.append((int(neg.iloc[5][node_col]), int(pos.iloc[5][node_col]), "basin_to_basin_mid"))
    if len(neg) > 8:
        pairs.append((int(neg.iloc[1][node_col]), int(neg.iloc[8][node_col]), "same_phase_negative"))
    if len(pos) > 8:
        pairs.append((int(pos.iloc[1][node_col]), int(pos.iloc[8][node_col]), "same_phase_positive"))
    if len(neutral) > 0 and len(pos) > 2:
        pairs.append((int(neutral.iloc[0][node_col]), int(pos.iloc[2][node_col]), "seam_to_positive"))

    dedup = []
    seen = set()
    for a, b, label in pairs:
        key = tuple(sorted((a, b)))
        if a == b or key in seen:
            continue
        seen.add(key)
        dedup.append((a, b, label))
        if len(dedup) >= n_pairs:
            break
    return dedup


def compute_path_metrics(path_nodes, path_df: pd.DataFrame):
    signed_phase = path_df["signed_phase"].to_numpy(dtype=float)
    seam_dist = path_df["distance_to_seam"].to_numpy(dtype=float) if "distance_to_seam" in path_df.columns else np.full(len(path_df), np.nan)
    curvature = path_df["scalar_curvature"].to_numpy(dtype=float) if "scalar_curvature" in path_df.columns else np.full(len(path_df), np.nan)

    finite_phase = signed_phase[np.isfinite(signed_phase)]
    phase_start = float(signed_phase[0]) if len(signed_phase) else np.nan
    phase_end = float(signed_phase[-1]) if len(signed_phase) else np.nan
    phase_span = float(phase_end - phase_start) if np.isfinite(phase_start) and np.isfinite(phase_end) else np.nan

    has_pos = np.any(finite_phase > 0)
    has_neg = np.any(finite_phase < 0)
    crosses_seam = int(has_pos and has_neg)

    min_distance_to_seam = float(np.nanmin(seam_dist)) if np.isfinite(seam_dist).any() else np.nan
    max_abs_curvature = float(np.nanmax(np.abs(curvature))) if np.isfinite(curvature).any() else np.nan
    lazarus_hit = 0
    if np.isfinite(min_distance_to_seam) and np.isfinite(max_abs_curvature):
        lazarus_hit = int((min_distance_to_seam <= 0.25) and (max_abs_curvature >= 50.0))

    return {
        "num_steps": len(path_nodes),
        "phase_start": phase_start,
        "phase_end": phase_end,
        "phase_span": phase_span,
        "crosses_seam": crosses_seam,
        "min_distance_to_seam": min_distance_to_seam,
        "max_abs_curvature": max_abs_curvature,
        "lazarus_hit": lazarus_hit,
    }


def render_paths(all_nodes: pd.DataFrame, seam_df: pd.DataFrame, paths_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    sc = ax.scatter(
        all_nodes["mds1"],
        all_nodes["mds2"],
        c=all_nodes["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=55,
        alpha=0.45,
    )
    fig.colorbar(sc, ax=ax, label="signed phase")

    if not seam_df.empty:
        seam_ord = seam_df.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], linewidth=2.0, color="black", alpha=0.9)

    for path_id, g in paths_df.groupby("path_id"):
        g = g.sort_values("step_idx")
        ax.plot(g["mds1"], g["mds2"], linewidth=2.2, alpha=0.95, label=str(path_id))
        ax.scatter(g.iloc[0]["mds1"], g.iloc[0]["mds2"], s=90, marker="o")
        ax.scatter(g.iloc[-1]["mds1"], g.iloc[-1]["mds2"], s=110, marker="X")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Operator S on the PAM manifold")
    if paths_df["path_id"].nunique() <= 8:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Canonical GE₀ / Operator S on the PAM manifold.")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops")
    parser.add_argument("--n-pairs", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges_csv)
    mds = pd.read_csv(args.mds_csv)
    phase = pd.read_csv(args.signed_phase_csv)
    curv = pd.read_csv(args.curvature_csv)
    seam = pd.read_csv(args.seam_csv)

    G = build_graph(edges)

    node_col = infer_node_column(mds)
    nodes = mds.copy()
    if node_col != "node_id":
        nodes = nodes.rename(columns={node_col: "node_id"})

    nodes = nodes.merge(
        phase[[c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]],
        on=["r", "alpha"],
        how="left",
    )

    if "scalar_curvature" in curv.columns:
        nodes = nodes.merge(curv[["r", "alpha", "scalar_curvature"]], on=["r", "alpha"], how="left")

    seam_small = seam.copy()
    if "mds1" not in seam_small.columns or "mds2" not in seam_small.columns:
        seam_small = seam_small[["r", "alpha"]].merge(
            nodes[["r", "alpha", "mds1", "mds2"]],
            on=["r", "alpha"],
            how="left",
        )
    seam_small = seam_small.dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"])

    pairs = select_representative_pairs(nodes, n_pairs=args.n_pairs)

    summary_rows = []
    path_rows = []

    for idx, (start_node, end_node, label) in enumerate(pairs, start=1):
        path_id = f"S{idx:02d}"
        path_nodes = nx.shortest_path(G, start_node, end_node, weight="weight")
        path_length = float(nx.shortest_path_length(G, start_node, end_node, weight="weight"))

        path_df = nodes[nodes["node_id"].isin(path_nodes)].copy()
        order = {n: i for i, n in enumerate(path_nodes)}
        path_df["step_idx"] = path_df["node_id"].map(order)
        path_df = path_df.sort_values("step_idx")

        metrics = compute_path_metrics(path_nodes, path_df)

        start_row = nodes[nodes["node_id"] == start_node].iloc[0]
        end_row = nodes[nodes["node_id"] == end_node].iloc[0]

        summary_rows.append(
            {
                "path_id": path_id,
                "label": label,
                "start_node": start_node,
                "end_node": end_node,
                "start_r": float(start_row["r"]),
                "start_alpha": float(start_row["alpha"]),
                "end_r": float(end_row["r"]),
                "end_alpha": float(end_row["alpha"]),
                "path_length": path_length,
                **metrics,
            }
        )

        for _, row in path_df.iterrows():
            path_rows.append(
                {
                    "path_id": path_id,
                    "label": label,
                    "step_idx": int(row["step_idx"]),
                    "node_id": int(row["node_id"]),
                    "r": float(row["r"]),
                    "alpha": float(row["alpha"]),
                    "mds1": float(row["mds1"]),
                    "mds2": float(row["mds2"]),
                    "signed_phase": float(row["signed_phase"]) if pd.notna(row.get("signed_phase")) else np.nan,
                    "distance_to_seam": float(row["distance_to_seam"]) if "distance_to_seam" in row and pd.notna(row["distance_to_seam"]) else np.nan,
                    "scalar_curvature": float(row["scalar_curvature"]) if "scalar_curvature" in row and pd.notna(row["scalar_curvature"]) else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    paths_df = pd.DataFrame(path_rows)

    summary_csv = outdir / "operator_S_summary.csv"
    paths_csv = outdir / "operator_S_paths.csv"
    plot_png = outdir / "operator_S_on_mds.png"

    summary_df.to_csv(summary_csv, index=False)
    paths_df.to_csv(paths_csv, index=False)
    render_paths(nodes, seam_small, paths_df, plot_png)

    print(summary_csv)
    print(paths_csv)
    print(plot_png)


if __name__ == "__main__":
    main()
