import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heapq import heappop, heappush
from pathlib import Path


__all__ = [
    "REQUIRED_COLS",
    "run_distance_graph",
]


"""Canonical Fisher-distance graph stage for the PAM geometry pipeline."""


REQUIRED_COLS = [
    "r",
    "alpha",
    "fim_rr",
    "fim_ra",
    "fim_aa",
    "fim_valid",
]


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def pivot_grid(df: pd.DataFrame, value_col: str, r_values: np.ndarray, a_values: np.ndarray):
    grid = df.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
    return grid.reindex(index=r_values, columns=a_values).to_numpy(dtype=float)


def metric_at(i, j, g_rr, g_ra, g_aa):
    return np.array([[g_rr[i, j], g_ra[i, j]], [g_ra[i, j], g_aa[i, j]]], dtype=float)


def edge_cost(delta, G1, G2=None, mode="midpoint"):
    delta = np.asarray(delta, dtype=float).reshape(2)
    if mode == "midpoint":
        G = G1 if G2 is None else 0.5 * (G1 + G2)
        quad = float(delta.T @ G @ delta)
        return np.sqrt(max(quad, 0.0))

    if mode == "endpoint_avg":
        if G2 is None:
            G2 = G1
        c1 = float(delta.T @ G1 @ delta)
        c2 = float(delta.T @ G2 @ delta)
        return 0.5 * (np.sqrt(max(c1, 0.0)) + np.sqrt(max(c2, 0.0)))

    raise ValueError(f"Unknown cost mode: {mode}")


def build_graph(r_values, a_values, valid, g_rr, g_ra, g_aa, neighbor_mode="4", cost_mode="midpoint"):
    nr, na = valid.shape
    offsets = [(1, 0), (0, 1)] if neighbor_mode == "4" else [(1, 0), (0, 1), (1, 1), (1, -1)]

    node_ids = {}
    nodes = []
    next_id = 0
    for i in range(nr):
        for j in range(na):
            if valid[i, j]:
                node_ids[(i, j)] = next_id
                nodes.append((next_id, i, j, float(r_values[i]), float(a_values[j])))
                next_id += 1

    adjacency = {nid: [] for nid, *_ in nodes}
    edge_rows = []

    for i in range(nr):
        for j in range(na):
            if not valid[i, j]:
                continue

            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nr and 0 <= nj < na):
                    continue
                if not valid[ni, nj]:
                    continue

                G1 = metric_at(i, j, g_rr, g_ra, g_aa)
                G2 = metric_at(ni, nj, g_rr, g_ra, g_aa)
                delta = np.array([r_values[ni] - r_values[i], a_values[nj] - a_values[j]], dtype=float)
                cost = edge_cost(delta, G1, G2, mode=cost_mode)

                u = node_ids[(i, j)]
                v = node_ids[(ni, nj)]
                adjacency[u].append((v, cost))
                adjacency[v].append((u, cost))

                edge_rows.append(
                    {
                        "src_id": u,
                        "dst_id": v,
                        "src_i": i,
                        "src_j": j,
                        "dst_i": ni,
                        "dst_j": nj,
                        "src_r": float(r_values[i]),
                        "src_alpha": float(a_values[j]),
                        "dst_r": float(r_values[ni]),
                        "dst_alpha": float(a_values[nj]),
                        "delta_r": float(delta[0]),
                        "delta_alpha": float(delta[1]),
                        "edge_cost": float(cost),
                    }
                )

    node_df = pd.DataFrame(nodes, columns=["node_id", "i", "j", "r", "alpha"])
    edge_df = pd.DataFrame(edge_rows)
    return node_df, edge_df, adjacency


def dijkstra(adjacency, start):
    dist = {node: np.inf for node in adjacency}
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        cur_d, u = heappop(pq)
        if cur_d > dist[u]:
            continue

        for v, w in adjacency[u]:
            nd = cur_d + w
            if nd < dist[v]:
                dist[v] = nd
                heappush(pq, (nd, v))

    return dist


def all_pairs_shortest_paths(adjacency, n_nodes: int):
    D = np.full((n_nodes, n_nodes), np.inf, dtype=float)
    for start in range(n_nodes):
        dist = dijkstra(adjacency, start)
        for node, d in dist.items():
            D[start, node] = d
    return D


def choose_anchor(node_df: pd.DataFrame, anchor_r=None, anchor_alpha=None):
    if anchor_r is None or anchor_alpha is None:
        return int(node_df.iloc[0]["node_id"])

    coords = node_df[["r", "alpha"]].to_numpy(dtype=float)
    target = np.array([anchor_r, anchor_alpha], dtype=float)
    idx = int(np.argmin(np.sum((coords - target) ** 2, axis=1)))
    return int(node_df.iloc[idx]["node_id"])


def render_distance_heatmap(node_df, distances, r_values, a_values, title, outpath):
    grid = np.full((len(r_values), len(a_values)), np.nan, dtype=float)
    for _, row in node_df.iterrows():
        grid[int(row["i"]), int(row["j"])] = distances[int(row["node_id"])]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(a_values)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_values], rotation=45, ha="right")
    ax.set_yticks(range(len(r_values)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_values])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fisher distance")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_distance_graph(
    fim_csv,
    outdir,
    neighbor_mode="4",
    cost_mode="midpoint",
    anchor_r=None,
    anchor_alpha=None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fim_csv)
    ensure_columns(df, REQUIRED_COLS)

    r_values = np.sort(df["r"].unique())
    a_values = np.sort(df["alpha"].unique())

    g_rr = pivot_grid(df, "fim_rr", r_values, a_values)
    g_ra = pivot_grid(df, "fim_ra", r_values, a_values)
    g_aa = pivot_grid(df, "fim_aa", r_values, a_values)
    valid = pivot_grid(df, "fim_valid", r_values, a_values).astype(bool)

    node_df, edge_df, adjacency = build_graph(
        r_values=r_values,
        a_values=a_values,
        valid=valid,
        g_rr=g_rr,
        g_ra=g_ra,
        g_aa=g_aa,
        neighbor_mode=neighbor_mode,
        cost_mode=cost_mode,
    )

    if node_df.empty:
        raise ValueError("No valid FIM nodes found; cannot build Fisher-distance graph.")

    n_nodes = len(node_df)
    D = all_pairs_shortest_paths(adjacency, n_nodes)
    dist_df = pd.DataFrame(D, index=node_df["node_id"], columns=node_df["node_id"])
    dist_df.index.name = "src_node_id"

    node_out = outdir / "fisher_nodes.csv"
    edge_out = outdir / "fisher_edges.csv"
    dist_out = outdir / "fisher_distance_matrix.csv"
    meta_out = outdir / "fisher_paths_metadata.txt"

    node_df.to_csv(node_out, index=False)
    edge_df.to_csv(edge_out, index=False)
    dist_df.to_csv(dist_out)

    anchor_id = choose_anchor(node_df, anchor_r, anchor_alpha)
    anchor_dist = dijkstra(adjacency, anchor_id)
    anchor_row = node_df[node_df["node_id"] == anchor_id].iloc[0]

    anchor_png = outdir / f"fisher_distance_from_r{anchor_row['r']:.3f}_a{anchor_row['alpha']:.6f}.png"
    render_distance_heatmap(
        node_df=node_df,
        distances=anchor_dist,
        r_values=r_values,
        a_values=a_values,
        title=f"Fisher distance from (r={anchor_row['r']:.3f}, α={anchor_row['alpha']:.6f})",
        outpath=anchor_png,
    )

    with meta_out.open("w", encoding="utf-8") as f:
        f.write("PAM Fisher-distance graph\n")
        f.write(f"fim_csv={fim_csv}\n")
        f.write(f"neighbor_mode={neighbor_mode}\n")
        f.write(f"cost_mode={cost_mode}\n")
        f.write(f"n_nodes={len(node_df)}\n")
        f.write(f"n_edges={len(edge_df)}\n")
        f.write(f"anchor_node_id={anchor_id}\n")
        f.write(f"anchor_r={float(anchor_row['r'])}\n")
        f.write(f"anchor_alpha={float(anchor_row['alpha'])}\n")

    print(f"Wrote {node_out}")
    print(f"Wrote {edge_out}")
    print(f"Wrote {dist_out}")
    print(f"Wrote {meta_out}")
    print(f"Wrote {anchor_png}")

    return {
        "nodes": node_df,
        "edges": edge_df,
        "distances": dist_df,
        "anchor_id": anchor_id,
    }