"""Canonical geodesic path and fan stages for the PAM geometry pipeline."""

import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_graph(edges: pd.DataFrame):
    graph = {}
    for _, e in edges.iterrows():
        u = int(e["src_id"])
        v = int(e["dst_id"])
        w = float(e["edge_cost"])
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w))
    return graph


def dijkstra(graph, start: int, goal: int):
    pq = [(0.0, start)]
    dist = {start: 0.0}
    prev = {}

    while pq:
        d, u = heapq.heappop(pq)

        if u == goal:
            break

        if d > dist.get(u, np.inf):
            continue

        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return []

    path = []
    cur = goal
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path


def closest_node(nodes: pd.DataFrame, r: float, alpha: float) -> int:
    d = (nodes["r"] - r) ** 2 + (nodes["alpha"] - alpha) ** 2
    return int(nodes.iloc[d.idxmin()]["node_id"])


def load_geodesic_inputs(nodes_csv, edges_csv, coords_csv):
    nodes = pd.read_csv(nodes_csv).copy()
    edges = pd.read_csv(edges_csv).copy()
    coords = pd.read_csv(coords_csv).copy().set_index("node_id")
    graph = build_graph(edges)
    return nodes, edges, coords, graph


def _build_path_nodes_df(
    path: list[int],
    path_id: str,
    nodes: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    source_node_id: int | None = None,
    target_node_id: int | None = None,
) -> pd.DataFrame:
    node_lookup = nodes.set_index("node_id")

    rows: list[dict[str, object]] = []
    for step, node_id in enumerate(path):
        rec: dict[str, object] = {
            "path_id": path_id,
            "step": int(step),
            "node_id": int(node_id),
            "source_node_id": int(source_node_id) if source_node_id is not None else None,
            "target_node_id": int(target_node_id) if target_node_id is not None else None,
            "r": np.nan,
            "alpha": np.nan,
            "mds1": np.nan,
            "mds2": np.nan,
        }

        if int(node_id) in node_lookup.index:
            rec["r"] = float(node_lookup.loc[int(node_id), "r"])
            rec["alpha"] = float(node_lookup.loc[int(node_id), "alpha"])

        if int(node_id) in coords.index:
            rec["mds1"] = float(coords.loc[int(node_id), "mds1"])
            rec["mds2"] = float(coords.loc[int(node_id), "mds2"])

        rows.append(rec)

    return pd.DataFrame(rows)


def _write_path_artifacts(
    path_nodes_df: pd.DataFrame,
    out_csv: Path,
    *,
    path_summary_csv: Path | None = None,
) -> None:
    path_nodes_df.to_csv(out_csv, index=False)

    if path_summary_csv is not None and not path_nodes_df.empty:
        summary = (
            path_nodes_df.groupby("path_id", dropna=False)
            .agg(
                n_nodes=("node_id", "size"),
                source_node_id=("source_node_id", "first"),
                target_node_id=("target_node_id", "first"),
                start_r=("r", "first"),
                start_alpha=("alpha", "first"),
                end_r=("r", "last"),
                end_alpha=("alpha", "last"),
            )
            .reset_index()
        )
        summary["n_steps"] = summary["n_nodes"] - 1
        summary.to_csv(path_summary_csv, index=False)


def run_geodesic(
    nodes_csv,
    edges_csv,
    coords_csv,
    r0,
    a0,
    r1,
    a1,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, coords, graph = load_geodesic_inputs(nodes_csv, edges_csv, coords_csv)

    start = closest_node(nodes, r0, a0)
    goal = closest_node(nodes, r1, a1)

    path = dijkstra(graph, start, goal)

    if not path:
        raise ValueError("No geodesic path found between the requested endpoints.")

    xs = [coords.loc[n, "mds1"] for n in path]
    ys = [coords.loc[n, "mds2"] for n in path]

    fig, ax = plt.subplots()
    ax.scatter(coords["mds1"], coords["mds2"], s=30)
    ax.plot(xs, ys, linewidth=3)
    ax.set_title("Fisher geodesic")
    fig.tight_layout()

    outpath = outdir / "geodesic_path.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    path_nodes_df = _build_path_nodes_df(
        path,
        path_id="main",
        nodes=nodes,
        coords=coords,
        source_node_id=start,
        target_node_id=goal,
    )
    _write_path_artifacts(
        path_nodes_df,
        outdir / "geodesic_path_nodes.csv",
        path_summary_csv=outdir / "geodesic_paths.csv",
    )

    print("path length:", len(path))
    print(f"Wrote {outpath}")
    print(f"Wrote {outdir / 'geodesic_path_nodes.csv'}")
    return path


def run_geodesic_fan(
    nodes_csv,
    edges_csv,
    coords_csv,
    r0,
    a0,
    r1,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, edges, coords, graph = load_geodesic_inputs(nodes_csv, edges_csv, coords_csv)

    start = closest_node(nodes, r0, a0)
    alphas = np.sort(nodes["alpha"].unique())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(coords["mds1"], coords["mds2"], s=20, alpha=0.5)

    fan_paths: list[list[int]] = []
    fan_frames: list[pd.DataFrame] = []

    for k, a in enumerate(alphas):
        goal = closest_node(nodes, r1, a)
        path = dijkstra(graph, start, goal)
        if not path:
            continue

        path_id = f"fan_{k:03d}"
        fan_paths.append(path)

        fan_frames.append(
            _build_path_nodes_df(
                path,
                path_id=path_id,
                nodes=nodes,
                coords=coords,
                source_node_id=start,
                target_node_id=goal,
            )
        )

        xs = [coords.loc[n, "mds1"] for n in path]
        ys = [coords.loc[n, "mds2"] for n in path]
        ax.plot(xs, ys, linewidth=1)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Fisher geodesic fan")
    fig.tight_layout()

    outpath = outdir / "geodesic_fan.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    if fan_frames:
        fan_nodes_df = pd.concat(fan_frames, ignore_index=True)
    else:
        fan_nodes_df = pd.DataFrame(
            columns=[
                "path_id",
                "step",
                "node_id",
                "source_node_id",
                "target_node_id",
                "r",
                "alpha",
                "mds1",
                "mds2",
            ]
        )

    _write_path_artifacts(
        fan_nodes_df,
        outdir / "geodesic_fan_path_nodes.csv",
        path_summary_csv=outdir / "geodesic_fan_paths.csv",
    )

    print(f"Wrote {outpath}")
    print(f"Wrote {outdir / 'geodesic_fan_path_nodes.csv'}")
    return fan_paths