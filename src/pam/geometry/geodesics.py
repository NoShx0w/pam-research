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
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    coords = pd.read_csv(coords_csv).set_index("node_id")
    graph = build_graph(edges)
    return nodes, edges, coords, graph


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

    print("path length:", len(path))
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

    fan_paths = []

    for a in alphas:
        goal = closest_node(nodes, r1, a)
        path = dijkstra(graph, start, goal)
        if not path:
            continue

        fan_paths.append(path)
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

    print(f"Wrote {outpath}")
    return fan_paths