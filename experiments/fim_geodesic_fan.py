import argparse
import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Build graph from fisher_edges.csv
# ---------------------------------------------------------
def build_graph(edges):
    graph = {}

    for _, e in edges.iterrows():
        u = int(e["src_id"])
        v = int(e["dst_id"])
        w = float(e["edge_cost"])

        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w))

    return graph


# ---------------------------------------------------------
# Dijkstra shortest path
# ---------------------------------------------------------
def dijkstra(graph, start, goal):

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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--coords", default="outputs/fim_mds/mds_coords.csv")

    parser.add_argument("--r0", type=float, required=True)
    parser.add_argument("--a0", type=float, required=True)

    parser.add_argument("--r1", type=float, required=True)

    parser.add_argument("--outdir", default="outputs/fim_geodesic")

    args = parser.parse_args()

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    coords = pd.read_csv(args.coords)

    graph = build_graph(edges)

    coords = coords.set_index("node_id")

    # -----------------------------------------------------
    # nearest grid node
    # -----------------------------------------------------
    def closest(r, a):
        d = (nodes["r"] - r) ** 2 + (nodes["alpha"] - a) ** 2
        return int(nodes.iloc[d.idxmin()]["node_id"])

    start = closest(args.r0, args.a0)

    alphas = np.sort(nodes["alpha"].unique())

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(coords["mds1"], coords["mds2"], s=20, alpha=0.5)

    # -----------------------------------------------------
    # compute fan
    # -----------------------------------------------------
    for a in alphas:

        goal = closest(args.r1, a)

        path = dijkstra(graph, start, goal)

        if not path:
            continue

        xs = [coords.loc[n, "mds1"] for n in path]
        ys = [coords.loc[n, "mds2"] for n in path]

        ax.plot(xs, ys, linewidth=1)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Fisher geodesic fan")

    fig.tight_layout()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / "geodesic_fan.png"

    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()