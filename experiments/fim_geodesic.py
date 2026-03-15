import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt


def dijkstra(edges, start, goal):

    graph = {}

    for _,e in edges.iterrows():
        graph.setdefault(e.src_id, []).append((e.dst_id, e.edge_cost))
        graph.setdefault(e.dst_id, []).append((e.src_id, e.edge_cost))

    pq = [(0,start)]
    dist = {start:0}
    prev = {}

    while pq:
        d,u = heapq.heappop(pq)

        if u == goal:
            break

        for v,w in graph.get(u,[]):
            nd = d + w
            if nd < dist.get(v,np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq,(nd,v))

    path = []
    cur = goal

    while cur in prev:
        path.append(cur)
        cur = prev[cur]

    path.append(start)
    path.reverse()

    return path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--coords", default="outputs/fim_mds/mds_coords.csv")

    parser.add_argument("--r0", type=float)
    parser.add_argument("--a0", type=float)

    parser.add_argument("--r1", type=float)
    parser.add_argument("--a1", type=float)

    parser.add_argument("--outdir", default="outputs/fim_geodesic")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)
    coords = pd.read_csv(args.coords)

    # find closest nodes to requested parameters

    def closest(r,a):
        d = (nodes.r-r)**2 + (nodes.alpha-a)**2
        return nodes.iloc[d.idxmin()].node_id

    start = closest(args.r0,args.a0)
    goal = closest(args.r1,args.a1)

    path = dijkstra(edges,start,goal)

    coords = coords.set_index("node_id")

    xs = [coords.loc[n].mds1 for n in path]
    ys = [coords.loc[n].mds2 for n in path]

    fig,ax = plt.subplots()

    ax.scatter(coords.mds1,coords.mds2,s=30)

    ax.plot(xs,ys,linewidth=3)

    ax.set_title("Fisher geodesic")

    fig.tight_layout()

    fig.savefig(outdir/"geodesic_path.png",dpi=180)

    plt.close()

    print("path length:",len(path))


if __name__ == "__main__":
    main()
