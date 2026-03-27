import argparse

from pam.geometry.geodesics import run_geodesic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--coords", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--r0", type=float)
    parser.add_argument("--a0", type=float)
    parser.add_argument("--r1", type=float)
    parser.add_argument("--a1", type=float)
    parser.add_argument("--outdir", default="outputs/fim_geodesic")
    return parser.parse_args()


def main():
    args = parse_args()
    run_geodesic(
        nodes_csv=args.nodes,
        edges_csv=args.edges,
        coords_csv=args.coords,
        r0=args.r0,
        a0=args.a0,
        r1=args.r1,
        a1=args.a1,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()