import argparse

from pam.geometry.embedding import run_embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-csv", default="outputs/fim_distance/fisher_distance_matrix.csv")
    parser.add_argument("--nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--outdir", default="outputs/fim_mds")
    parser.add_argument("--color-by", default="fim_det")
    return parser.parse_args()


def main():
    args = parse_args()
    run_embedding(
        distance_csv=args.distance_csv,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        fim_csv=args.fim_csv,
        outdir=args.outdir,
        color_by=args.color_by,
    )


if __name__ == "__main__":
    main()