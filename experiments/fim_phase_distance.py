import argparse

from pam.phase.seam_distance import run_seam_distance


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Fisher-geodesic distance from each grid node to the extracted phase seam."
    )
    p.add_argument("--distance-csv", default="outputs/fim_distance/fisher_distance_matrix.csv")
    p.add_argument("--nodes-csv", default="outputs/fim_distance/fisher_nodes.csv")
    p.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    p.add_argument("--outdir", default="outputs/fim_phase")
    return p.parse_args()


def main():
    args = parse_args()
    run_seam_distance(
        distance_csv=args.distance_csv,
        nodes_csv=args.nodes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()