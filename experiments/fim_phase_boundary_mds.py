import argparse

from pam.phase.seam_embedding import run_seam_embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-csv", default="outputs/fim_phase/phase_boundary_points.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--outdir", default="outputs/fim_phase")
    parser.add_argument("--n-samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    run_seam_embedding(
        boundary_csv=args.boundary_csv,
        mds_csv=args.mds_csv,
        outdir=args.outdir,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()