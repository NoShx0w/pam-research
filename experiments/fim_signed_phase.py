import argparse

from pam.phase.signed_phase import run_signed_phase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute signed phase coordinates from seam distance and seam orientation in MDS space."
    )
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_phase")
    return parser.parse_args()


def main():
    args = parse_args()
    run_signed_phase(
        mds_csv=args.mds_csv,
        seam_csv=args.seam_csv,
        phase_distance_csv=args.phase_distance_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()