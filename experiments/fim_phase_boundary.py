import argparse

from pam.phase.seam import run_seam_extraction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvature", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--outdir", default="outputs/fim_phase")
    parser.add_argument("--threshold", type=float, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    run_seam_extraction(
        curvature_csv=args.curvature,
        outdir=args.outdir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()