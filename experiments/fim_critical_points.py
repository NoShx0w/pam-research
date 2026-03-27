import argparse

from pam.topology.critical_points import run_critical_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_critical")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    run_critical_points(
        fim_csv=args.fim_csv,
        curvature_csv=args.curvature_csv,
        phase_distance_csv=args.phase_distance_csv,
        outdir=args.outdir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()