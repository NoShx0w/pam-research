import argparse

from pam.geometry.curvature import run_curvature


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    p.add_argument("--outdir", default="outputs/fim_curvature")
    return p.parse_args()


def main():
    args = parse_args()
    run_curvature(
        fim_csv=args.fim_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()