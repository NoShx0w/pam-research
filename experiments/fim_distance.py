import argparse

from pam.geometry.distance_graph import run_distance_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute discrete Fisher/geodesic distances on the PAM (r, alpha) grid."
    )
    parser.add_argument(
        "--fim-csv",
        default="outputs/fim/fim_surface.csv",
        help="Path to fim_surface.csv produced by experiments/fim.py",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_distance",
        help="Directory for Fisher-distance outputs",
    )
    parser.add_argument(
        "--neighbor-mode",
        choices=["4", "8"],
        default="4",
        help="Grid connectivity for the graph.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=["midpoint", "endpoint_avg"],
        default="midpoint",
        help="How to evaluate edge cost from local metrics.",
    )
    parser.add_argument(
        "--anchor-r",
        type=float,
        default=None,
        help="Optional anchor r for a distance heatmap. Defaults to the first valid node.",
    )
    parser.add_argument(
        "--anchor-alpha",
        type=float,
        default=None,
        help="Optional anchor alpha for a distance heatmap. Defaults to the first valid node.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_distance_graph(
        fim_csv=args.fim_csv,
        outdir=args.outdir,
        neighbor_mode=args.neighbor_mode,
        cost_mode=args.cost_mode,
        anchor_r=args.anchor_r,
        anchor_alpha=args.anchor_alpha,
    )


if __name__ == "__main__":
    main()