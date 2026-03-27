import argparse

from src.pam.geometry.fisher_metric import DEFAULT_OBSERVABLES, run_fisher_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate a Fisher-type metric on the PAM (r, alpha) control manifold."
    )
    parser.add_argument(
        "--index-csv",
        default="outputs/index.csv",
        help="Path to outputs/index.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim",
        help="Directory for FIM outputs",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus filter, e.g. C",
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        default=DEFAULT_OBSERVABLES,
        help="Observable columns used to induce the metric",
    )
    parser.add_argument(
        "--ridge-eps",
        type=float,
        default=1e-8,
        help="Regularization added to residual covariance before inversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_fisher_metric(
        index_csv=args.index_csv,
        outdir=args.outdir,
        corpus=args.corpus,
        observables=args.observables,
        ridge_eps=args.ridge_eps,
    )


if __name__ == "__main__":
    main()
