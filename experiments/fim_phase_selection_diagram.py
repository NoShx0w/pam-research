#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.topology.organization import run_organization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render phase selection diagram.")
    parser.add_argument(
        "--summary-csv",
        default="outputs/fim_initial_conditions/initial_conditions_outcome_summary.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_initial_conditions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_organization(
        summary_csv=args.summary_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()