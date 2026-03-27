#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.operators.transition_rate import run_transition_rate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate phase-transition rate as a function of Lazarus exposure along operator trajectories."
    )
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_transition_rate")
    parser.add_argument("--within-k", type=int, default=2, help="Count transition if phase flip occurs within k future steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_transition_rate(
        paths_csv=args.paths_csv,
        outdir=args.outdir,
        within_k=args.within_k,
    )


if __name__ == "__main__":
    main()