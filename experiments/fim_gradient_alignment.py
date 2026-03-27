#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.topology.flow import run_gradient_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate local gradient alignment between phase and Lazarus fields.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--outdir", default="outputs/fim_gradient_alignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gradient_alignment(
        mds_csv=args.mds_csv,
        phase_csv=args.phase_csv,
        lazarus_csv=args.lazarus_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()