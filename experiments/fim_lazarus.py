#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.operators.lazarus import run_lazarus


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute first-pass Lazarus regime scores from seam distance and curvature."
    )
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_lazarus")
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.85,
        help="Quantile threshold for lazarus_hit on lazarus_score.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_lazarus(
        signed_phase_csv=args.signed_phase_csv,
        curvature_csv=args.curvature_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        threshold_quantile=args.threshold_quantile,
    )


if __name__ == "__main__":
    main()