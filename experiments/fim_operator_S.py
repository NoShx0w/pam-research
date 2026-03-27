#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.operators.geodesic_extraction import run_geodesic_extraction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upgraded canonical Operator S with path annotations on the PAM manifold."
    )
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops")
    return parser.parse_args()


def main():
    args = parse_args()
    run_geodesic_extraction(
        edges_csv=args.edges_csv,
        mds_csv=args.mds_csv,
        signed_phase_csv=args.signed_phase_csv,
        curvature_csv=args.curvature_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()