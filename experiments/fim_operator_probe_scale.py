#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.operators.scaled_probes import run_scaled_probes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale GE probe experiment to many endpoint pairs and test Lazarus predictive structure."
    )
    parser.add_argument("--edges-csv", default="outputs/fim_distance/fisher_edges.csv")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--signed-phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--outdir", default="outputs/fim_ops_scaled")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-draw", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    run_scaled_probes(
        edges_csv=args.edges_csv,
        mds_csv=args.mds_csv,
        signed_phase_csv=args.signed_phase_csv,
        curvature_csv=args.curvature_csv,
        lazarus_csv=args.lazarus_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        n_pairs=args.n_pairs,
        seed=args.seed,
        max_draw=args.max_draw,
    )


if __name__ == "__main__":
    main()