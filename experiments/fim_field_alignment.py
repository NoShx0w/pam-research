#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.topology.field import run_field_alignment


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze seam/Lazarus field alignment on the PAM manifold.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_field_alignment")
    return parser.parse_args()


def main():
    args = parse_args()
    run_field_alignment(
        mds_csv=args.mds_csv,
        phase_csv=args.phase_csv,
        lazarus_csv=args.lazarus_csv,
        paths_csv=args.paths_csv,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()