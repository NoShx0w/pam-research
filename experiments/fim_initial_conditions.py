#!/usr/bin/env python3
from __future__ import annotations

import argparse

from pam.topology.initial_conditions import run_initial_conditions_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract initial conditions -> outcomes from existing trajectory data."
    )
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--metrics-csv", default="outputs/fim_ops_scaled/scaled_probe_metrics.csv")
    parser.add_argument("--corpora-py", default="src/corpora.py")
    parser.add_argument("--graze-threshold", type=float, default=0.15)
    parser.add_argument("--outdir", default="outputs/fim_initial_conditions")
    args = parser.parse_args()

    outputs = run_initial_conditions_summary(
        paths_csv=args.paths_csv,
        metrics_csv=args.metrics_csv,
        corpora_py=args.corpora_py,
        graze_threshold=args.graze_threshold,
        outdir=args.outdir,
    )

    print(outputs["initial_conditions_outcomes_csv"])
    print(outputs["initial_conditions_outcome_summary_csv"])
    print(outputs["link_geometry_summary_csv"])


if __name__ == "__main__":
    main()