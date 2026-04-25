#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_INPUT_COLS = [
    "probe_id",
    "step",
    "node_id",
    "r",
    "alpha",
    "mds1",
    "mds2",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare scaled probe paths for family stratification."
    )
    parser.add_argument(
        "--paths-csv",
        default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv",
    )
    args = parser.parse_args()

    paths_csv = Path(args.paths_csv)
    out_csv = Path(args.out_csv)

    if not paths_csv.exists():
        raise FileNotFoundError(paths_csv)

    df = pd.read_csv(paths_csv)

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input paths CSV missing required columns: {missing}")

    out = df[["probe_id", "step", "node_id", "r", "alpha", "mds1", "mds2"]].copy()
    out = out.rename(columns={"probe_id": "path_id"})

    for col in ["path_id", "step", "node_id", "r", "alpha", "mds1", "mds2"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(out_csv)


if __name__ == "__main__":
    main()
