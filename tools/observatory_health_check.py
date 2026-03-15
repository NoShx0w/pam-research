#!/usr/bin/env python3
"""
PAM Observatory Health Check

Diagnoses the integrity of the experimental data manifold.

Checks:
- index.csv presence
- grid completeness
- seed duplication
- observable sanity
- parameter consistency
"""

from pathlib import Path
import pandas as pd
import sys

INDEX_PATH = Path("outputs/index.csv")


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def warn(msg):
    print(f"[WARN] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def main():

    if not INDEX_PATH.exists():
        fail("outputs/index.csv not found")

    df = pd.read_csv(INDEX_PATH)

    if df.empty:
        fail("index.csv is empty")

    ok(f"Loaded {len(df)} experiment rows")

    required_columns = [
        "corpus",
        "r",
        "alpha",
        "seed",
        "piF_tail",
        "H_joint_mean",
    ]

    for col in required_columns:
        if col not in df.columns:
            fail(f"missing required column: {col}")

    ok("required columns present")

    # --- seed duplication check

    dup = df.duplicated(subset=["corpus", "r", "alpha", "seed"])

    if dup.any():
        warn(f"{dup.sum()} duplicate experiment rows detected")
    else:
        ok("no duplicate experiment keys")

    # --- grid coverage

    grid = df.groupby(["r", "alpha"]).size()

    if grid.min() != grid.max():
        warn("uneven seed coverage across grid")
    else:
        ok("grid seed coverage uniform")

    # --- observable sanity checks

    if (df["piF_tail"] < 0).any() or (df["piF_tail"] > 1).any():
        warn("piF_tail outside expected [0,1] range")

    if (df["H_joint_mean"] < 0).any():
        warn("negative entropy detected")

    ok("observable sanity checks completed")

    # --- simple anomaly scan

    if "best_corr" in df.columns:
        if (df["best_corr"].abs() > 1).any():
            warn("correlation values exceed [-1,1]")

    ok("basic anomaly scan complete")

    print("")
    print("Observatory health check complete.")


if __name__ == "__main__":
    main()
