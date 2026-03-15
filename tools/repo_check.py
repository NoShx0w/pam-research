#!/usr/bin/env python3
"""
repo_check.py

Lightweight repository state verifier for the PAM Observatory.

Checks:
- expected directory structure
- core pipeline files
- documentation charts
- output artifact directories

The goal is not strict CI validation, but **architectural drift detection**.
"""

import sys
from pathlib import Path

ROOT = Path(".")

REQUIRED_DIRS = [
    "experiments",
    "tui",
    "tools",
    "docs",
    "outputs",
]

REQUIRED_FILES = [
    "experiments/exp_batch.py",
    "experiments/common_quench_metrics.py",
    "tools/phase_movie.py",
    "tui/app.py",
    "docs/architecture.md",
    "docs/geometry_pipeline.md",
    "docs/allspark.md",
    "README.md",
]

OPTIONAL_OUTPUT_DIRS = [
    "outputs/trajectories",
    "outputs/fim",
    "outputs/fim_distance",
    "outputs/fim_mds",
    "outputs/fim_curvature",
]

INDEX_FILE = "outputs/index.csv"


def check_path(path):
    return (ROOT / path).exists()


def report(status, path, note=""):
    if note:
        print(f"[{status}] {path} — {note}")
    else:
        print(f"[{status}] {path}")


def main():
    drift = False

    print("PAM Observatory — Repository State Check")
    print("-" * 42)

    print("\nChecking directories:\n")

    for d in REQUIRED_DIRS:
        if check_path(d):
            report("OK", d)
        else:
            report("MISSING", d)
            drift = True

    print("\nChecking core files:\n")

    for f in REQUIRED_FILES:
        if check_path(f):
            report("OK", f)
        else:
            report("MISSING", f)
            drift = True

    print("\nChecking output state:\n")

    if check_path(INDEX_FILE):
        report("OK", INDEX_FILE)
    else:
        report("WARN", INDEX_FILE, "experiment ledger not found yet")

    for d in OPTIONAL_OUTPUT_DIRS:
        if check_path(d):
            report("OK", d)
        else:
            report("WARN", d, "derived artifacts not present yet")

    print("\nSummary:\n")

    if drift:
        print("Repository state: DRIFT DETECTED")
        sys.exit(1)
    else:
        print("Repository state: coherent")
        sys.exit(0)


if __name__ == "__main__":
    main()
