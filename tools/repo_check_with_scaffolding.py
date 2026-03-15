#!/usr/bin/env python3
"""
repo_check.py

Repository state verifier for the PAM Observatory.

Checks:
- directory structure
- core pipeline files
- documentation charts
- output artifact directories
- evidence of scaffolding formation

Goal:
Detect architectural drift and report structural coherence.
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

# --- Scaffolding signals ---

DOC_CHARTS = [
    "README.md",
    "docs/architecture.md",
    "docs/geometry_pipeline.md",
    "docs/allspark.md",
]

GEOMETRY_PIPELINE = [
    "experiments/fim.py",
    "experiments/fim_distance.py",
    "experiments/fim_mds.py",
    "experiments/fim_curvature.py",
]

OBSERVATORY_STACK = [
    "tui/app.py",
    "tools/phase_movie.py",
]


def exists(path):
    return (ROOT / path).exists()


def report(status, path, note=""):
    if note:
        print(f"[{status}] {path} — {note}")
    else:
        print(f"[{status}] {path}")


def scaffold_report(label, ok, note=""):
    if ok:
        print(f"[SCAFFOLD] {label}")
    else:
        print(f"[SCAFFOLD-WARN] {label} — {note}")


def main():
    drift = False

    print("PAM Observatory — Repository State Check")
    print("-" * 44)

    # --- directory checks ---
    print("\nChecking directories:\n")
    for d in REQUIRED_DIRS:
        if exists(d):
            report("OK", d)
        else:
            report("MISSING", d)
            drift = True

    # --- core file checks ---
    print("\nChecking core files:\n")
    for f in REQUIRED_FILES:
        if exists(f):
            report("OK", f)
        else:
            report("MISSING", f)
            drift = True

    # --- output state ---
    print("\nChecking output state:\n")

    if exists(INDEX_FILE):
        report("OK", INDEX_FILE)
    else:
        report("WARN", INDEX_FILE, "experiment ledger not present yet")

    for d in OPTIONAL_OUTPUT_DIRS:
        if exists(d):
            report("OK", d)
        else:
            report("WARN", d, "derived artifacts not present yet")

    # --- scaffolding detection ---
    print("\nChecking scaffolding formation:\n")

    docs_ok = all(exists(p) for p in DOC_CHARTS)
    scaffold_report(
        "canonical documentation manifold",
        docs_ok,
        "expected charts incomplete",
    )

    geom_ok = any(exists(p) for p in GEOMETRY_PIPELINE)
    scaffold_report(
        "geometry analysis pipeline",
        geom_ok,
        "Fisher geometry stages not detected",
    )

    obs_ok = all(exists(p) for p in OBSERVATORY_STACK)
    scaffold_report(
        "observatory instrumentation layer",
        obs_ok,
        "TUI or artifact tools missing",
    )

    outputs_ok = any(exists(p) for p in OPTIONAL_OUTPUT_DIRS)
    scaffold_report(
        "derived output tree",
        outputs_ok,
        "no geometry output directories yet",
    )

    # --- summary ---
    print("\nSummary:\n")

    if drift:
        print("Repository state: DRIFT DETECTED")
        sys.exit(1)
    else:
        print("Repository state: coherent")
        sys.exit(0)


if __name__ == "__main__":
    main()
