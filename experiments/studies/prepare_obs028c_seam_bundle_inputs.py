#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    python_bin: str = sys.executable


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare OBS-028c canonical seam bundle inputs."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    cfg = Config(python_bin=args.python_bin)
    project_root = Path(__file__).resolve().parents[2]

    steps = [
        "experiments/studies/fim_response_complex_compatibility.py",
        "experiments/studies/fim_response_operator_decomposition.py",
        "experiments/studies/obs025_anisotropy_vs_relational_obstruction.py",
        "experiments/studies/obs025_two_field_seam_panel.py",
        "experiments/studies/obs026_family_two_field_occupancy.py",
        "experiments/studies/obs028_embedding_comparison.py",
        "experiments/studies/obs028b_diffusion_mode_analysis.py",
        "experiments/studies/obs028c_export_canonical_seam_bundle.py",
    ]

    for rel in steps:
        run([cfg.python_bin, str(project_root / rel)])

    print()
    print("=== OBS-028c Seam Bundle Inputs Prepared ===")
    print("Produced or refreshed:")
    print("  - response complex compatibility")
    print("  - response operator decomposition")
    print("  - anisotropy vs relational obstruction")
    print("  - two-field seam panel")
    print("  - family two-field occupancy")
    print("  - embedding comparison")
    print("  - diffusion mode analysis")
    print("  - canonical seam bundle")


if __name__ == "__main__":
    main()
