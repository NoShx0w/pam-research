#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OBS-022 scene inputs.")
    parser.add_argument("--scale-root", default="outputs/scales/100000")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    run([
        args.python_bin,
        str(project_root / "experiments/toy/build_scale_family_substrate.py"),
        "--scale-root", args.scale_root,
        "--outputs-root", args.outputs_root,
    ])

    run([
        args.python_bin,
        str(project_root / "experiments/toy/export_obs022_scene_bundle.py"),
    ])

    run([
        args.python_bin,
        str(project_root / "experiments/toy/obs024_family_hotspot_occupancy.py"),
    ])

    run([
        args.python_bin,
        str(project_root / "experiments/studies/obs028c_export_canonical_seam_bundle.py"),
    ])


if __name__ == "__main__":
    main()
