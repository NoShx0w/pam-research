#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    scale_root: str = "outputs/scales/100000"
    outputs_root: str = "outputs"
    python_bin: str = sys.executable
    run_hotspot_occupancy: bool = False
    run_canonical_seam_bundle: bool = False
    run_pass2_annotations: bool = False


def require_optional_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare OBS-022 scene inputs from the formalized scale family substrate."
    )
    parser.add_argument("--scale-root", default="outputs/scales/100000")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-hotspot-occupancy", action="store_true")
    parser.add_argument("--run-canonical-seam-bundle", action="store_true")
    parser.add_argument("--run-pass2-annotations", action="store_true")
    args = parser.parse_args()
    
    cfg = Config(
        scale_root=args.scale_root,
        outputs_root=args.outputs_root,
        python_bin=args.python_bin,
        run_hotspot_occupancy=args.run_hotspot_occupancy,
        run_canonical_seam_bundle=args.run_canonical_seam_bundle,
        run_pass2_annotations=args.run_pass2_annotations,
    )

    project_root = Path(__file__).resolve().parents[2]

    # 1) Build the scale-conditioned family substrate.
    run([
        cfg.python_bin,
        str(project_root / "experiments/toy/build_scale_family_substrate.py"),
        "--scale-root", cfg.scale_root,
        "--outputs-root", cfg.outputs_root,
    ])

    # 2) Export the OBS-022 scene bundle.
    run([
        cfg.python_bin,
        str(project_root / "experiments/toy/export_obs022_scene_bundle.py"),
    ])

    # 3) Build hotspot occupancy study outputs.
    if cfg.run_hotspot_occupancy:
        require_optional_file(
            Path(cfg.outputs_root) / "obs023_local_direction_mismatch" / "local_direction_mismatch_nodes.csv",
            "OBS-023 mismatch nodes for OBS-024 hotspot occupancy",
        )
        run([
            cfg.python_bin,
            str(project_root / "experiments/toy/obs024_family_hotspot_occupancy.py"),
        ])

    # 4) Export canonical seam bundle.
    if cfg.run_canonical_seam_bundle:
        require_optional_file(
            Path(cfg.outputs_root) / "obs023_local_direction_mismatch" / "local_direction_mismatch_nodes.csv",
            "OBS-023 mismatch nodes for OBS-028c seam bundle",
        )
        require_optional_file(
            Path(cfg.outputs_root) / "obs025_anisotropy_vs_relational_obstruction" / "obs025_anisotropy_vs_relational_obstruction_nodes.csv",
            "OBS-025 anisotropy/relational obstruction nodes for OBS-028c seam bundle",
        )
        run([
            cfg.python_bin,
            str(project_root / "experiments/studies/obs028c_export_canonical_seam_bundle.py"),
        ])

    # 5) Optionally mirror pass-2 canonical annotation artifacts.
    if cfg.run_pass2_annotations:
        run([
            cfg.python_bin,
            str(project_root / "experiments/canonicalize_pass2_annotations.py"),
        ])

    print()
    print("=== OBS-022 Scene Inputs Prepared ===")
    print(f"scale_root={cfg.scale_root}")
    print(f"outputs_root={cfg.outputs_root}")
    print("Produced or refreshed:")
    print("  - scale family substrate")
    print("  - OBS-022 scene bundle")
    if cfg.run_hotspot_occupancy:
        print("  - OBS-024 family hotspot occupancy")
    if cfg.run_canonical_seam_bundle:
        print("  - OBS-028c canonical seam bundle")
    if cfg.run_pass2_annotations:
        print("  - pass-2 annotation mirrors")


if __name__ == "__main__":
    main()
