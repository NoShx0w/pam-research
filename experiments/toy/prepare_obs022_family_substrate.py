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


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare OBS-022 family substrate for a given scale."
    )
    parser.add_argument("--scale-root", default="outputs/scales/100000")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    cfg = Config(
        scale_root=args.scale_root,
        outputs_root=args.outputs_root,
        python_bin=args.python_bin,
    )

    project_root = Path(__file__).resolve().parents[2]
    scale_root = Path(cfg.scale_root)
    outputs_root = Path(cfg.outputs_root)

    scaled_paths_csv = scale_root / "fim_ops_scaled" / "scaled_probe_paths.csv"
    cleaned_paths_csv = scale_root / "fim_ops_scaled" / "scaled_probe_paths_for_family_clean.csv"
    diagnostics_outdir = scale_root / "toy_scaled_probe_path_diagnostics"
    diagnostics_csv = diagnostics_outdir / "path_diagnostics.csv"
    families_outdir = scale_root / "toy_scaled_probe_path_families"
    family_assignments_csv = families_outdir / "geodesic_path_family_assignments.csv"

    require_file(scaled_paths_csv, "scaled probe paths")
    require_file(outputs_root / "fim_identity" / "identity_field_nodes.csv", "base identity field nodes")
    require_file(outputs_root / "fim_identity_obstruction" / "identity_obstruction_nodes.csv", "base identity obstruction nodes")
    require_file(outputs_root / "fim_identity_obstruction" / "identity_obstruction_signed_nodes.csv", "base signed identity obstruction nodes")
    require_file(outputs_root / "fim_phase" / "signed_phase_coords.csv", "base signed phase")
    require_file(outputs_root / "fim_phase" / "phase_distance_to_seam.csv", "base seam distance")
    require_file(outputs_root / "fim_critical" / "criticality_surface.csv", "base criticality surface")

    # 1) Prepare family-ready path nodes.
    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/prepare_scaled_probe_paths_for_family.py"),
        "--paths-csv", str(scaled_paths_csv),
        "--out-csv", str(cleaned_paths_csv),
    ])

    # 2) Build path diagnostics using scale-local paths + base observatory fields.
    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/geodesic_path_diagnostics.py"),
        "--paths-csv", str(cleaned_paths_csv),
        "--outputs-root", str(outputs_root),
        "--outdir", str(diagnostics_outdir),
    ])

    # 3) Stratify path families from diagnostics.
    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/geodesic_path_family_stratification.py"),
        "--path-csv", str(diagnostics_csv),
        "--outdir", str(families_outdir),
    ])

    print()
    print("Prepared OBS-022 family substrate:")
    print(cleaned_paths_csv)
    print(diagnostics_csv)
    print(family_assignments_csv)


if __name__ == "__main__":
    main()
