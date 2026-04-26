#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Config:
    scale_root: str = "outputs/scales/100000"
    outputs_root: str = "outputs"
    outdir: str | None = None
    python_bin: str = sys.executable


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def copy_required(src: Path, dst: Path, label: str) -> None:
    require_file(src, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_n_rows(path: Path) -> int:
    return len(pd.read_csv(path))


def write_metadata(
    outpath: Path,
    *,
    scale_root: Path,
    outputs_root: Path,
    scaled_paths_csv: Path,
    prepared_paths_csv: Path,
    path_node_diagnostics_csv: Path,
    path_diagnostics_csv: Path,
    family_assignments_csv: Path,
    family_summary_csv: Path,
) -> None:
    lines = [
        "=== Scale Family Substrate Metadata ===",
        "",
        "Purpose",
        "- derive a scale-conditioned family substrate from scale-local probe paths",
        "- evaluate those paths against the base canonical observatory field stack",
        "- expose a stable family layer for downstream scene / hub / hotspot consumers",
        "",
        "Inputs",
        f"scale_root={scale_root}",
        f"outputs_root={outputs_root}",
        f"scaled_paths_csv={scaled_paths_csv}",
        f"prepared_paths_csv={prepared_paths_csv}",
        "",
        "Conventions",
        "- path data is scale-local",
        "- observatory field references come from the base outputs root",
        "- family assignments are derived from path diagnostics, not directly from raw path nodes",
        "",
        "Outputs",
        f"path_node_diagnostics_csv={path_node_diagnostics_csv}",
        f"path_diagnostics_csv={path_diagnostics_csv}",
        f"family_assignments_csv={family_assignments_csv}",
        f"family_summary_csv={family_summary_csv}",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a scale-conditioned family substrate from scaled probe paths."
    )
    parser.add_argument("--scale-root", default="outputs/scales/100000")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    scale_root = Path(args.scale_root)
    outputs_root = Path(args.outputs_root)
    outdir = Path(args.outdir) if args.outdir else scale_root / "family_substrate"

    cfg = Config(
        scale_root=str(scale_root),
        outputs_root=str(outputs_root),
        outdir=str(outdir),
        python_bin=args.python_bin,
    )

    project_root = Path(__file__).resolve().parents[2]

    scaled_paths_csv = scale_root / "fim_ops_scaled" / "scaled_probe_paths.csv"
    prepared_paths_csv = outdir / "path_nodes_for_family.csv"

    diagnostics_workdir = outdir / "_diagnostics_tmp"
    families_workdir = outdir / "_families_tmp"

    path_node_diagnostics_tmp = diagnostics_workdir / "path_node_diagnostics.csv"
    path_diagnostics_tmp = diagnostics_workdir / "path_diagnostics.csv"
    family_assignments_tmp = families_workdir / "geodesic_path_family_assignments.csv"
    family_summary_tmp = families_workdir / "geodesic_path_family_summary.csv"

    path_node_diagnostics_csv = outdir / "path_node_diagnostics.csv"
    path_diagnostics_csv = outdir / "path_diagnostics.csv"
    family_assignments_csv = outdir / "path_family_assignments.csv"
    family_summary_csv = outdir / "path_family_summary.csv"
    metadata_txt = outdir / "family_substrate_metadata.txt"

    require_file(scaled_paths_csv, "scaled probe paths")
    require_file(outputs_root / "fim_identity" / "identity_field_nodes.csv", "base identity field nodes")
    require_file(outputs_root / "fim_identity_obstruction" / "identity_obstruction_nodes.csv", "base identity obstruction nodes")
    require_file(outputs_root / "fim_identity_obstruction" / "identity_obstruction_signed_nodes.csv", "base signed identity obstruction nodes")
    require_file(outputs_root / "fim_phase" / "signed_phase_coords.csv", "base signed phase")
    require_file(outputs_root / "fim_phase" / "phase_distance_to_seam.csv", "base seam distance")
    require_file(outputs_root / "fim_critical" / "criticality_surface.csv", "base criticality surface")
    require_file(outputs_root / "fim_lazarus" / "lazarus_scores.csv", "base Lazarus scores")
    require_file(outputs_root / "fim_response_operator" / "response_operator_nodes.csv", "base response operator nodes")

    outdir.mkdir(parents=True, exist_ok=True)

    if diagnostics_workdir.exists():
        shutil.rmtree(diagnostics_workdir)
    if families_workdir.exists():
        shutil.rmtree(families_workdir)

    diagnostics_workdir.mkdir(parents=True, exist_ok=True)
    families_workdir.mkdir(parents=True, exist_ok=True)

    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/prepare_scaled_probe_paths_for_family.py"),
        "--paths-csv", str(scaled_paths_csv),
        "--out-csv", str(prepared_paths_csv),
    ])

    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/geodesic_path_diagnostics.py"),
        "--paths-csv", str(prepared_paths_csv),
        "--outputs-root", str(outputs_root),
        "--outdir", str(diagnostics_workdir),
    ])

    run_cmd([
        cfg.python_bin,
        str(project_root / "experiments/toy/geodesic_path_family_stratification.py"),
        "--path-csv", str(path_diagnostics_tmp),
        "--outdir", str(families_workdir),
    ])

    copy_required(path_node_diagnostics_tmp, path_node_diagnostics_csv, "path node diagnostics")
    copy_required(path_diagnostics_tmp, path_diagnostics_csv, "path diagnostics")
    copy_required(family_assignments_tmp, family_assignments_csv, "family assignments")
    copy_required(family_summary_tmp, family_summary_csv, "family summary")

    write_metadata(
        metadata_txt,
        scale_root=scale_root,
        outputs_root=outputs_root,
        scaled_paths_csv=scaled_paths_csv,
        prepared_paths_csv=prepared_paths_csv,
        path_node_diagnostics_csv=path_node_diagnostics_csv,
        path_diagnostics_csv=path_diagnostics_csv,
        family_assignments_csv=family_assignments_csv,
        family_summary_csv=family_summary_csv,
    )

    print()
    print("=== Scale Family Substrate ===")
    print(f"path_nodes_for_family: {prepared_paths_csv} ({load_n_rows(prepared_paths_csv)} rows)")
    print(f"path_node_diagnostics: {path_node_diagnostics_csv} ({load_n_rows(path_node_diagnostics_csv)} rows)")
    print(f"path_diagnostics: {path_diagnostics_csv} ({load_n_rows(path_diagnostics_csv)} rows)")
    print(f"path_family_assignments: {family_assignments_csv} ({load_n_rows(family_assignments_csv)} rows)")
    print(f"path_family_summary: {family_summary_csv} ({load_n_rows(family_summary_csv)} rows)")
    print(metadata_txt)


if __name__ == "__main__":
    main()