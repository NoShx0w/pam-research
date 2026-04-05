#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_family_scale_persistence.py

Cross-scale persistence pass for geodesic route families.

Goal
----
Test whether stable seam corridors persist across scale, and whether they
remain the highest-response / Lazarus-rich family.

Per scale, this script runs:

1. endpoint manifest
2. endpoint sweep
3. path diagnostics
4. family stratification
5. Lazarus overlay
6. response overlay

Then it collects a compact per-scale summary.

Expected layout
---------------
Base root:
    outputs/

Scaled roots:
    outputs/scales/<scale>/

Each scale root should contain the same canonical subdirectories used by the
toy scripts, e.g.
    fim_distance/
    fim_mds/
    fim_identity/
    fim_identity_obstruction/
    fim_phase/
    fim_critical/
    fim_lazarus/
    fim_response_operator/

Outputs
-------
- scale_family_persistence_summary.csv
- scale_family_persistence_summary.txt
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


@dataclass(frozen=True)
class Config:
    root: str = "outputs"
    scales: tuple[str, ...] = ("base", "10", "100", "1000")
    outdir: str = "outputs/toy_geodesic_family_scale_persistence"
    n_per_bucket: int = 2
    python_bin: str = sys.executable


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def scale_root(base_root: Path, scale: str) -> Path:
    return base_root if scale == "base" else base_root / "scales" / scale


def find_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_family_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-scale persistence pass for geodesic families.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--scales", default="base,10,100,1000")
    parser.add_argument("--outdir", default="outputs/toy_geodesic_family_scale_persistence")
    parser.add_argument("--n-per-bucket", type=int, default=2)
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        scales=tuple(x.strip() for x in args.scales.split(",") if x.strip()),
        outdir=args.outdir,
        n_per_bucket=args.n_per_bucket,
        python_bin=args.python_bin,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_root = Path(cfg.root).resolve()
    project_root = Path(__file__).resolve().parents[2]

    rows: list[dict[str, object]] = []

    for scale in cfg.scales:
        sroot = scale_root(base_root, scale)
        tag = f"scale_{scale}"
        print(f"\n=== Running scale: {scale} @ {sroot} ===")

        # Common paths inside this scale root
        nodes_csv = find_existing(sroot / "fim_distance" / "fisher_nodes.csv")
        edges_csv = find_existing(sroot / "fim_distance" / "fisher_edges.csv")
        coords_csv = find_existing(sroot / "fim_mds" / "mds_coords.csv")

        lazarus_csv = find_existing(sroot / "fim_lazarus" / "lazarus_scores.csv")
        response_csv = find_existing(sroot / "fim_response_operator" / "response_operator_nodes.csv")

        manifest_out = outdir / tag / "manifest"
        sweep_out = outdir / tag / "sweep"
        diag_out = outdir / tag / "diagnostics"
        fam_out = outdir / tag / "families"
        laz_out = outdir / tag / "lazarus"
        rsp_out = outdir / tag / "response"

        manifest_out.mkdir(parents=True, exist_ok=True)
        sweep_out.mkdir(parents=True, exist_ok=True)
        diag_out.mkdir(parents=True, exist_ok=True)
        fam_out.mkdir(parents=True, exist_ok=True)
        laz_out.mkdir(parents=True, exist_ok=True)
        rsp_out.mkdir(parents=True, exist_ok=True)

        # 1) endpoint manifest
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_endpoint_manifest.py"),
            "--outputs-root", str(sroot),
            "--outdir", str(manifest_out),
            "--n-per-bucket", str(cfg.n_per_bucket),
        ])

        manifest_csv = manifest_out / "geodesic_endpoint_manifest.csv"

        # 2) endpoint sweep
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_endpoint_sweep.py"),
            "--manifest-csv", str(manifest_csv),
            "--nodes-csv", str(nodes_csv),
            "--edges-csv", str(edges_csv),
            "--coords-csv", str(coords_csv),
            "--outdir", str(sweep_out),
        ])

        path_nodes_csv = sweep_out / "geodesic_endpoint_sweep_path_nodes.csv"

        # 3) path diagnostics
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_path_diagnostics.py"),
            "--paths-csv", str(path_nodes_csv),
            "--outputs-root", str(sroot),
            "--outdir", str(diag_out),
        ])

        path_diag_csv = diag_out / "path_diagnostics.csv"

        # 4) family stratification
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_path_family_stratification.py"),
            "--path-csv", str(path_diag_csv),
            "--outdir", str(fam_out),
        ])

        fam_assign_csv = fam_out / "geodesic_path_family_assignments.csv"
        fam_summary_csv = fam_out / "geodesic_path_family_summary.csv"

        # 5) Lazarus overlay
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_family_operator_overlay.py"),
            "--family-csv", str(fam_assign_csv),
            "--path-nodes-csv", str(path_nodes_csv),
            "--operator-nodes-csv", str(lazarus_csv),
            "--outdir", str(laz_out),
        ])

        laz_summary_csv = laz_out / "geodesic_family_operator_family_summary.csv"

        # 6) Response overlay
        run_cmd([
            cfg.python_bin,
            str(project_root / "experiments/toy/geodesic_family_transition_overlay.py"),
            "--family-csv", str(fam_assign_csv),
            "--path-nodes-csv", str(path_nodes_csv),
            "--transition-nodes-csv", str(response_csv),
            "--outdir", str(rsp_out),
        ])

        rsp_summary_csv = rsp_out / "geodesic_family_transition_family_summary.csv"

        fam = read_family_summary(fam_summary_csv)
        laz = pd.read_csv(laz_summary_csv).copy()
        rsp = pd.read_csv(rsp_summary_csv).copy()

        merged = fam.merge(laz, on=["path_family", "n_paths"], how="left", suffixes=("", "_laz"))
        merged = merged.merge(rsp, on=["path_family", "n_paths"], how="left", suffixes=("", "_rsp"))

        # winners
        top_rsp_family = None
        top_rsp_value = None
        if "mean_transition_exposure" in merged.columns and len(merged.dropna(subset=["mean_transition_exposure"])):
            idx = merged["mean_transition_exposure"].astype(float).idxmax()
            top_rsp_family = str(merged.loc[idx, "path_family"])
            top_rsp_value = float(merged.loc[idx, "mean_transition_exposure"])

        top_laz_family = None
        top_laz_value = None
        if "mean_lazarus" in merged.columns and len(merged.dropna(subset=["mean_lazarus"])):
            idx = merged["mean_lazarus"].astype(float).idxmax()
            top_laz_family = str(merged.loc[idx, "path_family"])
            top_laz_value = float(merged.loc[idx, "mean_lazarus"])

        stable = merged[merged["path_family"] == "stable_seam_corridor"].copy()
        if len(stable):
            srow = stable.iloc[0]
            rows.append({
                "scale": scale,
                "n_families": int(len(merged)),
                "n_paths_stable_seam_corridor": int(srow["n_paths"]),
                "stable_mean_length": float(srow["mean_length"]) if "mean_length" in srow else float("nan"),
                "stable_mean_near_fraction": float(srow["mean_near_fraction"]) if "mean_near_fraction" in srow else float("nan"),
                "stable_mean_lazarus": float(srow["mean_lazarus"]) if "mean_lazarus" in srow else float("nan"),
                "stable_mean_transition_exposure": float(srow["mean_transition_exposure"]) if "mean_transition_exposure" in srow else float("nan"),
                "top_lazarus_family": top_laz_family,
                "top_lazarus_value": top_laz_value,
                "top_response_family": top_rsp_family,
                "top_response_value": top_rsp_value,
                "stable_is_top_lazarus": int(top_laz_family == "stable_seam_corridor") if top_laz_family is not None else 0,
                "stable_is_top_response": int(top_rsp_family == "stable_seam_corridor") if top_rsp_family is not None else 0,
            })
        else:
            rows.append({
                "scale": scale,
                "n_families": int(len(merged)),
                "n_paths_stable_seam_corridor": 0,
                "stable_mean_length": float("nan"),
                "stable_mean_near_fraction": float("nan"),
                "stable_mean_lazarus": float("nan"),
                "stable_mean_transition_exposure": float("nan"),
                "top_lazarus_family": top_laz_family,
                "top_lazarus_value": top_laz_value,
                "top_response_family": top_rsp_family,
                "top_response_value": top_rsp_value,
                "stable_is_top_lazarus": 0,
                "stable_is_top_response": 0,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "scale_family_persistence_summary.csv", index=False)

    lines = ["=== Geodesic Family Scale Persistence Summary ===", ""]
    for _, row in summary.iterrows():
        lines.append(
            f"scale={row['scale']}: "
            f"stable_n={int(row['n_paths_stable_seam_corridor'])}, "
            f"stable_mean_lazarus={row['stable_mean_lazarus']:.4f}, "
            f"stable_mean_response={row['stable_mean_transition_exposure']:.4f}, "
            f"top_lazarus_family={row['top_lazarus_family']}, "
            f"top_response_family={row['top_response_family']}, "
            f"stable_is_top_lazarus={int(row['stable_is_top_lazarus'])}, "
            f"stable_is_top_response={int(row['stable_is_top_response'])}"
        )
    (outdir / "scale_family_persistence_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(outdir / "scale_family_persistence_summary.csv")
    print(outdir / "scale_family_persistence_summary.txt")


if __name__ == "__main__":
    main()
