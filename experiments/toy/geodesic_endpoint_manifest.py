#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_endpoint_manifest.py

Build a structured endpoint manifest for geodesic sweeps.

Goal
----
Select representative nodes across:
- seam bins
- identity sectors

so geodesic paths can be sampled across meaningful regime pairs.

Outputs
-------
- geodesic_endpoint_manifest.csv
- geodesic_endpoint_manifest_summary.txt
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


BEST_BASIS = (
    "absolute_holonomy_node",
    "obstruction_mean_abs_holonomy",
    "obstruction_signed_sum_holonomy",
)


@dataclass(frozen=True)
class Config:
    outputs_root: str = "outputs"
    outdir: str = "outputs/toy_geodesic_endpoint_manifest"
    n_per_bucket: int = 2
    eps: float = 1e-12
    near_q: float = 0.33
    mid_q: float = 0.66
    random_seed: int = 42


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).copy() if path.exists() else pd.DataFrame()


def _zscore(series: pd.Series, eps: float) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean(skipna=True)) if len(x.dropna()) else 0.0
    sd = float(x.std(skipna=True)) if len(x.dropna()) else 0.0
    return (x - mu) / max(sd, eps)


def load_nodes(outputs_root: Path) -> pd.DataFrame:
    identity_csv = outputs_root / "fim_identity" / "identity_field_nodes.csv"
    obstruction_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_nodes.csv"
    obstruction_signed_csv = outputs_root / "fim_identity_obstruction" / "identity_obstruction_signed_nodes.csv"
    phase_distance_csv = outputs_root / "fim_phase" / "phase_distance_to_seam.csv"
    criticality_csv = outputs_root / "fim_critical" / "criticality_surface.csv"

    identity = _safe_read_csv(identity_csv)
    obstruction = _safe_read_csv(obstruction_csv)
    obstruction_signed = _safe_read_csv(obstruction_signed_csv)
    phase_distance = _safe_read_csv(phase_distance_csv)
    criticality = _safe_read_csv(criticality_csv)

    if identity.empty:
        raise FileNotFoundError(f"Required input missing or empty: {identity_csv}")

    base_cols = [c for c in ["node_id", "r", "alpha"] if c in identity.columns]
    nodes = identity[base_cols].copy()

    if not obstruction.empty:
        keep = [c for c in ["node_id", "obstruction_mean_abs_holonomy"] if c in obstruction.columns]
        if keep:
            nodes = nodes.merge(obstruction[keep], on="node_id", how="left")

    if not obstruction_signed.empty:
        keep = [c for c in ["node_id", "obstruction_signed_sum_holonomy"] if c in obstruction_signed.columns]
        if keep:
            nodes = nodes.merge(obstruction_signed[keep], on="node_id", how="left")

    if "obstruction_mean_abs_holonomy" in nodes.columns:
        nodes["absolute_holonomy_node"] = nodes["obstruction_mean_abs_holonomy"]
    else:
        nodes["absolute_holonomy_node"] = np.nan

    if not phase_distance.empty:
        keep = [c for c in ["r", "alpha", "distance_to_seam"] if c in phase_distance.columns]
        if keep:
            for c in keep:
                phase_distance[c] = pd.to_numeric(phase_distance[c], errors="coerce")
            nodes = nodes.merge(phase_distance[keep], on=["r", "alpha"], how="left")

    if not criticality.empty:
        keep = [c for c in ["r", "alpha", "criticality"] if c in criticality.columns]
        if keep:
            for c in keep:
                criticality[c] = pd.to_numeric(criticality[c], errors="coerce")
            nodes = nodes.merge(criticality[keep], on=["r", "alpha"], how="left")

    for col in nodes.columns:
        if col != "node_id":
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    return nodes.sort_values(["r", "alpha"]).reset_index(drop=True)


def add_angle_and_bins(nodes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = nodes.copy()

    z1 = _zscore(work[BEST_BASIS[0]], cfg.eps)
    z2 = _zscore(work[BEST_BASIS[1]], cfg.eps)
    z3 = _zscore(work[BEST_BASIS[2]], cfg.eps)

    mean_z = (z1 + z2 + z3) / 3.0
    d1 = z1 - mean_z
    d2 = z2 - mean_z
    d3 = z3 - mean_z

    x = 0.5 * (2.0 * d1 - d2 - d3)
    y = (math.sqrt(3.0) / 2.0) * (d2 - d3)
    work["dew_angle_deg"] = np.degrees(np.arctan2(y, x))
    work["dew_sector"] = pd.cut(
        work["dew_angle_deg"],
        bins=[-180, -120, -60, 0, 60, 120, 180],
        labels=["S1", "S2", "S3", "S4", "S5", "S6"],
        include_lowest=True,
    )

    seam = pd.to_numeric(work["distance_to_seam"], errors="coerce")
    q1 = float(seam.quantile(cfg.near_q))
    q2 = float(seam.quantile(cfg.mid_q))

    def seam_bin(v: float) -> str | None:
        if pd.isna(v):
            return None
        if v <= q1:
            return "near"
        if v <= q2:
            return "mid"
        return "far"

    work["seam_bin"] = seam.map(seam_bin)
    return work


def select_manifest(nodes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows: list[pd.DataFrame] = []

    candidates = nodes.dropna(subset=["seam_bin", "dew_sector"]).copy()
    candidates["bucket"] = candidates["seam_bin"].astype(str) + "__" + candidates["dew_sector"].astype(str)

    for bucket, sub in candidates.groupby("bucket", dropna=False):
        sub = sub.copy()
        # Prefer structurally informative nodes: larger criticality + obstruction magnitude.
        crit = pd.to_numeric(sub["criticality"], errors="coerce").fillna(sub["criticality"].mean())
        obs = pd.to_numeric(sub["obstruction_mean_abs_holonomy"], errors="coerce").fillna(sub["obstruction_mean_abs_holonomy"].mean())
        sub["priority"] = (
            _zscore(crit, cfg.eps).fillna(0.0)
            + _zscore(obs, cfg.eps).fillna(0.0)
            + rng.normal(0.0, 1e-6, len(sub))
        )
        chosen = sub.sort_values("priority", ascending=False).head(cfg.n_per_bucket).copy()
        rows.append(chosen)

    if not rows:
        return pd.DataFrame()

    manifest = pd.concat(rows, ignore_index=True).copy()
    manifest = manifest.sort_values(["seam_bin", "dew_sector", "r", "alpha"]).reset_index(drop=True)
    manifest.insert(0, "endpoint_id", [f"ep{i+1:03d}" for i in range(len(manifest))])
    manifest["label"] = (
        manifest["seam_bin"].astype(str).str.lower()
        + "_"
        + manifest["dew_sector"].astype(str).str.lower()
        + "_"
        + manifest.groupby(["seam_bin", "dew_sector"]).cumcount().add(1).astype(str)
    )

    keep = [
        "endpoint_id",
        "label",
        "node_id",
        "r",
        "alpha",
        "seam_bin",
        "dew_sector",
        "distance_to_seam",
        "criticality",
        "obstruction_mean_abs_holonomy",
        "obstruction_signed_sum_holonomy",
        "dew_angle_deg",
    ]
    return manifest[keep]


def summarize(manifest: pd.DataFrame) -> str:
    lines = []
    lines.append("=== Geodesic Endpoint Manifest Summary ===")
    lines.append("")
    lines.append(f"n_endpoints = {len(manifest)}")
    lines.append("")
    if manifest.empty:
        return "\n".join(lines)

    counts = (
        manifest.groupby(["seam_bin", "dew_sector"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    lines.append("Bucket counts")
    for _, row in counts.iterrows():
        lines.append(f"  {row['seam_bin']} / {row['dew_sector']}: {int(row['n'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured geodesic endpoint manifest.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--outdir", default="outputs/toy_geodesic_endpoint_manifest")
    parser.add_argument("--n-per-bucket", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        outputs_root=args.outputs_root,
        outdir=args.outdir,
        n_per_bucket=args.n_per_bucket,
        random_seed=args.random_seed,
    )
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(Path(cfg.outputs_root))
    nodes = add_angle_and_bins(nodes, cfg)
    manifest = select_manifest(nodes, cfg)

    manifest.to_csv(outdir / "geodesic_endpoint_manifest.csv", index=False)
    (outdir / "geodesic_endpoint_manifest_summary.txt").write_text(
        summarize(manifest),
        encoding="utf-8",
    )

    print(outdir / "geodesic_endpoint_manifest.csv")
    print(outdir / "geodesic_endpoint_manifest_summary.txt")


if __name__ == "__main__":
    main()
