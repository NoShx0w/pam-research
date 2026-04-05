#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_family_representatives.py

Select representative geodesic paths per family.

Expected inputs
---------------
- path diagnostics / family assignments CSV containing:
  path_id, near_fraction, mean_angle_jump_deg, n_sector_changes, path_family
- path-node CSV containing:
  path_id, step, node_id, r, alpha, mds1, mds2

Outputs
-------
- geodesic_family_representatives.csv
- geodesic_family_representative_path_nodes.csv
- geodesic_family_representatives_summary.txt
"""

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    family_csv: str
    path_nodes_csv: str
    outdir: str = "outputs/toy_geodesic_family_representatives"


def _zscore(series: pd.Series, eps: float = 1e-12) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean(skipna=True)) if len(x.dropna()) else 0.0
    sd = float(x.std(skipna=True)) if len(x.dropna()) else 0.0
    return (x - mu) / max(sd, eps)


def choose_family_reps(df: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: list[pd.Series] = []

    for fam, sub in df.groupby("path_family", dropna=False):
        sub = sub.copy().reset_index(drop=True)
        if sub.empty:
            continue

        near = pd.to_numeric(sub["near_fraction"], errors="coerce").fillna(0.0)
        rough = pd.to_numeric(sub["mean_angle_jump_deg"], errors="coerce").fillna(0.0)
        changes = pd.to_numeric(sub["n_sector_changes"], errors="coerce").fillna(0.0)

        if fam == "reorganization_heavy":
            score = _zscore(near) + _zscore(rough) + _zscore(changes)
        elif fam == "stable_seam_corridor":
            score = _zscore(near) - _zscore(rough) - _zscore(changes)
        elif fam == "off_seam_reorganizing":
            score = -_zscore(near) + _zscore(rough) + _zscore(changes)
        else:  # settled_distant
            score = -_zscore(near) - _zscore(rough) - _zscore(changes)

        sub["family_score"] = score

        # extreme high
        top = sub.sort_values("family_score", ascending=False).head(1).copy()
        top["rep_kind"] = "extreme_high"
        chosen_rows.append(top.iloc[0])

        # extreme low / contrast within family
        bottom = sub.sort_values("family_score", ascending=True).head(1).copy()
        bottom["rep_kind"] = "extreme_low"
        if bottom.iloc[0]["path_id"] != top.iloc[0]["path_id"]:
            chosen_rows.append(bottom.iloc[0])

        # median / centroid-ish
        med_score = float(sub["family_score"].median())
        sub["dist_to_median"] = (sub["family_score"] - med_score).abs()
        median = sub.sort_values("dist_to_median", ascending=True).head(1).copy()
        median["rep_kind"] = "median"
        median_path_id = median.iloc[0]["path_id"]
        if median_path_id not in {r["path_id"] for r in chosen_rows if isinstance(r, pd.Series)}:
            chosen_rows.append(median.iloc[0])

    reps = pd.DataFrame(chosen_rows).copy()
    reps = reps.reset_index(drop=True)
    reps.insert(0, "rep_id", [f"rep_{i+1:03d}" for i in range(len(reps))])
    return reps


def summarize(reps: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Geodesic Family Representatives Summary ===")
    lines.append("")
    lines.append(f"n_representatives = {len(reps)}")
    lines.append("")
    for fam, sub in reps.groupby("path_family", dropna=False):
        lines.append(f"{fam}: {len(sub)}")
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['rep_id']} / {row['rep_kind']} / path_id={row['path_id']}: "
                f"near_fraction={float(row['near_fraction']):.4f}, "
                f"mean_angle_jump={float(row['mean_angle_jump_deg']):.4f}, "
                f"sector_changes={float(row['n_sector_changes']):.0f}, "
                f"mean_criticality={float(row['mean_criticality']):.4f}, "
                f"mean_unsigned_obstruction={float(row['mean_unsigned_obstruction']):.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative geodesic paths per family.")
    parser.add_argument("--family-csv", required=True)
    parser.add_argument("--path-nodes-csv", required=True)
    parser.add_argument("--outdir", default="outputs/toy_geodesic_family_representatives")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fam = pd.read_csv(args.family_csv).copy()
    path_nodes = pd.read_csv(args.path_nodes_csv).copy()

    reps = choose_family_reps(fam)

    rep_nodes = path_nodes.merge(
        reps[["rep_id", "path_id", "path_family", "rep_kind"]],
        on="path_id",
        how="inner",
    )

    reps.to_csv(outdir / "geodesic_family_representatives.csv", index=False)
    rep_nodes.to_csv(outdir / "geodesic_family_representative_path_nodes.csv", index=False)
    (outdir / "geodesic_family_representatives_summary.txt").write_text(
        summarize(reps),
        encoding="utf-8",
    )

    print(outdir / "geodesic_family_representatives.csv")
    print(outdir / "geodesic_family_representative_path_nodes.csv")
    print(outdir / "geodesic_family_representatives_summary.txt")


if __name__ == "__main__":
    main()
