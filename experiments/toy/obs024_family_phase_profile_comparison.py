#!/usr/bin/env python3
"""
OBS-024 — Family phase-profile comparison.

Compare phase-segmented relational mismatch across:
- branch_exit
- stable_seam_corridor
- reorganization_heavy

Inputs
------
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs024_family_phase_profile_comparison/
  family_phase_profile_rows.csv
  family_phase_profile_paths.csv
  obs024_family_phase_profile_comparison_summary.txt
  obs024_family_phase_profile_comparison_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    outdir: str = "outputs/obs024_family_phase_profile_comparison"
    seam_threshold: float = 0.15
    min_post_exit_steps: int = 2
    max_paths_per_class: int = 200


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def safe_sem(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    if n <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(n))


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for c in [
        "node_id",
        "neighbor_direction_mismatch_deg",
        "local_direction_mismatch_deg",
        "transport_align_mean_deg",
    ]:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

    for c in [
        "step", "node_id", "r", "alpha", "mds1", "mds2",
        "distance_to_seam", "is_branch_away", "is_representative",
    ]:
        if c in routes.columns:
            routes[c] = pd.to_numeric(routes[c], errors="coerce")

    return nodes, routes


def prepare_routes(nodes: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    enrich_cols = [
        c for c in [
            "node_id",
            "neighbor_direction_mismatch_deg",
            "local_direction_mismatch_deg",
            "transport_align_mean_deg",
        ] if c in nodes.columns
    ]
    out = routes.merge(
        nodes[enrich_cols].drop_duplicates(subset=["node_id"]),
        on="node_id",
        how="left",
    )
    out = out.sort_values(["path_id", "step"]).reset_index(drop=True)
    return out


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    out["route_class"] = np.select(
        [
            out["is_branch_away"] == 1,
            (out["is_representative"] == 1) & (out["path_family"] == "stable_seam_corridor"),
            (out["is_representative"] == 1) & (out["path_family"] == "reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def find_post_exit_start(seam_mask: np.ndarray, min_post_exit_steps: int) -> int | None:
    n = len(seam_mask)
    for i in range(n):
        if seam_mask[i]:
            continue
        j = i
        while j < n and (not seam_mask[j]):
            j += 1
        if (j - i) >= min_post_exit_steps:
            return i
    return None


def segment_paths(routes: pd.DataFrame, seam_threshold: float, min_post_exit_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = routes[routes["route_class"].isin(CLASS_ORDER)].copy()
    row_records = []
    path_records = []

    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        cls = str(grp["route_class"].iloc[0])

        d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce")
        seam_mask = (d2s <= seam_threshold).fillna(False).to_numpy(dtype=bool)

        if seam_mask.sum() == 0:
            continue

        first_contact = int(np.argmax(seam_mask))
        last_contact = int(len(seam_mask) - 1 - np.argmax(seam_mask[::-1]))

        tail = seam_mask[last_contact + 1 :]
        post_exit_start = find_post_exit_start(tail, min_post_exit_steps)
        if post_exit_start is not None:
            post_exit_start = int(post_exit_start + last_contact + 1)

        phase = np.array(["pre_contact"] * len(grp), dtype=object)
        phase[first_contact:last_contact + 1] = "seam_contact"
        if post_exit_start is not None:
            phase[post_exit_start:] = "post_exit"

        grp["phase"] = phase

        for _, row in grp.iterrows():
            row_records.append(
                {
                    "path_id": row["path_id"],
                    "route_class": cls,
                    "path_family": row.get("path_family", np.nan),
                    "step": row["step"],
                    "node_id": row.get("node_id", np.nan),
                    "distance_to_seam": row.get("distance_to_seam", np.nan),
                    "neighbor_direction_mismatch_deg": row.get("neighbor_direction_mismatch_deg", np.nan),
                    "local_direction_mismatch_deg": row.get("local_direction_mismatch_deg", np.nan),
                    "phase": row["phase"],
                }
            )

        rec = {
            "path_id": path_id,
            "route_class": cls,
            "n_steps": int(pd.to_numeric(grp["step"], errors="coerce").max()),
            "first_contact_idx": first_contact,
            "last_contact_idx": last_contact,
            "post_exit_start_idx": post_exit_start if post_exit_start is not None else np.nan,
        }

        for ph in ["pre_contact", "seam_contact", "post_exit"]:
            sub = grp[grp["phase"] == ph]
            rec[f"{ph}_n_rows"] = len(sub)
            rec[f"{ph}_mean_relational_mismatch"] = safe_mean(sub["neighbor_direction_mismatch_deg"])
            rec[f"{ph}_mean_local_mismatch"] = safe_mean(sub["local_direction_mismatch_deg"])
            rec[f"{ph}_mean_distance_to_seam"] = safe_mean(sub["distance_to_seam"])

        path_records.append(rec)

    return pd.DataFrame(row_records), pd.DataFrame(path_records)


def summarize_paths(path_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, grp in path_df.groupby("route_class", sort=False):
        row = {"route_class": cls, "n_paths": len(grp)}
        for ph in ["pre_contact", "seam_contact", "post_exit"]:
            row[f"{ph}_mean_relational_mismatch"] = safe_mean(grp[f"{ph}_mean_relational_mismatch"])
            row[f"{ph}_sem_relational_mismatch"] = safe_sem(grp[f"{ph}_mean_relational_mismatch"])
            row[f"{ph}_mean_local_mismatch"] = safe_mean(grp[f"{ph}_mean_local_mismatch"])
            row[f"{ph}_mean_distance_to_seam"] = safe_mean(grp[f"{ph}_mean_distance_to_seam"])
            row[f"{ph}_mean_rows"] = safe_mean(grp[f"{ph}_n_rows"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["order"] = out["route_class"].map({k: i for i, k in enumerate(CLASS_ORDER)})
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(summary_df: pd.DataFrame) -> str:
    lines = [
        "=== OBS-024 Family Phase Profile Comparison ===",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"{row['route_class']} (n_paths={int(row['n_paths'])})")
        for ph in ["pre_contact", "seam_contact", "post_exit"]:
            lines.append(
                f"  {ph}: "
                f"rel={float(row[f'{ph}_mean_relational_mismatch']):.4f}, "
                f"local={float(row[f'{ph}_mean_local_mismatch']):.4f}, "
                f"d2s={float(row[f'{ph}_mean_distance_to_seam']):.4f}, "
                f"rows={float(row[f'{ph}_mean_rows']):.4f}"
            )
        lines.append("")
    lines.extend(
        [
            "Interpretive guide",
            "- branch_exit should relax after sustained exit if seam obstruction is residency-dependent",
            "- stable_seam_corridor should remain elevated during seam contact",
            "- reorganization_heavy may remain elevated with more fragmented seam engagement",
        ]
    )
    return "\n".join(lines)


def render_figure(summary_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    phases = ["pre_contact", "seam_contact", "post_exit"]
    phase_labels = ["pre-contact", "seam-contact", "post-exit"]
    classes = summary_df["route_class"].tolist()
    x = np.arange(len(phases))
    width = 0.22

    # relational mismatch
    ax = axes[0, 0]
    for i, cls in enumerate(classes):
        row = summary_df[summary_df["route_class"] == cls].iloc[0]
        vals = [row[f"{ph}_mean_relational_mismatch"] for ph in phases]
        errs = [row[f"{ph}_sem_relational_mismatch"] for ph in phases]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, label=cls, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("mean relational mismatch (deg)")
    ax.set_title("Relational phase profile")
    ax.grid(alpha=0.15, axis="y")
    ax.legend()

    # local mismatch
    ax = axes[0, 1]
    for i, cls in enumerate(classes):
        row = summary_df[summary_df["route_class"] == cls].iloc[0]
        vals = [row[f"{ph}_mean_local_mismatch"] for ph in phases]
        ax.bar(x + (i - 1) * width, vals, width, label=cls, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("mean local mismatch (deg)")
    ax.set_title("Pointwise control profile")
    ax.grid(alpha=0.15, axis="y")

    # distance to seam
    ax = axes[1, 0]
    for i, cls in enumerate(classes):
        row = summary_df[summary_df["route_class"] == cls].iloc[0]
        vals = [row[f"{ph}_mean_distance_to_seam"] for ph in phases]
        ax.bar(x + (i - 1) * width, vals, width, label=cls, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("mean distance to seam")
    ax.set_title("Distance-to-seam profile")
    ax.grid(alpha=0.15, axis="y")

    # rows per path
    ax = axes[1, 1]
    for i, cls in enumerate(classes):
        row = summary_df[summary_df["route_class"] == cls].iloc[0]
        vals = [row[f"{ph}_mean_rows"] for ph in phases]
        ax.bar(x + (i - 1) * width, vals, width, label=cls, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("mean rows per path")
    ax.set_title("Phase occupancy")
    ax.grid(alpha=0.15, axis="y")

    fig.suptitle("PAM Observatory — OBS-024 family phase-profile comparison", fontsize=18)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-024 family phase-profile comparison.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--min-post-exit-steps", type=int, default=Config.min_post_exit_steps)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        min_post_exit_steps=args.min_post_exit_steps,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, routes = load_inputs(cfg)
    routes = classify_routes(prepare_routes(nodes, routes))
    row_df, path_df = segment_paths(routes, cfg.seam_threshold, cfg.min_post_exit_steps)
    summary_df = summarize_paths(path_df)

    rows_csv = outdir / "family_phase_profile_rows.csv"
    paths_csv = outdir / "family_phase_profile_paths.csv"
    txt_path = outdir / "obs024_family_phase_profile_comparison_summary.txt"
    png_path = outdir / "obs024_family_phase_profile_comparison_figure.png"

    row_df.to_csv(rows_csv, index=False)
    path_df.to_csv(paths_csv, index=False)
    txt_path.write_text(build_summary(summary_df), encoding="utf-8")
    render_figure(summary_df, png_path)

    print(rows_csv)
    print(paths_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
