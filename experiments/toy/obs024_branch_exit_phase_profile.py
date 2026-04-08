#!/usr/bin/env python3
"""
OBS-024 — Branch-exit phase profile.

Segment branch-exit paths into:
- pre_contact
- seam_contact
- post_exit

and summarize:
- relational mismatch
- local mismatch
- distance to seam
- pathwise parallel-transport residuals

Inputs
------
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv
outputs/obs022_scene_bundle/scene_nodes.csv
outputs/obs022_scene_bundle/scene_edges.csv

Outputs
-------
outputs/obs024_branch_exit_phase_profile/
  branch_exit_phase_profile_rows.csv
  branch_exit_phase_profile_paths.csv
  obs024_branch_exit_phase_profile_summary.txt
  obs024_branch_exit_phase_profile_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.geometry.directional_field import DirectionalField
from pam.geometry.parallel_transport import parallel_transport_along_path


@dataclass(frozen=True)
class Config:
    mismatch_nodes_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    scene_nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    scene_edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    outdir: str = "outputs/obs024_branch_exit_phase_profile"
    seam_threshold: float = 0.15
    min_post_exit_steps: int = 2


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def safe_sem(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    if n <= 1:
        return 0.0
    return float(s.std(ddof=1) / np.sqrt(n))


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, DirectionalField]:
    nodes = pd.read_csv(cfg.mismatch_nodes_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for c in [
        "node_id",
        "neighbor_direction_mismatch_mean",
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

    field = DirectionalField.from_csv(
        cfg.scene_nodes_csv,
        cfg.scene_edges_csv,
        connection_theta_col="fim_theta",
        response_theta_col="rsp_theta",
    )
    return nodes, routes, field


def prepare_routes(nodes: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    enrich_cols = [
        c for c in [
            "node_id",
            "neighbor_direction_mismatch_mean",
            "local_direction_mismatch_deg",
            "transport_align_mean_deg",
        ] if c in nodes.columns
    ]
    out = routes.merge(nodes[enrich_cols].drop_duplicates(subset=["node_id"]), on="node_id", how="left")
    out = out.sort_values(["path_id", "step"]).reset_index(drop=True)
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


def path_transport_metrics(field: DirectionalField, node_ids: list[int]) -> dict[str, float]:
    if len(node_ids) < 2:
        return {
            "path_transport_total_misalignment_deg": np.nan,
            "path_transport_mean_misalignment_deg": np.nan,
            "path_transport_max_misalignment_deg": np.nan,
            "path_transport_n_edges": 0,
            "endpoint_residual_deg": np.nan,
        }

    res = parallel_transport_along_path(field, node_ids)
    return {
        "path_transport_total_misalignment_deg": res.path_transport_total_misalignment_deg,
        "path_transport_mean_misalignment_deg": res.path_transport_mean_misalignment_deg,
        "path_transport_max_misalignment_deg": res.path_transport_max_misalignment_deg,
        "path_transport_n_edges": res.path_transport_n_edges,
        "endpoint_residual_deg": res.endpoint_residual_deg,
    }


def segment_branch_exit_paths(
    routes: pd.DataFrame,
    field: DirectionalField,
    seam_threshold: float,
    min_post_exit_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = routes[routes["is_branch_away"] == 1].copy()
    if len(work) == 0:
        return pd.DataFrame(), pd.DataFrame()

    row_records: list[dict] = []
    path_records: list[dict] = []

    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce")
        seam_mask = (d2s <= seam_threshold).fillna(False).to_numpy(dtype=bool)

        if seam_mask.sum() == 0:
            continue

        first_contact = int(np.argmax(seam_mask))
        last_contact = int(len(seam_mask) - 1 - np.argmax(seam_mask[::-1]))
        post_exit_start = find_post_exit_start(seam_mask[last_contact + 1 :], min_post_exit_steps)
        if post_exit_start is not None:
            post_exit_start = int(post_exit_start + last_contact + 1)

        phase = np.array(["pre_contact"] * len(grp), dtype=object)
        phase[first_contact : last_contact + 1] = "seam_contact"
        if post_exit_start is not None:
            phase[post_exit_start:] = "post_exit"

        grp["phase"] = phase

        for _, row in grp.iterrows():
            row_records.append(
                {
                    "path_id": row["path_id"],
                    "path_family": row.get("path_family", np.nan),
                    "step": row["step"],
                    "node_id": row.get("node_id", np.nan),
                    "r": row.get("r", np.nan),
                    "alpha": row.get("alpha", np.nan),
                    "mds1": row.get("mds1", np.nan),
                    "mds2": row.get("mds2", np.nan),
                    "distance_to_seam": row.get("distance_to_seam", np.nan),
                    "neighbor_direction_mismatch_mean": row.get("neighbor_direction_mismatch_mean", np.nan),
                    "local_direction_mismatch_deg": row.get("local_direction_mismatch_deg", np.nan),
                    "transport_align_mean_deg": row.get("transport_align_mean_deg", np.nan),
                    "phase": row["phase"],
                }
            )

        rec = {
            "path_id": path_id,
            "n_steps": int(pd.to_numeric(grp["step"], errors="coerce").max()),
            "first_contact_idx": first_contact,
            "last_contact_idx": last_contact,
            "post_exit_start_idx": post_exit_start if post_exit_start is not None else np.nan,
            "min_distance_to_seam": float(d2s.min()) if d2s.notna().any() else np.nan,
            "mean_distance_to_seam": safe_mean(d2s),
        }

        for ph in ["pre_contact", "seam_contact", "post_exit"]:
            sub = grp[grp["phase"] == ph]
            node_ids = [int(x) for x in pd.to_numeric(sub["node_id"], errors="coerce").dropna().tolist()]
            tm = path_transport_metrics(field, node_ids)

            rec[f"{ph}_n_rows"] = len(sub)
            rec[f"{ph}_mean_relational_mismatch"] = safe_mean(sub["neighbor_direction_mismatch_mean"])
            rec[f"{ph}_max_relational_mismatch"] = float(pd.to_numeric(sub["neighbor_direction_mismatch_mean"], errors="coerce").max()) if len(sub) else np.nan
            rec[f"{ph}_mean_local_mismatch"] = safe_mean(sub["local_direction_mismatch_deg"])
            rec[f"{ph}_mean_distance_to_seam"] = safe_mean(sub["distance_to_seam"])
            rec[f"{ph}_path_transport_mean_misalignment_deg"] = tm["path_transport_mean_misalignment_deg"]
            rec[f"{ph}_path_transport_max_misalignment_deg"] = tm["path_transport_max_misalignment_deg"]
            rec[f"{ph}_endpoint_residual_deg"] = tm["endpoint_residual_deg"]

        path_records.append(rec)

    return pd.DataFrame(row_records), pd.DataFrame(path_records)


def build_summary(path_df: pd.DataFrame, seam_threshold: float) -> str:
    lines = [
        "=== OBS-024 Branch Exit Phase Profile Summary ===",
        "",
        f"seam_threshold = {seam_threshold:.4f}",
        f"n_branch_exit_paths = {int(path_df['path_id'].nunique()) if len(path_df) else 0}",
        "",
    ]

    if len(path_df) == 0:
        lines.append("No branch-exit paths available.")
        return "\n".join(lines)

    for ph in ["pre_contact", "seam_contact", "post_exit"]:
        lines.extend(
            [
                f"{ph}",
                f"  mean relational mismatch      = {safe_mean(path_df[f'{ph}_mean_relational_mismatch']):.4f}",
                f"  mean local mismatch           = {safe_mean(path_df[f'{ph}_mean_local_mismatch']):.4f}",
                f"  mean distance to seam         = {safe_mean(path_df[f'{ph}_mean_distance_to_seam']):.4f}",
                f"  mean path-transport mismatch  = {safe_mean(path_df[f'{ph}_path_transport_mean_misalignment_deg']):.4f}",
                f"  mean endpoint residual        = {safe_mean(path_df[f'{ph}_endpoint_residual_deg']):.4f}",
                f"  mean rows per path            = {safe_mean(path_df[f'{ph}_n_rows']):.4f}",
                "",
            ]
        )

    pre = pd.to_numeric(path_df["pre_contact_mean_relational_mismatch"], errors="coerce")
    con = pd.to_numeric(path_df["seam_contact_mean_relational_mismatch"], errors="coerce")
    post = pd.to_numeric(path_df["post_exit_mean_relational_mismatch"], errors="coerce")

    lines.extend(
        [
            "Phase deltas",
            f"  mean(seam_contact - pre_contact) = {safe_mean(con - pre):.4f}",
            f"  mean(post_exit - seam_contact)   = {safe_mean(post - con):.4f}",
            f"  mean(post_exit - pre_contact)    = {safe_mean(post - pre):.4f}",
            "",
            "Interpretive summary",
            "- compare whether branch-exit paths sustain relational stress before and during seam contact",
            "- compare whether both relational mismatch and pathwise transport mismatch relax after sustained exit",
        ]
    )
    return "\n".join(lines)


def render_figure(path_df: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(12.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_trn = fig.add_subplot(gs[1, 1])

    phases = ["pre_contact", "seam_contact", "post_exit"]
    labels = ["pre-contact", "seam-contact", "post-exit"]

    rel_means = [safe_mean(path_df[f"{ph}_mean_relational_mismatch"]) for ph in phases]
    rel_sems = [safe_sem(path_df[f"{ph}_mean_relational_mismatch"]) for ph in phases]

    ax_bar.bar(labels, rel_means, yerr=rel_sems, alpha=0.9)
    ax_bar.set_ylabel("mean relational mismatch (deg)")
    ax_bar.set_title("Branch-exit relational profile", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    box_vals = [
        pd.to_numeric(path_df[f"{ph}_mean_relational_mismatch"], errors="coerce").dropna().to_numpy()
        for ph in phases
    ]
    ax_box.boxplot(box_vals, labels=labels)
    for i, vals in enumerate(box_vals, start=1):
        if len(vals):
            x = np.full(len(vals), i, dtype=float) + np.random.normal(0, 0.04, size=len(vals))
            ax_box.scatter(x, vals, s=18, alpha=0.55)
    ax_box.set_ylabel("path mean relational mismatch (deg)")
    ax_box.set_title("Path-level relational distribution", fontsize=14, pad=8)
    ax_box.grid(alpha=0.15, axis="y")

    dist_means = [safe_mean(path_df[f"{ph}_mean_distance_to_seam"]) for ph in phases]
    ax_dist.bar(labels, dist_means, alpha=0.9)
    ax_dist.set_ylabel("mean distance to seam")
    ax_dist.set_title("Distance-to-seam profile", fontsize=14, pad=8)
    ax_dist.grid(alpha=0.15, axis="y")

    trn_means = [safe_mean(path_df[f"{ph}_path_transport_mean_misalignment_deg"]) for ph in phases]
    ax_trn.bar(labels, trn_means, alpha=0.9)
    ax_trn.set_ylabel("mean path transport mismatch (deg)")
    ax_trn.set_title("Pathwise transport profile", fontsize=14, pad=8)
    ax_trn.grid(alpha=0.15, axis="y")

    fig.suptitle("PAM Observatory — OBS-024 branch-exit phase profile", fontsize=18)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-024 branch-exit phase profile.")
    parser.add_argument("--mismatch-nodes-csv", default=Config.mismatch_nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--scene-nodes-csv", default=Config.scene_nodes_csv)
    parser.add_argument("--scene-edges-csv", default=Config.scene_edges_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--min-post-exit-steps", type=int, default=Config.min_post_exit_steps)
    args = parser.parse_args()

    cfg = Config(
        mismatch_nodes_csv=args.mismatch_nodes_csv,
        routes_csv=args.routes_csv,
        scene_nodes_csv=args.scene_nodes_csv,
        scene_edges_csv=args.scene_edges_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        min_post_exit_steps=args.min_post_exit_steps,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, routes, field = load_inputs(cfg)
    routes = prepare_routes(nodes, routes)
    row_df, path_df = segment_branch_exit_paths(routes, field, cfg.seam_threshold, cfg.min_post_exit_steps)

    rows_csv = outdir / "branch_exit_phase_profile_rows.csv"
    paths_csv = outdir / "branch_exit_phase_profile_paths.csv"
    txt_path = outdir / "obs024_branch_exit_phase_profile_summary.txt"
    png_path = outdir / "obs024_branch_exit_phase_profile_figure.png"

    row_df.to_csv(rows_csv, index=False)
    path_df.to_csv(paths_csv, index=False)
    txt_path.write_text(build_summary(path_df, cfg.seam_threshold), encoding="utf-8")
    render_figure(path_df, png_path)

    print(rows_csv)
    print(paths_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()