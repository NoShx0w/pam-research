#!/usr/bin/env python3
from __future__ import annotations

"""
OBS-052 — Attractor basin mapping from recurrence, boundedness, and recovery landing.

Core question
-------------
Do recovery-like coupled windows preferentially terminate in recurrent,
low-divergence, low-roughness, low-drift regions, and do these regions form
distinct seam-core/near alignment sinks versus seam-far decoupled sinks?

Motivation
----------
OBS-050 established that recovery-like roughness-escalation windows are much
more likely to remain seam-coupled than nonrecovering windows.

OBS-051 established that, among coupled windows, recovery-like regimes are
dynamically more bounded.

OBS-052 asks the next natural question:
where do those bounded, recovery-like windows tend to land?

This script builds a first node-level basin analysis using:
- recurrence / visitation density
- local divergence from OBS-051
- roughness from OBS-050
- seam-drift from OBS-050
- recovery landing density

This is a first attractor-basin pass, not yet a final recovery-basin theory.

Inputs
------
1. OBS-022 scene bundle:
   outputs/obs022_scene_bundle/scene_nodes.csv
   outputs/obs022_scene_bundle/scene_routes.csv

2. Family substrate:
   outputs/scales/100000/family_substrate/path_family_assignments.csv

3. OBS-050:
   outputs/obs050_structural_coupling_persistence/structural_coupling_segments.csv

4. OBS-051:
   outputs/obs051_local_divergence_in_coupled_windows/obs051_window_divergence.csv

Outputs
-------
<outdir>/
  obs052_node_basin_table.csv
  obs052_recovery_landing_table.csv
  obs052_top_basin_candidates.csv
  obs052_attractor_basin_summary.txt
  obs052_attractor_map_mds.png
  obs052_recovery_landing_map_mds.png
  obs052_phase_portrait_roughness_vs_lambda.png

Interpretation
--------------
Node-level attractor score A(x) is a first heuristic composite:

    A(x) = z(recurrence_paths)
           - z(mean_lambda_local)
           - z(mean_roughness)
           - z(mean_abs_m_seam)
           + z(recovery_landings)

High A(x):
    recurrent, bounded, low-roughness, low-drift, recovery-attracting node

Provisional classes:
- alignment_sink: seam core/near and high A(x)
- decoupled_sink: seam far and high A(x)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECOVERING_FAMILIES = {"stable_seam_corridor", "reorganization_heavy"}
NONRECOVERING_FAMILIES = {"off_seam_reorganizing", "settled_distant"}


@dataclass(frozen=True)
class Config:
    scene_nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    scene_routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    family_csv: str = "outputs/scales/100000/family_substrate/path_family_assignments.csv"
    obs050_segments_csv: str = "outputs/obs050_structural_coupling_persistence/structural_coupling_segments.csv"
    obs051_windows_csv: str = "outputs/obs051_local_divergence_in_coupled_windows/obs051_window_divergence.csv"
    outdir: str = "outputs/obs052_attractor_basin_mapping"
    seam_core_max: float = 0.05
    seam_near_max: float = 0.15
    top_k: int = 20


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).copy()


def to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def classify_outcome(path_family: str) -> str:
    if path_family in RECOVERING_FAMILIES:
        return "recovering"
    if path_family in NONRECOVERING_FAMILIES:
        return "nonrecovering"
    return "other"


def classify_seam_band(mean_distance_to_seam: float, core_max: float, near_max: float) -> str:
    if not np.isfinite(mean_distance_to_seam):
        return "unknown"
    if mean_distance_to_seam <= core_max:
        return "core"
    if mean_distance_to_seam <= near_max:
        return "near"
    return "far"


def zscore_series(s: pd.Series, eps: float = 1e-12) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(x.mean(skipna=True)) if x.notna().any() else 0.0
    sd = float(x.std(skipna=True)) if x.notna().any() else 0.0
    return (x - mu) / max(sd, eps)


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = safe_read_csv(Path(cfg.scene_nodes_csv))
    routes = safe_read_csv(Path(cfg.scene_routes_csv))
    fam = safe_read_csv(Path(cfg.family_csv))
    segs = safe_read_csv(Path(cfg.obs050_segments_csv))
    obs051 = safe_read_csv(Path(cfg.obs051_windows_csv))

    for df in [nodes, routes, fam, segs, obs051]:
        if "path_id" in df.columns:
            df["path_id"] = df["path_id"].astype(str)
        if "segment_id" in df.columns:
            df["segment_id"] = df["segment_id"].astype(str)

    to_numeric_inplace(
        nodes,
        [
            "node_id", "r", "alpha", "mds1", "mds2",
            "distance_to_seam", "signed_phase", "lazarus_score",
            "response_strength", "node_holonomy_proxy",
        ],
    )
    to_numeric_inplace(
        routes,
        [
            "node_id", "step", "r", "alpha", "mds1", "mds2",
            "distance_to_seam", "signed_phase", "lazarus_score",
            "response_strength",
        ],
    )
    to_numeric_inplace(
        segs,
        [
            "start_step", "end_step", "center_step",
            "mean_roughness", "mean_roughness_smoothed",
            "mean_distance_to_seam", "min_distance_to_seam",
            "center_distance_to_seam", "m_r", "m_seam",
        ],
    )
    to_numeric_inplace(
        obs051,
        [
            "mean_d_start", "mean_d_end",
            "mean_lambda_local", "median_lambda_local", "max_lambda_local",
        ],
    )

    if "path_family" not in fam.columns:
        raise ValueError("family assignments file must contain path_family")

    return nodes, routes, fam, segs, obs051


def build_segment_endpoint_nodes(routes: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    """
    Map each OBS-050 segment to its terminal node via scene_routes.
    Uses exact path_id + end_step match.
    """
    use_routes = routes.copy()
    use_routes["path_id"] = use_routes["path_id"].astype(str)
    use_routes["step"] = pd.to_numeric(use_routes["step"], errors="coerce")

    use_segs = segs[["segment_id", "path_id", "path_family", "outcome_group", "start_step", "end_step", "seam_band", "coupling_class"]].copy()
    use_segs["path_id"] = use_segs["path_id"].astype(str)
    use_segs["end_step"] = pd.to_numeric(use_segs["end_step"], errors="coerce")

    end_nodes = use_segs.merge(
        use_routes[
            [
                "path_id", "step", "node_id", "mds1", "mds2",
                "distance_to_seam", "path_family", "is_representative", "is_branch_away"
            ]
            if "is_representative" in use_routes.columns and "is_branch_away" in use_routes.columns
            else [c for c in ["path_id", "step", "node_id", "mds1", "mds2", "distance_to_seam", "path_family"] if c in use_routes.columns]
        ],
        left_on=["path_id", "end_step"],
        right_on=["path_id", "step"],
        how="left",
        suffixes=("", "_route"),
    )

    if "path_family_route" in end_nodes.columns:
        end_nodes["path_family"] = end_nodes["path_family"].fillna(end_nodes["path_family_route"])
        end_nodes = end_nodes.drop(columns=["path_family_route"])

    return end_nodes


def build_node_recurrence_table(nodes: pd.DataFrame, routes: pd.DataFrame, fam: pd.DataFrame) -> pd.DataFrame:
    fam_use = fam[["path_id", "path_family"]].drop_duplicates().copy()
    fam_use["outcome_group"] = fam_use["path_family"].map(classify_outcome)

    route_use = routes.copy()
    route_use["path_id"] = route_use["path_id"].astype(str)
    route_use = route_use.merge(fam_use, on="path_id", how="left", suffixes=("", "_fam"))
    if "path_family_fam" in route_use.columns:
        route_use["path_family"] = route_use["path_family"].fillna(route_use["path_family_fam"])
        route_use = route_use.drop(columns=["path_family_fam"])

    agg = (
        route_use.groupby("node_id", dropna=False)
        .agg(
            n_visits=("path_id", "size"),
            n_unique_paths=("path_id", "nunique"),
            n_recovering_paths=("path_id", lambda s: int(route_use.loc[s.index, "outcome_group"].eq("recovering").groupby(level=0).first().sum()) if len(s) else 0),
        )
        .reset_index()
    )

    # Better explicit counts by node
    counts = (
        route_use.groupby(["node_id", "outcome_group"], dropna=False)["path_id"]
        .nunique()
        .reset_index(name="n_unique_paths_by_outcome")
    )
    pivot = (
        counts.pivot(index="node_id", columns="outcome_group", values="n_unique_paths_by_outcome")
        .fillna(0)
        .reset_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    for c in ["recovering", "nonrecovering", "other"]:
        if c not in pivot.columns:
            pivot[c] = 0
        pivot = pivot.rename(columns={c: f"n_unique_paths_{c}"})

    out = nodes.copy()
    out = out.merge(agg, on="node_id", how="left")
    out = out.merge(pivot, on="node_id", how="left")

    for c in ["n_visits", "n_unique_paths", "n_unique_paths_recovering", "n_unique_paths_nonrecovering", "n_unique_paths_other"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    out["seam_band"] = out["distance_to_seam"].map(
        lambda x: classify_seam_band(float(x), 0.05, 0.15) if pd.notna(x) else "unknown"
    )
    return out


def build_obs051_window_summary(obs051: pd.DataFrame) -> pd.DataFrame:
    """
    OBS-051 input file name suggests window divergence, but the content is the
    per-window summary table. We aggregate defensively by segment_id.
    """
    required = {"segment_id", "path_id", "outcome_group"}
    missing = required - set(obs051.columns)
    if missing:
        raise ValueError(f"OBS-051 windows file missing required columns: {sorted(missing)}")

    keep = [
        c for c in [
            "segment_id", "path_id", "path_family", "outcome_group",
            "seam_band", "coupling_class",
            "mean_lambda_local", "median_lambda_local", "max_lambda_local",
            "mean_d_start", "mean_d_end",
        ] if c in obs051.columns
    ]
    return obs051[keep].drop_duplicates().copy()


def build_recovery_landing_table(
    seg_end_nodes: pd.DataFrame,
    obs051_win: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count where windows land, especially recovering coupled windows.
    """
    work = seg_end_nodes.copy()
    work = work.merge(
        obs051_win[
            [c for c in [
                "segment_id", "mean_lambda_local", "median_lambda_local", "max_lambda_local",
                "coupling_class", "seam_band", "outcome_group", "path_family"
            ] if c in obs051_win.columns]
        ].drop_duplicates(subset=["segment_id"]),
        on="segment_id",
        how="left",
        suffixes=("", "_obs051"),
    )

    if "outcome_group_obs051" in work.columns and "outcome_group" in work.columns:
        work["outcome_group"] = work["outcome_group"].fillna(work["outcome_group_obs051"])
        work = work.drop(columns=["outcome_group_obs051"])
    if "path_family_obs051" in work.columns and "path_family" in work.columns:
        work["path_family"] = work["path_family"].fillna(work["path_family_obs051"])
        work = work.drop(columns=["path_family_obs051"])

    out = (
        work.groupby(["node_id", "outcome_group"], dropna=False)
        .agg(
            n_landings=("segment_id", "size"),
            n_unique_paths=("path_id", "nunique"),
            mean_lambda_local=("mean_lambda_local", "mean"),
        )
        .reset_index()
    )

    pivot_counts = (
        out.pivot(index="node_id", columns="outcome_group", values="n_landings")
        .fillna(0)
        .reset_index()
    )
    pivot_paths = (
        out.pivot(index="node_id", columns="outcome_group", values="n_unique_paths")
        .fillna(0)
        .reset_index()
    )

    pivot_counts.columns = [str(c) for c in pivot_counts.columns]
    pivot_paths.columns = [str(c) for c in pivot_paths.columns]

    for c in ["recovering", "nonrecovering", "other"]:
        if c not in pivot_counts.columns:
            pivot_counts[c] = 0
        if c not in pivot_paths.columns:
            pivot_paths[c] = 0

    pivot_counts = pivot_counts.rename(
        columns={
            "recovering": "recovering_landings",
            "nonrecovering": "nonrecovering_landings",
            "other": "other_landings",
        }
    )
    pivot_paths = pivot_paths.rename(
        columns={
            "recovering": "recovering_unique_path_landings",
            "nonrecovering": "nonrecovering_unique_path_landings",
            "other": "other_unique_path_landings",
        }
    )

    landing_table = pivot_counts.merge(pivot_paths, on="node_id", how="outer").fillna(0)
    for c in landing_table.columns:
        if c != "node_id":
            landing_table[c] = pd.to_numeric(landing_table[c], errors="coerce").fillna(0).astype(int)

    return landing_table


def build_node_basin_table(
    node_table: pd.DataFrame,
    seg_end_nodes: pd.DataFrame,
    obs051_win: pd.DataFrame,
    landing_table: pd.DataFrame,
) -> pd.DataFrame:
    # Join OBS-050 per-segment variables onto end nodes, then aggregate by node
    seg_attrs = seg_end_nodes[
        [c for c in [
            "segment_id", "node_id", "path_id", "path_family", "outcome_group",
            "seam_band", "coupling_class",
            "mean_roughness", "mean_roughness_smoothed", "m_seam", "m_r",
            "mean_distance_to_seam", "min_distance_to_seam"
        ] if c in seg_end_nodes.columns]
    ].copy()

    if "mean_roughness" not in seg_attrs.columns:
        # pull from OBS-050 if segment mapping retained only subset
        pass

    seg_merged = seg_attrs.merge(
        obs051_win[
            [c for c in [
                "segment_id", "mean_lambda_local", "median_lambda_local", "max_lambda_local"
            ] if c in obs051_win.columns]
        ].drop_duplicates(subset=["segment_id"]),
        on="segment_id",
        how="left",
    )

    node_dyn = (
        seg_merged.groupby("node_id", dropna=False)
        .agg(
            mean_lambda_local=("mean_lambda_local", "mean"),
            median_lambda_local=("mean_lambda_local", "median"),
            mean_roughness=("mean_roughness", "mean"),
            mean_abs_m_seam=("m_seam", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
            mean_m_seam=("m_seam", "mean"),
            n_segment_landings=("segment_id", "size"),
            n_unique_segment_paths=("path_id", "nunique"),
        )
        .reset_index()
    )

    out = node_table.merge(node_dyn, on="node_id", how="left")
    out = out.merge(landing_table, on="node_id", how="left")

    for c in [
        "recovering_landings", "nonrecovering_landings", "other_landings",
        "recovering_unique_path_landings", "nonrecovering_unique_path_landings", "other_unique_path_landings",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # composite score
    out["z_recurrence"] = zscore_series(out["n_unique_paths"])
    out["z_lambda"] = zscore_series(out["mean_lambda_local"])
    out["z_roughness"] = zscore_series(out["mean_roughness"])
    out["z_abs_m_seam"] = zscore_series(out["mean_abs_m_seam"])
    out["z_recovery_landings"] = zscore_series(out["recovering_landings"])

    out["attractor_score"] = (
        out["z_recurrence"].fillna(0.0)
        - out["z_lambda"].fillna(0.0)
        - out["z_roughness"].fillna(0.0)
        - out["z_abs_m_seam"].fillna(0.0)
        + out["z_recovery_landings"].fillna(0.0)
    )

    def _classify(row: pd.Series) -> str:
        seam_band = str(row.get("seam_band", "unknown"))
        if not pd.notna(row.get("attractor_score")):
            return "unknown"
        if seam_band in {"core", "near"}:
            return "alignment_sink"
        if seam_band == "far":
            return "decoupled_sink"
        return "unknown"

    out["basin_class"] = out.apply(_classify, axis=1)

    return out


def build_top_basin_candidates(node_basin: pd.DataFrame, top_k: int) -> pd.DataFrame:
    cols = [
        c for c in [
            "node_id", "r", "alpha", "mds1", "mds2", "seam_band", "basin_class",
            "n_visits", "n_unique_paths",
            "recovering_landings", "nonrecovering_landings",
            "mean_lambda_local", "mean_roughness", "mean_abs_m_seam",
            "attractor_score",
        ] if c in node_basin.columns
    ]
    return (
        node_basin[cols]
        .sort_values("attractor_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def summarize_text(
    node_basin: pd.DataFrame,
    top_basin: pd.DataFrame,
    cfg: Config,
) -> str:
    lines: list[str] = []
    lines.append("=== OBS-052 Attractor Basin Mapping Summary ===")
    lines.append("")
    lines.append(f"top_k = {cfg.top_k}")
    lines.append(f"n_nodes = {len(node_basin)}")
    lines.append("")

    if len(node_basin):
        for cls in ["alignment_sink", "decoupled_sink", "unknown"]:
            sub = node_basin[node_basin["basin_class"] == cls]
            if len(sub) == 0:
                continue
            lines.append(
                f"{cls}: "
                f"n_nodes={len(sub)}, "
                f"mean_attractor_score={pd.to_numeric(sub['attractor_score'], errors='coerce').mean():.6f}, "
                f"mean_lambda_local={pd.to_numeric(sub['mean_lambda_local'], errors='coerce').mean():.6f}, "
                f"mean_roughness={pd.to_numeric(sub['mean_roughness'], errors='coerce').mean():.6f}, "
                f"mean_recovering_landings={pd.to_numeric(sub['recovering_landings'], errors='coerce').mean():.6f}"
            )

    lines.append("")
    lines.append("Top basin candidates")
    for _, row in top_basin.iterrows():
        lines.append(
            f"node_id={int(row['node_id'])} | "
            f"seam_band={row.get('seam_band')} | "
            f"basin_class={row.get('basin_class')} | "
            f"attractor_score={float(row['attractor_score']):.6f} | "
            f"n_unique_paths={int(row['n_unique_paths']) if pd.notna(row.get('n_unique_paths')) else 0} | "
            f"recovering_landings={int(row['recovering_landings']) if pd.notna(row.get('recovering_landings')) else 0} | "
            f"nonrecovering_landings={int(row['nonrecovering_landings']) if pd.notna(row.get('nonrecovering_landings')) else 0} | "
            f"mean_lambda_local={float(row['mean_lambda_local']) if pd.notna(row.get('mean_lambda_local')) else float('nan'):.6f} | "
            f"mean_roughness={float(row['mean_roughness']) if pd.notna(row.get('mean_roughness')) else float('nan'):.6f} | "
            f"mean_abs_m_seam={float(row['mean_abs_m_seam']) if pd.notna(row.get('mean_abs_m_seam')) else float('nan'):.6f}"
        )

    return "\n".join(lines)


def plot_attractor_map(node_basin: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = node_basin.dropna(subset=["mds1", "mds2", "attractor_score"]).copy()

    sizes = 20 + 20 * zscore_series(plot_df["n_unique_paths"]).fillna(0.0).clip(lower=-1.0, upper=3.0).to_numpy()
    sizes = np.maximum(sizes, 12)

    sc = ax.scatter(
        plot_df["mds1"],
        plot_df["mds2"],
        c=plot_df["attractor_score"],
        s=sizes,
    )
    ax.set_title("OBS-052: attractor score on manifold")
    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    fig.colorbar(sc, ax=ax, label="attractor_score")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_recovery_landing_map(node_basin: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = node_basin.dropna(subset=["mds1", "mds2"]).copy()
    vals = pd.to_numeric(plot_df.get("recovering_landings", 0), errors="coerce").fillna(0)
    sizes = 12 + 8 * np.sqrt(vals.to_numpy(dtype=float))

    sc = ax.scatter(
        plot_df["mds1"],
        plot_df["mds2"],
        c=vals,
        s=sizes,
    )
    ax.set_title("OBS-052: recovery landing density on manifold")
    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    fig.colorbar(sc, ax=ax, label="recovering_landings")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_phase_portrait(node_basin: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = node_basin.dropna(subset=["mean_roughness", "mean_lambda_local", "seam_band"]).copy()

    for seam_band in ["core", "near", "far", "unknown"]:
        sub = plot_df[plot_df["seam_band"] == seam_band]
        if len(sub) == 0:
            continue
        ax.scatter(sub["mean_roughness"], sub["mean_lambda_local"], s=24, label=seam_band)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("mean_roughness")
    ax.set_ylabel("mean_lambda_local")
    ax.set_title("OBS-052: phase portrait (roughness vs local divergence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBS-052 attractor basin mapping.")
    parser.add_argument("--scene-nodes-csv", default=Config.scene_nodes_csv)
    parser.add_argument("--scene-routes-csv", default=Config.scene_routes_csv)
    parser.add_argument("--family-csv", default=Config.family_csv)
    parser.add_argument("--obs050-segments-csv", default=Config.obs050_segments_csv)
    parser.add_argument("--obs051-windows-csv", default=Config.obs051_windows_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-core-max", type=float, default=Config.seam_core_max)
    parser.add_argument("--seam-near-max", type=float, default=Config.seam_near_max)
    parser.add_argument("--top-k", type=int, default=Config.top_k)
    args = parser.parse_args()

    cfg = Config(
        scene_nodes_csv=args.scene_nodes_csv,
        scene_routes_csv=args.scene_routes_csv,
        family_csv=args.family_csv,
        obs050_segments_csv=args.obs050_segments_csv,
        obs051_windows_csv=args.obs051_windows_csv,
        outdir=args.outdir,
        seam_core_max=args.seam_core_max,
        seam_near_max=args.seam_near_max,
        top_k=args.top_k,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, routes, fam, segs, obs051 = load_inputs(cfg)

    if "outcome_group" not in segs.columns and "path_family" in segs.columns:
        segs["outcome_group"] = segs["path_family"].map(classify_outcome)
    if "coupling_class" not in segs.columns and "seam_band" in segs.columns:
        segs["coupling_class"] = np.where(
            segs["seam_band"].isin(["core", "near"]),
            "coupled",
            np.where(segs["seam_band"] == "far", "decoupled", "unknown"),
        )

    node_recurrence = build_node_recurrence_table(nodes, routes, fam)
    obs051_win = build_obs051_window_summary(obs051)
    seg_end_nodes = build_segment_endpoint_nodes(routes, segs)
    # add obs050 fields needed for basin aggregation
    seg_end_nodes = seg_end_nodes.merge(
        segs[
            [c for c in [
                "segment_id", "path_id", "path_family", "outcome_group",
                "seam_band", "coupling_class",
                "mean_roughness", "mean_roughness_smoothed", "m_seam", "m_r",
                "mean_distance_to_seam", "min_distance_to_seam"
            ] if c in segs.columns]
        ].drop_duplicates(subset=["segment_id"]),
        on=["segment_id", "path_id"],
        how="left",
        suffixes=("", "_seg"),
    )

    landing_table = build_recovery_landing_table(seg_end_nodes, obs051_win)
    node_basin = build_node_basin_table(node_recurrence, seg_end_nodes, obs051_win, landing_table)
    top_basin = build_top_basin_candidates(node_basin, cfg.top_k)

    node_basin.to_csv(outdir / "obs052_node_basin_table.csv", index=False)
    landing_table.to_csv(outdir / "obs052_recovery_landing_table.csv", index=False)
    top_basin.to_csv(outdir / "obs052_top_basin_candidates.csv", index=False)

    summary_txt = summarize_text(node_basin, top_basin, cfg)
    (outdir / "obs052_attractor_basin_summary.txt").write_text(summary_txt, encoding="utf-8")

    plot_attractor_map(node_basin, outdir / "obs052_attractor_map_mds.png")
    plot_recovery_landing_map(node_basin, outdir / "obs052_recovery_landing_map_mds.png")
    plot_phase_portrait(node_basin, outdir / "obs052_phase_portrait_roughness_vs_lambda.png")

    print(outdir / "obs052_node_basin_table.csv")
    print(outdir / "obs052_recovery_landing_table.csv")
    print(outdir / "obs052_top_basin_candidates.csv")
    print(outdir / "obs052_attractor_basin_summary.txt")
    print(outdir / "obs052_attractor_map_mds.png")
    print(outdir / "obs052_recovery_landing_map_mds.png")
    print(outdir / "obs052_phase_portrait_roughness_vs_lambda.png")


if __name__ == "__main__":
    main()
