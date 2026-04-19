#!/usr/bin/env python3
"""
obs045_release_recovery_sweep.py

OBS-045 — Controlled release recovery under continuous support expansion.

Purpose
-------
Run a disciplined parameter sweep over the OBS-044 continuous response-flow
solver, focusing on support expansion and optional local interpolation changes,
to test whether broader cross-phase release can be recovered without losing
seam engagement or support honesty.

This study is intentionally conservative:
- it reuses the OBS-044 continuous solver as the core engine
- it reuses OBS-044c family classification logic
- it varies only a small number of interpretable parameters
- it writes all sweep outputs into a dedicated OBS-045 directory

Primary question
----------------
Can controlled expansion of the continuous solver's local support region recover
cross-phase release behavior while preserving the seam-engaged structure
established in OBS-044?

Outputs
-------
Directory:
    outputs/obs045_release_recovery_sweep/

Files:
    release_recovery_sweep_summary.csv
    release_recovery_family_comparison.csv
    release_recovery_phase_crossing_figure.png
    release_recovery_seam_contact_figure.png
    release_recovery_release_cross_phase_figure.png
    release_recovery_summary.txt

Per-configuration subdirectories:
    outputs/obs045_release_recovery_sweep/<config_label>/
        continuous_flow_trajectories.csv
        continuous_flow_trajectory_points.csv
        continuous_flow_summary.csv
        continuous_flow_support_region.csv
        continuous_flow_path_family_assignments.csv
        continuous_flow_path_family_summary.csv
        continuous_flow_paths.png
        continuous_flow_quiver.png

Notes
-----
- This script contains the minimal OBS-044 continuous solver internally so that
  the sweep is self-contained and reproducible.
- It also contains the OBS-044c family classifier for continuous trajectories.
- Default sweep varies support_radius_scale only, keeping knn and step size fixed.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/fim_response_operator/response_operator_nodes.csv")
DEFAULT_SEAM_NODES = Path("outputs/obs028c_canonical_seam_bundle/seam_nodes.csv")
DEFAULT_OUTDIR = Path("outputs/obs045_release_recovery_sweep")

REQUIRED_COLUMNS = [
    "node_id",
    "mds1",
    "mds2",
    "T_xx",
    "T_xy",
    "T_yx",
    "T_yy",
]

FAMILY_ORDER = [
    "seam_hugging",
    "release_directed",
    "short_trapped",
    "mixed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-045 controlled release recovery sweep.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Response-operator node CSV.")
    p.add_argument("--seam-nodes-csv", type=Path, default=DEFAULT_SEAM_NODES, help="Optional seam-bundle enrichment CSV.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Sweep output directory.")

    p.add_argument("--n-seeds", type=int, default=26, help="Number of seeds.")
    p.add_argument("--seed-mode", choices=["uniform", "seam_near", "high_response"], default="seam_near")
    p.add_argument("--seam-near-quantile", type=float, default=0.35)
    p.add_argument("--high-response-quantile", type=float, default=0.75)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--seam-contact-threshold", type=float, default=0.25)
    p.add_argument("--min-angular-consistency", type=float, default=0.50)
    p.add_argument("--quiver-stride", type=int, default=3)

    # Sweep axes: space-separated values
    p.add_argument("--support-radius-scales", nargs="+", type=float, default=[3.5, 4.0, 4.5, 5.0])
    p.add_argument("--knn-values", nargs="+", type=int, default=[8])
    p.add_argument("--step-size-scales", nargs="+", type=float, default=[0.15])

    # Family classification settings
    p.add_argument("--short-path-quantile", type=float, default=0.33)
    p.add_argument("--seam-hugging-quantile", type=float, default=0.50)
    p.add_argument("--release-length-quantile", type=float, default=0.50)
    p.add_argument("--seam-contact-required", action="store_true")

    return p.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "node_id" not in out.columns:
        if "id" in out.columns:
            out = out.rename(columns={"id": "node_id"})
        elif {"r", "alpha"}.issubset(out.columns):
            out["node_id"] = [f"r{float(r):.6f}_a{float(a):.6f}" for r, a in zip(out["r"], out["alpha"])]
        else:
            out["node_id"] = [f"node_{i:03d}" for i in range(len(out))]
    out["node_id"] = out["node_id"].astype(str)
    return out


def choose_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if {"r", "alpha"}.issubset(left.columns) and {"r", "alpha"}.issubset(right.columns):
        return ["r", "alpha"]
    if "node_id" in left.columns and "node_id" in right.columns:
        return ["node_id"]
    raise ValueError("Need either shared (r, alpha) or shared node_id to merge seam nodes.")


def load_nodes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_csv(path)
    df = normalize_node_ids(df)
    require_columns(df, REQUIRED_COLUMNS)
    return df


def maybe_merge_seam_nodes(response_df: pd.DataFrame, seam_nodes_csv: Path | None) -> tuple[pd.DataFrame, list[str]]:
    if seam_nodes_csv is None or not seam_nodes_csv.exists():
        return response_df, []

    seam_df = pd.read_csv(seam_nodes_csv)
    seam_df = normalize_node_ids(seam_df)
    merge_keys = choose_merge_keys(response_df, seam_df)

    seam_cols = [
        "distance_to_seam",
        "signed_phase",
        "lazarus_score",
        "response_strength",
        "node_holonomy_proxy",
        "local_direction_mismatch_deg",
        "neighbor_direction_mismatch_mean",
        "sym_traceless_norm",
        "scalar_norm",
        "antisymmetric_norm",
        "commutator_norm_rsp",
        "hotspot_class",
    ]
    seam_cols = [c for c in seam_cols if c in seam_df.columns]
    keep = merge_keys + seam_cols
    seam_df = seam_df[keep].copy().drop_duplicates(subset=merge_keys)

    overlap = [c for c in seam_cols if c in response_df.columns]
    if overlap:
        seam_df = seam_df.rename(columns={c: f"{c}_seam" for c in overlap})

    merged = response_df.merge(seam_df, on=merge_keys, how="left", validate="one_to_one")
    return merged, merge_keys


def compute_response_eigensystem(df: pd.DataFrame) -> pd.DataFrame:
    vals_1 = []
    vec1_x = []
    vec1_y = []
    eig_valid = []

    for _, row in df.iterrows():
        T = np.array(
            [[float(row["T_xx"]), float(row["T_xy"])], [float(row["T_yx"]), float(row["T_yy"])]],
            dtype=float,
        )
        evals, evecs = np.linalg.eig(T)
        if np.max(np.abs(np.imag(evals))) > 1e-8:
            vals_1.append(np.nan)
            vec1_x.append(np.nan)
            vec1_y.append(np.nan)
            eig_valid.append(False)
            continue

        evals = np.real(evals)
        evecs = np.real(evecs)
        order = np.argsort(-np.abs(evals))
        evals = evals[order]
        evecs = evecs[:, order]
        v1 = evecs[:, 0].astype(float)
        n1 = np.linalg.norm(v1)
        if n1 <= 0:
            vals_1.append(np.nan)
            vec1_x.append(np.nan)
            vec1_y.append(np.nan)
            eig_valid.append(False)
            continue
        v1 = v1 / n1
        vals_1.append(float(evals[0]))
        vec1_x.append(float(v1[0]))
        vec1_y.append(float(v1[1]))
        eig_valid.append(True)

    out = df.copy()
    out["eigval_1"] = vals_1
    out["eigvec1_x"] = vec1_x
    out["eigvec1_y"] = vec1_y
    out["dominant_response_strength"] = np.abs(pd.to_numeric(out["eigval_1"], errors="coerce"))
    out["eigensystem_valid"] = eig_valid
    return out


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def median_nearest_neighbor_distance(xy: np.ndarray) -> float:
    D = pairwise_distances(xy)
    np.fill_diagonal(D, np.inf)
    return float(np.median(np.min(D, axis=1)))


def choose_seed_indices(
    df: pd.DataFrame,
    n_seeds: int,
    seed_mode: str,
    seam_near_quantile: float,
    high_response_quantile: float,
) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.array([], dtype=int)

    candidate_idx = np.arange(n)
    seam_col = choose_optional_column(df, ["distance_to_seam", "distance_to_seam_seam"])

    if seed_mode == "seam_near" and seam_col is not None:
        thresh = float(pd.to_numeric(df[seam_col], errors="coerce").quantile(seam_near_quantile))
        candidate_idx = np.where(pd.to_numeric(df[seam_col], errors="coerce").to_numpy() <= thresh)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(n)
    elif seed_mode == "high_response":
        thresh = float(pd.to_numeric(df["dominant_response_strength"], errors="coerce").quantile(high_response_quantile))
        candidate_idx = np.where(pd.to_numeric(df["dominant_response_strength"], errors="coerce").to_numpy() >= thresh)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(n)

    if len(candidate_idx) <= n_seeds:
        return np.array(candidate_idx, dtype=int)

    chosen = np.unique(np.linspace(0, len(candidate_idx) - 1, n_seeds).round().astype(int))
    return np.array(candidate_idx[chosen], dtype=int)


def knn_indices(xy: np.ndarray, z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt(np.sum((xy - z[None, :]) ** 2, axis=1))
    order = np.argsort(d)
    idx = order[:k]
    return idx, d[idx]


def interpolate_vector(z: np.ndarray, xy: np.ndarray, vecs: np.ndarray, k: int, prev_dir: np.ndarray | None) -> dict:
    idx, dist = knn_indices(xy, z, k)
    local_vecs = vecs[idx].copy()
    local_dist = dist.copy()

    anchor = local_vecs[0].copy()
    if prev_dir is not None and float(np.dot(anchor, prev_dir)) < 0:
        anchor = -anchor

    aligned = []
    for v in local_vecs:
        vv = v.copy()
        if float(np.dot(vv, anchor)) < 0:
            vv = -vv
        aligned.append(vv)
    aligned = np.array(aligned, dtype=float)

    weights = 1.0 / np.maximum(local_dist, 1e-9)
    vbar = np.sum(aligned * weights[:, None], axis=0)
    norm = np.linalg.norm(vbar)
    if norm <= 0:
        return {
            "valid": False,
            "vec": np.array([np.nan, np.nan]),
            "support_radius": float(np.max(local_dist)),
            "angular_consistency": np.nan,
            "neighbor_count": len(idx),
        }

    vbar = vbar / norm
    if prev_dir is not None and float(np.dot(vbar, prev_dir)) < 0:
        vbar = -vbar

    cons = float(np.average(np.clip(aligned @ vbar, -1, 1), weights=weights))
    return {
        "valid": True,
        "vec": vbar,
        "support_radius": float(np.max(local_dist)),
        "angular_consistency": cons,
        "neighbor_count": len(idx),
    }


def interpolate_scalar(z: np.ndarray, xy: np.ndarray, values: np.ndarray, k: int) -> float:
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return np.nan
    xyv = xy[mask]
    vv = values[mask]
    idx, dist = knn_indices(xyv, z, min(k, len(xyv)))
    w = 1.0 / np.maximum(dist, 1e-9)
    return float(np.sum(vv[idx] * w) / np.sum(w))


def integrate_continuous_path(
    seed_idx: int,
    df: pd.DataFrame,
    xy: np.ndarray,
    vecs: np.ndarray,
    k: int,
    step_size: float,
    max_steps: int,
    support_radius_threshold: float,
    min_angular_consistency: float,
    seam_contact_threshold: float,
) -> tuple[list[dict], dict]:
    seam_col = choose_optional_column(df, ["distance_to_seam", "distance_to_seam_seam"])
    phase_col = choose_optional_column(df, ["signed_phase", "signed_phase_seam"])
    laz_col = choose_optional_column(df, ["lazarus_score", "lazarus_score_seam"])
    mismatch_col = choose_optional_column(df, ["neighbor_direction_mismatch_mean", "neighbor_direction_mismatch_mean_seam"])
    hol_col = choose_optional_column(df, ["node_holonomy_proxy", "node_holonomy_proxy_seam"])

    seed_row = df.iloc[seed_idx]
    z = np.array([float(seed_row["mds1"]), float(seed_row["mds2"])], dtype=float)
    prev_dir = None
    point_rows = []

    seam_vals = []
    phase_vals = []
    laz_vals = []
    mismatch_vals = []
    hol_vals = []
    path_length = 0.0
    termination_reason = "max_steps"

    seam_values = pd.to_numeric(df[seam_col], errors="coerce").to_numpy(dtype=float) if seam_col else np.full(len(df), np.nan)
    phase_values = pd.to_numeric(df[phase_col], errors="coerce").to_numpy(dtype=float) if phase_col else np.full(len(df), np.nan)
    laz_values = pd.to_numeric(df[laz_col], errors="coerce").to_numpy(dtype=float) if laz_col else np.full(len(df), np.nan)
    mismatch_values = pd.to_numeric(df[mismatch_col], errors="coerce").to_numpy(dtype=float) if mismatch_col else np.full(len(df), np.nan)
    hol_values = pd.to_numeric(df[hol_col], errors="coerce").to_numpy(dtype=float) if hol_col else np.full(len(df), np.nan)

    for step in range(max_steps + 1):
        interp = interpolate_vector(z, xy, vecs, k=k, prev_dir=prev_dir)

        if not interp["valid"]:
            termination_reason = "invalid_local_vector"
            break
        if interp["support_radius"] > support_radius_threshold:
            termination_reason = "support_radius_exceeded"
            break
        if interp["angular_consistency"] < min_angular_consistency:
            termination_reason = "low_angular_consistency"
            break

        seam_val = interpolate_scalar(z, xy, seam_values, k)
        phase_val = interpolate_scalar(z, xy, phase_values, k)
        laz_val = interpolate_scalar(z, xy, laz_values, k)
        mismatch_val = interpolate_scalar(z, xy, mismatch_values, k)
        hol_val = interpolate_scalar(z, xy, hol_values, k)

        seam_vals.append(seam_val)
        phase_vals.append(phase_val)
        laz_vals.append(laz_val)
        mismatch_vals.append(mismatch_val)
        hol_vals.append(hol_val)

        point_rows.append(
            {
                "trajectory_id": None,
                "step": step,
                "x": float(z[0]),
                "y": float(z[1]),
                "vec_x": float(interp["vec"][0]),
                "vec_y": float(interp["vec"][1]),
                "support_radius": interp["support_radius"],
                "angular_consistency": interp["angular_consistency"],
                "neighbor_count": interp["neighbor_count"],
                "distance_to_seam": seam_val,
                "signed_phase": phase_val,
                "lazarus_score": laz_val,
                "neighbor_direction_mismatch_mean": mismatch_val,
                "node_holonomy_proxy": hol_val,
            }
        )

        if step == max_steps:
            termination_reason = "max_steps"
            break

        z_next = z + step_size * interp["vec"]
        path_length += float(np.linalg.norm(z_next - z))
        prev_dir = interp["vec"]
        z = z_next

    phase_arr = np.array(phase_vals, dtype=float)
    seam_arr = np.array(seam_vals, dtype=float)
    summary = {
        "trajectory_id": None,
        "seed_node_id": str(seed_row["node_id"]),
        "seed_node_index": int(seed_idx),
        "n_steps": max(0, len(point_rows) - 1),
        "n_points": len(point_rows),
        "termination_reason": termination_reason,
        "path_length_mds": path_length,
        "mean_distance_to_seam": float(np.nanmean(seam_arr)) if len(seam_arr) else np.nan,
        "min_distance_to_seam": float(np.nanmin(seam_arr)) if len(seam_arr) else np.nan,
        "mean_signed_phase": float(np.nanmean(phase_arr)) if len(phase_arr) else np.nan,
        "phase_span": float(np.nanmax(phase_arr) - np.nanmin(phase_arr)) if len(phase_arr) else np.nan,
        "phase_sign_change": bool(np.nanmin(phase_arr) < 0 < np.nanmax(phase_arr)) if len(phase_arr) else False,
        "mean_lazarus_score": float(np.nanmean(laz_vals)) if len(laz_vals) else np.nan,
        "mean_neighbor_direction_mismatch_mean": float(np.nanmean(mismatch_vals)) if len(mismatch_vals) else np.nan,
        "mean_node_holonomy_proxy": float(np.nanmean(hol_vals)) if len(hol_vals) else np.nan,
        "mean_support_radius": float(np.nanmean([r["support_radius"] for r in point_rows])) if point_rows else np.nan,
        "min_angular_consistency": float(np.nanmin([r["angular_consistency"] for r in point_rows])) if point_rows else np.nan,
        "seam_contact": bool(np.nanmin(seam_arr) <= seam_contact_threshold) if len(seam_arr) else False,
    }
    return point_rows, summary


def classify_family_frame(
    df: pd.DataFrame,
    short_path_quantile: float,
    seam_hugging_quantile: float,
    release_length_quantile: float,
    seam_contact_required: bool,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    path_len = pd.to_numeric(out["path_length_mds"], errors="coerce")
    mean_seam = pd.to_numeric(out["mean_distance_to_seam"], errors="coerce")

    short_len_thresh = float(path_len.quantile(short_path_quantile))
    seam_hugging_thresh = float(mean_seam.quantile(seam_hugging_quantile))
    release_len_thresh = float(path_len.quantile(release_length_quantile))

    labels = []
    reasons = []
    cross_flags = []

    for _, row in out.iterrows():
        phase_cross = bool(row["phase_sign_change"])
        seam_contact = bool(row["seam_contact"])
        path_length = float(row["path_length_mds"])
        mean_seam_dist = float(row["mean_distance_to_seam"]) if pd.notna(row["mean_distance_to_seam"]) else np.nan
        n_steps = float(row["n_steps"])
        term = str(row["termination_reason"])

        cross_flags.append(phase_cross)

        if (path_length <= short_len_thresh) and (
            n_steps <= 2 or term in {"no_forward_neighbor", "low_angular_consistency", "support_radius_exceeded"}
        ):
            labels.append("short_trapped")
            reasons.append("short_path_and_early_termination")
            continue

        seam_ok = seam_contact if seam_contact_required else True
        if seam_ok and pd.notna(mean_seam_dist) and (mean_seam_dist <= seam_hugging_thresh):
            labels.append("seam_hugging")
            reasons.append("low_mean_seam_distance")
            continue

        seam_ok = seam_contact if seam_contact_required else True
        if seam_ok and (path_length >= release_len_thresh):
            labels.append("release_directed")
            reasons.append("longer_path_extent")
            continue

        labels.append("mixed")
        reasons.append("fallback_mixed")

    out["path_family"] = labels
    out["family_reason"] = reasons
    out["cross_phase_flag"] = cross_flags

    thresholds = {
        "short_path_length_threshold": short_len_thresh,
        "seam_hugging_mean_seam_threshold": seam_hugging_thresh,
        "release_length_threshold": release_len_thresh,
    }
    return out, thresholds


def build_family_summary(assign_df: pd.DataFrame, run_name: str, thresholds: dict) -> pd.DataFrame:
    rows = []
    total = len(assign_df)

    for fam in FAMILY_ORDER:
        sub = assign_df[assign_df["path_family"] == fam].copy()
        if len(sub) == 0:
            rows.append(
                {
                    "run_name": run_name,
                    "path_family": fam,
                    "n_paths": 0,
                    "path_share": 0.0,
                    "cross_phase_share": np.nan,
                    "mean_path_length_mds": np.nan,
                    "mean_n_steps": np.nan,
                    "seam_contact_share": np.nan,
                    "share_support_radius_exceeded": np.nan,
                    "share_low_angular_consistency": np.nan,
                    "mean_distance_to_seam": np.nan,
                    "primary_termination_reason": "",
                    **thresholds,
                }
            )
            continue

        rows.append(
            {
                "run_name": run_name,
                "path_family": fam,
                "n_paths": len(sub),
                "path_share": len(sub) / total if total else np.nan,
                "cross_phase_share": float(sub["cross_phase_flag"].astype(bool).mean()),
                "mean_path_length_mds": float(pd.to_numeric(sub["path_length_mds"], errors="coerce").mean()),
                "mean_n_steps": float(pd.to_numeric(sub["n_steps"], errors="coerce").mean()),
                "seam_contact_share": float(sub["seam_contact"].astype(bool).mean()),
                "share_support_radius_exceeded": float((sub["termination_reason"] == "support_radius_exceeded").mean()),
                "share_low_angular_consistency": float((sub["termination_reason"] == "low_angular_consistency").mean()),
                "mean_distance_to_seam": float(pd.to_numeric(sub["mean_distance_to_seam"], errors="coerce").mean()),
                "primary_termination_reason": str(sub["termination_reason"].mode(dropna=False).iloc[0]),
                **thresholds,
            }
        )

    rows.append(
        {
            "run_name": run_name,
            "path_family": "__overall__",
            "n_paths": total,
            "path_share": 1.0 if total else np.nan,
            "cross_phase_share": float(assign_df["cross_phase_flag"].astype(bool).mean()) if total else np.nan,
            "mean_path_length_mds": float(pd.to_numeric(assign_df["path_length_mds"], errors="coerce").mean()) if total else np.nan,
            "mean_n_steps": float(pd.to_numeric(assign_df["n_steps"], errors="coerce").mean()) if total else np.nan,
            "seam_contact_share": float(assign_df["seam_contact"].astype(bool).mean()) if total else np.nan,
            "share_support_radius_exceeded": float((assign_df["termination_reason"] == "support_radius_exceeded").mean()) if total else np.nan,
            "share_low_angular_consistency": float((assign_df["termination_reason"] == "low_angular_consistency").mean()) if total else np.nan,
            "mean_distance_to_seam": float(pd.to_numeric(assign_df["mean_distance_to_seam"], errors="coerce").mean()) if total else np.nan,
            "primary_termination_reason": "",
            **thresholds,
        }
    )

    out = pd.DataFrame(rows)
    fam_order = FAMILY_ORDER + ["__overall__"]
    out["path_family"] = pd.Categorical(out["path_family"], categories=fam_order, ordered=True)
    return out.sort_values("path_family").reset_index(drop=True)


def plot_paths(points_df: pd.DataFrame, nodes_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    seam_col = choose_optional_column(nodes_df, ["distance_to_seam", "distance_to_seam_seam"])
    if seam_col is not None:
        sc = ax.scatter(
            nodes_df["mds1"],
            nodes_df["mds2"],
            c=pd.to_numeric(nodes_df[seam_col], errors="coerce"),
            s=18,
            alpha=0.45,
        )
        fig.colorbar(sc, ax=ax, label=seam_col)
    else:
        ax.scatter(nodes_df["mds1"], nodes_df["mds2"], s=18, alpha=0.25)

    for _, sub in points_df.groupby("trajectory_id", sort=False):
        ax.plot(sub["x"], sub["y"], linewidth=1.1, alpha=0.9)

    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    ax.set_title("OBS-045 continuous release-recovery sweep")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_quiver(nodes_df: pd.DataFrame, outpath: Path, stride: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    bg_col = choose_optional_column(nodes_df, ["signed_phase", "signed_phase_seam"])
    if bg_col is not None:
        sc = ax.scatter(
            nodes_df["mds1"],
            nodes_df["mds2"],
            c=pd.to_numeric(nodes_df[bg_col], errors="coerce"),
            s=16,
            alpha=0.45,
        )
        fig.colorbar(sc, ax=ax, label=bg_col)
    else:
        ax.scatter(nodes_df["mds1"], nodes_df["mds2"], s=16, alpha=0.25)

    qdf = nodes_df.iloc[::max(1, stride)].copy()
    ax.quiver(
        qdf["mds1"],
        qdf["mds2"],
        qdf["eigvec1_x"],
        qdf["eigvec1_y"],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        alpha=0.9,
    )
    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    ax.set_title("OBS-045 response direction anchors")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_config_label(support_radius_scale: float, knn: int, step_size_scale: float) -> str:
    s = str(support_radius_scale).replace(".", "p")
    k = str(knn)
    t = str(step_size_scale).replace(".", "p")
    return f"sr{s}_k{k}_ss{t}"


def plot_metric_over_configs(df: pd.DataFrame, metric: str, ylabel: str, outpath: Path) -> None:
    plot_df = df.copy().sort_values(["support_radius_scale", "knn", "step_size_scale"])
    labels = plot_df["config_label"].tolist()
    vals = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(vals)), vals, marker="o")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_summary_text(summary_df: pd.DataFrame, outpath: Path) -> None:
    best_phase = summary_df.sort_values("phase_sign_change_share", ascending=False).iloc[0]
    best_release_cross = summary_df.sort_values("release_directed_cross_phase_share", ascending=False).iloc[0]

    text = f"""OBS-045 release recovery sweep summary

Configurations tested: {len(summary_df)}

Best overall phase crossing:
- config_label: {best_phase['config_label']}
- support_radius_scale: {best_phase['support_radius_scale']}
- knn: {best_phase['knn']}
- step_size_scale: {best_phase['step_size_scale']}
- phase_sign_change_share: {best_phase['phase_sign_change_share']:.6f}
- seam_contact_share: {best_phase['seam_contact_share']:.6f}
- release_directed_cross_phase_share: {best_phase['release_directed_cross_phase_share']:.6f}

Best release-directed cross-phase recovery:
- config_label: {best_release_cross['config_label']}
- support_radius_scale: {best_release_cross['support_radius_scale']}
- knn: {best_release_cross['knn']}
- step_size_scale: {best_release_cross['step_size_scale']}
- release_directed_cross_phase_share: {best_release_cross['release_directed_cross_phase_share']:.6f}
- phase_sign_change_share: {best_release_cross['phase_sign_change_share']:.6f}
- seam_contact_share: {best_release_cross['seam_contact_share']:.6f}
"""
    outpath.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_nodes = load_nodes(args.input)
    raw_nodes, merge_keys = maybe_merge_seam_nodes(raw_nodes, args.seam_nodes_csv)
    nodes = compute_response_eigensystem(raw_nodes)
    nodes = nodes.replace([np.inf, -np.inf], np.nan)
    nodes = nodes.dropna(subset=["mds1", "mds2", "eigvec1_x", "eigvec1_y", "eigval_1"]).reset_index(drop=True)

    if len(nodes) == 0:
        raise ValueError("No valid nodes remain after eigensystem filtering.")

    xy = nodes[["mds1", "mds2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    vecs = nodes[["eigvec1_x", "eigvec1_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    median_nn = median_nearest_neighbor_distance(xy)

    seed_idx = choose_seed_indices(
        nodes,
        n_seeds=args.n_seeds,
        seed_mode=args.seed_mode,
        seam_near_quantile=args.seam_near_quantile,
        high_response_quantile=args.high_response_quantile,
    )

    sweep_rows = []
    family_frames = []

    configs = list(itertools.product(args.support_radius_scales, args.knn_values, args.step_size_scales))

    for support_radius_scale, knn, step_size_scale in configs:
        config_label = make_config_label(support_radius_scale, knn, step_size_scale)
        config_dir = args.outdir / config_label
        config_dir.mkdir(parents=True, exist_ok=True)

        step_size = step_size_scale * median_nn
        support_radius_threshold = support_radius_scale * median_nn

        point_rows = []
        traj_rows = []

        for i, sidx in enumerate(seed_idx, start=1):
            tid = f"cflow_{i:03d}"
            pts, summ = integrate_continuous_path(
                seed_idx=int(sidx),
                df=nodes,
                xy=xy,
                vecs=vecs,
                k=min(knn, len(nodes)),
                step_size=step_size,
                max_steps=args.max_steps,
                support_radius_threshold=support_radius_threshold,
                min_angular_consistency=args.min_angular_consistency,
                seam_contact_threshold=args.seam_contact_threshold,
            )
            for r in pts:
                r["trajectory_id"] = tid
                point_rows.append(r)
            summ["trajectory_id"] = tid
            traj_rows.append(summ)

        points_df = pd.DataFrame(point_rows)
        traj_df = pd.DataFrame(traj_rows)

        summary_df = pd.DataFrame(
            [
                {
                    "config_label": config_label,
                    "support_radius_scale": support_radius_scale,
                    "knn": min(knn, len(nodes)),
                    "step_size_scale": step_size_scale,
                    "n_nodes_used": len(nodes),
                    "n_trajectories": len(traj_df),
                    "seed_mode": args.seed_mode,
                    "median_nearest_neighbor_distance": median_nn,
                    "step_size": step_size,
                    "support_radius_threshold": support_radius_threshold,
                    "mean_path_length_mds": float(pd.to_numeric(traj_df["path_length_mds"], errors="coerce").mean()) if len(traj_df) else np.nan,
                    "mean_n_steps": float(pd.to_numeric(traj_df["n_steps"], errors="coerce").mean()) if len(traj_df) else np.nan,
                    "seam_contact_share": float(traj_df["seam_contact"].astype(bool).mean()) if len(traj_df) else np.nan,
                    "phase_sign_change_share": float(traj_df["phase_sign_change"].astype(bool).mean()) if len(traj_df) else np.nan,
                    "mean_distance_to_seam": float(pd.to_numeric(traj_df["mean_distance_to_seam"], errors="coerce").mean()) if len(traj_df) else np.nan,
                    "share_support_radius_exceeded": float((traj_df["termination_reason"] == "support_radius_exceeded").mean()) if len(traj_df) else np.nan,
                    "share_low_angular_consistency": float((traj_df["termination_reason"] == "low_angular_consistency").mean()) if len(traj_df) else np.nan,
                    "share_max_steps": float((traj_df["termination_reason"] == "max_steps").mean()) if len(traj_df) else np.nan,
                    "seam_nodes_csv": str(args.seam_nodes_csv) if args.seam_nodes_csv else "",
                    "merge_keys_used": ",".join(merge_keys),
                }
            ]
        )

        # Family classification
        assign_df, thresholds = classify_family_frame(
            traj_df,
            short_path_quantile=args.short_path_quantile,
            seam_hugging_quantile=args.seam_hugging_quantile,
            release_length_quantile=args.release_length_quantile,
            seam_contact_required=args.seam_contact_required,
        )
        assign_df["run_name"] = config_label
        family_summary_df = build_family_summary(assign_df, config_label, thresholds)

        # extract family metrics for sweep row
        fam_map = {row["path_family"]: row for row in family_summary_df.to_dict(orient="records")}
        sweep_row = summary_df.iloc[0].to_dict()
        for fam in ["seam_hugging", "release_directed", "short_trapped", "mixed"]:
            fr = fam_map.get(fam, {})
            sweep_row[f"{fam}_share"] = fr.get("path_share", np.nan)
            sweep_row[f"{fam}_cross_phase_share"] = fr.get("cross_phase_share", np.nan)
        sweep_rows.append(sweep_row)

        family_summary_df["config_label"] = config_label
        family_frames.append(family_summary_df)

        support_region_df = points_df[
            ["trajectory_id", "step", "x", "y", "support_radius", "angular_consistency", "neighbor_count"]
        ].copy()

        # write per-config artifacts
        traj_df.to_csv(config_dir / "continuous_flow_trajectories.csv", index=False)
        points_df.to_csv(config_dir / "continuous_flow_trajectory_points.csv", index=False)
        summary_df.to_csv(config_dir / "continuous_flow_summary.csv", index=False)
        support_region_df.to_csv(config_dir / "continuous_flow_support_region.csv", index=False)
        assign_df.to_csv(config_dir / "continuous_flow_path_family_assignments.csv", index=False)
        family_summary_df.to_csv(config_dir / "continuous_flow_path_family_summary.csv", index=False)
        plot_paths(points_df, nodes, config_dir / "continuous_flow_paths.png")
        plot_quiver(nodes, config_dir / "continuous_flow_quiver.png", args.quiver_stride)

        print(f"Wrote configuration: {config_dir}")

    sweep_df = pd.DataFrame(sweep_rows).sort_values(["support_radius_scale", "knn", "step_size_scale"]).reset_index(drop=True)
    family_comparison_df = pd.concat(family_frames, ignore_index=True)

    # Primary summary artifacts
    sweep_path = args.outdir / "release_recovery_sweep_summary.csv"
    family_path = args.outdir / "release_recovery_family_comparison.csv"
    txt_path = args.outdir / "release_recovery_summary.txt"

    sweep_df.to_csv(sweep_path, index=False)
    family_comparison_df.to_csv(family_path, index=False)
    write_summary_text(sweep_df, txt_path)

    # Figures
    plot_metric_over_configs(sweep_df, "phase_sign_change_share", "phase crossing share", args.outdir / "release_recovery_phase_crossing_figure.png")
    plot_metric_over_configs(sweep_df, "seam_contact_share", "seam contact share", args.outdir / "release_recovery_seam_contact_figure.png")
    plot_metric_over_configs(
        sweep_df,
        "release_directed_cross_phase_share",
        "release-directed cross-phase share",
        args.outdir / "release_recovery_release_cross_phase_figure.png",
    )

    print(f"Wrote: {sweep_path}")
    print(f"Wrote: {family_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
