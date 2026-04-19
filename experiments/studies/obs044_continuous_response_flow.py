#!/usr/bin/env python3
"""
obs044_continuous_response_flow.py

OBS-044 — Continuous response-flow reconstruction.

Purpose
-------
Lift the discrete response-guided flow from OBS-043 into a conservative
semi-continuous embedded reconstruction using local kNN vector interpolation.

This study is intentionally conservative:
- embedded-space only
- local interpolation only
- trust-region constrained
- no PDE or physical-flow claims
- no scalar shaping in the baseline reconstruction

Primary question
----------------
Does the response-guided flow established in OBS-043 persist under local
continuous reconstruction, reducing graph-sparsity termination while preserving
seam-engaged routing structure?

Inputs
------
Primary node field:
    outputs/fim_response_operator/response_operator_nodes.csv

Recommended enrichment:
    outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

Outputs
-------
Directory:
    outputs/obs044_continuous_response_flow/

Files:
    continuous_flow_trajectories.csv
    continuous_flow_trajectory_points.csv
    continuous_flow_summary.csv
    continuous_flow_support_region.csv
    continuous_flow_paths.png
    continuous_flow_quiver.png

Notes
-----
- Reconstruction uses kNN inverse-distance weighted vector averaging
- Local vector sign is aligned to nearest-anchor / previous-step continuity
- Integration stops when support radius or angular consistency become poor
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/fim_response_operator/response_operator_nodes.csv")
DEFAULT_SEAM_NODES = Path("outputs/obs028c_canonical_seam_bundle/seam_nodes.csv")
DEFAULT_OUTDIR = Path("outputs/obs044_continuous_response_flow")

REQUIRED_COLUMNS = [
    "node_id",
    "mds1",
    "mds2",
    "T_xx",
    "T_xy",
    "T_yx",
    "T_yy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-044 continuous response-flow reconstruction.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Response-operator node CSV.")
    p.add_argument(
        "--seam-nodes-csv",
        type=Path,
        default=DEFAULT_SEAM_NODES,
        help="Optional seam-bundle enrichment CSV.",
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument("--n-seeds", type=int, default=26, help="Number of seed nodes.")
    p.add_argument(
        "--seed-mode",
        choices=["uniform", "seam_near", "high_response"],
        default="seam_near",
        help="Seed selection strategy.",
    )
    p.add_argument("--knn", type=int, default=8, help="k nearest neighbors for local interpolation.")
    p.add_argument(
        "--step-size-scale",
        type=float,
        default=0.15,
        help="Step size as a fraction of median nearest-neighbor distance.",
    )
    p.add_argument("--max-steps", type=int, default=80, help="Maximum continuous integration steps.")
    p.add_argument(
        "--support-radius-scale",
        type=float,
        default=2.5,
        help="Terminate when nearest-support radius exceeds this multiple of median NN distance.",
    )
    p.add_argument(
        "--min-angular-consistency",
        type=float,
        default=0.50,
        help="Terminate when local vector consistency falls below this cosine-like score.",
    )
    p.add_argument(
        "--seam-near-quantile",
        type=float,
        default=0.35,
        help="Quantile threshold for seam-near seed selection.",
    )
    p.add_argument(
        "--high-response-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold for high-response seed selection.",
    )
    p.add_argument(
        "--seam-contact-threshold",
        type=float,
        default=0.25,
        help="Threshold for marking seam contact.",
    )
    p.add_argument(
        "--quiver-stride",
        type=int,
        default=3,
        help="Subsampling stride for quiver plot.",
    )
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
            out["node_id"] = [
                f"r{float(r):.6f}_a{float(a):.6f}"
                for r, a in zip(out["r"], out["alpha"])
            ]
        else:
            out["node_id"] = [f"node_{i:03d}" for i in range(len(out))]
    out["node_id"] = out["node_id"].astype(str)
    return out


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "node_id" in left.columns and "node_id" in right.columns:
        # only use node_id if it clearly matches; for these artifacts r,alpha is safer
        pass
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
            [
                [float(row["T_xx"]), float(row["T_xy"])],
                [float(row["T_yx"]), float(row["T_yy"])],
            ],
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


def choose_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def interpolate_vector(
    z: np.ndarray,
    xy: np.ndarray,
    vecs: np.ndarray,
    k: int,
    prev_dir: np.ndarray | None,
) -> dict:
    idx, dist = knn_indices(xy, z, k)
    local_vecs = vecs[idx].copy()
    local_dist = dist.copy()

    # anchor on nearest vector
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

    # consistency: weighted mean alignment of aligned vecs to averaged vec
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
    d = dist
    w = 1.0 / np.maximum(d, 1e-9)
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
        ax.plot(sub["x"], sub["y"], linewidth=1.2, alpha=0.9)
        seed = sub.iloc[0]
        ax.scatter([seed["x"]], [seed["y"]], marker="x", s=36)

    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    ax.set_title("OBS-044 continuous response-flow paths")
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
    ax.set_title("OBS-044 response direction anchors")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


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
    step_size = args.step_size_scale * median_nn
    support_radius_threshold = args.support_radius_scale * median_nn

    seed_idx = choose_seed_indices(
        nodes,
        n_seeds=args.n_seeds,
        seed_mode=args.seed_mode,
        seam_near_quantile=args.seam_near_quantile,
        high_response_quantile=args.high_response_quantile,
    )

    point_rows = []
    traj_rows = []

    for i, sidx in enumerate(seed_idx, start=1):
        tid = f"cflow_{i:03d}"
        pts, summ = integrate_continuous_path(
            seed_idx=int(sidx),
            df=nodes,
            xy=xy,
            vecs=vecs,
            k=min(args.knn, len(nodes)),
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
                "n_nodes_used": len(nodes),
                "n_trajectories": len(traj_df),
                "seed_mode": args.seed_mode,
                "knn": min(args.knn, len(nodes)),
                "median_nearest_neighbor_distance": median_nn,
                "step_size": step_size,
                "support_radius_threshold": support_radius_threshold,
                "mean_path_length_mds": float(pd.to_numeric(traj_df["path_length_mds"], errors="coerce").mean()) if len(traj_df) else np.nan,
                "mean_n_steps": float(pd.to_numeric(traj_df["n_steps"], errors="coerce").mean()) if len(traj_df) else np.nan,
                "seam_contact_share": float(traj_df["seam_contact"].astype(bool).mean()) if len(traj_df) else np.nan,
                "phase_sign_change_share": float(traj_df["phase_sign_change"].astype(bool).mean()) if len(traj_df) else np.nan,
                "share_support_radius_exceeded": float((traj_df["termination_reason"] == "support_radius_exceeded").mean()) if len(traj_df) else np.nan,
                "share_low_angular_consistency": float((traj_df["termination_reason"] == "low_angular_consistency").mean()) if len(traj_df) else np.nan,
                "share_max_steps": float((traj_df["termination_reason"] == "max_steps").mean()) if len(traj_df) else np.nan,
                "seam_nodes_csv": str(args.seam_nodes_csv) if args.seam_nodes_csv else "",
                "merge_keys_used": ",".join(merge_keys),
            }
        ]
    )

    support_region_df = points_df[
        ["trajectory_id", "step", "x", "y", "support_radius", "angular_consistency", "neighbor_count"]
    ].copy()

    traj_path = args.outdir / "continuous_flow_trajectories.csv"
    pts_path = args.outdir / "continuous_flow_trajectory_points.csv"
    sum_path = args.outdir / "continuous_flow_summary.csv"
    sup_path = args.outdir / "continuous_flow_support_region.csv"
    paths_fig = args.outdir / "continuous_flow_paths.png"
    quiver_fig = args.outdir / "continuous_flow_quiver.png"

    traj_df.to_csv(traj_path, index=False)
    points_df.to_csv(pts_path, index=False)
    summary_df.to_csv(sum_path, index=False)
    support_region_df.to_csv(sup_path, index=False)

    plot_paths(points_df, nodes, paths_fig)
    plot_quiver(nodes, quiver_fig, args.quiver_stride)

    print(f"Wrote: {traj_path}")
    print(f"Wrote: {pts_path}")
    print(f"Wrote: {sum_path}")
    print(f"Wrote: {sup_path}")
    print(f"Wrote: {paths_fig}")
    print(f"Wrote: {quiver_fig}")


if __name__ == "__main__":
    main()
