#!/usr/bin/env python3
"""
fim_response_flow.py

First-pass response-flow study for the PAM Observatory.

Purpose
-------
Construct a dynamical flow picture on the manifold by treating the dominant
eigenvector of the local 2x2 response tensor as a preferred direction field,
then integrating short piecewise trajectories through embedded node space.

This study is intentionally conservative:
- it does not claim physical Bohmian mechanics
- it treats the response eigenvector field as a mathematical direction field
- it works directly from the response-operator node table
- it can optionally enrich nodes from the canonical seam bundle
- it exports paths, diagnostics, and inspection figures

Primary question
----------------
Does the response eigenvector field induce coherent preferred trajectories on
the manifold, and if so:
- do they align with seam corridors?
- do they avoid high-barrier / high-mismatch regions?
- do they exhibit structured circulation or trapping?

Expected input
--------------
By default this script reads:

    outputs/fim_response_operator/response_operator_nodes.csv

Optional enrichment source
--------------------------
Recommended enrichment source:

    outputs/obs028c_canonical_seam_bundle/seam_nodes.csv

This canonical seam-bundle node table provides:
- distance_to_seam
- lazarus_score
- response_strength
- node_holonomy_proxy
- local_direction_mismatch_deg
- neighbor_direction_mismatch_mean
- decomposition context fields
- hotspot labels

Merge priority:
1. node_id
2. fallback r, alpha

Outputs
-------
Writes:

    outputs/fim_response_flow/response_flow_paths.csv
    outputs/fim_response_flow/response_flow_path_nodes.csv
    outputs/fim_response_flow/response_flow_summary.csv
    outputs/fim_response_flow/response_flow_paths.png
    outputs/fim_response_flow/response_flow_quiver.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/fim_response_operator/response_operator_nodes.csv")
DEFAULT_OUTDIR = Path("outputs/fim_response_flow")
DEFAULT_SEAM_NODES = Path("outputs/obs028c_canonical_seam_bundle/seam_nodes.csv")

REQUIRED_COLUMNS = [
    "node_id",
    "mds1",
    "mds2",
    "T_xx",
    "T_xy",
    "T_yx",
    "T_yy",
]

SUPPORTED_PENALTY_COLS = [
    "local_direction_mismatch_deg",
    "neighbor_direction_mismatch_mean",
    "node_holonomy_proxy",
    "lazarus_score",
    "distance_to_seam",
    "response_strength",
    "sym_traceless_norm",
    "scalar_norm",
    "antisymmetric_norm",
    "commutator_norm_rsp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate response-eigenvector flow on the manifold.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input response-operator node CSV.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument("--n-seeds", type=int, default=60, help="Number of seed nodes to launch from.")
    p.add_argument("--max-steps", type=int, default=25, help="Maximum node-hops per path.")
    p.add_argument(
        "--step-radius-scale",
        type=float,
        default=1.75,
        help="Multiplier on median nearest-neighbor distance for admissible hop radius.",
    )
    p.add_argument(
        "--min-forward-cos",
        type=float,
        default=0.15,
        help="Minimum cosine alignment between local direction and candidate hop.",
    )
    p.add_argument(
        "--seed-mode",
        choices=["uniform", "seam_near", "high_response"],
        default="uniform",
        help="How to choose seed nodes.",
    )
    p.add_argument(
        "--seam-near-quantile",
        type=float,
        default=0.35,
        help="When seam distance exists, seam-near seed quantile threshold.",
    )
    p.add_argument(
        "--high-response-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold on dominant eigenvalue magnitude for high_response seed mode.",
    )
    p.add_argument(
        "--allow-revisit",
        action="store_true",
        help="Allow revisiting nodes during path integration.",
    )
    p.add_argument(
        "--penalty-col",
        type=str,
        default="",
        help=(
            "Optional flow-shaping scalar column. Recommended: "
            "local_direction_mismatch_deg, neighbor_direction_mismatch_mean, "
            "node_holonomy_proxy, lazarus_score."
        ),
    )
    p.add_argument(
        "--penalty-weight",
        type=float,
        default=0.0,
        help="Penalty weight applied to the selected penalty column when scoring candidate steps.",
    )
    p.add_argument(
        "--seam-contact-threshold",
        type=float,
        default=0.25,
        help="Threshold used to mark seam contact in path summaries.",
    )
    p.add_argument(
        "--quiver-stride",
        type=int,
        default=3,
        help="Subsampling stride for quiver plot readability.",
    )
    p.add_argument(
        "--seam-nodes-csv",
        type=Path,
        default=None,
        help=(
            "Optional canonical seam-bundle node CSV for enrichment. "
            f"Recommended: {DEFAULT_SEAM_NODES}"
        ),
    )
    p.add_argument(
        "--merge-on-cols",
        nargs="+",
        default=["node_id", "r", "alpha"],
        help="Preferred merge keys in order of use.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def choose_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def load_nodes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_csv(path)
    df = normalize_node_ids(df)
    require_columns(df, REQUIRED_COLUMNS)
    return df.copy()


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame, preferred_keys: list[str]) -> list[str]:
    if "node_id" in preferred_keys and "node_id" in left.columns and "node_id" in right.columns:
        return ["node_id"]
    if {"r", "alpha"}.issubset(preferred_keys) and {"r", "alpha"}.issubset(left.columns) and {"r", "alpha"}.issubset(right.columns):
        return ["r", "alpha"]
    raise ValueError(
        "Could not determine merge keys. Need either node_id in both tables or r, alpha in both tables."
    )


def maybe_merge_seam_nodes(
    response_df: pd.DataFrame,
    seam_nodes_csv: Path | None,
    preferred_merge_keys: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if seam_nodes_csv is None:
        return response_df, [], []

    if not seam_nodes_csv.exists():
        raise FileNotFoundError(f"Seam nodes CSV not found: {seam_nodes_csv}")

    seam_df = pd.read_csv(seam_nodes_csv)
    seam_df = normalize_node_ids(seam_df)

    merge_keys = choose_merge_keys(response_df, seam_df, preferred_merge_keys)

    preferred_cols = [
        "distance_to_seam",
        "lazarus_score",
        "response_strength",
        "node_holonomy_proxy",
        "local_direction_mismatch_deg",
        "neighbor_direction_mismatch_mean",
        "sym_traceless_norm",
        "scalar_norm",
        "antisymmetric_norm",
        "commutator_norm_rsp",
        "anisotropy_hotspot",
        "relational_hotspot",
        "shared_hotspot",
        "seam_band",
        "hotspot_class",
    ]
    present_cols = [c for c in preferred_cols if c in seam_df.columns]

    keep_cols = merge_keys + present_cols
    seam_df = seam_df[keep_cols].copy().drop_duplicates(subset=merge_keys)

    overlap_nonkeys = [c for c in present_cols if c in response_df.columns]
    if overlap_nonkeys:
        seam_df = seam_df.rename(columns={c: f"{c}_seam" for c in overlap_nonkeys})
        present_cols = [f"{c}_seam" if c in overlap_nonkeys else c for c in present_cols]

    merged = response_df.merge(
        seam_df,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    return merged, merge_keys, present_cols


def compute_response_eigensystem(df: pd.DataFrame) -> pd.DataFrame:
    vals_1 = []
    vals_2 = []
    vec1_x = []
    vec1_y = []
    vec2_x = []
    vec2_y = []
    dom_mag = []
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
            vals_2.append(np.nan)
            vec1_x.append(np.nan)
            vec1_y.append(np.nan)
            vec2_x.append(np.nan)
            vec2_y.append(np.nan)
            dom_mag.append(np.nan)
            eig_valid.append(False)
            continue

        evals = np.real(evals)
        evecs = np.real(evecs)

        order = np.argsort(-np.abs(evals))
        evals = evals[order]
        evecs = evecs[:, order]

        v1 = evecs[:, 0].astype(float)
        v2 = evecs[:, 1].astype(float)

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 0 or n2 <= 0:
            vals_1.append(np.nan)
            vals_2.append(np.nan)
            vec1_x.append(np.nan)
            vec1_y.append(np.nan)
            vec2_x.append(np.nan)
            vec2_y.append(np.nan)
            dom_mag.append(np.nan)
            eig_valid.append(False)
            continue

        v1 = v1 / n1
        v2 = v2 / n2

        vals_1.append(float(evals[0]))
        vals_2.append(float(evals[1]))
        vec1_x.append(float(v1[0]))
        vec1_y.append(float(v1[1]))
        vec2_x.append(float(v2[0]))
        vec2_y.append(float(v2[1]))
        dom_mag.append(float(abs(evals[0])))
        eig_valid.append(True)

    out = df.copy()
    out["eigval_1"] = vals_1
    out["eigval_2"] = vals_2
    out["eigvec1_x"] = vec1_x
    out["eigvec1_y"] = vec1_y
    out["eigvec2_x"] = vec2_x
    out["eigvec2_y"] = vec2_y
    out["dominant_response_strength"] = dom_mag
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
        thresh = float(df[seam_col].quantile(seam_near_quantile))
        candidate_idx = np.where(pd.to_numeric(df[seam_col], errors="coerce").to_numpy() <= thresh)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(n)

    elif seed_mode == "high_response":
        thresh = float(df["dominant_response_strength"].quantile(high_response_quantile))
        candidate_idx = np.where(
            pd.to_numeric(df["dominant_response_strength"], errors="coerce").to_numpy() >= thresh
        )[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(n)

    if len(candidate_idx) <= n_seeds:
        return np.array(candidate_idx, dtype=int)

    chosen = np.unique(np.linspace(0, len(candidate_idx) - 1, n_seeds).round().astype(int))
    return np.array(candidate_idx[chosen], dtype=int)


def orient_vector(current_vec: np.ndarray, prev_vec: np.ndarray | None) -> np.ndarray:
    if prev_vec is None:
        return current_vec
    if float(np.dot(current_vec, prev_vec)) < 0:
        return -current_vec
    return current_vec


def first_present_numeric(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                continue
    return None


def resolve_penalty_column(df: pd.DataFrame, requested: str) -> str | None:
    if requested:
        if requested in df.columns:
            return requested
        if f"{requested}_seam" in df.columns:
            return f"{requested}_seam"
        raise ValueError(f"Requested penalty column not found: {requested}")

    return None


def integrate_path(
    seed_idx: int,
    df: pd.DataFrame,
    xy: np.ndarray,
    D: np.ndarray,
    max_steps: int,
    admissible_radius: float,
    min_forward_cos: float,
    allow_revisit: bool,
    penalty_col: str | None,
    penalty_weight: float,
    seam_contact_threshold: float,
) -> tuple[list[dict], dict]:
    visited: set[int] = set()
    current = int(seed_idx)
    prev_dir: np.ndarray | None = None
    path_rows: list[dict] = []

    path_distance = 0.0
    seam_vals: list[float] = []
    phase_vals: list[float] = []
    penalty_vals: list[float] = []
    response_vals: list[float] = []
    lazarus_vals: list[float] = []

    termination_reason = "max_steps"

    seam_col = choose_optional_column(df, ["distance_to_seam", "distance_to_seam_seam"])
    phase_col = choose_optional_column(df, ["signed_phase", "signed_phase_seam"])
    lazarus_col = choose_optional_column(df, ["lazarus_score", "lazarus_score_seam"])

    for step in range(max_steps + 1):
        row = df.iloc[current]
        local_dir = np.array([float(row["eigvec1_x"]), float(row["eigvec1_y"])], dtype=float)
        local_dir = orient_vector(local_dir, prev_dir)

        seam_val = first_present_numeric(row, [seam_col] if seam_col else [])
        phase_val = first_present_numeric(row, [phase_col] if phase_col else [])
        penalty_val = first_present_numeric(row, [penalty_col] if penalty_col else [])
        response_val = first_present_numeric(row, ["dominant_response_strength"])
        lazarus_val = first_present_numeric(row, [lazarus_col] if lazarus_col else [])

        if seam_val is not None:
            seam_vals.append(seam_val)
        if phase_val is not None:
            phase_vals.append(phase_val)
        if penalty_val is not None:
            penalty_vals.append(penalty_val)
        if response_val is not None:
            response_vals.append(response_val)
        if lazarus_val is not None:
            lazarus_vals.append(lazarus_val)

        path_rows.append(
            {
                "path_id": None,
                "step": step,
                "node_id": row["node_id"],
                "node_index": current,
                "mds1": float(row["mds1"]),
                "mds2": float(row["mds2"]),
                "eigvec1_x": float(local_dir[0]),
                "eigvec1_y": float(local_dir[1]),
                "eigval_1": float(row["eigval_1"]),
                "eigval_2": float(row["eigval_2"]),
                "dominant_response_strength": response_val,
                "distance_to_seam": seam_val,
                "signed_phase": phase_val,
                "penalty_scalar": penalty_val,
                "lazarus_score": lazarus_val,
            }
        )

        if step == max_steps:
            termination_reason = "max_steps"
            break

        if not allow_revisit:
            visited.add(current)

        candidates = np.where((D[current] > 0) & (D[current] <= admissible_radius))[0]
        best_next = None
        best_score = -np.inf

        for j in candidates:
            if (not allow_revisit) and (int(j) in visited):
                continue

            disp = xy[j] - xy[current]
            norm = np.linalg.norm(disp)
            if norm <= 0:
                continue

            disp_unit = disp / norm
            cos_align = float(np.dot(local_dir, disp_unit))
            if cos_align < min_forward_cos:
                continue

            scalar_penalty = 0.0
            if penalty_col is not None and pd.notna(df.iloc[j][penalty_col]):
                scalar_penalty = penalty_weight * float(df.iloc[j][penalty_col])

            score = cos_align - 0.05 * (norm / max(admissible_radius, 1e-12)) - scalar_penalty
            if score > best_score:
                best_score = score
                best_next = int(j)

        if best_next is None:
            termination_reason = "no_forward_neighbor"
            break

        path_distance += float(np.linalg.norm(xy[best_next] - xy[current]))
        prev_dir = local_dir
        current = best_next

    path_summary = {
        "path_id": None,
        "seed_node_id": df.iloc[seed_idx]["node_id"],
        "seed_node_index": int(seed_idx),
        "n_steps": len(path_rows) - 1,
        "n_nodes": len(path_rows),
        "termination_reason": termination_reason,
        "path_length_mds": path_distance,
        "mean_distance_to_seam": float(np.mean(seam_vals)) if seam_vals else np.nan,
        "min_distance_to_seam": float(np.min(seam_vals)) if seam_vals else np.nan,
        "mean_signed_phase": float(np.mean(phase_vals)) if phase_vals else np.nan,
        "phase_span": (float(np.max(phase_vals)) - float(np.min(phase_vals))) if phase_vals else np.nan,
        "phase_sign_change": bool(len(phase_vals) > 1 and np.nanmin(phase_vals) < 0 < np.nanmax(phase_vals))
        if phase_vals
        else False,
        "mean_penalty_scalar": float(np.mean(penalty_vals)) if penalty_vals else np.nan,
        "max_penalty_scalar": float(np.max(penalty_vals)) if penalty_vals else np.nan,
        "mean_dominant_response_strength": float(np.mean(response_vals)) if response_vals else np.nan,
        "max_dominant_response_strength": float(np.max(response_vals)) if response_vals else np.nan,
        "min_lazarus_score": float(np.min(lazarus_vals)) if lazarus_vals else np.nan,
        "max_lazarus_score": float(np.max(lazarus_vals)) if lazarus_vals else np.nan,
        "seam_contact": bool(np.any(np.array(seam_vals) <= seam_contact_threshold)) if seam_vals else False,
    }
    return path_rows, path_summary


def build_outputs(
    df: pd.DataFrame,
    n_seeds: int,
    max_steps: int,
    step_radius_scale: float,
    min_forward_cos: float,
    seed_mode: str,
    seam_near_quantile: float,
    high_response_quantile: float,
    allow_revisit: bool,
    penalty_col: str | None,
    penalty_weight: float,
    seam_contact_threshold: float,
    n_nodes_input_raw: int,
    n_nodes_valid_raw: int,
    seam_nodes_csv: Path | None,
    merge_keys_used: list[str],
    seam_cols_merged: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xy = df[["mds1", "mds2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    D = pairwise_distances(xy)
    median_nn = median_nearest_neighbor_distance(xy)
    admissible_radius = step_radius_scale * median_nn

    seed_indices = choose_seed_indices(
        df=df,
        n_seeds=n_seeds,
        seed_mode=seed_mode,
        seam_near_quantile=seam_near_quantile,
        high_response_quantile=high_response_quantile,
    )

    path_node_rows: list[dict] = []
    path_rows: list[dict] = []

    for k, seed_idx in enumerate(seed_indices, start=1):
        pid = f"flow_{k:03d}"
        nodes_k, summary_k = integrate_path(
            seed_idx=int(seed_idx),
            df=df,
            xy=xy,
            D=D,
            max_steps=max_steps,
            admissible_radius=admissible_radius,
            min_forward_cos=min_forward_cos,
            allow_revisit=allow_revisit,
            penalty_col=penalty_col,
            penalty_weight=penalty_weight,
            seam_contact_threshold=seam_contact_threshold,
        )

        for r in nodes_k:
            r["path_id"] = pid
            path_node_rows.append(r)

        summary_k["path_id"] = pid
        summary_k["seed_mode"] = seed_mode
        summary_k["admissible_radius"] = admissible_radius
        summary_k["min_forward_cos"] = min_forward_cos
        summary_k["penalty_col"] = penalty_col if penalty_col is not None else ""
        summary_k["penalty_weight"] = penalty_weight
        summary_k["seam_contact_threshold"] = seam_contact_threshold
        path_rows.append(summary_k)

    path_nodes_df = pd.DataFrame(path_node_rows)
    paths_df = pd.DataFrame(path_rows)

    summary_df = pd.DataFrame(
        [
            {
                "n_nodes_input_raw": n_nodes_input_raw,
                "n_nodes_valid_raw": n_nodes_valid_raw,
                "valid_node_share": (n_nodes_valid_raw / n_nodes_input_raw) if n_nodes_input_raw > 0 else np.nan,
                "n_nodes_used": len(df),
                "n_seed_candidates": len(df),
                "n_seed_used": len(seed_indices),
                "n_paths": len(paths_df),
                "seed_mode": seed_mode,
                "median_nearest_neighbor_distance": median_nn,
                "admissible_radius": admissible_radius,
                "mean_path_length_mds": float(paths_df["path_length_mds"].mean()) if len(paths_df) else np.nan,
                "mean_n_steps": float(paths_df["n_steps"].mean()) if len(paths_df) else np.nan,
                "seam_contact_share": float(paths_df["seam_contact"].mean()) if len(paths_df) else np.nan,
                "phase_sign_change_share": float(paths_df["phase_sign_change"].mean()) if len(paths_df) else np.nan,
                "share_no_forward_neighbor": float((paths_df["termination_reason"] == "no_forward_neighbor").mean())
                if len(paths_df)
                else np.nan,
                "penalty_col": penalty_col if penalty_col is not None else "",
                "penalty_weight": penalty_weight,
                "seam_contact_threshold": seam_contact_threshold,
                "seam_nodes_csv": str(seam_nodes_csv) if seam_nodes_csv is not None else "",
                "merge_keys_used": ",".join(merge_keys_used),
                "n_seam_cols_merged": len(seam_cols_merged),
            }
        ]
    )

    return paths_df, path_nodes_df, summary_df


def plot_paths_figure(df: pd.DataFrame, path_nodes_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    color_col = choose_optional_column(df, ["distance_to_seam", "distance_to_seam_seam"])
    if color_col is not None:
        sc = ax.scatter(
            df["mds1"],
            df["mds2"],
            c=pd.to_numeric(df[color_col], errors="coerce"),
            s=18,
            alpha=0.55,
        )
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(df["mds1"], df["mds2"], s=18, alpha=0.35)

    if not path_nodes_df.empty:
        for _, sub in path_nodes_df.groupby("path_id", sort=False):
            ax.plot(sub["mds1"], sub["mds2"], linewidth=1.2, alpha=0.9)
            seed = sub.iloc[0]
            ax.scatter([seed["mds1"]], [seed["mds2"]], marker="x", s=36)

    ax.set_xlabel("mds1")
    ax.set_ylabel("mds2")
    ax.set_title("Response flow paths on manifold")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_quiver_figure(df: pd.DataFrame, outpath: Path, stride: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    bg_col = choose_optional_column(df, ["signed_phase", "signed_phase_seam"])
    if bg_col is not None:
        sc = ax.scatter(
            df["mds1"],
            df["mds2"],
            c=pd.to_numeric(df[bg_col], errors="coerce"),
            s=16,
            alpha=0.45,
        )
        fig.colorbar(sc, ax=ax, label=bg_col)
    else:
        ax.scatter(df["mds1"], df["mds2"], s=16, alpha=0.25)

    qdf = df.iloc[::max(1, stride)].copy()
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
    ax.set_title("Dominant response eigenvector field")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_nodes = load_nodes(args.input)
    raw_nodes, merge_keys_used, seam_cols_merged = maybe_merge_seam_nodes(
        response_df=raw_nodes,
        seam_nodes_csv=args.seam_nodes_csv,
        preferred_merge_keys=args.merge_on_cols,
    )
    nodes = compute_response_eigensystem(raw_nodes)

    n_nodes_input_raw = len(nodes)
    n_nodes_valid_raw = int(pd.Series(nodes["eigensystem_valid"]).fillna(False).astype(bool).sum())

    nodes = nodes.replace([np.inf, -np.inf], np.nan)
    nodes = nodes.dropna(
        subset=["mds1", "mds2", "eigvec1_x", "eigvec1_y", "eigval_1", "eigval_2"]
    ).reset_index(drop=True)

    if len(nodes) == 0:
        raise ValueError("No valid nodes remain after eigensystem/coordinate filtering.")

    penalty_col = resolve_penalty_column(nodes, args.penalty_col)

    paths_df, path_nodes_df, summary_df = build_outputs(
        df=nodes,
        n_seeds=args.n_seeds,
        max_steps=args.max_steps,
        step_radius_scale=args.step_radius_scale,
        min_forward_cos=args.min_forward_cos,
        seed_mode=args.seed_mode,
        seam_near_quantile=args.seam_near_quantile,
        high_response_quantile=args.high_response_quantile,
        allow_revisit=args.allow_revisit,
        penalty_col=penalty_col,
        penalty_weight=args.penalty_weight,
        seam_contact_threshold=args.seam_contact_threshold,
        n_nodes_input_raw=n_nodes_input_raw,
        n_nodes_valid_raw=n_nodes_valid_raw,
        seam_nodes_csv=args.seam_nodes_csv,
        merge_keys_used=merge_keys_used,
        seam_cols_merged=seam_cols_merged,
    )

    paths_path = args.outdir / "response_flow_paths.csv"
    path_nodes_path = args.outdir / "response_flow_path_nodes.csv"
    summary_path = args.outdir / "response_flow_summary.csv"
    paths_fig_path = args.outdir / "response_flow_paths.png"
    quiver_fig_path = args.outdir / "response_flow_quiver.png"

    paths_df.to_csv(paths_path, index=False)
    path_nodes_df.to_csv(path_nodes_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_paths_figure(nodes, path_nodes_df, paths_fig_path)
    plot_quiver_figure(nodes, quiver_fig_path, args.quiver_stride)

    print(f"Wrote: {paths_path}")
    print(f"Wrote: {path_nodes_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {paths_fig_path}")
    print(f"Wrote: {quiver_fig_path}")


if __name__ == "__main__":
    main()