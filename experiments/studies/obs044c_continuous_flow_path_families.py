#!/usr/bin/env python3
"""
obs044c_continuous_flow_path_families.py

OBS-044c — First-pass path-family analysis for continuous response-flow outputs.

Purpose
-------
Classify OBS-044 continuous response-flow trajectories into the same first-pass
route-family taxonomy used for OBS-043b, then compare continuous and discrete
family structure.

This study is intentionally conservative:
- it uses path-level observables only
- it reuses the OBS-043b family logic as closely as possible
- it treats cross-phase as a path attribute, not a family
- it supports comparison between one continuous run and one discrete reference

Route-family taxonomy
---------------------
Each trajectory receives exactly one route family:

- seam_hugging
- release_directed
- short_trapped
- mixed

and one attribute flag:

- cross_phase_flag

Inputs
------
Continuous run directory (default):
    outputs/obs044_continuous_response_flow/

Expected files:
    continuous_flow_trajectories.csv
    continuous_flow_trajectory_points.csv

Optional discrete reference directory (default):
    outputs/fim_response_flow_relaxed/

Expected files:
    response_flow_paths.csv
    response_flow_path_nodes.csv

Outputs
-------
Continuous run directory:
    continuous_flow_path_family_assignments.csv
    continuous_flow_path_family_summary.csv

Cross-run comparison directory:
    outputs/obs044c_continuous_flow_path_families/

Files:
    continuous_vs_discrete_path_family_regime_comparison.csv
    continuous_vs_discrete_path_family_summary.png

Usage
-----
python experiments/studies/obs044c_continuous_flow_path_families.py

python experiments/studies/obs044c_continuous_flow_path_families.py \
    --continuous-dir outputs/obs044_continuous_response_flow \
    --discrete-dir outputs/fim_response_flow_relaxed \
    --make-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONTINUOUS_DIR = Path("outputs/obs044_continuous_response_flow")
DEFAULT_DISCRETE_DIR = Path("outputs/fim_response_flow_relaxed")
DEFAULT_OUTDIR = Path("outputs/obs044c_continuous_flow_path_families")

FAMILY_ORDER = [
    "seam_hugging",
    "release_directed",
    "short_trapped",
    "mixed",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-044c continuous flow path-family analysis.")
    p.add_argument("--continuous-dir", type=Path, default=DEFAULT_CONTINUOUS_DIR)
    p.add_argument("--discrete-dir", type=Path, default=DEFAULT_DISCRETE_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--make-figure", action="store_true")
    p.add_argument(
        "--short-path-quantile",
        type=float,
        default=0.33,
        help="Quantile threshold used to define short_trapped by path length.",
    )
    p.add_argument(
        "--seam-hugging-quantile",
        type=float,
        default=0.50,
        help="Quantile threshold used to define seam_hugging by mean seam distance.",
    )
    p.add_argument(
        "--release-length-quantile",
        type=float,
        default=0.50,
        help="Quantile threshold used to define release_directed by path length.",
    )
    p.add_argument(
        "--seam-contact-required",
        action="store_true",
        help="Require seam_contact=True for seam_hugging and release_directed.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_continuous(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj_path = run_dir / "continuous_flow_trajectories.csv"
    pts_path = run_dir / "continuous_flow_trajectory_points.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing: {traj_path}")
    if not pts_path.exists():
        raise FileNotFoundError(f"Missing: {pts_path}")

    traj_df = pd.read_csv(traj_path)
    pts_df = pd.read_csv(pts_path)

    require_columns(
        traj_df,
        [
            "trajectory_id",
            "seed_node_id",
            "n_steps",
            "path_length_mds",
            "mean_distance_to_seam",
            "min_distance_to_seam",
            "phase_sign_change",
            "seam_contact",
            "termination_reason",
        ],
        "continuous_flow_trajectories.csv",
    )
    require_columns(
        pts_df,
        ["trajectory_id", "step", "x", "y"],
        "continuous_flow_trajectory_points.csv",
    )
    return traj_df, pts_df


def load_discrete(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_path = run_dir / "response_flow_paths.csv"
    nodes_path = run_dir / "response_flow_path_nodes.csv"
    if not paths_path.exists():
        raise FileNotFoundError(f"Missing: {paths_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing: {nodes_path}")

    paths_df = pd.read_csv(paths_path)
    nodes_df = pd.read_csv(nodes_path)

    require_columns(
        paths_df,
        [
            "path_id",
            "seed_node_id",
            "n_steps",
            "path_length_mds",
            "mean_distance_to_seam",
            "min_distance_to_seam",
            "phase_sign_change",
            "seam_contact",
            "termination_reason",
        ],
        "response_flow_paths.csv",
    )
    require_columns(
        nodes_df,
        ["path_id", "step", "mds1", "mds2"],
        "response_flow_path_nodes.csv",
    )
    return paths_df, nodes_df


def classify_family_frame(
    df: pd.DataFrame,
    id_col: str,
    short_path_quantile: float,
    seam_hugging_quantile: float,
    release_length_quantile: float,
    seam_contact_required: bool,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    path_len = pd.to_numeric(out["path_length_mds"], errors="coerce")
    mean_seam = pd.to_numeric(out["mean_distance_to_seam"], errors="coerce")
    steps = pd.to_numeric(out["n_steps"], errors="coerce")

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

        if (path_length <= short_len_thresh) and (n_steps <= 2 or term in {"no_forward_neighbor", "low_angular_consistency", "support_radius_exceeded"}):
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


def build_family_summary(assign_df: pd.DataFrame, run_name: str, flow_type: str, thresholds: dict, id_col: str) -> pd.DataFrame:
    rows = []
    total = len(assign_df)

    for fam in FAMILY_ORDER:
        sub = assign_df[assign_df["path_family"] == fam].copy()
        if len(sub) == 0:
            rows.append(
                {
                    "run_name": run_name,
                    "flow_type": flow_type,
                    "path_family": fam,
                    "n_paths": 0,
                    "path_share": 0.0,
                    "cross_phase_share": np.nan,
                    "mean_path_length_mds": np.nan,
                    "mean_n_steps": np.nan,
                    "seam_contact_share": np.nan,
                    "share_termination_primary": np.nan,
                    "mean_distance_to_seam": np.nan,
                    **thresholds,
                }
            )
            continue

        primary_term = str(sub["termination_reason"].mode(dropna=False).iloc[0])
        rows.append(
            {
                "run_name": run_name,
                "flow_type": flow_type,
                "path_family": fam,
                "n_paths": len(sub),
                "path_share": len(sub) / total if total else np.nan,
                "cross_phase_share": float(sub["cross_phase_flag"].astype(bool).mean()),
                "mean_path_length_mds": float(pd.to_numeric(sub["path_length_mds"], errors="coerce").mean()),
                "mean_n_steps": float(pd.to_numeric(sub["n_steps"], errors="coerce").mean()),
                "seam_contact_share": float(sub["seam_contact"].astype(bool).mean()),
                "share_termination_primary": float((sub["termination_reason"] == primary_term).mean()),
                "mean_distance_to_seam": float(pd.to_numeric(sub["mean_distance_to_seam"], errors="coerce").mean()),
                "primary_termination_reason": primary_term,
                **thresholds,
            }
        )

    rows.append(
        {
            "run_name": run_name,
            "flow_type": flow_type,
            "path_family": "__overall__",
            "n_paths": total,
            "path_share": 1.0 if total else np.nan,
            "cross_phase_share": float(assign_df["cross_phase_flag"].astype(bool).mean()) if total else np.nan,
            "mean_path_length_mds": float(pd.to_numeric(assign_df["path_length_mds"], errors="coerce").mean()) if total else np.nan,
            "mean_n_steps": float(pd.to_numeric(assign_df["n_steps"], errors="coerce").mean()) if total else np.nan,
            "seam_contact_share": float(assign_df["seam_contact"].astype(bool).mean()) if total else np.nan,
            "share_termination_primary": np.nan,
            "mean_distance_to_seam": float(pd.to_numeric(assign_df["mean_distance_to_seam"], errors="coerce").mean()) if total else np.nan,
            "primary_termination_reason": "",
            **thresholds,
        }
    )

    out = pd.DataFrame(rows)
    fam_order = FAMILY_ORDER + ["__overall__"]
    out["path_family"] = pd.Categorical(out["path_family"], categories=fam_order, ordered=True)
    out = out.sort_values("path_family").reset_index(drop=True)
    return out


def plot_family_comparison(regime_df: pd.DataFrame, outpath: Path) -> None:
    plot_df = regime_df[regime_df["path_family"].isin(FAMILY_ORDER)].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    run_labels = plot_df["run_name"].tolist()
    run_names = list(dict.fromkeys(run_labels))
    x = np.arange(len(run_names))
    width = 0.18

    for i, fam in enumerate(FAMILY_ORDER):
        sub = plot_df[plot_df["path_family"] == fam].copy()
        sub = sub.set_index("run_name").reindex(run_names).reset_index()
        vals = pd.to_numeric(sub["path_share"], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fam)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=18, ha="right")
    ax.set_ylabel("path share")
    ax.set_title("OBS-044c continuous vs discrete path-family comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # continuous
    cont_traj_df, cont_pts_df = load_continuous(args.continuous_dir)
    cont_assign_df, cont_thresholds = classify_family_frame(
        cont_traj_df,
        id_col="trajectory_id",
        short_path_quantile=args.short_path_quantile,
        seam_hugging_quantile=args.seam_hugging_quantile,
        release_length_quantile=args.release_length_quantile,
        seam_contact_required=args.seam_contact_required,
    )
    cont_assign_df["run_name"] = args.continuous_dir.name
    cont_assign_df["flow_type"] = "continuous_obs044"

    cont_assign_out = args.continuous_dir / "continuous_flow_path_family_assignments.csv"
    cont_summary_out = args.continuous_dir / "continuous_flow_path_family_summary.csv"

    cont_summary_df = build_family_summary(
        cont_assign_df,
        run_name=args.continuous_dir.name,
        flow_type="continuous_obs044",
        thresholds=cont_thresholds,
        id_col="trajectory_id",
    )

    cont_assign_df.to_csv(cont_assign_out, index=False)
    cont_summary_df.to_csv(cont_summary_out, index=False)

    print(f"Wrote: {cont_assign_out}")
    print(f"Wrote: {cont_summary_out}")

    # discrete reference
    disc_paths_df, disc_nodes_df = load_discrete(args.discrete_dir)
    disc_assign_df, disc_thresholds = classify_family_frame(
        disc_paths_df,
        id_col="path_id",
        short_path_quantile=args.short_path_quantile,
        seam_hugging_quantile=args.seam_hugging_quantile,
        release_length_quantile=args.release_length_quantile,
        seam_contact_required=args.seam_contact_required,
    )
    disc_assign_df["run_name"] = args.discrete_dir.name
    disc_assign_df["flow_type"] = "discrete_relaxed"

    disc_summary_df = build_family_summary(
        disc_assign_df,
        run_name=args.discrete_dir.name,
        flow_type="discrete_relaxed",
        thresholds=disc_thresholds,
        id_col="path_id",
    )

    regime_df = pd.concat([disc_summary_df, cont_summary_df], ignore_index=True)
    fam_order = FAMILY_ORDER + ["__overall__"]
    regime_df["path_family"] = pd.Categorical(regime_df["path_family"], categories=fam_order, ordered=True)
    regime_df = regime_df.sort_values(["flow_type", "path_family"]).reset_index(drop=True)

    comparison_out = args.outdir / "continuous_vs_discrete_path_family_regime_comparison.csv"
    regime_df.to_csv(comparison_out, index=False)
    print(f"Wrote: {comparison_out}")

    if args.make_figure:
        fig_out = args.outdir / "continuous_vs_discrete_path_family_summary.png"
        plot_family_comparison(regime_df, fig_out)
        print(f"Wrote: {fig_out}")


if __name__ == "__main__":
    main()
