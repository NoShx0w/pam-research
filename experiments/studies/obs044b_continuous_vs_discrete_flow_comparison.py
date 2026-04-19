#!/usr/bin/env python3
"""
obs044b_continuous_vs_discrete_flow_comparison.py

OBS-044b — Compare OBS-043 relaxed discrete flow against OBS-044 continuous flow.

Purpose
-------
Construct a conservative comparison between:
- discrete response-guided flow from OBS-043
- continuous response-flow reconstruction from OBS-044

This study focuses on high-level observables and does not attempt one-to-one
trajectory matching.

Inputs
------
Discrete run directory (default):
    outputs/fim_response_flow_relaxed/

Expected files:
    response_flow_paths.csv
    response_flow_path_nodes.csv

Continuous run directory (default):
    outputs/obs044_continuous_response_flow/

Expected files:
    continuous_flow_trajectories.csv
    continuous_flow_trajectory_points.csv

Outputs
-------
Directory:
    outputs/obs044b_continuous_vs_discrete_flow_comparison/

Files:
    continuous_vs_discrete_summary.csv
    continuous_vs_discrete_trajectory_table.csv
    continuous_vs_discrete_figure.png
    continuous_vs_discrete_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DISCRETE_DIR = Path("outputs/fim_response_flow_relaxed")
DEFAULT_CONTINUOUS_DIR = Path("outputs/obs044_continuous_response_flow")
DEFAULT_OUTDIR = Path("outputs/obs044b_continuous_vs_discrete_flow_comparison")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare OBS-043 discrete and OBS-044 continuous flow.")
    p.add_argument("--discrete-dir", type=Path, default=DEFAULT_DISCRETE_DIR)
    p.add_argument("--continuous-dir", type=Path, default=DEFAULT_CONTINUOUS_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--seam-contact-threshold",
        type=float,
        default=0.25,
        help="Threshold used to report seam-contact summaries if needed.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_discrete(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = pd.read_csv(run_dir / "response_flow_paths.csv")
    nodes = pd.read_csv(run_dir / "response_flow_path_nodes.csv")

    require_columns(
        paths,
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
        nodes,
        ["path_id", "step", "mds1", "mds2"],
        "response_flow_path_nodes.csv",
    )
    return paths, nodes


def load_continuous(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = pd.read_csv(run_dir / "continuous_flow_trajectories.csv")
    pts = pd.read_csv(run_dir / "continuous_flow_trajectory_points.csv")

    require_columns(
        traj,
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
        pts,
        ["trajectory_id", "step", "x", "y"],
        "continuous_flow_trajectory_points.csv",
    )
    return traj, pts


def summarize_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flow_type": kind,
                "n_trajectories": len(df),
                "mean_path_length_mds": float(pd.to_numeric(df["path_length_mds"], errors="coerce").mean()),
                "median_path_length_mds": float(pd.to_numeric(df["path_length_mds"], errors="coerce").median()),
                "mean_n_steps": float(pd.to_numeric(df["n_steps"], errors="coerce").mean()),
                "median_n_steps": float(pd.to_numeric(df["n_steps"], errors="coerce").median()),
                "seam_contact_share": float(df["seam_contact"].astype(bool).mean()),
                "phase_sign_change_share": float(df["phase_sign_change"].astype(bool).mean()),
                "mean_distance_to_seam": float(pd.to_numeric(df["mean_distance_to_seam"], errors="coerce").mean()),
                "mean_min_distance_to_seam": float(pd.to_numeric(df["min_distance_to_seam"], errors="coerce").mean()),
            }
        ]
    )


def build_trajectory_table(discrete_df: pd.DataFrame, continuous_df: pd.DataFrame) -> pd.DataFrame:
    d = discrete_df.copy()
    d["flow_type"] = "discrete_relaxed"
    d = d.rename(columns={"path_id": "trajectory_id"})

    c = continuous_df.copy()
    c["flow_type"] = "continuous_obs044"

    keep = [
        "flow_type",
        "trajectory_id",
        "seed_node_id",
        "n_steps",
        "path_length_mds",
        "mean_distance_to_seam",
        "min_distance_to_seam",
        "phase_sign_change",
        "seam_contact",
        "termination_reason",
    ]
    out = pd.concat([d[keep], c[keep]], ignore_index=True)
    return out


def write_summary_text(summary_df: pd.DataFrame, outpath: Path) -> None:
    rows = summary_df.to_dict(orient="records")
    discrete = next(r for r in rows if r["flow_type"] == "discrete_relaxed")
    continuous = next(r for r in rows if r["flow_type"] == "continuous_obs044")

    text = f"""OBS-044b continuous vs discrete comparison

Discrete reference:
- n_trajectories: {discrete['n_trajectories']}
- mean_path_length_mds: {discrete['mean_path_length_mds']:.6f}
- mean_n_steps: {discrete['mean_n_steps']:.6f}
- seam_contact_share: {discrete['seam_contact_share']:.6f}
- phase_sign_change_share: {discrete['phase_sign_change_share']:.6f}
- mean_distance_to_seam: {discrete['mean_distance_to_seam']:.6f}

Continuous reconstruction:
- n_trajectories: {continuous['n_trajectories']}
- mean_path_length_mds: {continuous['mean_path_length_mds']:.6f}
- mean_n_steps: {continuous['mean_n_steps']:.6f}
- seam_contact_share: {continuous['seam_contact_share']:.6f}
- phase_sign_change_share: {continuous['phase_sign_change_share']:.6f}
- mean_distance_to_seam: {continuous['mean_distance_to_seam']:.6f}

Interpretive comparison:
- seam engagement is {'higher' if continuous['seam_contact_share'] > discrete['seam_contact_share'] else 'lower or equal'} in the continuous reconstruction
- path extent is {'higher' if continuous['mean_path_length_mds'] > discrete['mean_path_length_mds'] else 'lower'} in the continuous reconstruction
- phase crossing is {'higher' if continuous['phase_sign_change_share'] > discrete['phase_sign_change_share'] else 'lower'} in the continuous reconstruction
"""
    outpath.write_text(text, encoding="utf-8")


def plot_comparison(summary_df: pd.DataFrame, outpath: Path) -> None:
    metrics = [
        "mean_path_length_mds",
        "mean_n_steps",
        "seam_contact_share",
        "phase_sign_change_share",
    ]
    labels = [
        "mean path length",
        "mean steps",
        "seam contact share",
        "phase crossing share",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(metrics))
    width = 0.35

    disc = summary_df[summary_df["flow_type"] == "discrete_relaxed"].iloc[0]
    cont = summary_df[summary_df["flow_type"] == "continuous_obs044"].iloc[0]

    disc_vals = [float(disc[m]) for m in metrics]
    cont_vals = [float(cont[m]) for m in metrics]

    ax.bar(x - width / 2, disc_vals, width=width, label="discrete_relaxed")
    ax.bar(x + width / 2, cont_vals, width=width, label="continuous_obs044")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("OBS-044b continuous vs discrete flow comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    discrete_df, discrete_nodes = load_discrete(args.discrete_dir)
    continuous_df, continuous_pts = load_continuous(args.continuous_dir)

    summary_df = pd.concat(
        [
            summarize_table(discrete_df, "discrete_relaxed"),
            summarize_table(continuous_df, "continuous_obs044"),
        ],
        ignore_index=True,
    )

    traj_df = build_trajectory_table(discrete_df, continuous_df)

    summary_path = args.outdir / "continuous_vs_discrete_summary.csv"
    traj_path = args.outdir / "continuous_vs_discrete_trajectory_table.csv"
    fig_path = args.outdir / "continuous_vs_discrete_figure.png"
    txt_path = args.outdir / "continuous_vs_discrete_summary.txt"

    summary_df.to_csv(summary_path, index=False)
    traj_df.to_csv(traj_path, index=False)
    plot_comparison(summary_df, fig_path)
    write_summary_text(summary_df, txt_path)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {traj_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
