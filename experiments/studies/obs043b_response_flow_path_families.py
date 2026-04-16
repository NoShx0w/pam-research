#!/usr/bin/env python3
"""
obs043b_response_flow_path_families.py

OBS-043b — First-pass path-family analysis for response-guided flow outputs.

Purpose
-------
Classify response-flow paths into a small first-pass route-family taxonomy and
summarize how those families vary across integration regimes.

This revision keeps cross-phase as a path attribute rather than an overriding
family label.

Route-family taxonomy
---------------------
Each path receives exactly one route family:

- seam_hugging
- release_directed
- short_trapped
- mixed

and one attribute flag:

- cross_phase_flag

Design principles
-----------------
- simple, inspectable path-level rules
- data-driven thresholds within each run
- route family separated from phase-crossing attribute
- per-run outputs stay in each run directory
- cross-run comparison outputs go into:
    outputs/obs043b_response_flow_path_families/

Inputs
------
Each run directory is expected to contain:
    response_flow_paths.csv
    response_flow_path_nodes.csv

Outputs
-------
Per run:
    response_flow_path_family_assignments.csv
    response_flow_path_family_summary.csv

Cross-run:
    outputs/obs043b_response_flow_path_families/response_flow_path_family_regime_comparison.csv
    outputs/obs043b_response_flow_path_families/response_flow_path_family_summary.png  (optional)

Usage
-----
python experiments/studies/obs043b_response_flow_path_families.py \
    --run-dirs outputs/fim_response_flow_relaxed

python experiments/studies/obs043b_response_flow_path_families.py \
    --run-dirs \
        outputs/fim_response_flow_relaxed \
        outputs/fim_response_flow_relaxed_seambundle_mismatch \
        outputs/fim_response_flow_relaxed_seambundle_neighbor_mismatch \
    --make-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_PATHS = "response_flow_paths.csv"
EXPECTED_PATH_NODES = "response_flow_path_nodes.csv"

FAMILY_ORDER = [
    "seam_hugging",
    "release_directed",
    "short_trapped",
    "mixed",
]

DEFAULT_COMPARISON_OUTDIR = Path("outputs/obs043b_response_flow_path_families")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify response-flow paths into first-pass route families.")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more OBS-043 response-flow output directories.",
    )
    p.add_argument(
        "--comparison-outdir",
        type=Path,
        default=DEFAULT_COMPARISON_OUTDIR,
        help="Directory for cross-run OBS-043b comparison outputs.",
    )
    p.add_argument(
        "--make-figure",
        action="store_true",
        help="Write a simple family-share comparison figure.",
    )
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
        help="Require seam_contact=True for seam_hugging and release_directed labels.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_path = run_dir / EXPECTED_PATHS
    nodes_path = run_dir / EXPECTED_PATH_NODES

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
            "n_steps",
            "path_length_mds",
            "seam_contact",
            "mean_distance_to_seam",
            "phase_sign_change",
            "termination_reason",
        ],
        "response_flow_paths.csv",
    )
    require_columns(
        nodes_df,
        [
            "path_id",
            "step",
            "mds1",
            "mds2",
        ],
        "response_flow_path_nodes.csv",
    )

    return paths_df, nodes_df


def classify_paths(
    paths_df: pd.DataFrame,
    short_path_quantile: float,
    seam_hugging_quantile: float,
    release_length_quantile: float,
    seam_contact_required: bool,
) -> tuple[pd.DataFrame, dict]:
    df = paths_df.copy()

    path_len = pd.to_numeric(df["path_length_mds"], errors="coerce")
    mean_seam = pd.to_numeric(df["mean_distance_to_seam"], errors="coerce")
    n_steps = pd.to_numeric(df["n_steps"], errors="coerce")

    short_len_thresh = float(path_len.quantile(short_path_quantile))
    seam_hugging_thresh = float(mean_seam.quantile(seam_hugging_quantile))
    release_len_thresh = float(path_len.quantile(release_length_quantile))

    labels = []
    reasons = []
    cross_flags = []

    for _, row in df.iterrows():
        phase_cross = bool(row["phase_sign_change"])
        seam_contact = bool(row["seam_contact"])
        path_length = float(row["path_length_mds"])
        mean_seam_dist = float(row["mean_distance_to_seam"]) if pd.notna(row["mean_distance_to_seam"]) else np.nan
        steps = float(row["n_steps"])
        term = str(row["termination_reason"])

        cross_flags.append(phase_cross)

        # 1. short_trapped
        if (path_length <= short_len_thresh) and (steps <= 2 or term == "no_forward_neighbor"):
            labels.append("short_trapped")
            reasons.append("short_path_and_early_termination")
            continue

        # 2. seam_hugging
        seam_ok = seam_contact if seam_contact_required else True
        if seam_ok and pd.notna(mean_seam_dist) and (mean_seam_dist <= seam_hugging_thresh):
            labels.append("seam_hugging")
            reasons.append("low_mean_seam_distance")
            continue

        # 3. release_directed
        seam_ok = seam_contact if seam_contact_required else True
        if seam_ok and (path_length >= release_len_thresh):
            labels.append("release_directed")
            reasons.append("longer_path_extent")
            continue

        # 4. mixed
        labels.append("mixed")
        reasons.append("fallback_mixed")

    df["path_family"] = labels
    df["family_reason"] = reasons
    df["cross_phase_flag"] = cross_flags

    thresholds = {
        "short_path_length_threshold": short_len_thresh,
        "seam_hugging_mean_seam_threshold": seam_hugging_thresh,
        "release_length_threshold": release_len_thresh,
    }
    return df, thresholds


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
                    "share_no_forward_neighbor": np.nan,
                    "mean_distance_to_seam": np.nan,
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
                "share_no_forward_neighbor": float((sub["termination_reason"] == "no_forward_neighbor").mean()),
                "mean_distance_to_seam": float(pd.to_numeric(sub["mean_distance_to_seam"], errors="coerce").mean()),
                **thresholds,
            }
        )

    # add run-level overall row for convenience
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
            "share_no_forward_neighbor": float((assign_df["termination_reason"] == "no_forward_neighbor").mean()) if total else np.nan,
            "mean_distance_to_seam": float(pd.to_numeric(assign_df["mean_distance_to_seam"], errors="coerce").mean()) if total else np.nan,
            **thresholds,
        }
    )

    out = pd.DataFrame(rows)
    fam_order = FAMILY_ORDER + ["__overall__"]
    out["path_family"] = pd.Categorical(out["path_family"], categories=fam_order, ordered=True)
    out = out.sort_values("path_family").reset_index(drop=True)
    return out


def build_regime_comparison(summary_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not summary_frames:
        return pd.DataFrame()

    out = pd.concat(summary_frames, ignore_index=True)
    fam_order = FAMILY_ORDER + ["__overall__"]
    out["path_family"] = pd.Categorical(out["path_family"], categories=fam_order, ordered=True)
    out = out.sort_values(["run_name", "path_family"]).reset_index(drop=True)
    return out


def plot_family_summary(regime_df: pd.DataFrame, outpath: Path) -> None:
    if regime_df.empty:
        return

    plot_df = regime_df[regime_df["path_family"].isin(FAMILY_ORDER)].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    run_names = list(dict.fromkeys(plot_df["run_name"].tolist()))
    x = np.arange(len(run_names))
    width = 0.18

    for i, fam in enumerate(FAMILY_ORDER):
        sub = plot_df[plot_df["path_family"] == fam].copy()
        sub = sub.set_index("run_name").reindex(run_names).reset_index()
        vals = pd.to_numeric(sub["path_share"], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=fam)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=20, ha="right")
    ax.set_ylabel("path share")
    ax.set_title("OBS-043b response-flow path families")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.comparison_outdir.mkdir(parents=True, exist_ok=True)

    all_summary_frames = []

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        run_name = run_dir.name

        paths_df, nodes_df = load_run(run_dir)

        assign_df, thresholds = classify_paths(
            paths_df=paths_df,
            short_path_quantile=args.short_path_quantile,
            seam_hugging_quantile=args.seam_hugging_quantile,
            release_length_quantile=args.release_length_quantile,
            seam_contact_required=args.seam_contact_required,
        )

        assign_df["run_name"] = run_name

        summary_df = build_family_summary(assign_df, run_name, thresholds)
        all_summary_frames.append(summary_df)

        assign_out = run_dir / "response_flow_path_family_assignments.csv"
        summary_out = run_dir / "response_flow_path_family_summary.csv"

        assign_df.to_csv(assign_out, index=False)
        summary_df.to_csv(summary_out, index=False)

        print(f"Wrote: {assign_out}")
        print(f"Wrote: {summary_out}")

    regime_df = build_regime_comparison(all_summary_frames)

    comparison_out = args.comparison_outdir / "response_flow_path_family_regime_comparison.csv"
    regime_df.to_csv(comparison_out, index=False)
    print(f"Wrote: {comparison_out}")

    if args.make_figure:
        fig_out = args.comparison_outdir / "response_flow_path_family_summary.png"
        plot_family_summary(regime_df, fig_out)
        print(f"Wrote: {fig_out}")


if __name__ == "__main__":
    main()