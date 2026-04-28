#!/usr/bin/env python3
from __future__ import annotations

"""
OBS-051 — Local divergence within coupled roughness-escalation windows.

Core question
-------------
Among roughness-escalation windows that remain structurally coupled to the seam,
do recovery-like families exhibit lower local divergence than nonrecovery-like
families?

Motivation
----------
OBS-050 established that recovery-like roughness-escalation windows are much
more likely to remain seam-coupled than nonrecovering windows. The next question
is whether coupled instability is also dynamically more bounded.

This script implements a Lyapunov-like local divergence proxy, not a canonical
Lyapunov exponent. For each escalation window, it compares the initial and final
separation from nearby matched windows in a compact observatory state space.

Inputs
------
1. OBS-050 segment summary:
   outputs/obs050_structural_coupling_persistence/structural_coupling_segments.csv

2. Family substrate path-node diagnostics:
   outputs/scales/100000/family_substrate/path_node_diagnostics.csv

3. Family assignments:
   outputs/scales/100000/family_substrate/path_family_assignments.csv

Outputs
-------
<outdir>/
  obs051_window_divergence.csv
  obs051_outcome_summary.csv
  obs051_family_summary.csv
  obs051_local_divergence_summary.txt
  obs051_lambda_boxplot_by_outcome.png
  obs051_lambda_vs_initial_separation.png

Interpretation
--------------
For each coupled escalation window i, find nearby windows j by closeness of
their onset state. Then estimate

    lambda_local(i,j) = (1 / dt) * log((d_end + eps) / (d_start + eps))

where d_start is separation at onset and d_end is separation at window end.

Low / negative lambda_local:
    bounded or reconvergent instability

High positive lambda_local:
    locally divergent instability
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
    segments_csv: str = "outputs/obs050_structural_coupling_persistence/structural_coupling_segments.csv"
    nodes_csv: str = "outputs/scales/100000/family_substrate/path_node_diagnostics.csv"
    family_csv: str = "outputs/scales/100000/family_substrate/path_family_assignments.csv"
    outdir: str = "outputs/obs051_local_divergence_in_coupled_windows"
    coupling_class: str = "coupled"
    k_neighbors: int = 5
    max_initial_distance: float = 2.5
    eps: float = 1e-6
    min_dt: float = 1.0


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


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segs = safe_read_csv(Path(cfg.segments_csv))
    nodes = safe_read_csv(Path(cfg.nodes_csv))
    fam = safe_read_csv(Path(cfg.family_csv))

    segs["path_id"] = segs["path_id"].astype(str)
    nodes["path_id"] = nodes["path_id"].astype(str)
    fam["path_id"] = fam["path_id"].astype(str)

    to_numeric_inplace(
        segs,
        [
            "center_step",
            "start_step",
            "end_step",
            "mean_distance_to_seam",
            "min_distance_to_seam",
            "center_distance_to_seam",
            "mean_roughness",
            "mean_roughness_smoothed",
            "m_r",
            "m_seam",
            "mean_criticality",
            "mean_obstruction",
        ],
    )
    to_numeric_inplace(
        nodes,
        [
            "step",
            "distance_to_seam",
            "path_angle_jump_deg",
            "criticality",
            "obstruction_mean_abs_holonomy",
            "absolute_holonomy_node",
        ],
    )

    if "path_family" not in segs.columns:
        segs = segs.merge(fam[["path_id", "path_family"]].drop_duplicates(), on="path_id", how="left")
    if "outcome_group" not in segs.columns:
        segs["outcome_group"] = segs["path_family"].map(classify_outcome)

    return segs, nodes, fam


def add_coupling_class(segs: pd.DataFrame) -> pd.DataFrame:
    out = segs.copy()
    if "coupling_class" not in out.columns:
        out["coupling_class"] = np.where(
            out["seam_band"].isin(["core", "near"]),
            "coupled",
            np.where(out["seam_band"] == "far", "decoupled", "unknown"),
        )
    return out


def choose_obstruction_col(nodes: pd.DataFrame) -> str | None:
    for c in ["obstruction_mean_abs_holonomy", "absolute_holonomy_node"]:
        if c in nodes.columns:
            return c
    return None


def build_window_states(segs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    obstruction_col = choose_obstruction_col(nodes)
    records: list[dict[str, object]] = []

    node_cols = ["path_id", "step", "distance_to_seam", "path_angle_jump_deg"]
    if "criticality" in nodes.columns:
        node_cols.append("criticality")
    if obstruction_col is not None:
        node_cols.append(obstruction_col)

    use_nodes = nodes[node_cols].copy()

    for row in segs.itertuples(index=False):
        pid = str(row.path_id)
        start_step = float(row.start_step)
        end_step = float(row.end_step)

        sub = use_nodes[use_nodes["path_id"] == pid].copy()
        if len(sub) == 0:
            continue

        start_rows = sub[np.isclose(sub["step"], start_step)]
        end_rows = sub[np.isclose(sub["step"], end_step)]
        if len(start_rows) == 0 or len(end_rows) == 0:
            continue

        s0 = start_rows.iloc[0]
        s1 = end_rows.iloc[0]

        rec = {
            "segment_id": row.segment_id,
            "path_id": pid,
            "path_family": getattr(row, "path_family", None),
            "outcome_group": getattr(row, "outcome_group", None),
            "seam_band": getattr(row, "seam_band", None),
            "coupling_class": getattr(row, "coupling_class", None),
            "start_step": start_step,
            "end_step": end_step,
            "dt": max(float(end_step - start_step), 0.0),
            "start_distance_to_seam": float(s0["distance_to_seam"]) if pd.notna(s0["distance_to_seam"]) else np.nan,
            "end_distance_to_seam": float(s1["distance_to_seam"]) if pd.notna(s1["distance_to_seam"]) else np.nan,
            "start_roughness": float(s0["path_angle_jump_deg"]) if pd.notna(s0["path_angle_jump_deg"]) else np.nan,
            "end_roughness": float(s1["path_angle_jump_deg"]) if pd.notna(s1["path_angle_jump_deg"]) else np.nan,
        }

        if "criticality" in s0.index:
            rec["start_criticality"] = float(s0["criticality"]) if pd.notna(s0["criticality"]) else np.nan
            rec["end_criticality"] = float(s1["criticality"]) if pd.notna(s1["criticality"]) else np.nan

        if obstruction_col is not None:
            rec["start_obstruction"] = float(s0[obstruction_col]) if pd.notna(s0[obstruction_col]) else np.nan
            rec["end_obstruction"] = float(s1[obstruction_col]) if pd.notna(s1[obstruction_col]) else np.nan

        records.append(rec)

    return pd.DataFrame(records)


def feature_distance(a: pd.Series, b: pd.Series, cols: list[str]) -> float:
    vals = []
    for c in cols:
        av = pd.to_numeric(a.get(c), errors="coerce")
        bv = pd.to_numeric(b.get(c), errors="coerce")
        if pd.isna(av) or pd.isna(bv):
            continue
        vals.append((float(av) - float(bv)) ** 2)
    if not vals:
        return float("nan")
    return float(np.sqrt(np.sum(vals)))


def build_divergence_table(states: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if states.empty:
        return pd.DataFrame()

    feature_cols_start = ["start_distance_to_seam", "start_roughness"]
    if "start_criticality" in states.columns:
        feature_cols_start.append("start_criticality")
    if "start_obstruction" in states.columns:
        feature_cols_start.append("start_obstruction")

    feature_cols_end = ["end_distance_to_seam", "end_roughness"]
    if "end_criticality" in states.columns:
        feature_cols_end.append("end_criticality")
    if "end_obstruction" in states.columns:
        feature_cols_end.append("end_obstruction")

    rows: list[dict[str, object]] = []

    states = states.reset_index(drop=True)

    for i, a in states.iterrows():
        if pd.to_numeric(a["dt"], errors="coerce") < cfg.min_dt:
            continue

        dists: list[tuple[float, int]] = []
        for j, b in states.iterrows():
            if i == j:
                continue
            if str(a["path_id"]) == str(b["path_id"]):
                continue

            d0 = feature_distance(a, b, feature_cols_start)
            if not np.isfinite(d0):
                continue
            if d0 > cfg.max_initial_distance:
                continue
            dists.append((d0, j))

        if not dists:
            continue

        dists.sort(key=lambda x: x[0])
        for rank, (d_start, j) in enumerate(dists[: cfg.k_neighbors], start=1):
            b = states.iloc[j]
            d_end = feature_distance(a, b, feature_cols_end)
            if not np.isfinite(d_end):
                continue

            dt = max(float(a["dt"]), cfg.min_dt)
            lambda_local = (1.0 / dt) * np.log((d_end + cfg.eps) / (d_start + cfg.eps))

            rows.append(
                {
                    "segment_id": a["segment_id"],
                    "neighbor_segment_id": b["segment_id"],
                    "path_id": a["path_id"],
                    "neighbor_path_id": b["path_id"],
                    "path_family": a["path_family"],
                    "outcome_group": a["outcome_group"],
                    "seam_band": a["seam_band"],
                    "coupling_class": a["coupling_class"],
                    "neighbor_rank": rank,
                    "dt": dt,
                    "d_start": d_start,
                    "d_end": d_end,
                    "lambda_local": lambda_local,
                }
            )

    return pd.DataFrame(rows)


def build_window_summary(div: pd.DataFrame) -> pd.DataFrame:
    if div.empty:
        return pd.DataFrame()

    return (
        div.groupby(["segment_id", "path_id", "path_family", "outcome_group", "seam_band", "coupling_class"], dropna=False)
        .agg(
            n_neighbors=("neighbor_segment_id", "size"),
            mean_d_start=("d_start", "mean"),
            mean_d_end=("d_end", "mean"),
            mean_lambda_local=("lambda_local", "mean"),
            median_lambda_local=("lambda_local", "median"),
            max_lambda_local=("lambda_local", "max"),
        )
        .reset_index()
    )


def build_outcome_summary(win: pd.DataFrame) -> pd.DataFrame:
    if win.empty:
        return pd.DataFrame()

    return (
        win.groupby(["outcome_group"], dropna=False)
        .agg(
            n_windows=("segment_id", "size"),
            mean_lambda_local=("mean_lambda_local", "mean"),
            median_lambda_local=("mean_lambda_local", "median"),
            mean_d_start=("mean_d_start", "mean"),
            mean_d_end=("mean_d_end", "mean"),
        )
        .reset_index()
    )


def build_family_summary(win: pd.DataFrame) -> pd.DataFrame:
    if win.empty:
        return pd.DataFrame()

    return (
        win.groupby(["outcome_group", "path_family", "seam_band"], dropna=False)
        .agg(
            n_windows=("segment_id", "size"),
            mean_lambda_local=("mean_lambda_local", "mean"),
            median_lambda_local=("mean_lambda_local", "median"),
            mean_d_start=("mean_d_start", "mean"),
            mean_d_end=("mean_d_end", "mean"),
        )
        .reset_index()
        .sort_values(["outcome_group", "path_family", "seam_band"])
        .reset_index(drop=True)
    )


def summarize_text(
    states: pd.DataFrame,
    div: pd.DataFrame,
    win: pd.DataFrame,
    out_summary: pd.DataFrame,
    fam_summary: pd.DataFrame,
    cfg: Config,
) -> str:
    lines: list[str] = []
    lines.append("=== OBS-051 Local Divergence Summary ===")
    lines.append("")
    lines.append(f"coupling_class = {cfg.coupling_class}")
    lines.append(f"k_neighbors = {cfg.k_neighbors}")
    lines.append(f"max_initial_distance = {cfg.max_initial_distance:.6f}")
    lines.append(f"n_windows = {len(states)}")
    lines.append(f"n_window_pairs = {len(div)}")
    lines.append("")

    if not out_summary.empty:
        lines.append("Outcome summary")
        for _, row in out_summary.iterrows():
            lines.append(
                f"{row['outcome_group']}: "
                f"n_windows={int(row['n_windows'])}, "
                f"mean_lambda_local={row['mean_lambda_local']:.6f}, "
                f"median_lambda_local={row['median_lambda_local']:.6f}, "
                f"mean_d_start={row['mean_d_start']:.6f}, "
                f"mean_d_end={row['mean_d_end']:.6f}"
            )

        rec = out_summary[out_summary["outcome_group"] == "recovering"]
        non = out_summary[out_summary["outcome_group"] == "nonrecovering"]
        if len(rec) and len(non):
            lines.append("")
            lines.append(
                "recovering_vs_nonrecovering: "
                f"mean_lambda_local_diff="
                f"{float(rec['mean_lambda_local'].iloc[0]) - float(non['mean_lambda_local'].iloc[0]):.6f}"
            )

    if not fam_summary.empty:
        lines.append("")
        lines.append("Family summary")
        for _, row in fam_summary.iterrows():
            lines.append(
                f"{row['outcome_group']} | {row['path_family']} | {row['seam_band']}: "
                f"n_windows={int(row['n_windows'])}, "
                f"mean_lambda_local={row['mean_lambda_local']:.6f}, "
                f"median_lambda_local={row['median_lambda_local']:.6f}, "
                f"mean_d_start={row['mean_d_start']:.6f}, "
                f"mean_d_end={row['mean_d_end']:.6f}"
            )

    return "\n".join(lines)


def plot_lambda_boxplot(win: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []
    groups = []

    for group in ["recovering", "nonrecovering", "other"]:
        vals = pd.to_numeric(
            win.loc[win["outcome_group"] == group, "mean_lambda_local"],
            errors="coerce",
        ).dropna()
        if len(vals) == 0:
            continue
        labels.append(group)
        groups.append(vals.to_numpy(dtype=float))

    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("mean_lambda_local")
    ax.set_title("OBS-051: local divergence by outcome")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_lambda_scatter(win: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = win.dropna(subset=["mean_d_start", "mean_lambda_local", "outcome_group"]).copy()

    for group in ["recovering", "nonrecovering", "other"]:
        sub = plot_df[plot_df["outcome_group"] == group]
        if len(sub) == 0:
            continue
        ax.scatter(sub["mean_d_start"], sub["mean_lambda_local"], s=24, label=group)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("mean_d_start")
    ax.set_ylabel("mean_lambda_local")
    ax.set_title("OBS-051: local divergence vs initial separation")
    if not plot_df.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBS-051 local divergence in coupled escalation windows.")
    parser.add_argument("--segments-csv", default=Config.segments_csv)
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--family-csv", default=Config.family_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--coupling-class", default=Config.coupling_class)
    parser.add_argument("--k-neighbors", type=int, default=Config.k_neighbors)
    parser.add_argument("--max-initial-distance", type=float, default=Config.max_initial_distance)
    parser.add_argument("--eps", type=float, default=Config.eps)
    parser.add_argument("--min-dt", type=float, default=Config.min_dt)
    args = parser.parse_args()

    cfg = Config(
        segments_csv=args.segments_csv,
        nodes_csv=args.nodes_csv,
        family_csv=args.family_csv,
        outdir=args.outdir,
        coupling_class=args.coupling_class,
        k_neighbors=args.k_neighbors,
        max_initial_distance=args.max_initial_distance,
        eps=args.eps,
        min_dt=args.min_dt,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    segs, nodes, fam = load_inputs(cfg)
    segs = add_coupling_class(segs)
    segs = segs[segs["coupling_class"] == cfg.coupling_class].copy()

    states = build_window_states(segs, nodes)
    div = build_divergence_table(states, cfg)
    win = build_window_summary(div)
    out_summary = build_outcome_summary(win)
    fam_summary = build_family_summary(win)

    win.to_csv(outdir / "obs051_window_divergence.csv", index=False)
    out_summary.to_csv(outdir / "obs051_outcome_summary.csv", index=False)
    fam_summary.to_csv(outdir / "obs051_family_summary.csv", index=False)

    summary_txt = summarize_text(states, div, win, out_summary, fam_summary, cfg)
    (outdir / "obs051_local_divergence_summary.txt").write_text(summary_txt, encoding="utf-8")

    plot_lambda_boxplot(win, outdir / "obs051_lambda_boxplot_by_outcome.png")
    plot_lambda_scatter(win, outdir / "obs051_lambda_vs_initial_separation.png")

    print(outdir / "obs051_window_divergence.csv")
    print(outdir / "obs051_outcome_summary.csv")
    print(outdir / "obs051_family_summary.csv")
    print(outdir / "obs051_local_divergence_summary.txt")
    print(outdir / "obs051_lambda_boxplot_by_outcome.png")
    print(outdir / "obs051_lambda_vs_initial_separation.png")


if __name__ == "__main__":
    main()
