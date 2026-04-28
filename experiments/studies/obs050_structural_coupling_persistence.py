#!/usr/bin/env python3
from __future__ import annotations

"""
OBS-050 — Structural coupling persistence during roughness escalation.

Refinement
----------
This version treats seam proximity during escalation windows as the primary
structural-coupling observable, with seam-distance slope as a secondary
posture variable.

Core question
-------------
When trajectories enter roughness-escalation regimes, do paths that later
recover remain concentrated in seam-core / seam-near bands while paths that
fail occupy farther seam bands?
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
    substrate_root: str = "outputs/scales/100000/family_substrate"
    outdir: str = "outputs/obs050_structural_coupling_persistence"
    roughness_floor: float = 30.0
    slope_quantile: float = 0.75
    seam_slope_eps: float = 1e-3
    smooth_window: int = 5
    pre_steps: int = 2
    post_steps: int = 3
    dedup_gap: int = 3
    seam_core_max: float = 0.05
    seam_near_max: float = 0.15


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}\n"
            "Hint: run experiments/toy/build_scale_family_substrate.py first."
        )
    return pd.read_csv(path).copy()


def to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def rolling_mean_by_path(df: pd.DataFrame, value_col: str, window: int) -> pd.Series:
    if window <= 1:
        return pd.to_numeric(df[value_col], errors="coerce")
    return (
        df.groupby("path_id", dropna=False)[value_col]
        .transform(
            lambda s: pd.to_numeric(s, errors="coerce")
            .rolling(window, min_periods=1, center=True)
            .mean()
        )
    )


def centered_local_slope(values: pd.Series, window: int) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    half = max(1, window // 2)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        y = vals[lo:hi]
        x = np.arange(lo, hi, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        out[i] = np.polyfit(x[mask], y[mask], 1)[0]

    return pd.Series(out, index=values.index)


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    if np.allclose(x, x[0]):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def classify_outcome(path_family: str) -> str:
    if path_family in RECOVERING_FAMILIES:
        return "recovering"
    if path_family in NONRECOVERING_FAMILIES:
        return "nonrecovering"
    return "other"


def classify_posture(m_r: float, m_seam: float, eps: float) -> str:
    if not np.isfinite(m_r) or m_r <= 0:
        return "not_escalating"
    if not np.isfinite(m_seam):
        return "unknown"
    if m_seam < -eps:
        return "compression"
    if abs(m_seam) <= eps:
        return "graze"
    return "dissipation"


def classify_seam_band(mean_distance_to_seam: float, core_max: float, near_max: float) -> str:
    if not np.isfinite(mean_distance_to_seam):
        return "unknown"
    if mean_distance_to_seam <= core_max:
        return "core"
    if mean_distance_to_seam <= near_max:
        return "near"
    return "far"


def load_inputs(substrate_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes_csv = substrate_root / "path_node_diagnostics.csv"
    paths_csv = substrate_root / "path_diagnostics.csv"
    fam_csv = substrate_root / "path_family_assignments.csv"

    nodes = safe_read_csv(nodes_csv)
    paths = safe_read_csv(paths_csv)
    fam = safe_read_csv(fam_csv)

    required_nodes = {"path_id", "step", "distance_to_seam", "path_angle_jump_deg"}
    missing_nodes = required_nodes - set(nodes.columns)
    if missing_nodes:
        raise ValueError(f"path_node_diagnostics missing required columns: {sorted(missing_nodes)}")

    required_fam = {"path_id", "path_family"}
    missing_fam = required_fam - set(fam.columns)
    if missing_fam:
        raise ValueError(f"path_family_assignments missing required columns: {sorted(missing_fam)}")

    nodes["path_id"] = nodes["path_id"].astype(str)
    paths["path_id"] = paths["path_id"].astype(str)
    fam["path_id"] = fam["path_id"].astype(str)

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
    return nodes, paths, fam


def extract_roughness_escalation_windows(
    nodes: pd.DataFrame,
    fam: pd.DataFrame,
    roughness_floor: float,
    slope_threshold: float,
    seam_slope_eps: float,
    smooth_window: int,
    pre_steps: int,
    post_steps: int,
    dedup_gap: int,
    seam_core_max: float,
    seam_near_max: float,
) -> pd.DataFrame:
    work = nodes.copy()
    work["path_id"] = work["path_id"].astype(str)
    work = work.merge(fam[["path_id", "path_family"]].drop_duplicates(), on="path_id", how="left")
    work = work.sort_values(["path_id", "step"]).reset_index(drop=True)

    work["roughness_smoothed"] = rolling_mean_by_path(work, "path_angle_jump_deg", smooth_window)
    work["roughness_slope_local"] = (
        work.groupby("path_id", dropna=False)["roughness_smoothed"]
        .transform(lambda s: centered_local_slope(s, smooth_window))
    )

    rows: list[dict[str, object]] = []

    for path_id, sub in work.groupby("path_id", dropna=False, sort=False):
        sub = sub.sort_values("step").reset_index(drop=True)
        family = str(sub["path_family"].iloc[0]) if "path_family" in sub.columns else "unknown"
        outcome_group = classify_outcome(family)

        rough = pd.to_numeric(sub["roughness_smoothed"], errors="coerce")
        slope = pd.to_numeric(sub["roughness_slope_local"], errors="coerce")

        candidate_mask = (
            rough.ge(roughness_floor)
            & slope.ge(slope_threshold)
            & rough.notna()
            & slope.notna()
        )

        candidate_idx = np.flatnonzero(candidate_mask.to_numpy(dtype=bool))
        if len(candidate_idx) == 0:
            continue

        chosen: list[int] = []
        cluster: list[int] = [int(candidate_idx[0])]

        for idx in candidate_idx[1:]:
            idx = int(idx)
            if idx - cluster[-1] <= dedup_gap:
                cluster.append(idx)
            else:
                best = max(cluster, key=lambda j: float(slope.iloc[j]) if pd.notna(slope.iloc[j]) else -np.inf)
                chosen.append(best)
                cluster = [idx]

        if cluster:
            best = max(cluster, key=lambda j: float(slope.iloc[j]) if pd.notna(slope.iloc[j]) else -np.inf)
            chosen.append(best)

        for win_no, center in enumerate(chosen, start=1):
            i0 = max(0, center - pre_steps)
            i1 = min(len(sub) - 1, center + post_steps)
            seg = sub.iloc[i0 : i1 + 1].copy()

            x = pd.to_numeric(seg["step"], errors="coerce").to_numpy(dtype=float)
            rough_raw = pd.to_numeric(seg["path_angle_jump_deg"], errors="coerce").to_numpy(dtype=float)
            rough_smooth = pd.to_numeric(seg["roughness_smoothed"], errors="coerce").to_numpy(dtype=float)
            seam = pd.to_numeric(seg["distance_to_seam"], errors="coerce").to_numpy(dtype=float)

            m_r = fit_slope(x, rough_smooth)
            m_seam = fit_slope(x, seam)
            if not np.isfinite(m_r) or abs(m_r) < 1e-9:
                continue

            mean_distance_to_seam = float(np.nanmean(seam)) if np.isfinite(seam).any() else np.nan
            min_distance_to_seam = float(np.nanmin(seam)) if np.isfinite(seam).any() else np.nan

            row = {
                "path_id": str(path_id),
                "path_family": family,
                "outcome_group": outcome_group,
                "segment_id": f"{path_id}::win{win_no:03d}",
                "center_step": float(sub["step"].iloc[center]),
                "start_step": float(seg["step"].iloc[0]),
                "end_step": float(seg["step"].iloc[-1]),
                "segment_len": int(len(seg)),
                "mean_roughness": float(np.nanmean(rough_raw)) if np.isfinite(rough_raw).any() else np.nan,
                "mean_roughness_smoothed": float(np.nanmean(rough_smooth)) if np.isfinite(rough_smooth).any() else np.nan,
                "max_roughness": float(np.nanmax(rough_raw)) if np.isfinite(rough_raw).any() else np.nan,
                "mean_distance_to_seam": mean_distance_to_seam,
                "min_distance_to_seam": min_distance_to_seam,
                "center_distance_to_seam": float(sub["distance_to_seam"].iloc[center]) if pd.notna(sub["distance_to_seam"].iloc[center]) else np.nan,
                "m_r": m_r,
                "m_seam": m_seam,
                "center_roughness_slope": float(slope.iloc[center]) if pd.notna(slope.iloc[center]) else np.nan,
                "posture": classify_posture(m_r, m_seam, seam_slope_eps),
                "seam_band": classify_seam_band(mean_distance_to_seam, seam_core_max, seam_near_max),
            }

            if "criticality" in seg.columns:
                crit = pd.to_numeric(seg["criticality"], errors="coerce").to_numpy(dtype=float)
                row["mean_criticality"] = float(np.nanmean(crit)) if np.isfinite(crit).any() else np.nan
                row["max_criticality"] = float(np.nanmax(crit)) if np.isfinite(crit).any() else np.nan

            obstruction_col = None
            for c in ["obstruction_mean_abs_holonomy", "absolute_holonomy_node"]:
                if c in seg.columns:
                    obstruction_col = c
                    break
            if obstruction_col is not None:
                obs = pd.to_numeric(seg[obstruction_col], errors="coerce").to_numpy(dtype=float)
                row["mean_obstruction"] = float(np.nanmean(obs)) if np.isfinite(obs).any() else np.nan
                row["max_obstruction"] = float(np.nanmax(obs)) if np.isfinite(obs).any() else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def coupling_effect_sizes(coupling_summary: pd.DataFrame) -> dict[str, float]:
    if coupling_summary.empty:
        return {}

    def _count(group: str, cls: str) -> float:
        sub = coupling_summary[
            (coupling_summary["outcome_group"] == group)
            & (coupling_summary["coupling_class"] == cls)
        ]
        if len(sub) == 0:
            return 0.0
        return float(pd.to_numeric(sub["n_segments"], errors="coerce").iloc[0])

    rec_c = _count("recovering", "coupled")
    rec_d = _count("recovering", "decoupled")
    non_c = _count("nonrecovering", "coupled")
    non_d = _count("nonrecovering", "decoupled")

    out: dict[str, float] = {}

    rec_total = rec_c + rec_d
    non_total = non_c + non_d

    if rec_total > 0:
        out["recovering_coupled_share"] = rec_c / rec_total
    if non_total > 0:
        out["nonrecovering_coupled_share"] = non_c / non_total

    if rec_total > 0 and non_total > 0 and non_c > 0:
        out["coupled_risk_ratio"] = (rec_c / rec_total) / (non_c / non_total)

    # Haldane-Anscombe style continuity correction for robustness
    a = rec_c + 0.5
    b = rec_d + 0.5
    c = non_c + 0.5
    d = non_d + 0.5
    out["coupled_odds_ratio"] = (a * d) / (b * c)

    return out


def build_path_summary(segs: pd.DataFrame, path_diag: pd.DataFrame, fam: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        base = fam[["path_id", "path_family"]].drop_duplicates().copy()
        base["outcome_group"] = base["path_family"].map(classify_outcome)
        base["n_escalation_windows"] = 0
        return base

    agg = (
        segs.groupby(["path_id", "path_family", "outcome_group"], dropna=False)
        .agg(
            n_escalation_windows=("segment_id", "size"),
            mean_segment_m_r=("m_r", "mean"),
            mean_segment_m_seam=("m_seam", "mean"),
            mean_segment_mean_distance=("mean_distance_to_seam", "mean"),
            mean_segment_min_distance=("min_distance_to_seam", "mean"),
            core_windows=("seam_band", lambda s: int((s == "core").sum())),
            near_windows=("seam_band", lambda s: int((s == "near").sum())),
            far_windows=("seam_band", lambda s: int((s == "far").sum())),
            n_compression=("posture", lambda s: int((s == "compression").sum())),
            n_graze=("posture", lambda s: int((s == "graze").sum())),
            n_dissipation=("posture", lambda s: int((s == "dissipation").sum())),
        )
        .reset_index()
    )

    out = path_diag.copy()
    out["path_id"] = out["path_id"].astype(str)
    out = out.merge(
        fam[["path_id", "path_family"]].drop_duplicates(),
        on="path_id",
        how="left",
        suffixes=("", "_fam"),
    )
    if "path_family_fam" in out.columns:
        out["path_family"] = out["path_family"].fillna(out["path_family_fam"])
        out = out.drop(columns=["path_family_fam"])
    out["outcome_group"] = out["path_family"].map(classify_outcome)
    out = out.merge(agg, on=["path_id", "path_family", "outcome_group"], how="left")

    for col in [
        "n_escalation_windows", "core_windows", "near_windows", "far_windows",
        "n_compression", "n_graze", "n_dissipation",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out


def build_family_summary(segs: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame()

    return (
        segs.groupby(["outcome_group", "path_family", "seam_band", "posture"], dropna=False)
        .agg(
            n_segments=("segment_id", "size"),
            mean_m_r=("m_r", "mean"),
            mean_m_seam=("m_seam", "mean"),
            mean_roughness=("mean_roughness", "mean"),
            mean_mean_distance_to_seam=("mean_distance_to_seam", "mean"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
        )
        .reset_index()
        .sort_values(["outcome_group", "path_family", "seam_band", "posture"])
        .reset_index(drop=True)
    )


def build_seam_band_summary(segs: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame()

    out = (
        segs.groupby(["outcome_group", "seam_band"], dropna=False)
        .agg(
            n_segments=("segment_id", "size"),
            mean_m_seam=("m_seam", "mean"),
            mean_mean_distance_to_seam=("mean_distance_to_seam", "mean"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
        )
        .reset_index()
    )

    total_by_group = out.groupby("outcome_group")["n_segments"].transform("sum")
    out["segment_share"] = out["n_segments"] / total_by_group
    return out.sort_values(["outcome_group", "seam_band"]).reset_index(drop=True)




def summarize_text(
    segs: pd.DataFrame,
    path_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    seam_band_summary: pd.DataFrame,
    coupling_summary: pd.DataFrame,
    effect_sizes: dict[str, float],
    roughness_floor: float,
    slope_threshold: float,
) -> str:
    lines: list[str] = []
    lines.append("=== OBS-050 Structural Coupling Persistence Summary ===")
    lines.append("")
    lines.append("window_type = roughness_escalation")
    lines.append(f"roughness_threshold = {roughness_floor:.6f}")
    lines.append(f"slope_threshold = {slope_threshold:.6f}")
    lines.append(f"n_segments = {len(segs)}")
    lines.append(f"n_paths = {path_summary['path_id'].nunique() if 'path_id' in path_summary.columns else 0}")
    lines.append("")

    if not segs.empty:
        for group in ["recovering", "nonrecovering", "other"]:
            sub = segs[segs["outcome_group"] == group].copy()
            if len(sub) == 0:
                continue
            lines.append(
                f"{group}: "
                f"n_segments={len(sub)}, "
                f"mean_m_r={pd.to_numeric(sub['m_r'], errors='coerce').mean():.6f}, "
                f"mean_m_seam={pd.to_numeric(sub['m_seam'], errors='coerce').mean():.6f}, "
                f"mean_mean_distance_to_seam={pd.to_numeric(sub['mean_distance_to_seam'], errors='coerce').mean():.6f}, "
                f"mean_min_distance_to_seam={pd.to_numeric(sub['min_distance_to_seam'], errors='coerce').mean():.6f}"
            )

        lines.append("")

        rec = segs[segs["outcome_group"] == "recovering"].copy()
        non = segs[segs["outcome_group"] == "nonrecovering"].copy()
        if len(rec) and len(non):
            for col in ["m_seam", "mean_distance_to_seam", "min_distance_to_seam"]:
                r = pd.to_numeric(rec[col], errors="coerce").dropna()
                n = pd.to_numeric(non[col], errors="coerce").dropna()
                if len(r) and len(n):
                    lines.append(f"recovering_vs_nonrecovering: mean_{col}_diff={(r.mean() - n.mean()):.6f}")
                    lines.append(f"recovering_vs_nonrecovering: median_{col}_diff={(r.median() - n.median()):.6f}")

        lines.append("")
        lines.append("Seam-band summary")
        for _, row in seam_band_summary.iterrows():
            lines.append(
                f"{row['outcome_group']} | {row['seam_band']}: "
                f"n_segments={int(row['n_segments'])}, "
                f"segment_share={row['segment_share']:.4f}, "
                f"mean_m_seam={row['mean_m_seam']:.6f}, "
                f"mean_mean_distance_to_seam={row['mean_mean_distance_to_seam']:.6f}, "
                f"mean_min_distance_to_seam={row['mean_min_distance_to_seam']:.6f}"
            )

    if not coupling_summary.empty:
        lines.append("")
        lines.append("Coupled vs decoupled summary")
        for _, row in coupling_summary.iterrows():
            lines.append(
                f"{row['outcome_group']} | {row['coupling_class']}: "
                f"n_segments={int(row['n_segments'])}, "
                f"segment_share={row['segment_share']:.4f}"
            )

    if effect_sizes:
        lines.append("")
        lines.append("Coupling effect sizes")
        if "recovering_coupled_share" in effect_sizes:
            lines.append(
                f"recovering_coupled_share={effect_sizes['recovering_coupled_share']:.6f}"
            )
        if "nonrecovering_coupled_share" in effect_sizes:
            lines.append(
                f"nonrecovering_coupled_share={effect_sizes['nonrecovering_coupled_share']:.6f}"
            )
        if "coupled_risk_ratio" in effect_sizes:
            lines.append(
                f"coupled_risk_ratio={effect_sizes['coupled_risk_ratio']:.6f}"
            )
        if "coupled_odds_ratio" in effect_sizes:
            lines.append(
                f"coupled_odds_ratio={effect_sizes['coupled_odds_ratio']:.6f}"
            )


    if not family_summary.empty:
        lines.append("")
        lines.append("Family/seam-band/posture summary")
        for _, row in family_summary.iterrows():
            lines.append(
                f"{row['outcome_group']} | {row['path_family']} | {row['seam_band']} | {row['posture']}: "
                f"n_segments={int(row['n_segments'])}, "
                f"mean_m_r={row['mean_m_r']:.6f}, "
                f"mean_m_seam={row['mean_m_seam']:.6f}, "
                f"mean_roughness={row['mean_roughness']:.6f}, "
                f"mean_mean_distance_to_seam={row['mean_mean_distance_to_seam']:.6f}, "
                f"mean_min_distance_to_seam={row['mean_min_distance_to_seam']:.6f}"
            )

    return "\n".join(lines)


def build_coupling_summary(segs: pd.DataFrame) -> pd.DataFrame:
    if segs.empty:
        return pd.DataFrame()

    work = segs.copy()
    work["coupling_class"] = np.where(
        work["seam_band"].isin(["core", "near"]),
        "coupled",
        np.where(work["seam_band"] == "far", "decoupled", "unknown"),
    )

    out = (
        work.groupby(["outcome_group", "coupling_class"], dropna=False)
        .agg(n_segments=("segment_id", "size"))
        .reset_index()
    )

    total_by_group = out.groupby("outcome_group")["n_segments"].transform("sum")
    out["segment_share"] = out["n_segments"] / total_by_group
    return out.sort_values(["outcome_group", "coupling_class"]).reset_index(drop=True)


def plot_scatter_mseam_vs_mean_distance(segs: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = segs.dropna(subset=["m_seam", "mean_distance_to_seam", "outcome_group"]).copy()

    for group in ["recovering", "nonrecovering", "other"]:
        sub = plot_df[plot_df["outcome_group"] == group]
        if len(sub) == 0:
            continue
        ax.scatter(sub["m_seam"], sub["mean_distance_to_seam"], s=24, label=group)

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("m_seam")
    ax.set_ylabel("mean_distance_to_seam")
    ax.set_title("OBS-050: seam-distance slope vs mean seam distance")
    if not plot_df.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_seam_band_counts(seam_band_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if seam_band_summary.empty:
        fig.tight_layout()
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        return

    band_order = ["core", "near", "far", "unknown"]
    group_order = ["recovering", "nonrecovering", "other"]

    pivot = (
        seam_band_summary.pivot(index="seam_band", columns="outcome_group", values="segment_share")
        .reindex(index=band_order, columns=group_order)
        .fillna(0.0)
    )

    x = np.arange(len(pivot.index))
    width = 0.25

    for i, group in enumerate(group_order):
        ax.bar(x + (i - 1) * width, pivot[group].to_numpy(dtype=float), width=width, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel("segment_share")
    ax.set_title("OBS-050: seam-band distribution by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_boxplot_mseam_by_outcome(segs: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    labels: list[str] = []
    groups: list[np.ndarray] = []

    for group in ["recovering", "nonrecovering", "other"]:
        vals = pd.to_numeric(segs.loc[segs["outcome_group"] == group, "m_seam"], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        labels.append(group)
        groups.append(vals.to_numpy(dtype=float))

    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("m_seam")
    ax.set_title("OBS-050: seam-distance slope by outcome group")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_coupling_class_distribution(coupling_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if coupling_summary.empty:
        fig.tight_layout()
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        return

    order = ["recovering", "nonrecovering", "other"]
    coupled = []
    decoupled = []

    for group in order:
        sub = coupling_summary[coupling_summary["outcome_group"] == group]
        if len(sub) == 0:
            continue
        c = sub.loc[sub["coupling_class"] == "coupled", "segment_share"]
        d = sub.loc[sub["coupling_class"] == "decoupled", "segment_share"]
        coupled.append(float(c.iloc[0]) if len(c) else 0.0)
        decoupled.append(float(d.iloc[0]) if len(d) else 0.0)

    groups = [g for g in order if len(coupling_summary[coupling_summary["outcome_group"] == g]) > 0]
    x = np.arange(len(groups))

    ax.bar(x, coupled, label="coupled")
    ax.bar(x, decoupled, bottom=coupled, label="decoupled")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("segment_share")
    ax.set_title("OBS-050: coupled vs decoupled by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBS-050 structural coupling persistence during roughness escalation."
    )
    parser.add_argument("--substrate-root", default="outputs/scales/100000/family_substrate")
    parser.add_argument("--outdir", default="outputs/obs050_structural_coupling_persistence")
    parser.add_argument("--roughness-floor", type=float, default=30.0)
    parser.add_argument("--slope-quantile", type=float, default=0.75)
    parser.add_argument("--seam-slope-eps", type=float, default=1e-3)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--pre-steps", type=int, default=2)
    parser.add_argument("--post-steps", type=int, default=3)
    parser.add_argument("--dedup-gap", type=int, default=3)
    parser.add_argument("--seam-core-max", type=float, default=0.05)
    parser.add_argument("--seam-near-max", type=float, default=0.15)
    args = parser.parse_args()

    cfg = Config(
        substrate_root=args.substrate_root,
        outdir=args.outdir,
        roughness_floor=args.roughness_floor,
        slope_quantile=args.slope_quantile,
        seam_slope_eps=args.seam_slope_eps,
        smooth_window=args.smooth_window,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        dedup_gap=args.dedup_gap,
        seam_core_max=args.seam_core_max,
        seam_near_max=args.seam_near_max,
    )

    substrate_root = Path(cfg.substrate_root)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, path_diag, fam = load_inputs(substrate_root)

    tmp = nodes.copy()
    tmp["path_id"] = tmp["path_id"].astype(str)
    tmp = tmp.sort_values(["path_id", "step"]).reset_index(drop=True)
    tmp["roughness_smoothed"] = rolling_mean_by_path(tmp, "path_angle_jump_deg", cfg.smooth_window)
    tmp["roughness_slope_local"] = (
        tmp.groupby("path_id", dropna=False)["roughness_smoothed"]
        .transform(lambda s: centered_local_slope(s, cfg.smooth_window))
    )
    slope_vals = pd.to_numeric(tmp["roughness_slope_local"], errors="coerce").dropna()
    if len(slope_vals) == 0:
        raise ValueError("No valid local roughness slopes found.")
    slope_threshold = float(slope_vals.quantile(cfg.slope_quantile))

    segs = extract_roughness_escalation_windows(
        nodes=nodes,
        fam=fam,
        roughness_floor=cfg.roughness_floor,
        slope_threshold=slope_threshold,
        seam_slope_eps=cfg.seam_slope_eps,
        smooth_window=cfg.smooth_window,
        pre_steps=cfg.pre_steps,
        post_steps=cfg.post_steps,
        dedup_gap=cfg.dedup_gap,
        seam_core_max=cfg.seam_core_max,
        seam_near_max=cfg.seam_near_max,
    )

    path_summary = build_path_summary(segs, path_diag, fam)
    family_summary = build_family_summary(segs)
    seam_band_summary = build_seam_band_summary(segs)
    coupling_summary = build_coupling_summary(segs)
    effect_sizes = coupling_effect_sizes(coupling_summary)

    segs.to_csv(outdir / "structural_coupling_segments.csv", index=False)
    path_summary.to_csv(outdir / "structural_coupling_path_summary.csv", index=False)
    family_summary.to_csv(outdir / "structural_coupling_family_summary.csv", index=False)
    seam_band_summary.to_csv(outdir / "structural_coupling_seam_band_summary.csv", index=False)
    coupling_summary.to_csv(outdir / "structural_coupling_coupled_vs_decoupled_summary.csv", index=False)


    summary_txt = summarize_text(
        segs=segs,
        path_summary=path_summary,
        family_summary=family_summary,
        seam_band_summary=seam_band_summary,
        coupling_summary=coupling_summary,
        effect_sizes=effect_sizes,
        roughness_floor=cfg.roughness_floor,
        slope_threshold=slope_threshold,
    )
    (outdir / "obs050_structural_coupling_persistence_summary.txt").write_text(summary_txt, encoding="utf-8")

    plot_scatter_mseam_vs_mean_distance(segs, outdir / "obs050_m_seam_vs_mean_distance_to_seam.png")
    plot_seam_band_counts(seam_band_summary, outdir / "obs050_seam_band_distribution_by_outcome.png")
    plot_boxplot_mseam_by_outcome(segs, outdir / "obs050_m_seam_boxplot_by_outcome.png")
    plot_coupling_class_distribution(coupling_summary, outdir / "obs050_coupled_vs_decoupled_by_outcome.png")

    print(outdir / "structural_coupling_segments.csv")
    print(outdir / "structural_coupling_path_summary.csv")
    print(outdir / "structural_coupling_family_summary.csv")
    print(outdir / "structural_coupling_seam_band_summary.csv")
    print(outdir / "obs050_structural_coupling_persistence_summary.txt")
    print(outdir / "obs050_m_seam_vs_mean_distance_to_seam.png")
    print(outdir / "obs050_seam_band_distribution_by_outcome.png")
    print(outdir / "obs050_m_seam_boxplot_by_outcome.png")
    print(outdir / "obs050_coupled_vs_decoupled_by_outcome.png")


if __name__ == "__main__":
    main()