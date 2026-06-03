#!/usr/bin/env python3
"""
OBS-073 — Continuous-field groupoid reduction, v5.

Purpose
-------
Test whether symbolic / proto-groupoid route classes are recoverable from
continuous field geometry.

v5 patch
--------
v3 added shortcut-resistance tests for seam proximity and absolute grid-location.
v4 added OBS-072 locus targets connecting the Cp2 false-recovery compression
mode directly to the continuous-field reduction test.
v5 adds label-shuffle null controls for OBS-072 locus targets.

The label-shuffle control keeps the same eligible Cp2 row pool, same feature
matrix, same class balance, and same CV protocol, but randomly permutes labels.
This asks whether high OBS-072 separability survives only with the real semantic
assignment, rather than being a row-pool / class-imbalance / model artifact.

New OBS-072 target family
-------------------------
6A. obs072_cp2_locus_false_recovery
6B. obs072_cp2_locus_false_recovery_no_direct_seam
6C. obs072_cp2_locus_false_recovery_no_grid
6D. obs072_cp2_locus_false_recovery_no_direct_seam_no_grid

These targets are within-Cp2-only by default. They compare:

  false_recovery_compression_locus:
    corpus == Cp2
    outcome_group == nonrecovering
    coupling_class == coupled
    path_family == off_seam_reorganizing
    seam_band == near
    posture == compression
    center r ~= 0.2
    center alpha ~= 0.1328571428571428

against:

  true_bounded_recovery:
    corpus == Cp2
    outcome_group == recovering
    coupling_class == coupled

Interpretation
--------------
If the OBS-072 locus target survives no_direct_seam_no_grid, the false-recovery
mode has a broader continuous-field signature inside Cp2. If it collapses only
under no_grid_location, it is primarily a localized hotspot. If it collapses
under no_direct_seam, it is primarily seam-proximity defined.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRoot:
    label: str
    root: Path
    scale: str

    @property
    def family_substrate_dir(self) -> Path:
        return self.root / "scales" / self.scale / "family_substrate"

    @property
    def path_node_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_node_diagnostics.csv"

    @property
    def path_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_diagnostics.csv"

    @property
    def family_assignments_csv(self) -> Path:
        return self.family_substrate_dir / "path_family_assignments.csv"

    @property
    def obs050_segments_csv(self) -> Path:
        return self.root / "obs050_structural_coupling_persistence" / "structural_coupling_segments.csv"

    @property
    def obs051_dir(self) -> Path:
        return self.root / "obs051_local_divergence_in_coupled_windows"


@dataclass(frozen=True)
class Config:
    corpora: list[CorpusRoot]
    outdir: Path
    random_state: int = 73
    min_class_count: int = 20
    max_rows_per_corpus: int | None = None
    rf_n_estimators: int = 500
    rf_max_depth: int = 4
    rf_min_samples_leaf: int = 20
    permutation_repeats: int = 8
    label_shuffle_repeats: int = 20
    obs072_locus_corpus: str = "Cp2"
    obs072_locus_r: float = 0.2
    obs072_locus_alpha: float = 0.1328571428571428
    obs072_locus_r_tol: float = 1e-9
    obs072_locus_alpha_tol: float = 1e-6
    obs072_center_max_delta: int = 2


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target_col: str
    no_direct_seam: bool = False
    no_grid_location: bool = False
    allow_obs051_features: bool = False
    within_corpus_only: bool = False
    corpus_filter: str | None = None
    obs072_locus_target: bool = False
    description: str = ""


TARGET_SPECS = [
    # 1. path family
    TargetSpec("path_family", "target_path_family"),
    TargetSpec("path_family_no_direct_seam", "target_path_family", no_direct_seam=True),
    TargetSpec("path_family_no_grid", "target_path_family", no_grid_location=True),
    TargetSpec(
        "path_family_no_direct_seam_no_grid",
        "target_path_family",
        no_direct_seam=True,
        no_grid_location=True,
    ),

    # 2. coupling
    TargetSpec("coupling_class_full", "target_coupling_class"),
    TargetSpec("coupling_class_no_direct_seam", "target_coupling_class", no_direct_seam=True),
    TargetSpec("coupling_class_no_grid", "target_coupling_class", no_grid_location=True),
    TargetSpec(
        "coupling_class_no_direct_seam_no_grid",
        "target_coupling_class",
        no_direct_seam=True,
        no_grid_location=True,
    ),

    # 3. outcome
    TargetSpec("outcome_group", "target_outcome_group"),
    TargetSpec("outcome_group_no_direct_seam", "target_outcome_group", no_direct_seam=True),
    TargetSpec("outcome_group_no_grid", "target_outcome_group", no_grid_location=True),
    TargetSpec(
        "outcome_group_no_direct_seam_no_grid",
        "target_outcome_group",
        no_direct_seam=True,
        no_grid_location=True,
    ),

    # 4. coupled outcome
    TargetSpec("coupled_outcome_group", "target_coupled_outcome_group"),
    TargetSpec(
        "coupled_outcome_group_no_direct_seam",
        "target_coupled_outcome_group",
        no_direct_seam=True,
    ),
    TargetSpec(
        "coupled_outcome_group_no_grid",
        "target_coupled_outcome_group",
        no_grid_location=True,
    ),
    TargetSpec(
        "coupled_outcome_group_no_direct_seam_no_grid",
        "target_coupled_outcome_group",
        no_direct_seam=True,
        no_grid_location=True,
    ),

    # 5. recovery channel
    TargetSpec("recovery_channel_structural", "target_recovery_channel_structural"),
    TargetSpec(
        "recovery_channel_boundedness_strict",
        "target_recovery_channel_boundedness_strict",
        allow_obs051_features=True,
    ),
    TargetSpec(
        "recovery_channel_no_direct_seam",
        "target_recovery_channel_structural",
        no_direct_seam=True,
    ),
    TargetSpec(
        "recovery_channel_no_grid",
        "target_recovery_channel_structural",
        no_grid_location=True,
    ),
    TargetSpec(
        "recovery_channel_no_direct_seam_no_grid",
        "target_recovery_channel_structural",
        no_direct_seam=True,
        no_grid_location=True,
    ),

    # 6. OBS-072 Cp2 localized false-recovery channel.
    TargetSpec(
        "obs072_cp2_locus_false_recovery",
        "target_obs072_cp2_locus_false_recovery",
        within_corpus_only=True,
        corpus_filter="Cp2",
        obs072_locus_target=True,
        description="OBS-072 Cp2 localized false-recovery compression locus vs Cp2 true bounded recovery.",
    ),
    TargetSpec(
        "obs072_cp2_locus_false_recovery_no_direct_seam",
        "target_obs072_cp2_locus_false_recovery",
        no_direct_seam=True,
        within_corpus_only=True,
        corpus_filter="Cp2",
        obs072_locus_target=True,
    ),
    TargetSpec(
        "obs072_cp2_locus_false_recovery_no_grid",
        "target_obs072_cp2_locus_false_recovery",
        no_grid_location=True,
        within_corpus_only=True,
        corpus_filter="Cp2",
        obs072_locus_target=True,
    ),
    TargetSpec(
        "obs072_cp2_locus_false_recovery_no_direct_seam_no_grid",
        "target_obs072_cp2_locus_false_recovery",
        no_direct_seam=True,
        no_grid_location=True,
        within_corpus_only=True,
        corpus_filter="Cp2",
        obs072_locus_target=True,
    ),
]


# ---------------------------------------------------------------------
# Feature provenance / leakage rules
# ---------------------------------------------------------------------


ALWAYS_EXCLUDE_EXACT = {
    "path_id",
    "corpus",
    "source_root",
    "scale",
    "path_family",
    "target_path_family",
    "target_coupling_class",
    "target_outcome_group",
    "target_coupled_outcome_group",
    "target_recovery_channel_structural",
    "target_recovery_channel_boundedness_strict",
    "target_obs072_cp2_locus_false_recovery",
    "tortuosity_coordinate_basis",
}

TARGET_DERIVED_PREFIXES = ("obs050_", "obs072_")
BOUNDEDNESS_PREFIXES = ("obs051_",)

DIRECT_SEAM_PATTERNS = [
    r"distance_to_seam",
    r"(^|_)seam($|_)",
    r"near_fraction",
    r"mid_fraction",
    r"far_fraction",
    r"core_fraction",
    r"coupled",
    r"coupling",
    r"m_seam",
    r"min_distance",
    r"mean_distance",
]

SYMBOLIC_PATTERNS = [
    r"family$",
    r"path_family",
    r"outcome",
    r"posture",
    r"seam_band",
    r"coupling_class",
]

GRID_LOCATION_PATTERNS = [
    r"^pn_node_id_x_(mean|std|min|max|median|sum)$",
    r"^pn_node_id_y_(mean|std|min|max|median|sum)$",
    r"^pn_r_(mean|std|min|max|median|sum)$",
    r"^pn_alpha_(mean|std|min|max|median|sum)$",
    r"^pn_mds1_(mean|std|min|max|median|sum)$",
    r"^pn_mds2_(mean|std|min|max|median|sum)$",
    r"^pd_start_",
    r"^pd_end_",
    r"^pd_initial_",
    r"^pd_final_",
    r"^pd_mean_r$",
    r"^pd_mean_alpha$",
    r"^pd_min_r$",
    r"^pd_max_r$",
    r"^pd_min_alpha$",
    r"^pd_max_alpha$",
]

TORTUOSITY_FEATURES = {
    "path_arclength",
    "path_chord_length",
    "path_tortuosity",
    "turning_abs_sum",
    "turning_signed_sum",
    "turning_number_proxy",
    "turning_mean_abs",
    "turning_max_abs",
    "net_displacement",
}


def matches_any(col: str, patterns: list[str]) -> bool:
    return any(re.search(p, col, flags=re.IGNORECASE) for p in patterns)


def classify_feature_provenance(col: str) -> str:
    if col in ALWAYS_EXCLUDE_EXACT or col.startswith("target_"):
        return "target_or_metadata"
    if col.startswith(TARGET_DERIVED_PREFIXES):
        return "obs050_or_obs072_target_derived"
    if col.startswith(BOUNDEDNESS_PREFIXES):
        return "obs051_boundedness"
    if matches_any(col, SYMBOLIC_PATTERNS):
        return "symbolic"
    if matches_any(col, DIRECT_SEAM_PATTERNS):
        return "direct_seam"
    if matches_any(col, GRID_LOCATION_PATTERNS):
        return "grid_location"
    if col in TORTUOSITY_FEATURES:
        return "trajectory_tortuosity"
    if col.endswith("_last_minus_first"):
        return "coordinate_delta_or_field_delta"
    if "criticality" in col.lower():
        return "criticality"
    if "lazarus" in col.lower():
        return "lazarus"
    if "holonomy" in col.lower() or "obstruction" in col.lower():
        return "holonomy_obstruction"
    if "fim" in col.lower():
        return "fim_invariant"
    if "response" in col.lower() or "rsp_" in col.lower():
        return "response_field"
    if "roughness" in col.lower():
        return "roughness"
    if col.startswith("pn_"):
        return "path_node_continuous"
    if col.startswith("pd_"):
        return "path_diagnostic_continuous"
    return "continuous_other"


def feature_allowed_for_target(col: str, spec: TargetSpec) -> tuple[bool, str]:
    prov = classify_feature_provenance(col)

    if prov in {"target_or_metadata", "symbolic"}:
        return False, prov

    if prov == "obs050_or_obs072_target_derived":
        return False, prov

    if prov == "obs051_boundedness" and not spec.allow_obs051_features:
        return False, prov

    if spec.no_direct_seam and prov == "direct_seam":
        return False, "direct_seam_blinded"

    if spec.no_grid_location and prov == "grid_location":
        return False, "grid_location_blinded"

    return True, prov


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required {label} exists but is empty: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def normalize_path_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "path_id" not in out.columns and "probe_id" in out.columns:
        out = out.rename(columns={"probe_id": "path_id"})
    if "path_id" in out.columns:
        out["path_id"] = out["path_id"].astype(str)
    return out


def normalize_outcome_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome_group" not in out.columns and "outcome" in out.columns:
        out = out.rename(columns={"outcome": "outcome_group"})
    return out


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def finite_mean(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.mean()) if y.notna().any() else float("nan")


def finite_std(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.std()) if y.notna().sum() > 1 else float("nan")


def finite_min(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.min()) if y.notna().any() else float("nan")


def finite_max(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.max()) if y.notna().any() else float("nan")


def finite_median(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce")
    return float(y.median()) if y.notna().any() else float("nan")


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return float(a / b)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Tortuosity engine
# ---------------------------------------------------------------------


def choose_coordinate_basis(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    cols = set(df.columns)
    if {"mds1", "mds2"}.issubset(cols):
        return "mds1", "mds2", "mds1_mds2"
    if {"x", "y"}.issubset(cols):
        return "x", "y", "xy"
    if {"r", "alpha"}.issubset(cols):
        return "r", "alpha", "r_alpha"
    return None, None, "none"


def angle_diff(theta2: np.ndarray, theta1: np.ndarray) -> np.ndarray:
    d = theta2 - theta1
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def compute_tortuosity_for_path(grp: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    g = grp.sort_values("step").copy()
    x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return {c: np.nan for c in TORTUOSITY_FEATURES}

    dx = np.diff(x)
    dy = np.diff(y)
    step_len = np.sqrt(dx * dx + dy * dy)

    arclength = float(np.nansum(step_len))
    chord = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    tortuosity = safe_ratio(arclength, chord)

    if len(dx) < 2:
        turning_abs_sum = 0.0
        turning_signed_sum = 0.0
        turning_mean_abs = 0.0
        turning_max_abs = 0.0
    else:
        theta = np.arctan2(dy, dx)
        dtheta = angle_diff(theta[1:], theta[:-1])
        abs_dtheta = np.abs(dtheta)
        turning_abs_sum = float(np.nansum(abs_dtheta))
        turning_signed_sum = float(np.nansum(dtheta))
        turning_mean_abs = float(np.nanmean(abs_dtheta)) if len(abs_dtheta) else 0.0
        turning_max_abs = float(np.nanmax(abs_dtheta)) if len(abs_dtheta) else 0.0

    return {
        "path_arclength": arclength,
        "path_chord_length": chord,
        "path_tortuosity": tortuosity,
        "turning_abs_sum": turning_abs_sum,
        "turning_signed_sum": turning_signed_sum,
        "turning_number_proxy": turning_signed_sum / (2.0 * np.pi),
        "turning_mean_abs": turning_mean_abs,
        "turning_max_abs": turning_max_abs,
        "net_displacement": chord,
    }


def compute_tortuosity_features(path_nodes: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if "step" not in path_nodes.columns:
        out = path_nodes[["path_id"]].drop_duplicates().copy()
        out["tortuosity_coordinate_basis"] = "none"
        for c in TORTUOSITY_FEATURES:
            out[c] = np.nan
        return out, "none"

    x_col, y_col, basis = choose_coordinate_basis(path_nodes)
    if x_col is None or y_col is None:
        out = path_nodes[["path_id"]].drop_duplicates().copy()
        out["tortuosity_coordinate_basis"] = "none"
        for c in TORTUOSITY_FEATURES:
            out[c] = np.nan
        return out, "none"

    rows = []
    for path_id, grp in path_nodes.groupby("path_id", sort=False):
        row: dict[str, Any] = {
            "path_id": str(path_id),
            "tortuosity_coordinate_basis": basis,
        }
        row.update(compute_tortuosity_for_path(grp, x_col, y_col))
        rows.append(row)

    return pd.DataFrame(rows), basis


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------


def load_base_tables(root: CorpusRoot) -> dict[str, pd.DataFrame]:
    return {
        "path_nodes": normalize_path_id(
            read_csv_required(root.path_node_diagnostics_csv, f"{root.label} path_node_diagnostics")
        ),
        "path_diag": normalize_path_id(
            read_csv_required(root.path_diagnostics_csv, f"{root.label} path_diagnostics")
        ),
        "family": normalize_path_id(
            read_csv_required(root.family_assignments_csv, f"{root.label} path_family_assignments")
        ),
        "obs050": normalize_path_id(
            normalize_outcome_column(
                read_csv_required(root.obs050_segments_csv, f"{root.label} OBS-050 segments")
            )
        ),
    }


def summarize_numeric_by_path(path_nodes: pd.DataFrame) -> pd.DataFrame:
    ignore = {
        "path_id",
        "probe_id",
        "step",
        "node_id",
        "path_family",
        "outcome_group",
        "coupling_class",
        "seam_band",
        "posture",
    }

    numeric_cols = []
    for c in path_nodes.columns:
        if c in ignore:
            continue
        s = pd.to_numeric(path_nodes[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)

    rows = []
    for path_id, grp in path_nodes.groupby("path_id", sort=False):
        row: dict[str, Any] = {"path_id": str(path_id)}
        for col in numeric_cols:
            s = pd.to_numeric(grp[col], errors="coerce")
            if not s.notna().any():
                continue

            prefix = f"pn_{col}"
            row[f"{prefix}_mean"] = finite_mean(s)
            row[f"{prefix}_std"] = finite_std(s)
            row[f"{prefix}_min"] = finite_min(s)
            row[f"{prefix}_max"] = finite_max(s)
            row[f"{prefix}_median"] = finite_median(s)
            row[f"{prefix}_sum"] = float(s.sum(skipna=True))

            nz = s.dropna()
            row[f"{prefix}_last_minus_first"] = (
                float(nz.iloc[-1] - nz.iloc[0]) if len(nz) >= 2 else np.nan
            )

        row["pn_n_steps"] = int(len(grp))
        rows.append(row)

    return pd.DataFrame(rows)


def reduce_path_diagnostics(path_diag: pd.DataFrame) -> pd.DataFrame:
    path_diag = normalize_path_id(path_diag)
    out = path_diag.drop_duplicates("path_id").copy()

    keep = ["path_id"]
    for c in out.columns:
        if c == "path_id" or c == "path_family":
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().any():
            keep.append(c)

    out = out[keep].copy()
    return out.rename(columns={c: f"pd_{c}" for c in out.columns if c != "path_id"})


def normalize_obs050_classes(obs050: pd.DataFrame) -> pd.DataFrame:
    df = normalize_path_id(normalize_outcome_column(obs050))

    if "seam_band" not in df.columns:
        mean_d = numeric_series(df, "mean_distance_to_seam")
        min_d = numeric_series(df, "min_distance_to_seam")
        df["seam_band"] = np.where(
            min_d <= 1e-12,
            "core",
            np.where(mean_d <= 0.15, "near", "far"),
        )

    if "coupling_class" not in df.columns:
        df["coupling_class"] = np.where(
            df["seam_band"].astype(str).isin(["core", "near"]),
            "coupled",
            np.where(df["seam_band"].astype(str).eq("far"), "decoupled", "unknown"),
        )

    if "posture" not in df.columns:
        m = numeric_series(df, "m_seam")
        df["posture"] = np.where(
            m < -0.02,
            "compression",
            np.where(m > 0.02, "dissipation", "graze"),
        )

    return df


def enrich_obs050_segment_centers(
    obs050: pd.DataFrame,
    path_nodes: pd.DataFrame,
    *,
    max_delta: int = 2,
) -> pd.DataFrame:
    """
    Attach center-step coordinates from path_node_diagnostics to OBS-050 segments.

    OBS-050 segment rows do not always carry r/alpha. OBS-072 v2 enriched them
    by path_id + center_step. This function performs the same enrichment for
    OBS-073 v4 so that the Cp2 locus target is explicit and reproducible.
    """
    seg = obs050.copy()
    if "center_step" not in seg.columns or "step" not in path_nodes.columns:
        seg["obs072_center_r"] = np.nan
        seg["obs072_center_alpha"] = np.nan
        seg["obs072_center_step_delta"] = np.nan
        seg["obs072_center_exact"] = 0
        return seg

    keep = ["path_id", "step"]
    for c in ["r", "alpha", "mds1", "mds2", "lazarus_score", "criticality"]:
        if c in path_nodes.columns:
            keep.append(c)

    nodes = path_nodes[keep].copy()
    nodes["path_id"] = nodes["path_id"].astype(str)
    nodes["step"] = pd.to_numeric(nodes["step"], errors="coerce")
    nodes = nodes.dropna(subset=["path_id", "step"]).copy()
    nodes["step"] = nodes["step"].astype(int)

    seg["path_id"] = seg["path_id"].astype(str)
    seg["center_step_numeric"] = pd.to_numeric(seg["center_step"], errors="coerce")
    seg["center_step_int"] = seg["center_step_numeric"].round().astype("Int64")

    renamed = nodes.rename(
        columns={
            "step": "center_step_int",
            "r": "obs072_center_r",
            "alpha": "obs072_center_alpha",
            "mds1": "obs072_center_mds1",
            "mds2": "obs072_center_mds2",
            "lazarus_score": "obs072_center_lazarus_score",
            "criticality": "obs072_center_criticality",
        }
    )

    out = seg.merge(renamed, on=["path_id", "center_step_int"], how="left")
    out["obs072_center_step_delta"] = np.where(out["obs072_center_r"].notna(), 0, np.nan)
    out["obs072_center_exact"] = out["obs072_center_r"].notna().astype(int)

    missing = out["obs072_center_r"].isna() & out["center_step_int"].notna()
    if missing.any() and max_delta > 0:
        node_groups = {pid: g.sort_values("center_step_int") for pid, g in renamed.groupby("path_id", sort=False)}
        fill_cols = [c for c in renamed.columns if c.startswith("obs072_center_")]

        for idx in out.index[missing]:
            pid = out.at[idx, "path_id"]
            center = out.at[idx, "center_step_int"]
            if pd.isna(center) or pid not in node_groups:
                continue
            g = node_groups[pid]
            delta = (pd.to_numeric(g["center_step_int"], errors="coerce") - int(center)).abs()
            if not delta.notna().any():
                continue
            j = delta.idxmin()
            d = float(delta.loc[j])
            if d <= max_delta:
                for c in fill_cols:
                    if c in g.columns:
                        out.at[idx, c] = g.at[j, c]
                out.at[idx, "obs072_center_step_delta"] = d
                out.at[idx, "obs072_center_exact"] = int(d == 0)

    return out.drop(columns=["center_step_numeric", "center_step_int"], errors="ignore")


def derive_obs050_targets(obs050: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    df = normalize_obs050_classes(obs050)

    locus_r = 0.2 if cfg is None else cfg.obs072_locus_r
    locus_alpha = 0.1328571428571428 if cfg is None else cfg.obs072_locus_alpha
    locus_r_tol = 1e-9 if cfg is None else cfg.obs072_locus_r_tol
    locus_alpha_tol = 1e-6 if cfg is None else cfg.obs072_locus_alpha_tol

    rows = []
    for path_id, grp in df.groupby("path_id", sort=False):
        outcome_mode = (
            grp["outcome_group"].astype(str).mode().iloc[0]
            if "outcome_group" in grp.columns and not grp.empty
            else pd.NA
        )

        coupled_mask = grp["coupling_class"].astype(str).eq("coupled")
        coupled_share = float(coupled_mask.mean()) if len(grp) else 0.0
        has_coupled = bool(coupled_mask.any())

        fam_col = grp["path_family"].astype(str) if "path_family" in grp.columns else pd.Series("", index=grp.index)

        false_mask = (
            grp["outcome_group"].astype(str).eq("nonrecovering")
            & coupled_mask
            & fam_col.eq("off_seam_reorganizing")
            & grp["seam_band"].astype(str).eq("near")
            & grp["posture"].astype(str).eq("compression")
        )

        true_mask = grp["outcome_group"].astype(str).eq("recovering") & coupled_mask

        center_r = pd.to_numeric(grp.get("obs072_center_r", pd.Series(np.nan, index=grp.index)), errors="coerce")
        center_alpha = pd.to_numeric(grp.get("obs072_center_alpha", pd.Series(np.nan, index=grp.index)), errors="coerce")
        locus_mask = (
            false_mask
            & center_r.sub(locus_r).abs().le(locus_r_tol)
            & center_alpha.sub(locus_alpha).abs().le(locus_alpha_tol)
        )

        coupling_class = "coupled" if coupled_share >= 0.5 or has_coupled else "decoupled"

        if bool(false_mask.any()):
            channel = "false_recovery_compression"
        elif bool(true_mask.any()):
            channel = "true_bounded_recovery"
        else:
            channel = pd.NA

        if bool(locus_mask.any()):
            obs072_channel = "false_recovery_compression_locus"
        elif bool(true_mask.any()):
            obs072_channel = "true_bounded_recovery"
        else:
            obs072_channel = pd.NA

        rows.append(
            {
                "path_id": str(path_id),
                "target_outcome_group": outcome_mode,
                "target_coupling_class": coupling_class,
                "target_coupled_outcome_group": outcome_mode if has_coupled else pd.NA,
                "target_recovery_channel_structural": channel,
                "target_obs072_cp2_locus_false_recovery": obs072_channel,

                # Metadata only. Excluded from X.
                "obs050_n_segments": int(len(grp)),
                "obs050_coupled_share": coupled_share,
                "obs050_has_nonrecovering_coupled": int(bool(false_mask.any())),
                "obs072_has_locus_false_recovery": int(bool(locus_mask.any())),
                "obs072_locus_segment_count": int(locus_mask.sum()),
                "obs072_false_recovery_segment_count": int(false_mask.sum()),
                "obs072_center_exact_share": finite_mean(grp.get("obs072_center_exact", pd.Series(dtype=float))),
                "obs050_mean_m_seam": finite_mean(grp.get("m_seam", pd.Series(dtype=float))),
                "obs050_mean_m_r": finite_mean(grp.get("m_r", pd.Series(dtype=float))),
                "obs050_mean_distance_to_seam": finite_mean(grp.get("mean_distance_to_seam", pd.Series(dtype=float))),
                "obs050_min_distance_to_seam": finite_min(grp.get("min_distance_to_seam", pd.Series(dtype=float))),
                "obs050_mean_roughness": finite_mean(grp.get("mean_roughness", pd.Series(dtype=float))),
            }
        )

    return pd.DataFrame(rows)


def load_obs051_boundedness(root: CorpusRoot) -> pd.DataFrame | None:
    pieces = []
    for band in ["all", "core", "near"]:
        p = root.obs051_dir / f"obs051_window_divergence_{band}.csv"
        df = read_csv_optional(p)
        if df is None or "path_id" not in df.columns:
            continue
        df = normalize_path_id(df)
        df["obs051_band_source"] = band
        pieces.append(df)

    if not pieces:
        return None

    raw = pd.concat(pieces, ignore_index=True)
    value_cols = [
        c for c in [
            "lambda_local",
            "mean_lambda_local",
            "delta_d",
            "mean_delta_d",
            "bounded_share",
            "mean_bounded_share",
            "d_start",
            "d_end",
            "mean_d_start",
            "mean_d_end",
        ]
        if c in raw.columns
    ]

    rows = []
    for path_id, grp in raw.groupby("path_id", sort=False):
        row: dict[str, Any] = {"path_id": str(path_id)}
        for c in value_cols:
            row[f"obs051_{c}_mean"] = finite_mean(grp[c])
            row[f"obs051_{c}_min"] = finite_min(grp[c])
            row[f"obs051_{c}_max"] = finite_max(grp[c])
        row["obs051_n_windows"] = int(len(grp))
        rows.append(row)

    return pd.DataFrame(rows)


def add_boundedness_strict_target(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["target_recovery_channel_boundedness_strict"] = pd.NA

    base = out["target_recovery_channel_structural"].astype(str)
    false_mask = base.eq("false_recovery_compression")
    true_mask = base.eq("true_bounded_recovery")

    out.loc[false_mask, "target_recovery_channel_boundedness_strict"] = "false_recovery_compression"

    if not true_mask.any():
        return out

    strict = true_mask.copy()

    lambda_cols = [c for c in ["obs051_lambda_local_mean", "obs051_mean_lambda_local_mean"] if c in out.columns]
    delta_cols = [c for c in ["obs051_delta_d_mean", "obs051_mean_delta_d_mean"] if c in out.columns]
    bounded_cols = [c for c in ["obs051_bounded_share_mean", "obs051_mean_bounded_share_mean"] if c in out.columns]

    if lambda_cols:
        col = lambda_cols[0]
        q = pd.to_numeric(out.loc[true_mask, col], errors="coerce").quantile(0.60)
        strict &= pd.to_numeric(out[col], errors="coerce") <= q

    if delta_cols:
        col = delta_cols[0]
        q = pd.to_numeric(out.loc[true_mask, col], errors="coerce").quantile(0.60)
        strict &= pd.to_numeric(out[col], errors="coerce") <= q

    if bounded_cols:
        col = bounded_cols[0]
        q = pd.to_numeric(out.loc[true_mask, col], errors="coerce").quantile(0.40)
        strict &= pd.to_numeric(out[col], errors="coerce") >= q

    out.loc[strict, "target_recovery_channel_boundedness_strict"] = "true_bounded_recovery"
    return out


def build_feature_table_for_corpus(root: CorpusRoot, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    tables = load_base_tables(root)

    path_node_features = summarize_numeric_by_path(tables["path_nodes"])
    tort, basis = compute_tortuosity_features(tables["path_nodes"])
    diag = reduce_path_diagnostics(tables["path_diag"])
    obs050_enriched = enrich_obs050_segment_centers(
        tables["obs050"],
        tables["path_nodes"],
        max_delta=cfg.obs072_center_max_delta,
    )
    targets = derive_obs050_targets(obs050_enriched, cfg)

    feature = path_node_features.merge(tort, on="path_id", how="left")
    feature = feature.merge(diag, on="path_id", how="left")

    family = tables["family"]
    if "path_family" in family.columns:
        feature = feature.merge(
            family[["path_id", "path_family"]].drop_duplicates("path_id"),
            on="path_id",
            how="left",
        )
    else:
        feature["path_family"] = pd.NA

    feature["target_path_family"] = feature["path_family"]
    feature = feature.merge(targets, on="path_id", how="left")

    obs051 = load_obs051_boundedness(root)
    if obs051 is not None:
        feature = feature.merge(obs051, on="path_id", how="left")

    feature = add_boundedness_strict_target(feature)

    feature["corpus"] = root.label
    feature["source_root"] = str(root.root)
    feature["scale"] = root.scale
    feature["tortuosity_coordinate_basis"] = basis

    manifest = {
        "corpus": root.label,
        "root": str(root.root),
        "scale": root.scale,
        "n_paths": int(feature["path_id"].nunique()),
        "n_feature_rows": int(len(feature)),
        "tortuosity_coordinate_basis": basis,
        "obs072_locus_corpus": cfg.obs072_locus_corpus,
        "obs072_locus_r": cfg.obs072_locus_r,
        "obs072_locus_alpha": cfg.obs072_locus_alpha,
        "obs072_locus_r_tol": cfg.obs072_locus_r_tol,
        "obs072_locus_alpha_tol": cfg.obs072_locus_alpha_tol,
        "obs072_center_max_delta": cfg.obs072_center_max_delta,
    }

    return feature, manifest


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------


def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced",
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def make_tree(cfg: Config) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced",
        random_state=cfg.random_state,
    )


def filter_df_for_spec(df: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    out = df.copy()
    if spec.corpus_filter is not None and "corpus" in out.columns:
        out = out[out["corpus"].astype(str).eq(spec.corpus_filter)].copy()
    return out


def get_feature_columns_for_target(df: pd.DataFrame, spec: TargetSpec) -> tuple[list[str], pd.DataFrame]:
    rows = []
    features = []

    for col in df.columns:
        allowed, reason = feature_allowed_for_target(col, spec)
        numeric_ok = False

        if allowed:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_ok = bool(s.notna().any())
            allowed = numeric_ok

        rows.append(
            {
                "target": spec.name,
                "feature": col,
                "allowed": int(allowed),
                "provenance_or_exclusion": reason,
                "numeric_ok": int(numeric_ok),
                "no_direct_seam": int(spec.no_direct_seam),
                "no_grid_location": int(spec.no_grid_location),
                "allow_obs051_features": int(spec.allow_obs051_features),
                "within_corpus_only": int(spec.within_corpus_only),
                "corpus_filter": spec.corpus_filter or "",
                "obs072_locus_target": int(spec.obs072_locus_target),
            }
        )

        if allowed:
            features.append(col)

    return features, pd.DataFrame(rows)


def prepare_xy(
    df: pd.DataFrame,
    spec: TargetSpec,
    min_class_count: int,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame, pd.DataFrame]:
    df = filter_df_for_spec(df, spec)
    if spec.target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=str), [], pd.DataFrame(), pd.DataFrame()

    work = df[df[spec.target_col].notna()].copy()
    work[spec.target_col] = work[spec.target_col].astype(str)

    counts = work[spec.target_col].value_counts()
    valid_classes = counts[counts >= min_class_count].index.tolist()
    work = work[work[spec.target_col].isin(valid_classes)].copy()

    feature_cols, feature_manifest = get_feature_columns_for_target(work, spec)

    X = work[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=work.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    med = X.median(axis=0, skipna=True) if not X.empty else pd.Series(dtype=float)
    X = X.fillna(med).fillna(0.0)

    y = work[spec.target_col].astype(str)
    meta_cols = [c for c in ["corpus", "path_id"] if c in work.columns]
    meta = work[meta_cols].copy()

    return X, y, feature_cols, meta, feature_manifest


def score_common_fields(spec: TargetSpec) -> dict[str, int | str]:
    return {
        "no_direct_seam": int(spec.no_direct_seam),
        "no_grid_location": int(spec.no_grid_location),
        "allow_obs051_features": int(spec.allow_obs051_features),
        "within_corpus_only": int(spec.within_corpus_only),
        "corpus_filter": spec.corpus_filter or "",
        "obs072_locus_target": int(spec.obs072_locus_target),
    }


def evaluate_within_pooled_cv(
    cfg: Config,
    df: pd.DataFrame,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    X, y, feature_cols, _meta, feature_manifest = prepare_xy(df, spec, cfg.min_class_count)

    score_rows: list[dict[str, Any]] = []
    gini_rows: list[dict[str, Any]] = []
    perm_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    run_type = "within_corpus_cv" if not spec.within_corpus_only else "within_corpus_only_cv"
    train_test = spec.corpus_filter or "pooled"

    if len(y.unique()) < 2 or len(X) < cfg.min_class_count * 2 or len(feature_cols) == 0:
        score_rows.append(
            {
                "run_type": run_type,
                "target": spec.name,
                "train_corpus": train_test,
                "test_corpus": train_test,
                "status": "insufficient_classes_rows_or_features",
                "n_rows": len(X),
                "n_classes": int(y.nunique()),
                "feature_count": len(feature_cols),
                **score_common_fields(spec),
            }
        )
        return (
            pd.DataFrame(score_rows),
            pd.DataFrame(gini_rows),
            pd.DataFrame(perm_rows),
            pd.DataFrame(confusion_rows),
            "",
            feature_manifest,
        )

    min_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    rf = make_rf(cfg)
    y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)

    score_rows.append(
        {
            "run_type": run_type,
            "target": spec.name,
            "train_corpus": train_test,
            "test_corpus": train_test,
            "status": "ok",
            "n_rows": len(X),
            "n_classes": int(y.nunique()),
            "classes": json.dumps(sorted(y.unique().tolist())),
            "feature_count": len(feature_cols),
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro"),
            "weighted_f1": f1_score(y, y_pred, average="weighted"),
            **score_common_fields(spec),
        }
    )

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y, y_pred, labels=labels)
    for i, actual in enumerate(labels):
        for j, pred in enumerate(labels):
            confusion_rows.append(
                {
                    "run_type": run_type,
                    "target": spec.name,
                    "train_corpus": train_test,
                    "test_corpus": train_test,
                    "actual": actual,
                    "predicted": pred,
                    "n": int(cm[i, j]),
                }
            )

    rf.fit(X, y)

    for rank, idx in enumerate(np.argsort(rf.feature_importances_)[::-1], start=1):
        feat = feature_cols[idx]
        gini_rows.append(
            {
                "run_type": run_type,
                "target": spec.name,
                "train_corpus": train_test,
                "test_corpus": train_test,
                "rank": rank,
                "feature": feat,
                "importance": float(rf.feature_importances_[idx]),
                "feature_provenance": classify_feature_provenance(feat),
            }
        )

    try:
        perm = permutation_importance(
            rf,
            X,
            y,
            n_repeats=cfg.permutation_repeats,
            random_state=cfg.random_state,
            n_jobs=-1,
            scoring="balanced_accuracy",
        )
        order = np.argsort(perm.importances_mean)[::-1]
        for rank, idx in enumerate(order, start=1):
            feat = feature_cols[idx]
            perm_rows.append(
                {
                    "run_type": run_type,
                    "target": spec.name,
                    "train_corpus": train_test,
                    "test_corpus": train_test,
                    "rank": rank,
                    "feature": feat,
                    "importance_mean": float(perm.importances_mean[idx]),
                    "importance_std": float(perm.importances_std[idx]),
                    "feature_provenance": classify_feature_provenance(feat),
                }
            )
    except Exception as exc:
        warnings.warn(f"Permutation importance failed for {spec.name}: {exc}")

    tree = make_tree(cfg)
    tree.fit(X, y)
    rules = export_text(tree, feature_names=feature_cols, max_depth=4)

    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(gini_rows),
        pd.DataFrame(perm_rows),
        pd.DataFrame(confusion_rows),
        rules,
        feature_manifest,
    )


def evaluate_cross_corpus_transfer(
    cfg: Config,
    df: pd.DataFrame,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.within_corpus_only:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    score_rows: list[dict[str, Any]] = []
    gini_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    corpora = sorted(df["corpus"].dropna().astype(str).unique().tolist())
    if len(corpora) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for train_corpus in corpora:
        for test_corpus in corpora:
            if train_corpus == test_corpus:
                continue

            train = df[df["corpus"].astype(str) == train_corpus].copy()
            test = df[df["corpus"].astype(str) == test_corpus].copy()

            train = train[train[spec.target_col].notna()].copy()
            test = test[test[spec.target_col].notna()].copy()

            if train.empty or test.empty:
                continue

            train[spec.target_col] = train[spec.target_col].astype(str)
            test[spec.target_col] = test[spec.target_col].astype(str)

            train_counts = train[spec.target_col].value_counts()
            valid_train = train_counts[train_counts >= cfg.min_class_count].index.tolist()
            common = sorted(set(valid_train) & set(test[spec.target_col].unique().tolist()))

            if len(common) < 2:
                score_rows.append(
                    {
                        "run_type": "cross_corpus_transfer",
                        "target": spec.name,
                        "train_corpus": train_corpus,
                        "test_corpus": test_corpus,
                        "status": "insufficient_common_classes",
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_classes": len(common),
                        **score_common_fields(spec),
                    }
                )
                continue

            train = train[train[spec.target_col].isin(common)].copy()
            test = test[test[spec.target_col].isin(common)].copy()

            train_cols, _ = get_feature_columns_for_target(train, spec)
            test_cols, _ = get_feature_columns_for_target(test, spec)
            feature_cols = sorted(set(train_cols) & set(test_cols))

            if not feature_cols:
                score_rows.append(
                    {
                        "run_type": "cross_corpus_transfer",
                        "target": spec.name,
                        "train_corpus": train_corpus,
                        "test_corpus": test_corpus,
                        "status": "no_common_features",
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_classes": len(common),
                        **score_common_fields(spec),
                    }
                )
                continue

            X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

            med = X_train.median(axis=0, skipna=True)
            X_train = X_train.fillna(med).fillna(0.0)
            X_test = X_test.fillna(med).fillna(0.0)

            y_train = train[spec.target_col].astype(str)
            y_test = test[spec.target_col].astype(str)

            rf = make_rf(cfg)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            score_rows.append(
                {
                    "run_type": "cross_corpus_transfer",
                    "target": spec.name,
                    "train_corpus": train_corpus,
                    "test_corpus": test_corpus,
                    "status": "ok",
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_classes": len(common),
                    "classes": json.dumps(common),
                    "feature_count": len(feature_cols),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "macro_f1": f1_score(y_test, y_pred, average="macro"),
                    "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
                    **score_common_fields(spec),
                }
            )

            labels = common
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            for i, actual in enumerate(labels):
                for j, pred in enumerate(labels):
                    confusion_rows.append(
                        {
                            "run_type": "cross_corpus_transfer",
                            "target": spec.name,
                            "train_corpus": train_corpus,
                            "test_corpus": test_corpus,
                            "actual": actual,
                            "predicted": pred,
                            "n": int(cm[i, j]),
                        }
                    )

            for rank, idx in enumerate(np.argsort(rf.feature_importances_)[::-1], start=1):
                feat = feature_cols[idx]
                gini_rows.append(
                    {
                        "run_type": "cross_corpus_transfer",
                        "target": spec.name,
                        "train_corpus": train_corpus,
                        "test_corpus": test_corpus,
                        "rank": rank,
                        "feature": feat,
                        "importance": float(rf.feature_importances_[idx]),
                        "feature_provenance": classify_feature_provenance(feat),
                    }
                )

    return pd.DataFrame(score_rows), pd.DataFrame(gini_rows), pd.DataFrame(confusion_rows)


def run_target(
    cfg: Config,
    feature_table: pd.DataFrame,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    cv_scores, cv_gini, cv_perm, cv_conf, rules, feature_manifest = evaluate_within_pooled_cv(cfg, feature_table, spec)
    x_scores, x_gini, x_conf = evaluate_cross_corpus_transfer(cfg, feature_table, spec)

    scores = pd.concat([cv_scores, x_scores], ignore_index=True)
    gini = pd.concat([cv_gini, x_gini], ignore_index=True)
    confusion = pd.concat([cv_conf, x_conf], ignore_index=True)

    target_rows = []
    scoped = filter_df_for_spec(feature_table, spec)
    if spec.target_col in scoped.columns:
        tmp = scoped[scoped[spec.target_col].notna()].copy()
        for corpus, grp in tmp.groupby("corpus", dropna=False):
            counts = grp[spec.target_col].astype(str).value_counts()
            for cls, n in counts.items():
                target_rows.append(
                    {
                        "target": spec.name,
                        "target_col": spec.target_col,
                        "corpus": corpus,
                        "class": cls,
                        "n": int(n),
                        **score_common_fields(spec),
                    }
                )

    return scores, gini, cv_perm, confusion, rules, pd.DataFrame(target_rows), feature_manifest


# ---------------------------------------------------------------------
# Label-shuffle null controls
# ---------------------------------------------------------------------


def obs072_shuffle_specs() -> list[TargetSpec]:
    """
    Label-shuffle controls are scoped to OBS-072 locus targets.

    The strongest control is the blinded no_direct_seam_no_grid variant, but
    keeping all four variants makes the null diagnostic easier to read:
      - unblinded
      - no_direct_seam
      - no_grid
      - no_direct_seam_no_grid
    """
    return [spec for spec in TARGET_SPECS if spec.obs072_locus_target]


def evaluate_label_shuffle_nulls_for_spec(
    cfg: Config,
    feature_table: pd.DataFrame,
    spec: TargetSpec,
    *,
    observed_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate label-shuffle nulls for one target spec.

    The row pool and feature matrix are exactly those used by prepare_xy().
    Labels are permuted inside that same eligible pool. Each repeat runs the
    same stratified CV protocol against the shuffled labels.

    Interpretation:
      observed_balanced_accuracy >> shuffled_balanced_accuracy
    means the real semantic label carries field-structured signal beyond
    class balance / row-pool artifacts.
    """
    rows: list[dict[str, Any]] = []

    if cfg.label_shuffle_repeats <= 0:
        return pd.DataFrame(rows)

    X, y, feature_cols, _meta, _feature_manifest = prepare_xy(
        feature_table,
        spec,
        cfg.min_class_count,
    )

    if len(y.unique()) < 2 or len(X) < cfg.min_class_count * 2 or len(feature_cols) == 0:
        rows.append(
            {
                "target": spec.name,
                "status": "insufficient_classes_rows_or_features",
                "n_rows": len(X),
                "n_classes": int(y.nunique()),
                "feature_count": len(feature_cols),
                "label_shuffle_repeats": cfg.label_shuffle_repeats,
                **score_common_fields(spec),
            }
        )
        return pd.DataFrame(rows)

    observed = observed_scores[
        observed_scores.get("status", "").eq("ok")
        & observed_scores.get("target", "").eq(spec.name)
        & observed_scores.get("run_type", "").eq("within_corpus_only_cv")
    ].copy()

    if observed.empty:
        observed_ba = float("nan")
        observed_macro_f1 = float("nan")
    else:
        observed_ba = float(pd.to_numeric(observed["balanced_accuracy"], errors="coerce").max())
        observed_macro_f1 = float(pd.to_numeric(observed["macro_f1"], errors="coerce").max())

    min_count = int(y.value_counts().min())
    n_splits = max(2, min(5, min_count))

    rng = np.random.default_rng(cfg.random_state + 7300)
    labels = sorted(y.unique().tolist())
    y_values = y.to_numpy(copy=True)

    for repeat in range(cfg.label_shuffle_repeats):
        shuffled_values = rng.permutation(y_values)
        y_shuffle = pd.Series(shuffled_values, index=y.index, name=f"{spec.target_col}_shuffled")

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cfg.random_state + 1000 + repeat,
        )

        rf = RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            class_weight="balanced",
            random_state=cfg.random_state + 2000 + repeat,
            n_jobs=-1,
        )

        try:
            y_pred = cross_val_predict(rf, X, y_shuffle, cv=cv, n_jobs=-1)
            shuffle_ba = balanced_accuracy_score(y_shuffle, y_pred)
            shuffle_macro_f1 = f1_score(y_shuffle, y_pred, average="macro")
            status = "ok"
        except Exception as exc:
            shuffle_ba = float("nan")
            shuffle_macro_f1 = float("nan")
            status = f"failed: {type(exc).__name__}: {exc}"

        rows.append(
            {
                "target": spec.name,
                "run_type": "label_shuffle_null_cv",
                "status": status,
                "repeat": repeat,
                "n_rows": len(X),
                "n_classes": int(y.nunique()),
                "classes": json.dumps(labels),
                "feature_count": len(feature_cols),
                "observed_balanced_accuracy": observed_ba,
                "shuffle_balanced_accuracy": shuffle_ba,
                "delta_observed_minus_shuffle_ba": (
                    observed_ba - shuffle_ba
                    if np.isfinite(observed_ba) and np.isfinite(shuffle_ba)
                    else float("nan")
                ),
                "observed_macro_f1": observed_macro_f1,
                "shuffle_macro_f1": shuffle_macro_f1,
                "delta_observed_minus_shuffle_macro_f1": (
                    observed_macro_f1 - shuffle_macro_f1
                    if np.isfinite(observed_macro_f1) and np.isfinite(shuffle_macro_f1)
                    else float("nan")
                ),
                "label_shuffle_repeats": cfg.label_shuffle_repeats,
                **score_common_fields(spec),
            }
        )

    return pd.DataFrame(rows)


def evaluate_label_shuffle_nulls(
    cfg: Config,
    feature_table: pd.DataFrame,
    observed_scores: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for spec in obs072_shuffle_specs():
        pieces.append(
            evaluate_label_shuffle_nulls_for_spec(
                cfg,
                feature_table,
                spec,
                observed_scores=observed_scores,
            )
        )

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)


def summarize_label_shuffle_nulls(label_shuffle: pd.DataFrame) -> pd.DataFrame:
    if label_shuffle.empty:
        return pd.DataFrame()

    ok = label_shuffle[label_shuffle["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()

    rows = []
    for target, grp in ok.groupby("target", sort=False):
        obs_ba = pd.to_numeric(grp["observed_balanced_accuracy"], errors="coerce").max()
        shuffle_ba = pd.to_numeric(grp["shuffle_balanced_accuracy"], errors="coerce")
        obs_f1 = pd.to_numeric(grp["observed_macro_f1"], errors="coerce").max()
        shuffle_f1 = pd.to_numeric(grp["shuffle_macro_f1"], errors="coerce")

        n = int(shuffle_ba.notna().sum())
        if n > 0 and np.isfinite(obs_ba):
            p_ge_observed = float((1 + (shuffle_ba >= obs_ba).sum()) / (1 + n))
        else:
            p_ge_observed = float("nan")

        rows.append(
            {
                "target": target,
                "n_repeats_ok": n,
                "observed_balanced_accuracy": float(obs_ba) if np.isfinite(obs_ba) else float("nan"),
                "shuffle_balanced_accuracy_mean": float(shuffle_ba.mean()) if shuffle_ba.notna().any() else float("nan"),
                "shuffle_balanced_accuracy_std": float(shuffle_ba.std()) if shuffle_ba.notna().sum() > 1 else float("nan"),
                "shuffle_balanced_accuracy_max": float(shuffle_ba.max()) if shuffle_ba.notna().any() else float("nan"),
                "delta_observed_minus_shuffle_mean_ba": (
                    float(obs_ba - shuffle_ba.mean())
                    if np.isfinite(obs_ba) and shuffle_ba.notna().any()
                    else float("nan")
                ),
                "empirical_p_shuffle_ge_observed_ba": p_ge_observed,
                "observed_macro_f1": float(obs_f1) if np.isfinite(obs_f1) else float("nan"),
                "shuffle_macro_f1_mean": float(shuffle_f1.mean()) if shuffle_f1.notna().any() else float("nan"),
                "shuffle_macro_f1_std": float(shuffle_f1.std()) if shuffle_f1.notna().sum() > 1 else float("nan"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Shortcut risk / report
# ---------------------------------------------------------------------


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def add_shortcut_risk(model_scores: pd.DataFrame) -> pd.DataFrame:
    if model_scores.empty or "balanced_accuracy" not in model_scores.columns:
        return model_scores

    out = model_scores.copy()
    out["shortcut_risk"] = pd.NA

    ok = out[out["status"].eq("ok") & out["run_type"].isin(["within_corpus_cv", "within_corpus_only_cv"])].copy()
    if ok.empty:
        return out

    for target in ok["target"].unique():
        rows = ok[ok["target"] == target]
        ba = pd.to_numeric(rows["balanced_accuracy"], errors="coerce").max()
        no_seam = int(rows["no_direct_seam"].max()) if "no_direct_seam" in rows.columns else 0
        no_grid = int(rows["no_grid_location"].max()) if "no_grid_location" in rows.columns else 0
        obs072 = int(rows["obs072_locus_target"].max()) if "obs072_locus_target" in rows.columns else 0

        if no_seam and no_grid:
            risk = "low" if ba >= 0.70 else "medium" if ba >= 0.58 else "high"
        elif no_seam or no_grid:
            risk = "medium" if ba >= 0.70 else "high"
        else:
            risk = "unblinded"

        if obs072:
            risk = f"obs072_{risk}"

        out.loc[out["target"] == target, "shortcut_risk"] = risk

    return out


def obs072_locus_interpretation(model_scores: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if model_scores.empty:
        return lines

    ok = model_scores[
        model_scores.get("status", "").eq("ok")
        & model_scores.get("obs072_locus_target", 0).astype(str).isin(["1", "1.0", "True"])
        & model_scores.get("run_type", "").eq("within_corpus_only_cv")
    ].copy()
    if ok.empty:
        lines.append("No successful OBS-072 locus target runs.")
        return lines

    def get_ba(name: str) -> float:
        s = ok[ok["target"].eq(name)]["balanced_accuracy"]
        return float(pd.to_numeric(s, errors="coerce").max()) if len(s) else float("nan")

    full = get_ba("obs072_cp2_locus_false_recovery")
    no_seam = get_ba("obs072_cp2_locus_false_recovery_no_direct_seam")
    no_grid = get_ba("obs072_cp2_locus_false_recovery_no_grid")
    both = get_ba("obs072_cp2_locus_false_recovery_no_direct_seam_no_grid")

    lines.extend(
        [
            f"- full: `{fmt(full, 4)}`",
            f"- no_direct_seam: `{fmt(no_seam, 4)}`",
            f"- no_grid: `{fmt(no_grid, 4)}`",
            f"- no_direct_seam_no_grid: `{fmt(both, 4)}`",
            "",
        ]
    )

    if np.isfinite(full) and full < 0.65:
        verdict = "The OBS-072 composition label is weakly field-separable even before shortcut removal."
    elif np.isfinite(both) and both >= 0.70:
        verdict = "The OBS-072 Cp2 locus retains a continuous-field signature after removing direct seam and absolute grid-location shortcuts."
    elif np.isfinite(no_seam) and no_seam < 0.65:
        verdict = "The OBS-072 Cp2 locus appears primarily seam/proximity-defined."
    elif np.isfinite(no_grid) and no_grid < 0.65 or np.isfinite(both) and both < 0.65:
        verdict = "The OBS-072 Cp2 locus appears primarily localized/hotspot-defined rather than portable across field coordinates."
    else:
        verdict = "The OBS-072 Cp2 locus is partially field-separable, but the shortcut-resistance pattern is mixed."

    lines.append(f"**Provisional v5 read:** {verdict}")
    return lines


def write_summary(
    cfg: Config,
    corpus_manifests: list[dict[str, Any]],
    model_scores: pd.DataFrame,
    gini: pd.DataFrame,
    perm: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    label_shuffle: pd.DataFrame,
    label_shuffle_summary: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# OBS-073 — Continuous-field groupoid reduction v5",
        "",
        "## Scope",
        "",
        "OBS-073 tests whether symbolic route/groupoid classes are recoverable from continuous field geometry.",
        "v5 connects OBS-072 by adding within-Cp2 false-recovery locus targets and label-shuffle null controls.",
        "",
        "## Corpus roots",
        "",
    ]

    for m in corpus_manifests:
        lines.append(
            f"- `{m['corpus']}` root `{m['root']}` scale `{m['scale']}` "
            f"paths `{m['n_paths']}` tortuosity basis `{m['tortuosity_coordinate_basis']}`"
        )

    lines.extend(
        [
            "",
            "## OBS-072 locus configuration",
            "",
            f"- locus corpus: `{cfg.obs072_locus_corpus}`",
            f"- locus r: `{cfg.obs072_locus_r}` ± `{cfg.obs072_locus_r_tol}`",
            f"- locus alpha: `{cfg.obs072_locus_alpha}` ± `{cfg.obs072_locus_alpha_tol}`",
            f"- center-step max delta: `{cfg.obs072_center_max_delta}`",
            f"- label-shuffle repeats: `{cfg.label_shuffle_repeats}`",
            "",
            "## Feature guardrails",
            "",
            "- `obs050_*` and `obs072_*` features are excluded from all predictor matrices.",
            "- `obs051_*` features are excluded except for `recovery_channel_boundedness_strict`.",
            "- `target_*`, `path_family`, outcome labels, seam-band labels, coupling labels, and posture labels are excluded.",
            "- `no_direct_seam` targets remove seam/proximity/fraction features.",
            "- `no_grid_location` targets remove absolute node/grid/MDS coordinate-location features.",
            "- Coordinate deltas and tortuosity features remain allowed unless otherwise excluded.",
            "- OBS-072 locus targets are within-Cp2-only by default and do not run cross-corpus transfer.",
            "",
            "## Model scores",
            "",
        ]
    )

    ok = model_scores[model_scores.get("status", "") == "ok"].copy() if not model_scores.empty else pd.DataFrame()
    if ok.empty:
        lines.append("No successful model runs.")
    else:
        cols = [
            "run_type",
            "target",
            "train_corpus",
            "test_corpus",
            "n_rows",
            "n_train",
            "n_test",
            "balanced_accuracy",
            "macro_f1",
            "feature_count",
            "no_direct_seam",
            "no_grid_location",
            "allow_obs051_features",
            "within_corpus_only",
            "obs072_locus_target",
            "shortcut_risk",
        ]
        use_cols = [c for c in cols if c in ok.columns]
        lines.append("| " + " | ".join(use_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_cols)) + " |")
        for _, row in ok.sort_values(["target", "run_type", "train_corpus", "test_corpus"]).iterrows():
            vals = []
            for c in use_cols:
                val = row.get(c, "")
                vals.append(fmt(val) if isinstance(val, (float, np.floating)) else str(val))
            lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## OBS-072 locus target read", ""])
    lines.extend(obs072_locus_interpretation(model_scores))

    lines.extend(["", "## OBS-072 label-shuffle null controls", ""])

    if cfg.label_shuffle_repeats <= 0:
        lines.append("Label-shuffle controls were disabled for this run.")
    elif label_shuffle_summary.empty:
        lines.append("No successful label-shuffle null controls were produced.")
    else:
        lines.append(
            "The shuffle control preserves the eligible Cp2 row pool, feature matrix, "
            "class balance, and CV protocol, but permutes the OBS-072 labels."
        )
        lines.append("")
        cols = [
            "target",
            "n_repeats_ok",
            "observed_balanced_accuracy",
            "shuffle_balanced_accuracy_mean",
            "shuffle_balanced_accuracy_std",
            "shuffle_balanced_accuracy_max",
            "delta_observed_minus_shuffle_mean_ba",
            "empirical_p_shuffle_ge_observed_ba",
        ]
        use_cols = [c for c in cols if c in label_shuffle_summary.columns]
        lines.append("| " + " | ".join(use_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_cols)) + " |")
        for _, row in label_shuffle_summary.sort_values("target").iterrows():
            vals = []
            for c in use_cols:
                val = row.get(c, "")
                vals.append(fmt(val, 6) if isinstance(val, (float, np.floating)) else str(val))
            lines.append("| " + " | ".join(vals) + " |")

        blinded = label_shuffle_summary[
            label_shuffle_summary["target"].eq("obs072_cp2_locus_false_recovery_no_direct_seam_no_grid")
        ]
        if not blinded.empty:
            r = blinded.iloc[0]
            obs_ba = float(r.get("observed_balanced_accuracy", np.nan))
            null_mean = float(r.get("shuffle_balanced_accuracy_mean", np.nan))
            pval = float(r.get("empirical_p_shuffle_ge_observed_ba", np.nan))

            lines.append("")
            if np.isfinite(obs_ba) and np.isfinite(null_mean) and obs_ba - null_mean >= 0.20:
                lines.append(
                    f"**v5 null read:** The blinded OBS-072 locus result remains above the shuffled-label null "
                    f"(observed BA `{fmt(obs_ba, 4)}` vs shuffle mean `{fmt(null_mean, 4)}`, "
                    f"empirical p `{fmt(pval, 4)}`)."
                )
            else:
                lines.append(
                    f"**v5 null read:** The blinded OBS-072 locus result does not cleanly separate from the "
                    f"shuffled-label null (observed BA `{fmt(obs_ba, 4)}` vs shuffle mean `{fmt(null_mean, 4)}`, "
                    f"empirical p `{fmt(pval, 4)}`)."
                )

    lines.extend(["", "## Top pooled / within-corpus permutation importances", ""])
    pooled_perm = perm[perm["run_type"].isin(["within_corpus_cv", "within_corpus_only_cv"])].copy() if not perm.empty else pd.DataFrame()

    if pooled_perm.empty:
        lines.append("No permutation importances.")
    else:
        for spec in TARGET_SPECS:
            sub = pooled_perm[pooled_perm["target"].eq(spec.name)].sort_values("rank").head(10)
            if sub.empty:
                continue
            lines.append(f"### {spec.name}")
            lines.append("")
            for _, row in sub.iterrows():
                lines.append(
                    f"- `{row['feature']}` [{row.get('feature_provenance', '')}]: "
                    f"{fmt(row['importance_mean'], 6)} ± {fmt(row['importance_std'], 6)}"
                )
            lines.append("")

    lines.extend(["", "## Feature-manifest summary", ""])
    if feature_manifest.empty:
        lines.append("No feature manifest written.")
    else:
        summary = (
            feature_manifest
            .groupby(["target", "allowed", "provenance_or_exclusion"], as_index=False)
            .agg(n_features=("feature", "count"))
            .sort_values(["target", "allowed", "provenance_or_exclusion"])
        )
        lines.append("| target | allowed | provenance_or_exclusion | n_features |")
        lines.append("| --- | ---: | --- | ---: |")
        for _, row in summary.iterrows():
            lines.append(
                f"| {row['target']} | {int(row['allowed'])} | "
                f"{row['provenance_or_exclusion']} | {int(row['n_features'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Strong unblinded performance alone is not a pruning result.",
            "- Strong `no_direct_seam_no_grid` performance is the strongest evidence for field-level reduction.",
            "- Collapse under `no_grid_location` suggests hotspot/localization shortcut use.",
            "- Collapse under `no_direct_seam` suggests seam-proximity shortcut use.",
            "- OBS-072 locus targets are not cross-corpus transfer tests; they ask whether the Cp2 locus is internally field-separable.",
            "- Label-shuffle nulls test whether OBS-072 locus separability depends on real labels rather than row-pool artifacts.",
            "- Target 5B is boundedness-assisted and should not be treated as the primary non-leakage result.",
            "",
        ]
    )

    (cfg.outdir / "obs073_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS-073 continuous-field groupoid reduction v5.")

    p.add_argument("--left-label", default="C")
    p.add_argument("--left-root", default="outputs")
    p.add_argument("--right-label", default="Cp2")
    p.add_argument("--right-root", default="outputs/corpora/Cp2/campaigns/full_v2/pipeline")
    p.add_argument("--scale", default="100000")
    p.add_argument("--left-scale", default=None)
    p.add_argument("--right-scale", default=None)

    p.add_argument("--outdir", default="outputs/obs073_continuous_field_groupoid_reduction_v5")
    p.add_argument("--random-state", type=int, default=73)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--max-rows-per-corpus", type=int, default=0)

    p.add_argument("--rf-n-estimators", type=int, default=500)
    p.add_argument("--rf-max-depth", type=int, default=4)
    p.add_argument("--rf-min-samples-leaf", type=int, default=20)
    p.add_argument("--permutation-repeats", type=int, default=8)
    p.add_argument(
        "--label-shuffle-repeats",
        type=int,
        default=20,
        help="Number of label-shuffle null repeats for OBS-072 locus targets. Use 0 to disable.",
    )

    p.add_argument("--obs072-locus-corpus", default="Cp2")
    p.add_argument("--obs072-locus-r", type=float, default=0.2)
    p.add_argument("--obs072-locus-alpha", type=float, default=0.1328571428571428)
    p.add_argument("--obs072-locus-r-tol", type=float, default=1e-9)
    p.add_argument("--obs072-locus-alpha-tol", type=float, default=1e-6)
    p.add_argument("--obs072-center-max-delta", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    left_scale = args.left_scale or args.scale
    right_scale = args.right_scale or args.scale

    cfg = Config(
        corpora=[
            CorpusRoot(args.left_label, Path(args.left_root), str(left_scale)),
            CorpusRoot(args.right_label, Path(args.right_root), str(right_scale)),
        ],
        outdir=Path(args.outdir),
        random_state=args.random_state,
        min_class_count=args.min_class_count,
        max_rows_per_corpus=args.max_rows_per_corpus or None,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        permutation_repeats=args.permutation_repeats,
        label_shuffle_repeats=args.label_shuffle_repeats,
        obs072_locus_corpus=args.obs072_locus_corpus,
        obs072_locus_r=args.obs072_locus_r,
        obs072_locus_alpha=args.obs072_locus_alpha,
        obs072_locus_r_tol=args.obs072_locus_r_tol,
        obs072_locus_alpha_tol=args.obs072_locus_alpha_tol,
        obs072_center_max_delta=args.obs072_center_max_delta,
    )

    global TARGET_SPECS
    TARGET_SPECS = [
        spec if not spec.obs072_locus_target else TargetSpec(
            name=spec.name,
            target_col=spec.target_col,
            no_direct_seam=spec.no_direct_seam,
            no_grid_location=spec.no_grid_location,
            allow_obs051_features=spec.allow_obs051_features,
            within_corpus_only=True,
            corpus_filter=cfg.obs072_locus_corpus,
            obs072_locus_target=True,
            description=spec.description,
        )
        for spec in TARGET_SPECS
    ]

    ensure_dir(cfg.outdir)

    feature_tables = []
    corpus_manifests = []

    for root in cfg.corpora:
        ft, manifest = build_feature_table_for_corpus(root, cfg)

        if cfg.max_rows_per_corpus is not None and len(ft) > cfg.max_rows_per_corpus:
            # Preserve rare OBS-072 locus rows if present.
            locus = ft[ft.get("target_obs072_cp2_locus_false_recovery", pd.Series(index=ft.index)).astype(str).eq("false_recovery_compression_locus")].copy()
            remaining = ft.drop(index=locus.index, errors="ignore")
            n_remaining = max(0, cfg.max_rows_per_corpus - len(locus))
            sampled = remaining.sample(
                n=min(n_remaining, len(remaining)),
                random_state=cfg.random_state,
                replace=False,
            ).copy()
            ft = pd.concat([locus, sampled], ignore_index=True)

        feature_tables.append(ft)
        corpus_manifests.append(manifest)

    feature_table = pd.concat(feature_tables, ignore_index=True)
    feature_table.to_csv(cfg.outdir / "obs073_feature_table.csv", index=False)

    all_scores = []
    all_gini = []
    all_perm = []
    all_confusion = []
    all_target_manifest = []
    all_feature_manifest = []

    for spec in TARGET_SPECS:
        if spec.target_col not in feature_table.columns:
            continue

        print(f"==> OBS-073 v5 target: {spec.name}")

        scores, gini, perm, confusion, rules, target_manifest, feature_manifest = run_target(
            cfg,
            feature_table,
            spec,
        )

        all_scores.append(scores)
        all_gini.append(gini)
        all_perm.append(perm)
        all_confusion.append(confusion)
        all_target_manifest.append(target_manifest)
        all_feature_manifest.append(feature_manifest)

        if rules:
            (cfg.outdir / f"obs073_decision_rules_{spec.name}.txt").write_text(
                rules,
                encoding="utf-8",
            )

    model_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    gini = pd.concat(all_gini, ignore_index=True) if all_gini else pd.DataFrame()
    perm = pd.concat(all_perm, ignore_index=True) if all_perm else pd.DataFrame()
    confusion = pd.concat(all_confusion, ignore_index=True) if all_confusion else pd.DataFrame()
    target_manifest = pd.concat(all_target_manifest, ignore_index=True) if all_target_manifest else pd.DataFrame()
    feature_manifest = pd.concat(all_feature_manifest, ignore_index=True) if all_feature_manifest else pd.DataFrame()

    model_scores = add_shortcut_risk(model_scores)
    print("==> OBS-073 v5 label-shuffle null controls")
    label_shuffle = evaluate_label_shuffle_nulls(
        cfg,
        feature_table,
        observed_scores=model_scores,
    )
    label_shuffle_summary = summarize_label_shuffle_nulls(label_shuffle)

    target_manifest.to_csv(cfg.outdir / "obs073_target_manifest.csv", index=False)
    feature_manifest.to_csv(cfg.outdir / "obs073_feature_manifest.csv", index=False)
    model_scores.to_csv(cfg.outdir / "obs073_model_scores.csv", index=False)
    gini.to_csv(cfg.outdir / "obs073_feature_importance_gini.csv", index=False)
    perm.to_csv(cfg.outdir / "obs073_feature_importance_permutation.csv", index=False)
    confusion.to_csv(cfg.outdir / "obs073_confusion_matrices.csv", index=False)
    label_shuffle.to_csv(cfg.outdir / "obs073_label_shuffle_nulls.csv", index=False)
    label_shuffle_summary.to_csv(cfg.outdir / "obs073_label_shuffle_summary.csv", index=False)

    write_summary(
        cfg=cfg,
        corpus_manifests=corpus_manifests,
        model_scores=model_scores,
        gini=gini,
        perm=perm,
        feature_manifest=feature_manifest,
        label_shuffle=label_shuffle,
        label_shuffle_summary=label_shuffle_summary,
    )

    print(cfg.outdir / "obs073_feature_table.csv")
    print(cfg.outdir / "obs073_target_manifest.csv")
    print(cfg.outdir / "obs073_feature_manifest.csv")
    print(cfg.outdir / "obs073_model_scores.csv")
    print(cfg.outdir / "obs073_feature_importance_gini.csv")
    print(cfg.outdir / "obs073_feature_importance_permutation.csv")
    print(cfg.outdir / "obs073_confusion_matrices.csv")
    print(cfg.outdir / "obs073_label_shuffle_nulls.csv")
    print(cfg.outdir / "obs073_label_shuffle_summary.csv")
    print(cfg.outdir / "obs073_summary.md")


if __name__ == "__main__":
    main()
