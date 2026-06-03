#!/usr/bin/env python3
"""
OBS-072 — Cp2 nonrecovering seam-drift diagnostic.

Purpose
-------
OBS-071 showed that OBS-050 qualitatively replicates from C to Cp2, but with
a weakened contrast:

    C   nonrecovering coupled share ≈ 0.0377
    Cp2 nonrecovering coupled share ≈ 0.0822

Cp2 recovering segments remain strongly seam-coupled, but Cp2 nonrecovering
segments are also more seam-coupled than in C.

This script isolates that baseline drift.

v2 upgrade
----------
v1 showed that Cp2's elevated nonrecovering seam-coupled baseline is dominated
by:

    off_seam_reorganizing / near / compression

v2 enriches OBS-050 segment rows from path_node_diagnostics using:

    path_id + center_step

or nearest available step within the same path.

This recovers local/grid fields where available, including r, alpha,
center-step seam distance, Lazarus, signed phase, criticality, obstruction, and
other node/path diagnostics. That allows the diagnostic to localize Cp2 drift in
parameter/field space rather than only family/band/posture space.

Typical usage
-------------
PYTHONPATH=src:experiments .venv/bin/python \
  experiments/studies/obs072_cp2_nonrecovering_seam_drift.py \
  --left-label C \
  --left-root outputs \
  --right-label Cp2 \
  --right-root outputs/corpora/Cp2/campaigns/full_v2/pipeline \
  --scale 100000 \
  --outdir outputs/obs072_cp2_nonrecovering_seam_drift

Outputs
-------
<outdir>/
  obs072_input_manifest.csv
  obs072_nonrecovering_segment_enriched.csv
  obs072_nonrecovering_coupling_summary.csv
  obs072_family_band_posture_summary.csv
  obs072_grid_concentration_summary.csv
  obs072_center_step_field_summary.csv
  obs072_field_contrast_summary.csv
  obs072_corpus_delta_summary.csv
  obs072_cp2_nonrecovering_seam_drift_summary.md

Guardrails
----------
- This is a diagnostic, not a causal proof.
- The script compares artifact roots as supplied.
- C may be legacy-root while Cp2 may be campaign-scoped.
- Elevated nonrecovering seam coupling can reflect geometry, corpus dynamics,
  route-family composition, or measurement basis; this script distinguishes
  patterns but does not adjudicate tokenizer/embedding causality directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRoot:
    label: str
    root: Path
    scale: str

    @property
    def obs050_dir(self) -> Path:
        return self.root / "obs050_structural_coupling_persistence"

    @property
    def family_substrate_dir(self) -> Path:
        return self.root / "scales" / self.scale / "family_substrate"

    @property
    def scene_nodes_csv(self) -> Path:
        return self.root / "obs022_scene_bundle" / "scene_nodes.csv"

    @property
    def fim_surface_csv(self) -> Path:
        return self.root / "fim" / "fim_surface.csv"

    @property
    def lazarus_scores_csv(self) -> Path:
        return self.root / "fim_lazarus" / "lazarus_scores.csv"

    @property
    def phase_distance_csv(self) -> Path:
        return self.root / "fim_phase" / "phase_distance_to_seam.csv"

    @property
    def path_node_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_node_diagnostics.csv"

    @property
    def path_diagnostics_csv(self) -> Path:
        return self.family_substrate_dir / "path_diagnostics.csv"

    @property
    def family_assignments_csv(self) -> Path:
        return self.family_substrate_dir / "path_family_assignments.csv"


@dataclass(frozen=True)
class Config:
    left: CorpusRoot
    right: CorpusRoot
    outdir: Path
    max_step_delta: int = 2


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


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def get_first_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def as_float(x: Any) -> float:
    try:
        out = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite_mean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def finite_median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.median()) if x.notna().any() else float("nan")


def finite_min(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.min()) if x.notna().any() else float("nan")


def finite_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.max()) if x.notna().any() else float("nan")


def normalize_outcome_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome_group" not in out.columns and "outcome" in out.columns:
        out = out.rename(columns={"outcome": "outcome_group"})
    if "outcome_group" not in out.columns:
        raise ValueError(
            "Expected outcome column `outcome_group` or `outcome`; "
            f"found {list(out.columns)}"
        )
    return out


def normalize_coupling_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "coupling_class" in out.columns:
        return out

    if "coupled_class" in out.columns:
        return out.rename(columns={"coupled_class": "coupling_class"})

    # OBS-050 segment tables often do not carry coupling_class directly.
    # In that artifact, coupling is defined by seam_band:
    #   core/near => coupled
    #   far       => decoupled
    if "seam_band" in out.columns:
        band = out["seam_band"].astype(str)
        out["coupling_class"] = np.where(
            band.isin(["core", "near"]),
            "coupled",
            np.where(band.eq("far"), "decoupled", "unknown"),
        )
        return out

    # Fallback: derive seam_band from distances, then derive coupling.
    if "mean_distance_to_seam" in out.columns or "min_distance_to_seam" in out.columns:
        mean_d = numeric(out, "mean_distance_to_seam")
        min_d = numeric(out, "min_distance_to_seam")
        seam_band = np.where(
            min_d <= 1e-12,
            "core",
            np.where(mean_d <= 0.15, "near", "far"),
        )
        out["seam_band"] = seam_band
        out["coupling_class"] = np.where(
            out["seam_band"].isin(["core", "near"]),
            "coupled",
            np.where(out["seam_band"].eq("far"), "decoupled", "unknown"),
        )
        return out

    raise ValueError(
        "Expected coupling column `coupling_class`, or enough seam-band/distance "
        f"fields to derive it; found {list(out.columns)}"
    )


def normalize_path_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "path_id" not in out.columns and "probe_id" in out.columns:
        out = out.rename(columns={"probe_id": "path_id"})
    if "path_id" in out.columns:
        out["path_id"] = out["path_id"].astype(str)
    return out


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


def build_manifest(root: CorpusRoot) -> pd.DataFrame:
    checks = [
        ("obs050_segments", root.obs050_dir / "structural_coupling_segments.csv"),
        ("obs050_coupled_summary", root.obs050_dir / "structural_coupling_coupled_vs_decoupled_summary.csv"),
        ("obs050_family_summary", root.obs050_dir / "structural_coupling_family_summary.csv"),
        ("family_assignments", root.family_assignments_csv),
        ("path_diagnostics", root.path_diagnostics_csv),
        ("path_node_diagnostics", root.path_node_diagnostics_csv),
        ("scene_nodes", root.scene_nodes_csv),
        ("fim_surface", root.fim_surface_csv),
        ("lazarus_scores", root.lazarus_scores_csv),
        ("phase_distance_to_seam", root.phase_distance_csv),
    ]
    rows = []
    for artifact, path in checks:
        exists = path.exists()
        rows.append(
            {
                "corpus": root.label,
                "artifact": artifact,
                "path": str(path),
                "exists": int(exists),
                "bytes": int(path.stat().st_size) if exists else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Loading / enrichment
# ---------------------------------------------------------------------


def load_segments(root: CorpusRoot) -> pd.DataFrame:
    segs = read_csv_required(
        root.obs050_dir / "structural_coupling_segments.csv",
        f"{root.label} OBS-050 structural_coupling_segments.csv",
    )
    segs = normalize_outcome_column(segs)
    segs = normalize_coupling_column(segs)
    segs = normalize_path_id(segs)
    segs["corpus"] = root.label

    if "seam_band" not in segs.columns:
        if "segment_seam_band" in segs.columns:
            segs = segs.rename(columns={"segment_seam_band": "seam_band"})
        else:
            mean_d = numeric(segs, "mean_distance_to_seam")
            min_d = numeric(segs, "min_distance_to_seam")
            segs["seam_band"] = np.where(
                min_d <= 1e-12,
                "core",
                np.where(mean_d <= 0.15, "near", "far"),
            )

    if "posture" not in segs.columns:
        if "seam_posture" in segs.columns:
            segs = segs.rename(columns={"seam_posture": "posture"})
        elif "m_seam" in segs.columns:
            m = numeric(segs, "m_seam")
            segs["posture"] = np.where(
                m < -0.02,
                "compression",
                np.where(m > 0.02, "dissipation", "graze"),
            )
        else:
            segs["posture"] = "unknown"

    return segs


def load_family_assignments(root: CorpusRoot) -> pd.DataFrame | None:
    df = read_csv_optional(root.family_assignments_csv)
    if df is None:
        return None
    df = normalize_path_id(df)

    if "path_family" not in df.columns:
        return None

    keep = [c for c in ["path_id", "path_family"] if c in df.columns]
    return df[keep].drop_duplicates("path_id").copy()


def load_path_diagnostics(root: CorpusRoot) -> pd.DataFrame | None:
    df = read_csv_optional(root.path_diagnostics_csv)
    if df is None:
        return None
    df = normalize_path_id(df)

    keep_cols = [
        c
        for c in [
            "path_id",
            "path_family",
            "mean_distance_to_seam",
            "min_distance_to_seam",
            "mean_lazarus",
            "max_lazarus",
            "mean_lazarus_score",
            "max_lazarus_score",
            "mean_signed_phase",
            "phase_span",
            "roughness",
            "mean_roughness",
            "max_roughness",
            "path_length",
            "n_steps",
            "seam_fraction",
            "criticality_load",
            "mean_criticality",
            "max_criticality",
            "mean_obstruction",
            "max_obstruction",
        ]
        if c in df.columns
    ]
    if "path_id" not in keep_cols:
        return None
    return df[keep_cols].drop_duplicates("path_id").copy()


def load_path_node_diagnostics(root: CorpusRoot) -> pd.DataFrame | None:
    df = read_csv_optional(root.path_node_diagnostics_csv)
    if df is None:
        return None
    df = normalize_path_id(df)

    if "path_id" not in df.columns or "step" not in df.columns:
        return None

    # Keep only columns likely useful for local center-step enrichment.
    keep_cols = [
        c
        for c in [
            "path_id",
            "step",
            "node_id",
            "r",
            "alpha",
            "path_family",
            "mds1",
            "mds2",
            "signed_phase",
            "distance_to_seam",
            "lazarus_score",
            "lazarus_hit",
            "criticality",
            "criticality_score",
            "mean_criticality",
            "obstruction",
            "holonomy",
            "absolute_holonomy",
            "unsigned_obstruction",
            "node_holonomy_proxy",
            "response_strength",
            "rsp_anisotropy",
            "roughness",
            "mean_roughness",
            "roughness_smoothed",
            "roughness_slope",
            "m_r",
            "m_seam",
        ]
        if c in df.columns
    ]

    out = df[keep_cols].copy()
    out["path_id"] = out["path_id"].astype(str)
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out = out.dropna(subset=["path_id", "step"]).copy()
    out["step"] = out["step"].astype(int)
    return out


def load_node_fields(root: CorpusRoot) -> pd.DataFrame | None:
    """
    Load node/grid fields at r, alpha level for optional enrichment.
    Priority: scene_nodes, then FIM / Lazarus / phase if available.
    """
    scene = read_csv_optional(root.scene_nodes_csv)
    if scene is None:
        return None

    node = scene.copy()
    if not {"r", "alpha"}.issubset(node.columns):
        return None

    keep_scene = [
        c
        for c in [
            "r",
            "alpha",
            "distance_to_seam",
            "signed_phase",
            "lazarus_score",
            "lazarus_hit",
            "response_strength",
            "rsp_anisotropy",
            "node_holonomy_proxy",
        ]
        if c in node.columns
    ]
    node = node[keep_scene].drop_duplicates(["r", "alpha"]).copy()

    fim = read_csv_optional(root.fim_surface_csv)
    if fim is not None and {"r", "alpha"}.issubset(fim.columns):
        keep_fim = [
            c
            for c in [
                "r",
                "alpha",
                "fim_det",
                "fim_trace",
                "fim_eig1",
                "fim_eig2",
                "fim_cond",
                "fim_theta",
            ]
            if c in fim.columns
        ]
        node = node.merge(fim[keep_fim], on=["r", "alpha"], how="left")

    laz = read_csv_optional(root.lazarus_scores_csv)
    if laz is not None and {"r", "alpha"}.issubset(laz.columns):
        keep_laz = [
            c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns
        ]
        keep_laz = [c for c in keep_laz if c not in node.columns or c in ["r", "alpha"]]
        if len(keep_laz) > 2:
            node = node.merge(laz[keep_laz], on=["r", "alpha"], how="left")

    phase = read_csv_optional(root.phase_distance_csv)
    if phase is not None and {"r", "alpha"}.issubset(phase.columns):
        keep_phase = [c for c in ["r", "alpha", "distance_to_seam"] if c in phase.columns]
        keep_phase = [c for c in keep_phase if c not in node.columns or c in ["r", "alpha"]]
        if len(keep_phase) > 2:
            node = node.merge(phase[keep_phase], on=["r", "alpha"], how="left")

    return node


def nearest_center_rows(
    segments: pd.DataFrame,
    node_diag: pd.DataFrame,
    *,
    max_step_delta: int,
) -> pd.DataFrame:
    """
    Attach one center-step diagnostic row to each segment.

    Strategy:
    - exact path_id + center_step if available
    - otherwise nearest step in same path, bounded by max_step_delta
    """
    if "center_step" not in segments.columns:
        out = segments.copy()
        out["center_enrichment_status"] = "missing_center_step"
        out["center_step_matched"] = np.nan
        out["center_step_delta"] = np.nan
        return out

    seg = segments.copy().reset_index(drop=False).rename(columns={"index": "_segment_row_id"})
    seg["path_id"] = seg["path_id"].astype(str)
    seg["center_step"] = pd.to_numeric(seg["center_step"], errors="coerce")
    seg = seg.dropna(subset=["path_id", "center_step"]).copy()
    seg["center_step"] = seg["center_step"].astype(int)

    nd = node_diag.copy()
    nd["path_id"] = nd["path_id"].astype(str)
    nd["step"] = pd.to_numeric(nd["step"], errors="coerce")
    nd = nd.dropna(subset=["path_id", "step"]).copy()
    nd["step"] = nd["step"].astype(int)

    # Exact join first.
    nd_exact = nd.rename(columns={"step": "center_step"}).copy()
    enrich_cols = [c for c in nd_exact.columns if c not in {"path_id", "center_step"}]
    exact = seg[["_segment_row_id", "path_id", "center_step"]].merge(
        nd_exact[["path_id", "center_step"] + enrich_cols],
        on=["path_id", "center_step"],
        how="left",
    )
    exact["_has_exact"] = exact[enrich_cols].notna().any(axis=1) if enrich_cols else False

    exact_ids = set(exact.loc[exact["_has_exact"], "_segment_row_id"].tolist())

    # Nearest join for rows without exact match.
    missing = seg[~seg["_segment_row_id"].isin(exact_ids)][
        ["_segment_row_id", "path_id", "center_step"]
    ].copy()

    nearest_rows: list[dict[str, Any]] = []
    if not missing.empty:
        nd_by_path = {pid: g.sort_values("step") for pid, g in nd.groupby("path_id", sort=False)}

        for _, row in missing.iterrows():
            pid = str(row["path_id"])
            cstep = int(row["center_step"])
            g = nd_by_path.get(pid)
            if g is None or g.empty:
                nearest_rows.append(
                    {
                        "_segment_row_id": row["_segment_row_id"],
                        "center_enrichment_status": "missing_path",
                        "center_step_matched": np.nan,
                        "center_step_delta": np.nan,
                    }
                )
                continue

            steps = g["step"].to_numpy(dtype=int)
            idx = int(np.argmin(np.abs(steps - cstep)))
            matched = int(steps[idx])
            delta = int(matched - cstep)

            if abs(delta) > max_step_delta:
                nearest_rows.append(
                    {
                        "_segment_row_id": row["_segment_row_id"],
                        "center_enrichment_status": "nearest_too_far",
                        "center_step_matched": matched,
                        "center_step_delta": delta,
                    }
                )
                continue

            src = g.iloc[idx].to_dict()
            out_row = {
                "_segment_row_id": row["_segment_row_id"],
                "center_enrichment_status": "nearest",
                "center_step_matched": matched,
                "center_step_delta": delta,
            }
            for k, v in src.items():
                if k in {"path_id"}:
                    continue
                out_row[f"center_{k}"] = v
            nearest_rows.append(out_row)

    # Convert exact matches into same center_* schema.
    exact_rows = []
    for _, row in exact[exact["_has_exact"]].iterrows():
        out_row = {
            "_segment_row_id": row["_segment_row_id"],
            "center_enrichment_status": "exact",
            "center_step_matched": row["center_step"],
            "center_step_delta": 0,
        }
        for c in enrich_cols:
            out_row[f"center_{c}"] = row[c]
        exact_rows.append(out_row)

    center = pd.DataFrame(exact_rows + nearest_rows)
    if center.empty:
        out = segments.copy()
        out["center_enrichment_status"] = "no_center_matches"
        out["center_step_matched"] = np.nan
        out["center_step_delta"] = np.nan
        return out

    original = segments.copy().reset_index(drop=False).rename(columns={"index": "_segment_row_id"})
    out = original.merge(center, on="_segment_row_id", how="left")
    out["center_enrichment_status"] = out["center_enrichment_status"].fillna("unmatched")
    out = out.drop(columns=["_segment_row_id"])
    return out


def enrich_segments(root: CorpusRoot, *, max_step_delta: int) -> pd.DataFrame:
    segs = load_segments(root)

    fam = load_family_assignments(root)
    if fam is not None:
        if "path_family" in segs.columns:
            fam = fam.rename(columns={"path_family": "path_family_from_assignments"})
        segs = segs.merge(fam, on="path_id", how="left")
        if "path_family" not in segs.columns and "path_family_from_assignments" in segs.columns:
            segs["path_family"] = segs["path_family_from_assignments"]

    path_diag = load_path_diagnostics(root)
    if path_diag is not None:
        rename = {}
        for c in path_diag.columns:
            if c == "path_id":
                continue
            if c in segs.columns:
                rename[c] = f"path_{c}"
        path_diag = path_diag.rename(columns=rename)
        segs = segs.merge(path_diag, on="path_id", how="left")

    path_node_diag = load_path_node_diagnostics(root)
    if path_node_diag is not None:
        segs = nearest_center_rows(segs, path_node_diag, max_step_delta=max_step_delta)
    else:
        segs["center_enrichment_status"] = "missing_path_node_diagnostics"
        segs["center_step_matched"] = np.nan
        segs["center_step_delta"] = np.nan

    node_fields = load_node_fields(root)

    # Prefer center r/alpha for grid enrichment when available.
    if "center_r" in segs.columns and "r" not in segs.columns:
        segs["r"] = pd.to_numeric(segs["center_r"], errors="coerce")
    if "center_alpha" in segs.columns and "alpha" not in segs.columns:
        segs["alpha"] = pd.to_numeric(segs["center_alpha"], errors="coerce")

    if node_fields is not None and {"r", "alpha"}.issubset(segs.columns):
        rename = {}
        for c in node_fields.columns:
            if c in {"r", "alpha"}:
                continue
            if c in segs.columns:
                rename[c] = f"node_{c}"
        node_fields = node_fields.rename(columns=rename)
        segs = segs.merge(node_fields, on=["r", "alpha"], how="left")

    if "path_family" not in segs.columns:
        segs["path_family"] = "unknown"

    return segs


# ---------------------------------------------------------------------
# Summary functions
# ---------------------------------------------------------------------


def summarize_nonrecovering_coupling(segs: pd.DataFrame) -> pd.DataFrame:
    nonrec = segs[segs["outcome_group"].astype(str) == "nonrecovering"].copy()

    rows = []
    for corpus, grp in nonrec.groupby("corpus", dropna=False):
        n_total = len(grp)
        for coupling, sub in grp.groupby("coupling_class", dropna=False):
            n = len(sub)
            rows.append(
                {
                    "corpus": corpus,
                    "outcome_group": "nonrecovering",
                    "coupling_class": coupling,
                    "n_segments": n,
                    "segment_share_within_nonrecovering": n / n_total if n_total else np.nan,
                    "mean_m_seam": finite_mean(get_first_series(sub, ["m_seam", "center_m_seam"])),
                    "mean_mean_distance_to_seam": finite_mean(
                        get_first_series(sub, ["mean_distance_to_seam", "center_distance_to_seam"])
                    ),
                    "mean_min_distance_to_seam": finite_mean(
                        get_first_series(sub, ["min_distance_to_seam", "center_distance_to_seam"])
                    ),
                    "mean_roughness": finite_mean(
                        get_first_series(sub, ["mean_roughness", "roughness", "center_roughness"])
                    ),
                    "mean_lazarus_score": finite_mean(
                        get_first_series(sub, ["center_lazarus_score", "lazarus_score", "node_lazarus_score"])
                    ),
                    "mean_signed_phase": finite_mean(
                        get_first_series(sub, ["center_signed_phase", "signed_phase", "node_signed_phase"])
                    ),
                    "mean_fim_trace": finite_mean(get_first_series(sub, ["fim_trace", "node_fim_trace"])),
                    "mean_fim_det": finite_mean(get_first_series(sub, ["fim_det", "node_fim_det"])),
                    "center_exact_share": float(
                        (sub.get("center_enrichment_status", pd.Series(index=sub.index)) == "exact").mean()
                    )
                    if "center_enrichment_status" in sub.columns
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize_family_band_posture(segs: pd.DataFrame) -> pd.DataFrame:
    nonrec = segs[segs["outcome_group"].astype(str) == "nonrecovering"].copy()

    group_cols = [
        "corpus",
        "coupling_class",
        "path_family",
        "seam_band",
        "posture",
    ]

    for col in group_cols:
        if col not in nonrec.columns:
            nonrec[col] = "unknown"

    total_by_corpus = nonrec.groupby("corpus").size().to_dict()
    total_coupled_by_corpus = (
        nonrec[nonrec["coupling_class"].astype(str) == "coupled"]
        .groupby("corpus")
        .size()
        .to_dict()
    )

    rows = []
    for keys, grp in nonrec.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        corpus = row["corpus"]
        n = len(grp)
        coupled_den = total_coupled_by_corpus.get(corpus, 0)
        rows.append(
            {
                **row,
                "n_segments": n,
                "share_within_nonrecovering": n / total_by_corpus.get(corpus, np.nan),
                "share_within_nonrecovering_coupled": (
                    n / coupled_den
                    if row["coupling_class"] == "coupled" and coupled_den
                    else np.nan
                ),
                "mean_m_seam": finite_mean(get_first_series(grp, ["m_seam", "center_m_seam"])),
                "mean_m_r": finite_mean(get_first_series(grp, ["m_r", "center_m_r"])),
                "mean_roughness": finite_mean(
                    get_first_series(grp, ["mean_roughness", "roughness", "center_roughness"])
                ),
                "mean_mean_distance_to_seam": finite_mean(
                    get_first_series(grp, ["mean_distance_to_seam", "center_distance_to_seam"])
                ),
                "mean_min_distance_to_seam": finite_mean(
                    get_first_series(grp, ["min_distance_to_seam", "center_distance_to_seam"])
                ),
                "mean_lazarus_score": finite_mean(
                    get_first_series(grp, ["center_lazarus_score", "lazarus_score", "node_lazarus_score"])
                ),
                "mean_signed_phase": finite_mean(
                    get_first_series(grp, ["center_signed_phase", "signed_phase", "node_signed_phase"])
                ),
                "mean_fim_trace": finite_mean(get_first_series(grp, ["fim_trace", "node_fim_trace"])),
                "mean_fim_det": finite_mean(get_first_series(grp, ["fim_det", "node_fim_det"])),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["corpus", "coupling_class", "n_segments"],
            ascending=[True, True, False],
        )
    return out


def summarize_grid_concentration(segs: pd.DataFrame) -> pd.DataFrame:
    nonrec = segs[segs["outcome_group"].astype(str) == "nonrecovering"].copy()
    nonrec = nonrec[nonrec["coupling_class"].astype(str) == "coupled"].copy()

    if not {"r", "alpha"}.issubset(nonrec.columns):
        rows = []
        for corpus, grp in nonrec.groupby("corpus", dropna=False):
            rows.append(
                {
                    "corpus": corpus,
                    "status": "missing_r_alpha",
                    "n_nonrecovering_coupled_segments": int(len(grp)),
                }
            )
        return pd.DataFrame(rows)

    nonrec["r"] = pd.to_numeric(nonrec["r"], errors="coerce")
    nonrec["alpha"] = pd.to_numeric(nonrec["alpha"], errors="coerce")
    nonrec = nonrec.dropna(subset=["r", "alpha"]).copy()

    total_by_corpus = nonrec.groupby("corpus").size().to_dict()

    group_cols = ["corpus", "r", "alpha"]
    rows = []
    for keys, grp in nonrec.groupby(group_cols, dropna=False):
        corpus, r, alpha = keys
        n = len(grp)
        rows.append(
            {
                "corpus": corpus,
                "status": "ok",
                "r": r,
                "alpha": alpha,
                "n_nonrecovering_coupled_segments": n,
                "share_within_nonrecovering_coupled": n / total_by_corpus.get(corpus, np.nan),
                "mean_m_seam": finite_mean(get_first_series(grp, ["m_seam", "center_m_seam"])),
                "mean_mean_distance_to_seam": finite_mean(
                    get_first_series(grp, ["mean_distance_to_seam", "center_distance_to_seam"])
                ),
                "mean_min_distance_to_seam": finite_mean(
                    get_first_series(grp, ["min_distance_to_seam", "center_distance_to_seam"])
                ),
                "mean_roughness": finite_mean(
                    get_first_series(grp, ["mean_roughness", "roughness", "center_roughness"])
                ),
                "mean_lazarus_score": finite_mean(
                    get_first_series(grp, ["center_lazarus_score", "lazarus_score", "node_lazarus_score"])
                ),
                "mean_signed_phase": finite_mean(
                    get_first_series(grp, ["center_signed_phase", "signed_phase", "node_signed_phase"])
                ),
                "mean_fim_trace": finite_mean(get_first_series(grp, ["fim_trace", "node_fim_trace"])),
                "mean_fim_det": finite_mean(get_first_series(grp, ["fim_det", "node_fim_det"])),
                "center_exact_share": float(
                    (grp.get("center_enrichment_status", pd.Series(index=grp.index)) == "exact").mean()
                )
                if "center_enrichment_status" in grp.columns
                else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["corpus", "n_nonrecovering_coupled_segments"],
            ascending=[True, False],
        )
    return out


def summarize_center_step_fields(segs: pd.DataFrame) -> pd.DataFrame:
    nonrec = segs[segs["outcome_group"].astype(str) == "nonrecovering"].copy()

    fields = [
        "center_distance_to_seam",
        "center_lazarus_score",
        "center_signed_phase",
        "center_criticality",
        "center_criticality_score",
        "center_obstruction",
        "center_holonomy",
        "center_absolute_holonomy",
        "center_unsigned_obstruction",
        "center_response_strength",
        "center_rsp_anisotropy",
        "center_roughness",
        "center_roughness_smoothed",
        "center_roughness_slope",
    ]
    fields = [f for f in fields if f in nonrec.columns]

    rows = []
    for keys, grp in nonrec.groupby(["corpus", "coupling_class"], dropna=False):
        corpus, coupling = keys
        row_base = {
            "corpus": corpus,
            "outcome_group": "nonrecovering",
            "coupling_class": coupling,
            "n_segments": len(grp),
        }
        for field in fields:
            rows.append(
                {
                    **row_base,
                    "field": field,
                    "defined": int(pd.to_numeric(grp[field], errors="coerce").notna().sum()),
                    "mean": finite_mean(grp[field]),
                    "median": finite_median(grp[field]),
                    "min": finite_min(grp[field]),
                    "max": finite_max(grp[field]),
                }
            )

    return pd.DataFrame(rows)


def summarize_field_contrast(segs: pd.DataFrame) -> pd.DataFrame:
    nonrec = segs[segs["outcome_group"].astype(str) == "nonrecovering"].copy()

    fields = [
        "m_seam",
        "m_r",
        "mean_roughness",
        "roughness",
        "mean_distance_to_seam",
        "min_distance_to_seam",
        "center_distance_to_seam",
        "center_lazarus_score",
        "center_signed_phase",
        "center_criticality",
        "center_obstruction",
        "center_holonomy",
        "fim_trace",
        "fim_det",
        "fim_cond",
        "response_strength",
        "node_response_strength",
        "rsp_anisotropy",
        "node_rsp_anisotropy",
    ]
    fields = [f for f in fields if f in nonrec.columns]

    rows = []
    for corpus, grp in nonrec.groupby("corpus", dropna=False):
        coupled = grp[grp["coupling_class"].astype(str) == "coupled"]
        decoupled = grp[grp["coupling_class"].astype(str) == "decoupled"]

        for field in fields:
            c_mean = finite_mean(coupled[field])
            d_mean = finite_mean(decoupled[field])
            rows.append(
                {
                    "corpus": corpus,
                    "field": field,
                    "coupled_mean": c_mean,
                    "decoupled_mean": d_mean,
                    "coupled_minus_decoupled": c_mean - d_mean,
                    "coupled_median": finite_median(coupled[field]),
                    "decoupled_median": finite_median(decoupled[field]),
                    "coupled_defined": int(pd.to_numeric(coupled[field], errors="coerce").notna().sum()),
                    "decoupled_defined": int(pd.to_numeric(decoupled[field], errors="coerce").notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def summarize_corpus_delta(
    coupling: pd.DataFrame,
    family_band: pd.DataFrame,
    field_contrast: pd.DataFrame,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    c = coupling[
        (coupling["outcome_group"] == "nonrecovering")
        & (coupling["coupling_class"] == "coupled")
    ].copy()
    if set(c["corpus"]) >= {left_label, right_label}:
        l = c[c["corpus"] == left_label].iloc[0]
        r = c[c["corpus"] == right_label].iloc[0]
        for col in [
            "segment_share_within_nonrecovering",
            "mean_m_seam",
            "mean_mean_distance_to_seam",
            "mean_min_distance_to_seam",
            "mean_roughness",
            "mean_lazarus_score",
            "mean_signed_phase",
            "mean_fim_trace",
            "mean_fim_det",
            "center_exact_share",
        ]:
            if col in c.columns:
                rows.append(
                    {
                        "comparison": f"nonrecovering_coupled_{col}",
                        "left_label": left_label,
                        "right_label": right_label,
                        "left_value": l.get(col, np.nan),
                        "right_value": r.get(col, np.nan),
                        "right_minus_left": as_float(r.get(col, np.nan)) - as_float(l.get(col, np.nan)),
                    }
                )

    f = family_band[family_band["coupling_class"].astype(str) == "coupled"].copy()
    if not f.empty:
        key_cols = ["path_family", "seam_band", "posture"]
        value_col = "share_within_nonrecovering_coupled"
        lf = f[f["corpus"] == left_label][key_cols + [value_col]].rename(
            columns={value_col: "left_value"}
        )
        rf = f[f["corpus"] == right_label][key_cols + [value_col]].rename(
            columns={value_col: "right_value"}
        )
        merged = lf.merge(rf, on=key_cols, how="outer")
        merged["left_value"] = pd.to_numeric(merged["left_value"], errors="coerce").fillna(0.0)
        merged["right_value"] = pd.to_numeric(merged["right_value"], errors="coerce").fillna(0.0)
        merged["right_minus_left"] = merged["right_value"] - merged["left_value"]
        for _, row in merged.sort_values("right_minus_left", ascending=False).iterrows():
            rows.append(
                {
                    "comparison": (
                        "nonrecovering_coupled_composition:"
                        f"{row['path_family']}|{row['seam_band']}|{row['posture']}"
                    ),
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_value": row["left_value"],
                    "right_value": row["right_value"],
                    "right_minus_left": row["right_minus_left"],
                }
            )

    if not field_contrast.empty:
        lf = field_contrast[field_contrast["corpus"] == left_label][
            ["field", "coupled_minus_decoupled"]
        ].rename(columns={"coupled_minus_decoupled": "left_value"})
        rf = field_contrast[field_contrast["corpus"] == right_label][
            ["field", "coupled_minus_decoupled"]
        ].rename(columns={"coupled_minus_decoupled": "right_value"})
        merged = lf.merge(rf, on="field", how="inner")
        for _, row in merged.iterrows():
            rows.append(
                {
                    "comparison": f"field_contrast_coupled_minus_decoupled:{row['field']}",
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_value": row["left_value"],
                    "right_value": row["right_value"],
                    "right_minus_left": as_float(row["right_value"]) - as_float(row["left_value"]),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------


def fmt(x: Any, digits: int = 6) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def top_rows_markdown(
    df: pd.DataFrame,
    *,
    n: int,
    cols: list[str],
) -> list[str]:
    if df.empty:
        return ["No rows."]

    use = df.head(n).copy()
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in use.iterrows():
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(fmt(val, 4))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    cfg: Config,
    manifest: pd.DataFrame,
    coupling: pd.DataFrame,
    family_band: pd.DataFrame,
    grid: pd.DataFrame,
    center_fields: pd.DataFrame,
    field_contrast: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    left = cfg.left.label
    right = cfg.right.label

    missing = manifest[manifest["exists"] == 0]

    lines: list[str] = [
        "# OBS-072 — Cp2 nonrecovering seam-drift diagnostic",
        "",
        "## Scope",
        "",
        "This diagnostic compares nonrecovering OBS-050 segments across two artifact roots.",
        "It focuses on why Cp2 has a higher nonrecovering seam-coupled baseline than C.",
        "",
        "v2 enriches segment rows from path-node diagnostics at `path_id + center_step`.",
        "",
        "## Roots",
        "",
        f"- `{left}` root: `{cfg.left.root}`",
        f"- `{right}` root: `{cfg.right.root}`",
        f"- scale: `{cfg.left.scale}` / `{cfg.right.scale}`",
        f"- max center-step delta for nearest enrichment: `{cfg.max_step_delta}`",
        "",
        "## Artifact availability",
        "",
        f"- checked artifacts: `{len(manifest)}`",
        f"- missing artifacts: `{len(missing)}`",
        "",
    ]

    if not missing.empty:
        lines.append("Missing artifacts:")
        for _, row in missing.iterrows():
            lines.append(f"- `{row['corpus']}` `{row['artifact']}`: `{row['path']}`")
        lines.append("")

    lines.extend(["## Headline nonrecovering coupling", ""])

    c = coupling[
        (coupling["outcome_group"] == "nonrecovering")
        & (coupling["coupling_class"] == "coupled")
    ].copy()

    for corpus in [left, right]:
        sub = c[c["corpus"] == corpus]
        if sub.empty:
            lines.append(f"- `{corpus}`: missing nonrecovering coupled summary")
            continue
        r = sub.iloc[0]
        lines.append(
            f"- `{corpus}`: nonrecovering coupled share "
            f"`{fmt(r.get('segment_share_within_nonrecovering'))}`, "
            f"segments `{int(r.get('n_segments', 0))}`, "
            f"mean distance-to-seam `{fmt(r.get('mean_mean_distance_to_seam'))}`, "
            f"mean min-distance-to-seam `{fmt(r.get('mean_min_distance_to_seam'))}`, "
            f"mean roughness `{fmt(r.get('mean_roughness'))}`, "
            f"mean center Lazarus `{fmt(r.get('mean_lazarus_score'))}`"
        )
    lines.append("")

    headline = delta[delta["comparison"] == "nonrecovering_coupled_segment_share_within_nonrecovering"]
    if not headline.empty:
        r = headline.iloc[0]
        lines.append(
            f"Difference `{right} - {left}` in nonrecovering coupled share: "
            f"`{fmt(r['right_minus_left'])}`."
        )
        lines.append("")

    lines.extend(
        [
            "## Top nonrecovering-coupled composition shifts",
            "",
            "Rows are composition shares within nonrecovering-coupled segments.",
            "",
        ]
    )

    comp = delta[delta["comparison"].astype(str).str.startswith("nonrecovering_coupled_composition:")].copy()
    comp = comp.sort_values("right_minus_left", ascending=False)
    lines.extend(
        top_rows_markdown(
            comp,
            n=12,
            cols=["comparison", "left_value", "right_value", "right_minus_left"],
        )
    )
    lines.append("")

    lines.extend(["## Strongest Cp2 nonrecovering-coupled grid concentrations", ""])

    if "corpus" not in grid.columns or "r" not in grid.columns or "alpha" not in grid.columns:
        lines.append(
            "Grid concentration could not be computed because enriched segment rows "
            "do not carry `r, alpha` coordinates."
        )
    else:
        grid_right = grid[grid["corpus"] == right].copy()
        if not grid_right.empty and "n_nonrecovering_coupled_segments" in grid_right.columns:
            grid_right = grid_right.sort_values(
                "n_nonrecovering_coupled_segments", ascending=False
            )
        lines.extend(
            top_rows_markdown(
                grid_right,
                n=12,
                cols=[
                    "corpus",
                    "r",
                    "alpha",
                    "n_nonrecovering_coupled_segments",
                    "share_within_nonrecovering_coupled",
                    "mean_mean_distance_to_seam",
                    "mean_roughness",
                    "mean_lazarus_score",
                ],
            )
        )
    lines.append("")

    lines.extend(
        [
            "## Center-step field summary",
            "",
            "Fields are measured at the matched segment center step where available.",
            "",
        ]
    )
    cf = center_fields.copy()
    if not cf.empty:
        cf = cf.sort_values(["corpus", "coupling_class", "field"])
    lines.extend(
        top_rows_markdown(
            cf,
            n=60,
            cols=["corpus", "coupling_class", "field", "defined", "mean", "median"],
        )
    )
    lines.append("")

    lines.extend(
        [
            "## Field contrast: nonrecovering coupled minus decoupled",
            "",
            "Positive values mean the field is larger in nonrecovering-coupled segments than in nonrecovering-decoupled segments.",
            "",
        ]
    )

    fc = field_contrast.copy()
    if not fc.empty:
        fc = fc.sort_values(["corpus", "field"])
    lines.extend(
        top_rows_markdown(
            fc,
            n=60,
            cols=["corpus", "field", "coupled_mean", "decoupled_mean", "coupled_minus_decoupled"],
        )
    )
    lines.append("")

    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- This diagnostic isolates pattern structure; it does not prove whether the cause is tokenizer, embedding geometry, or corpus dynamics.",
            "- Elevated Cp2 nonrecovering seam coupling can arise from more nonrecovering traffic in seam-adjacent regions, more off-seam reorganizing paths that graze the seam, or a different geometry of the entropy manifold.",
            "- Center-step enrichment depends on the availability and semantics of `path_node_diagnostics.csv`.",
            "- OBS-072 should be read alongside OBS-071, not as a replacement for it.",
            "",
            "## Output tables",
            "",
            "- `obs072_input_manifest.csv`",
            "- `obs072_nonrecovering_segment_enriched.csv`",
            "- `obs072_nonrecovering_coupling_summary.csv`",
            "- `obs072_family_band_posture_summary.csv`",
            "- `obs072_grid_concentration_summary.csv`",
            "- `obs072_center_step_field_summary.csv`",
            "- `obs072_field_contrast_summary.csv`",
            "- `obs072_corpus_delta_summary.csv`",
            "",
        ]
    )

    (cfg.outdir / "obs072_cp2_nonrecovering_seam_drift_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OBS-072 Cp2 nonrecovering seam-drift diagnostic."
    )
    p.add_argument("--left-label", default="C")
    p.add_argument("--left-root", default="outputs")
    p.add_argument("--right-label", default="Cp2")
    p.add_argument("--right-root", default="outputs/corpora/Cp2/campaigns/full_v2/pipeline")
    p.add_argument("--scale", default="100000")
    p.add_argument("--left-scale", default=None)
    p.add_argument("--right-scale", default=None)
    p.add_argument("--outdir", default="outputs/obs072_cp2_nonrecovering_seam_drift")
    p.add_argument(
        "--max-step-delta",
        type=int,
        default=2,
        help="Maximum allowed nearest-step offset for center-step enrichment.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    left_scale = args.left_scale or args.scale
    right_scale = args.right_scale or args.scale

    cfg = Config(
        left=CorpusRoot(args.left_label, Path(args.left_root), str(left_scale)),
        right=CorpusRoot(args.right_label, Path(args.right_root), str(right_scale)),
        outdir=Path(args.outdir),
        max_step_delta=args.max_step_delta,
    )

    ensure_outdir(cfg.outdir)

    manifest = pd.concat(
        [build_manifest(cfg.left), build_manifest(cfg.right)],
        ignore_index=True,
    )
    manifest.to_csv(cfg.outdir / "obs072_input_manifest.csv", index=False)

    left_segments = enrich_segments(cfg.left, max_step_delta=cfg.max_step_delta)
    right_segments = enrich_segments(cfg.right, max_step_delta=cfg.max_step_delta)
    segments = pd.concat([left_segments, right_segments], ignore_index=True)

    nonrec = segments[segments["outcome_group"].astype(str) == "nonrecovering"].copy()
    nonrec.to_csv(cfg.outdir / "obs072_nonrecovering_segment_enriched.csv", index=False)

    coupling = summarize_nonrecovering_coupling(segments)
    coupling.to_csv(cfg.outdir / "obs072_nonrecovering_coupling_summary.csv", index=False)

    family_band = summarize_family_band_posture(segments)
    family_band.to_csv(cfg.outdir / "obs072_family_band_posture_summary.csv", index=False)

    grid = summarize_grid_concentration(segments)
    grid.to_csv(cfg.outdir / "obs072_grid_concentration_summary.csv", index=False)

    center_fields = summarize_center_step_fields(segments)
    center_fields.to_csv(cfg.outdir / "obs072_center_step_field_summary.csv", index=False)

    field_contrast = summarize_field_contrast(segments)
    field_contrast.to_csv(cfg.outdir / "obs072_field_contrast_summary.csv", index=False)

    delta = summarize_corpus_delta(
        coupling=coupling,
        family_band=family_band,
        field_contrast=field_contrast,
        left_label=cfg.left.label,
        right_label=cfg.right.label,
    )
    delta.to_csv(cfg.outdir / "obs072_corpus_delta_summary.csv", index=False)

    write_report(
        cfg=cfg,
        manifest=manifest,
        coupling=coupling,
        family_band=family_band,
        grid=grid,
        center_fields=center_fields,
        field_contrast=field_contrast,
        delta=delta,
    )

    print(cfg.outdir / "obs072_input_manifest.csv")
    print(cfg.outdir / "obs072_nonrecovering_segment_enriched.csv")
    print(cfg.outdir / "obs072_nonrecovering_coupling_summary.csv")
    print(cfg.outdir / "obs072_family_band_posture_summary.csv")
    print(cfg.outdir / "obs072_grid_concentration_summary.csv")
    print(cfg.outdir / "obs072_center_step_field_summary.csv")
    print(cfg.outdir / "obs072_field_contrast_summary.csv")
    print(cfg.outdir / "obs072_corpus_delta_summary.csv")
    print(cfg.outdir / "obs072_cp2_nonrecovering_seam_drift_summary.md")


if __name__ == "__main__":
    main()