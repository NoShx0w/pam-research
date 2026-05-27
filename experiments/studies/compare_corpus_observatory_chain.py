#!/usr/bin/env python3
"""
compare_corpus_observatory_chain.py

Compare two PAM observatory-chain artifact roots.

Purpose
-------
Produce a compact cross-corpus comparison over the OBS-028c → OBS-051 chain,
with emphasis on measurements that are already registry-visible and
scientifically interpretable:

1. OBS-050 structural coupling persistence
2. OBS-051 local divergence / boundedness by seam band
3. scale-family substrate composition
4. OBS-028c seam bundle summary
5. optional node-field summaries when available

This script does not recompute observatory artifacts. It only reads completed
artifact stores and writes comparison tables/reports.

Typical usage
-------------
Legacy C root vs scoped Cp2 root:

    PYTHONPATH=src:experiments .venv/bin/python \
      experiments/studies/compare_corpus_observatory_chain.py \
      --left-label C \
      --left-root outputs \
      --right-label Cp2 \
      --right-root outputs/corpora/Cp2/campaigns/full_v2/pipeline \
      --scale 100000 \
      --outdir outputs/comparisons/C_vs_Cp2_observatory_chain

Outputs
-------
<outdir>/
  corpus_root_manifest.csv
  obs050_coupling_comparison.csv
  obs051_banded_comparison.csv
  family_substrate_comparison.csv
  obs028c_seam_bundle_comparison.csv
  node_field_comparison.csv
  comparison_summary.md

Guardrails
----------
- Missing optional artifacts are recorded, not silently invented.
- Differences are directional measurements, not universality claims.
- OBS-051 remains provisional unless independently stabilized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
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
    def obs028c_dir(self) -> Path:
        return self.root / "obs028c_canonical_seam_bundle"

    @property
    def obs050_dir(self) -> Path:
        return self.root / "obs050_structural_coupling_persistence"

    @property
    def obs051_dir(self) -> Path:
        return self.root / "obs051_local_divergence_in_coupled_windows"

    @property
    def scene_bundle_dir(self) -> Path:
        return self.root / "obs022_scene_bundle"

    @property
    def fim_phase_dir(self) -> Path:
        return self.root / "fim_phase"

    @property
    def fim_lazarus_dir(self) -> Path:
        return self.root / "fim_lazarus"

    @property
    def fim_dir(self) -> Path:
        return self.root / "fim"


@dataclass(frozen=True)
class CompareConfig:
    left: CorpusRoot
    right: CorpusRoot
    outdir: Path


# ---------------------------------------------------------------------
# Basic IO helpers
# ---------------------------------------------------------------------


def read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_text_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite_mean(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def finite_median(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.median()) if x.notna().any() else float("nan")


def finite_min(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.min()) if x.notna().any() else float("nan")


def finite_max(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce")
    return float(x.max()) if x.notna().any() else float("nan")


def compare_wide(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key_cols: list[str],
    value_cols: list[str],
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    l = left[key_cols + value_cols].copy()
    r = right[key_cols + value_cols].copy()

    l = l.rename(columns={c: f"{left_label}_{c}" for c in value_cols})
    r = r.rename(columns={c: f"{right_label}_{c}" for c in value_cols})

    out = l.merge(r, on=key_cols, how="outer")

    for col in value_cols:
        lc = f"{left_label}_{col}"
        rc = f"{right_label}_{col}"
        if lc in out.columns and rc in out.columns:
            out[f"delta_{right_label}_minus_{left_label}_{col}"] = (
                pd.to_numeric(out[rc], errors="coerce")
                - pd.to_numeric(out[lc], errors="coerce")
            )
    return out


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------


def artifact_status(root: CorpusRoot) -> pd.DataFrame:
    checks = [
        ("family_substrate_summary", root.family_substrate_dir / "path_family_summary.csv"),
        ("family_substrate_assignments", root.family_substrate_dir / "path_family_assignments.csv"),
        ("family_substrate_path_diagnostics", root.family_substrate_dir / "path_diagnostics.csv"),
        ("obs028c_seam_nodes", root.obs028c_dir / "seam_nodes.csv"),
        ("obs028c_seam_family_summary", root.obs028c_dir / "seam_family_summary.csv"),
        ("obs050_segments", root.obs050_dir / "structural_coupling_segments.csv"),
        ("obs050_path_summary", root.obs050_dir / "structural_coupling_path_summary.csv"),
        ("obs050_coupled_summary", root.obs050_dir / "structural_coupling_coupled_vs_decoupled_summary.csv"),
        ("obs050_summary_txt", root.obs050_dir / "obs050_structural_coupling_persistence_summary.txt"),
        ("obs051_outcome_all", root.obs051_dir / "obs051_outcome_summary_all.csv"),
        ("obs051_outcome_core", root.obs051_dir / "obs051_outcome_summary_core.csv"),
        ("obs051_outcome_near", root.obs051_dir / "obs051_outcome_summary_near.csv"),
        ("obs051_summary_all", root.obs051_dir / "obs051_local_divergence_summary_all.txt"),
        ("obs051_summary_core", root.obs051_dir / "obs051_local_divergence_summary_core.txt"),
        ("obs051_summary_near", root.obs051_dir / "obs051_local_divergence_summary_near.txt"),
        ("scene_nodes", root.scene_bundle_dir / "scene_nodes.csv"),
        ("phase_distance_to_seam", root.fim_phase_dir / "phase_distance_to_seam.csv"),
        ("lazarus_scores", root.fim_lazarus_dir / "lazarus_scores.csv"),
        ("fim_surface", root.fim_dir / "fim_surface.csv"),
    ]

    rows = []
    for name, path in checks:
        exists = path.exists()
        rows.append(
            {
                "corpus": root.label,
                "artifact": name,
                "path": str(path),
                "exists": int(exists),
                "bytes": int(path.stat().st_size) if exists else 0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OBS-050
# ---------------------------------------------------------------------


def summarize_obs050(root: CorpusRoot) -> pd.DataFrame:
    coupled = read_csv_optional(root.obs050_dir / "structural_coupling_coupled_vs_decoupled_summary.csv")
    segments = read_csv_optional(root.obs050_dir / "structural_coupling_segments.csv")
    path_summary = read_csv_optional(root.obs050_dir / "structural_coupling_path_summary.csv")

    rows: list[dict[str, Any]] = []

    if coupled is None or coupled.empty:
        return pd.DataFrame(
            [
                {
                    "corpus": root.label,
                    "status": "missing_obs050_coupled_summary",
                }
            ]
        )

    # Expected columns are usually outcome, coupling_class, n_segments, segment_share.
    cols = set(coupled.columns)
    outcome_col = (
        "outcome"
        if "outcome" in cols
        else "outcome_group"
        if "outcome_group" in cols
        else None
    )
    coupling_col = "coupling_class" if "coupling_class" in cols else None

    if outcome_col is None or coupling_col is None:
        return pd.DataFrame(
            [
                {
                    "corpus": root.label,
                    "status": "schema_mismatch_obs050_coupled_summary",
                    "columns": ",".join(coupled.columns),
                }
            ]
        )

    def share(outcome: str, coupling: str) -> float:
        sub = coupled[
            (coupled[outcome_col].astype(str) == outcome)
            & (coupled[coupling_col].astype(str) == coupling)
        ]
        if sub.empty:
            return float("nan")
        if "segment_share" in sub.columns:
            return as_float(sub["segment_share"].iloc[0])
        if "n_segments" in sub.columns:
            n = pd.to_numeric(sub["n_segments"], errors="coerce").sum()
            total = pd.to_numeric(
                coupled[coupled[outcome_col].astype(str) == outcome]["n_segments"],
                errors="coerce",
            ).sum()
            return float(n / total) if total else float("nan")
        return float("nan")

    recovering_coupled = share("recovering", "coupled")
    nonrecovering_coupled = share("nonrecovering", "coupled")

    rr = (
        recovering_coupled / nonrecovering_coupled
        if np.isfinite(recovering_coupled)
        and np.isfinite(nonrecovering_coupled)
        and nonrecovering_coupled != 0
        else float("nan")
    )

    rec_odds = (
        recovering_coupled / (1.0 - recovering_coupled)
        if np.isfinite(recovering_coupled) and recovering_coupled < 1.0
        else float("nan")
    )
    nonrec_odds = (
        nonrecovering_coupled / (1.0 - nonrecovering_coupled)
        if np.isfinite(nonrecovering_coupled) and nonrecovering_coupled < 1.0
        else float("nan")
    )
    odds_ratio = (
        rec_odds / nonrec_odds
        if np.isfinite(rec_odds) and np.isfinite(nonrec_odds) and nonrec_odds != 0
        else float("nan")
    )

    row = {
        "corpus": root.label,
        "status": "ok",
        "recovering_coupled_share": recovering_coupled,
        "nonrecovering_coupled_share": nonrecovering_coupled,
        "coupled_share_diff": recovering_coupled - nonrecovering_coupled,
        "coupled_risk_ratio": rr,
        "coupled_odds_ratio": odds_ratio,
        "n_segments": int(len(segments)) if segments is not None else np.nan,
        "n_paths": int(len(path_summary)) if path_summary is not None else np.nan,
    }

    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OBS-051
# ---------------------------------------------------------------------


def load_obs051_band(root: CorpusRoot, band: str) -> pd.DataFrame | None:
    preferred = root.obs051_dir / f"obs051_outcome_summary_{band}.csv"
    fallback = root.obs051_dir / "obs051_outcome_summary.csv"
    if preferred.exists():
        return read_csv_optional(preferred)
    if band == "near" and fallback.exists():
        # OBS-051 runner writes generic files last; after all/core/near sequence,
        # generic file usually corresponds to near. Prefer explicit files when available.
        return read_csv_optional(fallback)
    return None


def summarize_obs051(root: CorpusRoot) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for band in ["all", "core", "near"]:
        df = load_obs051_band(root, band)
        if df is None or df.empty:
            rows.append(
                {
                    "corpus": root.label,
                    "band": band,
                    "status": "missing_obs051_outcome_summary",
                }
            )
            continue

        outcome_col = (
            "outcome"
            if "outcome" in df.columns
            else "outcome_group"
            if "outcome_group" in df.columns
            else None
        )

        if outcome_col is None:
            rows.append(
                {
                    "corpus": root.label,
                    "band": band,
                    "status": "schema_mismatch_obs051_outcome_summary",
                    "columns": ",".join(df.columns),
                }
            )
            continue

        def value(outcome: str, col: str) -> float:
            if col not in df.columns:
                return float("nan")
            sub = df[df[outcome_col].astype(str) == outcome]
            if sub.empty:
                return float("nan")
            return as_float(sub[col].iloc[0])

        rec_lambda = value("recovering", "mean_lambda_local")
        nonrec_lambda = value("nonrecovering", "mean_lambda_local")
        rec_delta_d = value("recovering", "mean_delta_d")
        nonrec_delta_d = value("nonrecovering", "mean_delta_d")
        rec_bounded = value("recovering", "mean_bounded_share")
        nonrec_bounded = value("nonrecovering", "mean_bounded_share")
        rec_n = value("recovering", "n_windows")
        nonrec_n = value("nonrecovering", "n_windows")

        rows.append(
            {
                "corpus": root.label,
                "band": band,
                "status": "ok",
                "recovering_n_windows": rec_n,
                "nonrecovering_n_windows": nonrec_n,
                "recovering_mean_lambda_local": rec_lambda,
                "nonrecovering_mean_lambda_local": nonrec_lambda,
                "mean_lambda_local_diff_recovering_minus_nonrecovering": rec_lambda - nonrec_lambda,
                "recovering_mean_delta_d": rec_delta_d,
                "nonrecovering_mean_delta_d": nonrec_delta_d,
                "mean_delta_d_diff_recovering_minus_nonrecovering": rec_delta_d - nonrec_delta_d,
                "recovering_mean_bounded_share": rec_bounded,
                "nonrecovering_mean_bounded_share": nonrec_bounded,
                "mean_bounded_share_diff_recovering_minus_nonrecovering": rec_bounded - nonrec_bounded,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Family substrate
# ---------------------------------------------------------------------


def summarize_family_substrate(root: CorpusRoot) -> pd.DataFrame:
    path = root.family_substrate_dir / "path_family_summary.csv"
    df = read_csv_optional(path)

    if df is None or df.empty:
        return pd.DataFrame(
            [
                {
                    "corpus": root.label,
                    "status": "missing_family_substrate_summary",
                }
            ]
        )

    work = df.copy()
    family_col = "path_family" if "path_family" in work.columns else None
    if family_col is None:
        return pd.DataFrame(
            [
                {
                    "corpus": root.label,
                    "status": "schema_mismatch_family_substrate_summary",
                    "columns": ",".join(work.columns),
                }
            ]
        )

    count_col = None
    for c in ["n_paths", "count", "path_count", "n"]:
        if c in work.columns:
            count_col = c
            break

    if count_col is None:
        # Fallback: each row gets unit mass.
        work["n_paths"] = 1
        count_col = "n_paths"

    work[count_col] = pd.to_numeric(work[count_col], errors="coerce").fillna(0)
    total = float(work[count_col].sum()) if work[count_col].sum() else float("nan")

    rows = []
    for _, row in work.iterrows():
        n = as_float(row[count_col])
        rows.append(
            {
                "corpus": root.label,
                "status": "ok",
                "path_family": str(row[family_col]),
                "n_paths": n,
                "path_share": n / total if np.isfinite(total) and total else float("nan"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# OBS-028c seam bundle
# ---------------------------------------------------------------------


def summarize_obs028c(root: CorpusRoot) -> pd.DataFrame:
    seam_nodes = read_csv_optional(root.obs028c_dir / "seam_nodes.csv")
    seam_family = read_csv_optional(root.obs028c_dir / "seam_family_summary.csv")
    seam_embedding = read_csv_optional(root.obs028c_dir / "seam_embedding_summary.csv")

    if seam_nodes is None or seam_nodes.empty:
        return pd.DataFrame(
            [
                {
                    "corpus": root.label,
                    "status": "missing_obs028c_seam_nodes",
                }
            ]
        )

    row: dict[str, Any] = {
        "corpus": root.label,
        "status": "ok",
        "n_seam_nodes": int(len(seam_nodes)),
        "n_seam_families": int(len(seam_family)) if seam_family is not None else np.nan,
        "n_embedding_rows": int(len(seam_embedding)) if seam_embedding is not None else np.nan,
    }

    for col in ["distance_to_seam", "signed_phase", "lazarus_score", "response_strength"]:
        if col in seam_nodes.columns:
            row[f"mean_{col}"] = finite_mean(seam_nodes[col])
            row[f"median_{col}"] = finite_median(seam_nodes[col])
            row[f"min_{col}"] = finite_min(seam_nodes[col])
            row[f"max_{col}"] = finite_max(seam_nodes[col])

    return pd.DataFrame([row])


# ---------------------------------------------------------------------
# Node fields
# ---------------------------------------------------------------------


def summarize_node_fields(root: CorpusRoot) -> pd.DataFrame:
    candidates = [
        ("scene_nodes", root.scene_bundle_dir / "scene_nodes.csv"),
        ("phase_distance_to_seam", root.fim_phase_dir / "phase_distance_to_seam.csv"),
        ("lazarus_scores", root.fim_lazarus_dir / "lazarus_scores.csv"),
        ("fim_surface", root.fim_dir / "fim_surface.csv"),
    ]

    rows: list[dict[str, Any]] = []

    for source, path in candidates:
        df = read_csv_optional(path)
        if df is None or df.empty:
            rows.append(
                {
                    "corpus": root.label,
                    "source": source,
                    "status": "missing_or_empty",
                    "path": str(path),
                }
            )
            continue

        numeric_cols = [
            c for c in [
                "distance_to_seam",
                "signed_phase",
                "lazarus_score",
                "fim_det",
                "fim_trace",
                "fim_eig1",
                "fim_eig2",
                "fim_cond",
                "response_strength",
                "rsp_anisotropy",
            ]
            if c in df.columns
        ]

        if not numeric_cols:
            rows.append(
                {
                    "corpus": root.label,
                    "source": source,
                    "status": "no_known_numeric_fields",
                    "path": str(path),
                    "rows": int(len(df)),
                }
            )
            continue

        for col in numeric_cols:
            x = pd.to_numeric(df[col], errors="coerce")
            rows.append(
                {
                    "corpus": root.label,
                    "source": source,
                    "field": col,
                    "status": "ok",
                    "path": str(path),
                    "rows": int(len(df)),
                    "defined": int(x.notna().sum()),
                    "mean": finite_mean(x),
                    "median": finite_median(x),
                    "min": finite_min(x),
                    "max": finite_max(x),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------


def fmt_float(x: Any, digits: int = 6) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def extract_obs050_line(df: pd.DataFrame, corpus: str) -> str:
    row = df[df["corpus"] == corpus]
    if row.empty:
        return f"- `{corpus}`: missing OBS-050 summary"
    r = row.iloc[0]
    if r.get("status") != "ok":
        return f"- `{corpus}`: {r.get('status')}"
    return (
        f"- `{corpus}`: recovering coupled share "
        f"{fmt_float(r.get('recovering_coupled_share'))}; "
        f"nonrecovering coupled share {fmt_float(r.get('nonrecovering_coupled_share'))}; "
        f"risk ratio {fmt_float(r.get('coupled_risk_ratio'))}; "
        f"odds ratio {fmt_float(r.get('coupled_odds_ratio'))}; "
        f"segments {r.get('n_segments')}; paths {r.get('n_paths')}"
    )


def extract_obs051_lines(df: pd.DataFrame, corpus: str) -> list[str]:
    rows = df[df["corpus"] == corpus].copy()
    if rows.empty:
        return [f"- `{corpus}`: missing OBS-051 summaries"]

    out = []
    for band in ["all", "core", "near"]:
        sub = rows[rows["band"] == band]
        if sub.empty:
            out.append(f"- `{corpus}` `{band}`: missing")
            continue
        r = sub.iloc[0]
        if r.get("status") != "ok":
            out.append(f"- `{corpus}` `{band}`: {r.get('status')}")
            continue
        out.append(
            f"- `{corpus}` `{band}`: "
            f"Δλ={fmt_float(r.get('mean_lambda_local_diff_recovering_minus_nonrecovering'))}; "
            f"Δmean_delta_d={fmt_float(r.get('mean_delta_d_diff_recovering_minus_nonrecovering'))}; "
            f"Δbounded_share={fmt_float(r.get('mean_bounded_share_diff_recovering_minus_nonrecovering'))}; "
            f"recovering windows={fmt_float(r.get('recovering_n_windows'), 0)}; "
            f"nonrecovering windows={fmt_float(r.get('nonrecovering_n_windows'), 0)}"
        )
    return out


def write_report(
    cfg: CompareConfig,
    manifest: pd.DataFrame,
    obs050: pd.DataFrame,
    obs051: pd.DataFrame,
    family: pd.DataFrame,
    obs028c: pd.DataFrame,
    node_fields: pd.DataFrame,
) -> None:
    left = cfg.left.label
    right = cfg.right.label

    missing = manifest[manifest["exists"] == 0]

    lines = [
        f"# Observatory-chain comparison: {left} vs {right}",
        "",
        "## Scope",
        "",
        "This report compares already-produced file-first observatory-chain artifacts.",
        "It does not recompute geometry, paths, families, coupling, or divergence.",
        "",
        "## Roots",
        "",
        f"- `{left}` root: `{cfg.left.root}`",
        f"- `{right}` root: `{cfg.right.root}`",
        f"- scale: `{cfg.left.scale}` / `{cfg.right.scale}`",
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

    lines.extend(
        [
            "## OBS-050 structural coupling persistence",
            "",
            extract_obs050_line(obs050, left),
            extract_obs050_line(obs050, right),
            "",
        ]
    )

    if set(obs050["corpus"]) >= {left, right}:
        lrow = obs050[(obs050["corpus"] == left) & (obs050["status"] == "ok")]
        rrow = obs050[(obs050["corpus"] == right) & (obs050["status"] == "ok")]
        if not lrow.empty and not rrow.empty:
            l = lrow.iloc[0]
            r = rrow.iloc[0]
            lines.extend(
                [
                    "Cross-corpus OBS-050 deltas:",
                    "",
                    f"- Δ recovering coupled share (`{right} - {left}`): "
                    f"`{fmt_float(r['recovering_coupled_share'] - l['recovering_coupled_share'])}`",
                    f"- Δ nonrecovering coupled share (`{right} - {left}`): "
                    f"`{fmt_float(r['nonrecovering_coupled_share'] - l['nonrecovering_coupled_share'])}`",
                    f"- Δ risk ratio (`{right} - {left}`): "
                    f"`{fmt_float(r['coupled_risk_ratio'] - l['coupled_risk_ratio'])}`",
                    f"- Δ odds ratio (`{right} - {left}`): "
                    f"`{fmt_float(r['coupled_odds_ratio'] - l['coupled_odds_ratio'])}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## OBS-051 local divergence / boundedness",
            "",
            "Convention: differences are `recovering - nonrecovering` within each corpus/band.",
            "Negative `Δλ` means recovering windows have lower mean local divergence.",
            "Negative `Δmean_delta_d` means recovering windows expand less in distance.",
            "Positive `Δbounded_share` means recovering windows have a larger bounded-neighbor share.",
            "",
        ]
    )
    lines.extend(extract_obs051_lines(obs051, left))
    lines.extend(extract_obs051_lines(obs051, right))
    lines.append("")

    lines.extend(
        [
            "## Family substrate",
            "",
            "Family proportions are written to `family_substrate_comparison.csv`.",
            "",
        ]
    )

    fam_ok = family[family["status"] == "ok"].copy()
    if not fam_ok.empty:
        for corpus in [left, right]:
            sub = fam_ok[fam_ok["corpus"] == corpus].sort_values("path_share", ascending=False)
            if sub.empty:
                lines.append(f"- `{corpus}`: missing usable family summary")
                continue
            parts = [
                f"{row['path_family']}={fmt_float(row['path_share'], 4)}"
                for _, row in sub.iterrows()
            ]
            lines.append(f"- `{corpus}`: " + "; ".join(parts))
    lines.append("")

    lines.extend(
        [
            "## OBS-028c seam bundle",
            "",
            "Seam bundle summaries are written to `obs028c_seam_bundle_comparison.csv`.",
            "",
        ]
    )

    for corpus in [left, right]:
        sub = obs028c[obs028c["corpus"] == corpus]
        if sub.empty:
            lines.append(f"- `{corpus}`: missing OBS-028c seam summary")
            continue
        r = sub.iloc[0]
        lines.append(
            f"- `{corpus}`: seam nodes={r.get('n_seam_nodes')}; "
            f"seam families={r.get('n_seam_families')}; "
            f"mean distance_to_seam={fmt_float(r.get('mean_distance_to_seam'))}; "
            f"mean lazarus_score={fmt_float(r.get('mean_lazarus_score'))}"
        )
    lines.append("")

    lines.extend(
        [
            "## Node-field summaries",
            "",
            "Node-field aggregate comparisons are written to `node_field_comparison.csv`.",
            "",
            "## Guardrails",
            "",
            "- OBS-050 is a stronger replication target than OBS-051.",
            "- OBS-051 should remain provisional unless band-specific effects survive repeated scoped runs.",
            "- Differences reported here are corpus/root comparisons, not universal claims.",
            "- Missing optional artifacts should be interpreted as incomplete chain coverage, not automatically as scientific absence.",
            "",
        ]
    )

    (cfg.outdir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two PAM observatory-chain artifact roots."
    )
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--right-root", required=True)
    parser.add_argument("--scale", default="100000")
    parser.add_argument(
        "--left-scale",
        default=None,
        help="Optional left scale override. Defaults to --scale.",
    )
    parser.add_argument(
        "--right-scale",
        default=None,
        help="Optional right scale override. Defaults to --scale.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/comparisons/observatory_chain_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    left_scale = args.left_scale or args.scale
    right_scale = args.right_scale or args.scale

    cfg = CompareConfig(
        left=CorpusRoot(args.left_label, Path(args.left_root), str(left_scale)),
        right=CorpusRoot(args.right_label, Path(args.right_root), str(right_scale)),
        outdir=Path(args.outdir),
    )

    ensure_outdir(cfg.outdir)

    manifest = pd.concat(
        [artifact_status(cfg.left), artifact_status(cfg.right)],
        ignore_index=True,
    )
    manifest.to_csv(cfg.outdir / "corpus_root_manifest.csv", index=False)

    obs050_left = summarize_obs050(cfg.left)
    obs050_right = summarize_obs050(cfg.right)
    obs050 = pd.concat([obs050_left, obs050_right], ignore_index=True)
    obs050.to_csv(cfg.outdir / "obs050_coupling_comparison.csv", index=False)

    obs051_left = summarize_obs051(cfg.left)
    obs051_right = summarize_obs051(cfg.right)
    obs051 = pd.concat([obs051_left, obs051_right], ignore_index=True)
    obs051.to_csv(cfg.outdir / "obs051_banded_comparison.csv", index=False)

    family_left = summarize_family_substrate(cfg.left)
    family_right = summarize_family_substrate(cfg.right)
    family = pd.concat([family_left, family_right], ignore_index=True)
    family.to_csv(cfg.outdir / "family_substrate_comparison.csv", index=False)

    obs028c_left = summarize_obs028c(cfg.left)
    obs028c_right = summarize_obs028c(cfg.right)
    obs028c = pd.concat([obs028c_left, obs028c_right], ignore_index=True)
    obs028c.to_csv(cfg.outdir / "obs028c_seam_bundle_comparison.csv", index=False)

    node_left = summarize_node_fields(cfg.left)
    node_right = summarize_node_fields(cfg.right)
    node_fields = pd.concat([node_left, node_right], ignore_index=True)
    node_fields.to_csv(cfg.outdir / "node_field_comparison.csv", index=False)

    write_report(
        cfg=cfg,
        manifest=manifest,
        obs050=obs050,
        obs051=obs051,
        family=family,
        obs028c=obs028c,
        node_fields=node_fields,
    )

    print(cfg.outdir / "corpus_root_manifest.csv")
    print(cfg.outdir / "obs050_coupling_comparison.csv")
    print(cfg.outdir / "obs051_banded_comparison.csv")
    print(cfg.outdir / "family_substrate_comparison.csv")
    print(cfg.outdir / "obs028c_seam_bundle_comparison.csv")
    print(cfg.outdir / "node_field_comparison.csv")
    print(cfg.outdir / "comparison_summary.md")


if __name__ == "__main__":
    main()
