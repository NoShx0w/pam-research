#!/usr/bin/env python3
"""
obs084a_canonical_lineage_and_observation_bridge.py

OBS-084a — Canonical Lineage and Observation Bridge
====================================================

Purpose
-------
Reduce the broad OBS-084 reconnaissance inventory to a small, explicit,
semantically joined evidence spine for the OBS-078 -> OBS-083 RIG lineage.

This bridge resolves:

* one canonical source artifact for each required scientific role;
* duplicate, smoke, legacy, mirror, and alternate-version artifacts;
* mappings from OBS-083 registry records to observation-level prediction or
  reconstructible-loss sources;
* candidate observation keys and cross-artifact key compatibility;
* carrier-to-feature definitions;
* structural cluster hierarchies and per-record partition balance;
* support-family availability and predeclared discretization requirements;
* field roles for outcome, grouping, provenance, predictor, and forbidden
  leakage use;
* source hashes and code provenance for a future frozen candidate manifest.

The script is deliberately pre-discovery. It does NOT:

* nominate failure supports;
* inspect or create reserved confirmation outcomes;
* compute localization contrasts;
* assign FL0-FL5 levels;
* create direct witnesses;
* propose repairs or interventions;
* establish causality, control, actionability, external generalization, or
  formal topology.

Default outputs
---------------
outputs/rig_registry/obs084_direct_failure_witness/lineage_bridge/
    obs084a_canonical_source_manifest.csv
    obs084a_artifact_duplicate_resolution.csv
    obs084a_registry_record_to_prediction_source.csv
    obs084a_observation_key_crosswalk.csv
    obs084a_carrier_feature_resolution.csv
    obs084a_cluster_hierarchy.csv
    obs084a_partition_balance_by_record.csv
    obs084a_support_family_resolution.csv
    obs084a_field_role_classification.csv
    obs084a_lineage_bridge_summary.csv
    obs084a_lineage_bridge_report.md

Run
---
PYTHONPATH=src .venv/bin/python \
    experiments/studies/obs084a_canonical_lineage_and_observation_bridge.py

Guardrail
---------
A positive bridge result establishes semantic-lineage feasibility only. It does
not establish that any failure support exists or can be confirmed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rig_registry/obs084_direct_failure_witness/lineage_bridge"
)

CANONICAL_ROLE_SPECS: dict[str, tuple[str, ...]] = {
    "canonical_feature_table": (
        "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv",
        "outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_feature_table.csv",
    ),
    "canonical_feature_manifest": (
        "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_manifest.csv",
        "outputs/comparisons/obs078a_mechanistic_signature_classifier/obs078a_feature_manifest.csv",
    ),
    "leave_structure_predictions": (
        "outputs/comparisons/obs079a_stability_signature_leave_structure_out/obs079a_leave_structure_predictions.csv",
    ),
    "pairwise_predictions": (
        "outputs/comparisons/obs079c_pairwise_stability_classifiers/obs079c_pairwise_predictions.csv",
    ),
    "numeric_transform_predictions": (
        "outputs/comparisons/obs080a_stability_core_transform_sensitivity/obs080a_transform_predictions.csv",
    ),
    "scale_band_predictions": (
        "outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/obs080b_scale_band_predictions.csv",
    ),
    "feature_contract_predictions": (
        "outputs/comparisons/obs080c_feature_family_contract_sensitivity/obs080c_feature_contract_predictions.csv",
    ),
    "structural_resampling_summary": (
        "outputs/comparisons/obs080d_structural_resampling_contract_sensitivity/obs080d_bootstrap_summary.csv",
        "outputs/comparisons/obs080d_structural_resampling_contract_sensitivity_smoke/obs080d_bootstrap_summary.csv",
    ),
    "registry": ("outputs/rig_registry/rig_relation_registry.csv",),
    "obs083_subclasses": (
        "outputs/rig_registry/obs083_negative_control_localization/obs083_diagnostic_subclass_assignments.csv",
    ),
    "obs083_relation_controls": (
        "outputs/rig_registry/obs083_negative_control_localization/obs083_relation_control_contrast.csv",
    ),
    "obs083_carrier_controls": (
        "outputs/rig_registry/obs083_negative_control_localization/obs083_carrier_control_contrast.csv",
    ),
}

# Lower is more authoritative. The rule is explicit rather than size-driven.
PATH_PENALTIES: tuple[tuple[str, int, str], ...] = (
    ("_smoke", 60, "smoke_run"),
    ("/smoke/", 60, "smoke_run"),
    ("canonical_legacy", 50, "legacy_copy"),
    ("/pipeline/", 20, "pipeline_mirror"),
    ("rig_navigator", 40, "derived_view"),
    ("_v2", -5, "version_preference"),
    ("comparisons/", -10, "comparison_canonical_preference"),
)

OBSERVATION_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "observation_id": ("observation_id", "sample_id", "row_id", "case_id", "id"),
    "case": ("case", "case_id", "regime", "condition", "corpus", "label", "true_label"),
    "object": ("object", "object_id", "object_key", "case_object"),
    "cohort": ("cohort", "cohort_id", "path_label_cohort", "transition_cohort"),
    "transition": ("transition", "transition_id", "transition_type", "transition_key"),
    "fold": ("fold", "fold_id", "split", "validation_fold"),
    "scheme": ("scheme", "validation_scheme", "holdout_scheme"),
    "path": ("path_id", "route_id", "trajectory_id"),
}

FIELD_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "outcome": (
        r"(^|_)(label|class|target|true_label|true_class|y_true|regime)($|_)",
        r"(^|_)prediction($|_)",
        r"predicted_(label|class|regime)",
        r"probability|proba|margin|correct|error|loss",
    ),
    "grouping_partition": (
        r"object|cohort|transition|route|path|trajectory|window|fold|split|scheme|cluster",
    ),
    "provenance": (
        r"corpus|campaign|model|prompt|source|artifact|manifest|provenance|commit|version|run_id",
    ),
    "forbidden_predictive_leakage": (
        r"record_id|relation|carrier|subclass|readiness|blocker",
        r"true_|predicted_|probability|proba|correct|confusion|score$|loss$",
        r"fold|split|holdout|validation",
    ),
}

CARRIER_FEATURE_HINTS: dict[str, tuple[str, ...]] = {
    "stability_core_3": ("mean_lambda_local_mean", "mean_delta_d_mean", "bounded_share_mean"),
    "geometry_scores_only": (
        "geometry", "distance", "seam", "response", "energy", "curvature", "mds"
    ),
    "path_shares_only": ("share", "path", "route", "cohort"),
    "stability_plus_geometry": (
        "mean_lambda_local_mean", "mean_delta_d_mean", "bounded_share_mean",
        "geometry", "distance", "seam", "response", "energy", "mds"
    ),
    "no_window": ("__exclude_window__",),
    "strict_numeric_all": ("__all_numeric_nonleakage__",),
}

SUPPORT_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "object": ("object",),
    "cohort": ("cohort",),
    "transition": ("transition",),
    "scale_band": ("scale_band", "scale", "diffusion_scale"),
    "contract_or_transform": ("contract", "transform", "transformation"),
    "feature_family": ("feature_family", "feature_group", "feature_set", "carrier"),
    "seam_relative": ("seam", "distance_to_seam", "seam_distance"),
    "boundary_relative": ("boundary", "distance_to_boundary"),
    "window": ("window", "window_id", "window_index"),
    "route_or_path": ("route", "path", "trajectory"),
    "provenance_slice": ("corpus", "campaign", "model", "prompt", "provenance"),
}


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    exists: bool
    size_bytes: int
    sha256: str
    rows: int | None
    columns: tuple[str, ...]
    read_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recon-dir", type=Path, default=Path(
        "outputs/rig_registry/obs084_direct_failure_witness/reconnaissance"
    ))
    parser.add_argument("--max-read-mb", type=float, default=256.0)
    parser.add_argument("--sample-rows", type=int, default=250_000)
    parser.add_argument("--min-two-way-clusters", type=int, default=12)
    parser.add_argument("--min-three-way-clusters", type=int, default=24)
    parser.add_argument("--min-per-class-clusters", type=int, default=3)
    parser.add_argument("--partition-seed", type=str, default="obs084-lineage-v1")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def read_csv_safe(path: Path, max_read_mb: float, sample_rows: int) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "missing"
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_read_mb:
            df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
            return df, "sampled_size_limit"
        return pd.read_csv(path, low_memory=False), "ok"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty_csv"
    except Exception as exc:
        try:
            return pd.read_csv(path, nrows=0), f"header_only_after_error:{type(exc).__name__}"
        except Exception:
            return pd.DataFrame(), f"read_error:{type(exc).__name__}"


def artifact_info(path: Path, max_read_mb: float, sample_rows: int) -> tuple[ArtifactInfo, pd.DataFrame]:
    if not path.exists():
        return ArtifactInfo(path, False, 0, "", None, (), "missing"), pd.DataFrame()
    df, status = read_csv_safe(path, max_read_mb, sample_rows)
    info = ArtifactInfo(
        path=path,
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        rows=len(df),
        columns=tuple(map(str, df.columns)),
        read_status=status,
    )
    return info, df


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def path_authority_score(path: Path) -> tuple[int, list[str]]:
    s = path.as_posix().lower()
    score = 100
    reasons: list[str] = []
    for token, delta, reason in PATH_PENALTIES:
        if token in s:
            score += delta
            reasons.append(reason)
    return score, reasons


def discover_role_candidates(repo_root: Path, role: str, explicit: Sequence[str]) -> list[Path]:
    candidates: list[Path] = []
    for rel in explicit:
        p = repo_root / rel
        if p.exists():
            candidates.append(p)
    # Add same-basename alternatives to make duplicate resolution inspectable.
    for rel in explicit:
        basename = Path(rel).name
        for p in (repo_root / "outputs").rglob(basename):
            if p not in candidates:
                candidates.append(p)
    return candidates


def choose_canonical(candidates: Sequence[Path]) -> tuple[Path | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ranked: list[tuple[int, str, Path, list[str]]] = []
    for p in candidates:
        score, reasons = path_authority_score(p)
        ranked.append((score, p.as_posix(), p, reasons))
    ranked.sort(key=lambda x: (x[0], x[1]))
    winner = ranked[0][2] if ranked else None
    winner_hash = sha256_file(winner) if winner and winner.exists() else ""
    for score, _, p, reasons in ranked:
        phash = sha256_file(p)
        rows.append({
            "artifact_path": p.as_posix(),
            "authority_score": score,
            "selection_reasons": ";".join(reasons) or "default",
            "selected_canonical": bool(winner and p == winner),
            "content_identical_to_selected": bool(winner_hash and phash == winner_hash),
            "sha256": phash,
            "resolution_status": (
                "selected" if winner and p == winner else
                "duplicate_mirror" if winner_hash and phash == winner_hash else
                "alternate_noncanonical"
            ),
        })
    return winner, rows


def first_present(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    lookup = {norm(c): c for c in columns}
    for a in aliases:
        if norm(a) in lookup:
            return lookup[norm(a)]
    return None


def infer_relation_column(df: pd.DataFrame) -> str | None:
    return first_present(df.columns, ("relation", "comparison", "pair", "task", "relation_id"))


def infer_carrier_column(df: pd.DataFrame) -> str | None:
    return first_present(df.columns, ("carrier", "feature_set", "feature_contract", "panel", "feature_family"))


def infer_record_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    record = first_present(df.columns, ("record_id", "registry_record_id", "rig_record_id"))
    return record, infer_relation_column(df), infer_carrier_column(df)


def canonical_registry_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = frames.get("obs083_subclasses", pd.DataFrame()).copy()
    if base.empty:
        base = frames.get("registry", pd.DataFrame()).copy()
    if base.empty:
        return pd.DataFrame(columns=["record_id", "relation", "carrier", "subclass"])
    rid, rel, car = infer_record_columns(base)
    out = pd.DataFrame(index=base.index)
    if rid:
        out["record_id"] = base[rid].astype(str)
    elif rel and car:
        out["record_id"] = base[rel].astype(str) + "__" + base[car].astype(str)
    else:
        out["record_id"] = [f"record_{i}" for i in range(len(base))]
    if rel:
        out["relation"] = base[rel].astype(str)
    else:
        out["relation"] = out["record_id"].str.split("__", n=1).str[0]
    if car:
        out["carrier"] = base[car].astype(str)
    else:
        out["carrier"] = out["record_id"].str.split("__", n=1).str[1]
    subclass_col = first_present(base.columns, ("subclass", "diagnostic_subclass"))
    out["subclass"] = base[subclass_col].astype(str) if subclass_col else "unknown"
    out["confirmation_eligibility"] = out["subclass"].map(
        lambda x: "fl3_confirmation_eligible" if str(x).startswith("C2") else "discovery_only"
    )
    return out.drop_duplicates("record_id").reset_index(drop=True)


def relation_matches(source_value: str, relation: str) -> bool:
    a = norm(source_value)
    b = norm(relation)
    if a == b:
        return True
    tokens = [t for t in b.split("_") if t not in {"vs", "three", "way"}]
    return all(t in a for t in tokens) if tokens else False


def carrier_matches(source_value: str, carrier: str) -> bool:
    a, b = norm(source_value), norm(carrier)
    if a == b:
        return True
    aliases = {
        "stability_core_3": ("stability_core", "core_3", "minimal_core"),
        "geometry_scores_only": ("geometry", "geometry_only"),
        "path_shares_only": ("path_share", "path_shares"),
        "stability_plus_geometry": ("stability_geometry", "core_plus_geometry"),
        "no_window": ("no_window", "without_window"),
        "strict_numeric_all": ("strict_numeric", "numeric_all", "all_numeric"),
    }
    return any(norm(x) in a for x in aliases.get(b, (b,)))


def score_prediction_source(df: pd.DataFrame, relation: str, carrier: str, role: str) -> tuple[int, str, int]:
    if df.empty:
        return 0, "empty_or_missing", 0
    rel_col = infer_relation_column(df)
    car_col = infer_carrier_column(df)
    rel_score, car_score = 0, 0
    matched_rows = len(df)
    mask = pd.Series(True, index=df.index)
    evidence: list[str] = []
    if rel_col:
        rmask = df[rel_col].astype(str).map(lambda x: relation_matches(x, relation))
        rel_score = 45 if rmask.any() else 0
        mask &= rmask
        evidence.append(f"relation_col={rel_col}")
    else:
        # three-way sources may omit a relation column.
        rel_score = 10 if norm(relation) == "three_way" else 0
        evidence.append("relation_col_absent")
    if car_col:
        cmask = df[car_col].astype(str).map(lambda x: carrier_matches(x, carrier))
        car_score = 35 if cmask.any() else 0
        mask &= cmask
        evidence.append(f"carrier_col={car_col}")
    else:
        # Source role gives partial carrier evidence.
        role_hint = {
            "pairwise_predictions": "stability_core_3",
            "leave_structure_predictions": "stability_core_3",
        }.get(role)
        car_score = 15 if role_hint == carrier else 0
        evidence.append("carrier_col_absent")
    matched_rows = int(mask.sum()) if len(mask) else 0
    prediction_cols = [c for c in df.columns if re.search(r"pred|proba|probability|margin|correct|score", norm(c))]
    pred_score = 15 if prediction_cols else 0
    key_cols = [first_present(df.columns, aliases) for aliases in OBSERVATION_KEY_ALIASES.values()]
    key_score = 5 if any(key_cols) else 0
    score = rel_score + car_score + pred_score + key_score
    evidence.extend([f"prediction_cols={len(prediction_cols)}", f"matched_rows={matched_rows}"])
    return score, ";".join(evidence), matched_rows


def build_record_source_map(registry: pd.DataFrame, frames: dict[str, pd.DataFrame], paths: dict[str, Path]) -> pd.DataFrame:
    prediction_roles = [
        "leave_structure_predictions", "pairwise_predictions", "numeric_transform_predictions",
        "scale_band_predictions", "feature_contract_predictions", "structural_resampling_summary",
    ]
    rows: list[dict[str, Any]] = []
    for rec in registry.to_dict("records"):
        candidates: list[dict[str, Any]] = []
        for role in prediction_roles:
            df = frames.get(role, pd.DataFrame())
            score, evidence, matched_rows = score_prediction_source(df, rec["relation"], rec["carrier"], role)
            candidates.append({
                "source_role": role,
                "source_path": paths.get(role, Path("")).as_posix(),
                "mapping_score": score,
                "mapping_evidence": evidence,
                "matched_rows": matched_rows,
            })
        candidates.sort(key=lambda x: (-x["mapping_score"], x["source_role"]))
        best = candidates[0] if candidates else {}
        tie_count = sum(c.get("mapping_score") == best.get("mapping_score") for c in candidates)
        status = (
            "canonical_observation_source_resolved" if best.get("mapping_score", 0) >= 65 and best.get("matched_rows", 0) > 0 else
            "reconstructible_source_candidate" if best.get("mapping_score", 0) >= 35 else
            "unresolved"
        )
        rows.append({
            **rec,
            "selected_source_role": best.get("source_role", ""),
            "selected_source_path": best.get("source_path", ""),
            "mapping_score": best.get("mapping_score", 0),
            "mapping_status": status,
            "matched_rows": best.get("matched_rows", 0),
            "mapping_evidence": best.get("mapping_evidence", ""),
            "top_score_tie_count": tie_count,
            "alternative_sources": json.dumps(candidates[1:4], sort_keys=True),
        })
    return pd.DataFrame(rows)


def build_key_crosswalk(feature_df: pd.DataFrame, frames: dict[str, pd.DataFrame], paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    canonical_cols = list(feature_df.columns)
    for role, df in frames.items():
        if role in {"canonical_feature_table", "registry", "obs083_subclasses"} or df.empty:
            continue
        for family, aliases in OBSERVATION_KEY_ALIASES.items():
            left = first_present(canonical_cols, aliases)
            right = first_present(df.columns, aliases)
            if not left and not right:
                continue
            left_unique = int(feature_df[left].nunique(dropna=True)) if left else 0
            right_unique = int(df[right].nunique(dropna=True)) if right else 0
            overlap = 0
            overlap_ratio_left = 0.0
            overlap_ratio_right = 0.0
            if left and right:
                lv = set(feature_df[left].dropna().astype(str).unique())
                rv = set(df[right].dropna().astype(str).unique())
                overlap = len(lv & rv)
                overlap_ratio_left = overlap / max(1, len(lv))
                overlap_ratio_right = overlap / max(1, len(rv))
            rows.append({
                "source_role": role,
                "source_path": paths.get(role, Path("")).as_posix(),
                "key_family": family,
                "canonical_column": left or "",
                "source_column": right or "",
                "canonical_unique": left_unique,
                "source_unique": right_unique,
                "value_overlap_count": overlap,
                "canonical_overlap_ratio": overlap_ratio_left,
                "source_overlap_ratio": overlap_ratio_right,
                "schema_bridge_status": (
                    "value_overlap_supported" if overlap > 0 else
                    "column_alias_only" if left and right else
                    "one_side_only"
                ),
                "warning": "shared names do not prove semantic equivalence",
            })
    return pd.DataFrame(rows)


def numeric_nonleakage_columns(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        n = norm(c)
        forbidden = any(re.search(p, n) for p in FIELD_ROLE_PATTERNS["forbidden_predictive_leakage"])
        if not forbidden:
            out.append(c)
    return out


def build_carrier_resolution(registry: pd.DataFrame, feature_df: pd.DataFrame, feature_manifest: pd.DataFrame) -> pd.DataFrame:
    columns = list(feature_df.columns)
    numeric_safe = numeric_nonleakage_columns(feature_df)
    rows: list[dict[str, Any]] = []
    for carrier in sorted(registry["carrier"].dropna().astype(str).unique()):
        hint = CARRIER_FEATURE_HINTS.get(norm(carrier), ())
        selected: list[str] = []
        method = "heuristic"
        if hint == ("__all_numeric_nonleakage__",):
            selected = numeric_safe
            method = "all_numeric_nonleakage"
        elif hint == ("__exclude_window__",):
            selected = [c for c in numeric_safe if "window" not in norm(c)]
            method = "numeric_nonleakage_excluding_window"
        else:
            for c in columns:
                nc = norm(c)
                if any(norm(h) == nc or norm(h) in nc for h in hint):
                    selected.append(c)
        # Prefer explicit manifest grouping when available.
        manifest_matches: list[str] = []
        if not feature_manifest.empty:
            name_col = first_present(feature_manifest.columns, ("feature", "feature_name", "column", "field"))
            group_col = first_present(feature_manifest.columns, ("feature_family", "feature_group", "panel", "carrier"))
            if name_col and group_col:
                mask = feature_manifest[group_col].astype(str).map(lambda x: carrier_matches(x, carrier))
                manifest_matches = [x for x in feature_manifest.loc[mask, name_col].astype(str) if x in columns]
                if manifest_matches:
                    selected = manifest_matches
                    method = "explicit_feature_manifest"
        selected = list(dict.fromkeys(selected))
        rows.append({
            "carrier": carrier,
            "resolution_method": method,
            "resolved_feature_count": len(selected),
            "resolved_features": json.dumps(selected),
            "feature_manifest_matches": len(manifest_matches),
            "carrier_resolution_status": (
                "resolved" if selected else "unresolved"
            ),
            "predictive_leakage_screened": True,
            "note": "heuristic resolutions require review before discovery freeze" if method != "explicit_feature_manifest" else "",
        })
    return pd.DataFrame(rows)


def stable_partition(value: str, seed: str, parts: int = 3) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) % parts


def build_cluster_hierarchy(feature_df: pd.DataFrame, seed: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, aliases in OBSERVATION_KEY_ALIASES.items():
        col = first_present(feature_df.columns, aliases)
        if not col:
            continue
        values = feature_df[col].dropna().astype(str)
        unique = sorted(values.unique())
        assignments = defaultdict(int)
        for v in unique:
            assignments[stable_partition(v, seed, 3)] += 1
        rows.append({
            "cluster_family": family,
            "cluster_column": col,
            "row_count": len(feature_df),
            "unique_clusters": len(unique),
            "discovery_clusters": assignments[0],
            "confirmation_clusters": assignments[1],
            "replication_clusters": assignments[2],
            "three_way_nonempty": all(assignments[i] > 0 for i in range(3)),
            "recommended_priority": {
                "object": 1, "path": 2, "transition": 3, "cohort": 4,
                "observation_id": 9, "case": 8, "fold": 10, "scheme": 10,
            }.get(family, 7),
            "independence_status": "candidate_only_not_proven",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["recommended_priority", "unique_clusters"], ascending=[True, False])
    return out


def build_partition_balance(
    registry: pd.DataFrame, feature_df: pd.DataFrame, cluster_df: pd.DataFrame, seed: str,
    min_per_class: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if feature_df.empty or cluster_df.empty:
        return pd.DataFrame()
    class_col = first_present(feature_df.columns, OBSERVATION_KEY_ALIASES["case"])
    top_clusters = cluster_df.head(6)
    for rec in registry.to_dict("records"):
        rel_tokens = [t for t in re.split(r"_vs_|__", rec["relation"]) if t]
        for _, crow in top_clusters.iterrows():
            ccol = str(crow["cluster_column"])

            # Avoid duplicate-label selection when the cluster column is also the
            # class column (for example, a case-level cluster candidate). Pandas
            # returns a DataFrame for duplicate labels, which cannot be assigned
            # to the single ``partition`` column. Keep each requested label once
            # and explicitly collapse any duplicate source columns to the first
            # physical column for this schema-level audit.
            selected_cols = [ccol]
            if class_col and class_col != ccol:
                selected_cols.append(class_col)
            work = feature_df.loc[:, selected_cols].copy()
            if work.columns.duplicated().any():
                work = work.loc[:, ~work.columns.duplicated(keep="first")].copy()

            cluster_values = work.loc[:, ccol]
            if isinstance(cluster_values, pd.DataFrame):
                cluster_values = cluster_values.iloc[:, 0]
            valid_mask = cluster_values.notna()
            work = work.loc[valid_mask].copy()
            cluster_values = cluster_values.loc[valid_mask]
            work["partition"] = cluster_values.astype(str).map(
                lambda x: stable_partition(x, seed, 3)
            )

            effective_class_col = class_col if class_col and class_col in work.columns else None
            if effective_class_col and norm(rec["relation"]) != "three_way":
                class_values = work.loc[:, effective_class_col]
                if isinstance(class_values, pd.DataFrame):
                    class_values = class_values.iloc[:, 0]
                mask = class_values.astype(str).map(
                    lambda x: any(norm(t) == norm(x) or norm(t) in norm(x) for t in rel_tokens)
                )
                work = work.loc[mask].copy()

            counts = work.groupby("partition")[ccol].nunique().to_dict()
            class_min = None
            if effective_class_col and not work.empty:
                tab = work.groupby(["partition", effective_class_col])[ccol].nunique()
                class_min = int(tab.min()) if len(tab) else 0
            rows.append({
                "record_id": rec["record_id"],
                "relation": rec["relation"],
                "carrier": rec["carrier"],
                "subclass": rec["subclass"],
                "cluster_family": crow["cluster_family"],
                "cluster_column": ccol,
                "discovery_clusters": counts.get(0, 0),
                "confirmation_clusters": counts.get(1, 0),
                "replication_clusters": counts.get(2, 0),
                "minimum_class_partition_clusters": class_min if class_min is not None else "unknown",
                "three_way_balance_candidate": bool(
                    counts.get(0, 0) > 0 and counts.get(1, 0) > 0 and counts.get(2, 0) > 0 and
                    (class_min is None or class_min >= min_per_class)
                ),
                "status": "schema_balance_candidate_not_confirmation_ready",
            })
    return pd.DataFrame(rows)


def build_support_resolution(frames: dict[str, pd.DataFrame], paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, patterns in SUPPORT_FAMILY_PATTERNS.items():
        matches: list[dict[str, Any]] = []
        for role, df in frames.items():
            if df.empty:
                continue
            for c in df.columns:
                nc = norm(c)
                if any(norm(p) in nc for p in patterns):
                    unique = int(df[c].nunique(dropna=True))
                    matches.append({"role": role, "column": c, "unique": unique})
        available = bool(matches)
        needs_discretization = family in {"seam_relative", "boundary_relative"} and any(m["unique"] > 12 for m in matches)
        rows.append({
            "support_family": family,
            "available": available,
            "artifact_column_count": len(matches),
            "candidate_columns": json.dumps(matches[:30], sort_keys=True),
            "requires_predeclared_discretization": needs_discretization,
            "discovery_status": (
                "available_after_discretization_freeze" if needs_discretization else
                "available" if available else
                "unavailable"
            ),
            "confirmation_status": "not_evaluated",
        })
    return pd.DataFrame(rows)


def classify_field(role: str, column: str, df: pd.DataFrame) -> dict[str, Any]:
    n = norm(column)
    matches = {k: any(re.search(p, n) for p in pats) for k, pats in FIELD_ROLE_PATTERNS.items()}
    if matches["forbidden_predictive_leakage"]:
        primary = "forbidden_predictive_leakage"
    elif matches["outcome"]:
        primary = "outcome_or_evaluation"
    elif matches["grouping_partition"]:
        primary = "grouping_partition"
    elif matches["provenance"]:
        primary = "provenance"
    elif pd.api.types.is_numeric_dtype(df[column]):
        primary = "candidate_predictor"
    else:
        primary = "metadata_unresolved"
    return {
        "source_role": role,
        "field": column,
        "dtype": str(df[column].dtype),
        "unique_values": int(df[column].nunique(dropna=True)),
        "primary_field_role": primary,
        "allowed_as_predictor": primary == "candidate_predictor",
        "allowed_for_matching": primary in {"grouping_partition", "provenance", "metadata_unresolved"},
        "allowed_for_partitioning": primary == "grouping_partition",
        "manual_review_required": primary == "metadata_unresolved",
    }


def build_field_roles(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, df in frames.items():
        if df.empty:
            continue
        for c in df.columns:
            rows.append(classify_field(role, c, df))
    return pd.DataFrame(rows)


def write_report(
    out_path: Path,
    canonical_manifest: pd.DataFrame,
    duplicate_resolution: pd.DataFrame,
    record_map: pd.DataFrame,
    crosswalk: pd.DataFrame,
    carrier_resolution: pd.DataFrame,
    cluster_hierarchy: pd.DataFrame,
    partition_balance: pd.DataFrame,
    support_resolution: pd.DataFrame,
    field_roles: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    status = summary.loc[0, "overall_status"] if not summary.empty else "unknown"
    canonical_table = df_to_markdown(canonical_manifest[[
        "source_role", "selected_path", "exists", "rows_read", "column_count", "read_status"
    ]]) if not canonical_manifest.empty else "No canonical sources resolved."
    record_counts = record_map["mapping_status"].value_counts().rename_axis("mapping_status").reset_index(name="count")
    support_table = support_resolution[["support_family", "available", "requires_predeclared_discretization", "discovery_status"]]
    text = f"""# OBS-084a — Canonical Lineage and Observation Bridge

## State

Bridge completed.

Overall status: `{status}`

This is a semantic-lineage and observation-bridge audit only. It performs no
candidate generation, confirmation, witness assignment, FL promotion, repair
design, intervention, or causal analysis.

## Canonical interpretation

The bridge asks whether the broad reconnaissance inventory can be reduced to a
small, explicit evidence spine connecting OBS-083 registry records to canonical
OBS-078–080 observation-level artifacts.

A positive mapping means only that a record has a plausible canonical or
reconstructible observation source. It does not establish a failure support.

## Canonical source manifest

{canonical_table}

## Registry-record source mapping

{df_to_markdown(record_counts)}

- Records audited: {len(record_map)}
- FL3-eligible C2 records: {(record_map['confirmation_eligibility'] == 'fl3_confirmation_eligible').sum() if not record_map.empty else 0}
- Canonically resolved observation sources: {(record_map['mapping_status'] == 'canonical_observation_source_resolved').sum() if not record_map.empty else 0}
- Reconstructible source candidates: {(record_map['mapping_status'] == 'reconstructible_source_candidate').sum() if not record_map.empty else 0}
- Unresolved records: {(record_map['mapping_status'] == 'unresolved').sum() if not record_map.empty else 0}

## Observation-key bridge

- Crosswalk rows: {len(crosswalk)}
- Value-overlap-supported bridges: {(crosswalk['schema_bridge_status'] == 'value_overlap_supported').sum() if not crosswalk.empty else 0}

Shared field names remain insufficient by themselves. A future discovery script
must use only reviewed value-overlap or explicitly reconstructed keys.

## Carrier-feature resolution

- Carriers audited: {len(carrier_resolution)}
- Resolved carriers: {(carrier_resolution['carrier_resolution_status'] == 'resolved').sum() if not carrier_resolution.empty else 0}
- Unresolved carriers: {(carrier_resolution['carrier_resolution_status'] == 'unresolved').sum() if not carrier_resolution.empty else 0}

Heuristic carrier definitions require manual review before candidate freeze.

## Cluster hierarchy and partition balance

- Candidate cluster families: {len(cluster_hierarchy)}
- Record-by-cluster balance rows: {len(partition_balance)}
- Three-way balance candidates: {partition_balance['three_way_balance_candidate'].sum() if not partition_balance.empty else 0}

Cluster counts and deterministic hash partitions are design candidates only.
They do not establish statistical independence or adequate matched support.

## Support-family resolution

{df_to_markdown(support_table)}

Continuous seam/boundary fields must be discretized under a predeclared rule
before outcome inspection. Unavailable families must not appear in the frozen
candidate vocabulary.

## Field-role classification

- Fields classified: {len(field_roles)}
- Candidate predictors: {(field_roles['primary_field_role'] == 'candidate_predictor').sum() if not field_roles.empty else 0}
- Grouping/partition fields: {(field_roles['primary_field_role'] == 'grouping_partition').sum() if not field_roles.empty else 0}
- Provenance fields: {(field_roles['primary_field_role'] == 'provenance').sum() if not field_roles.empty else 0}
- Forbidden predictive-leakage fields: {(field_roles['primary_field_role'] == 'forbidden_predictive_leakage').sum() if not field_roles.empty else 0}
- Manual-review fields: {field_roles['manual_review_required'].sum() if not field_roles.empty else 0}

A field may be valid for matching, grouping, or provenance while remaining
forbidden as a predictive carrier field.

## Duplicate and lineage resolution

- Candidate artifact variants audited: {len(duplicate_resolution)}
- Selected canonical artifacts: {duplicate_resolution['selected_canonical'].sum() if not duplicate_resolution.empty else 0}
- Content-identical mirrors: {duplicate_resolution['content_identical_to_selected'].sum() if not duplicate_resolution.empty else 0}
- Alternate noncanonical artifacts: {(duplicate_resolution['resolution_status'] == 'alternate_noncanonical').sum() if not duplicate_resolution.empty else 0}

Large OBS-073/075 route tables are not selected merely because they expose many
path identifiers. They remain optional enrichment unless an explicit semantic
bridge to the registry lineage is later established.

## Discovery gates

OBS-084 candidate discovery should remain blocked for any record lacking:

1. a reviewed canonical or reconstructible observation source;
2. a reviewed observation-key bridge;
3. an explicit carrier-feature definition;
4. a defensible cluster hierarchy;
5. per-record partition balance;
6. a frozen support vocabulary;
7. a reviewed field-role classification;
8. source hashes and repository commit identity.

## Outputs

- `obs084a_canonical_source_manifest.csv`
- `obs084a_artifact_duplicate_resolution.csv`
- `obs084a_registry_record_to_prediction_source.csv`
- `obs084a_observation_key_crosswalk.csv`
- `obs084a_carrier_feature_resolution.csv`
- `obs084a_cluster_hierarchy.csv`
- `obs084a_partition_balance_by_record.csv`
- `obs084a_support_family_resolution.csv`
- `obs084a_field_role_classification.csv`
- `obs084a_lineage_bridge_summary.csv`
- `obs084a_lineage_bridge_report.md`

## Limitations

- Canonical selection is rule-based and must be reviewed where multiple
  non-identical authoritative-looking artifacts exist.
- Value overlap does not prove that two columns have identical scientific
  semantics.
- Carrier resolution may be heuristic when upstream manifests do not encode the
  exact registry carrier.
- Deterministic partitions are proposals only; reserved evidence is not created
  or unlocked by this script.
- No localization predicate, support, contrast, witness, or FL level is created.

## Canonical result statement

OBS-084a lineage bridging reduces the broad repository artifact inventory to an
explicit OBS-078–083 evidence spine and audits whether registry records can be
connected to observation-level sources, carrier definitions, structural units,
support vocabularies, and leakage-safe field roles. It establishes semantic
lineage feasibility only and no direct failure support, causal origin, repair
target, actionability, external generalization, or formal topology.
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_paths: dict[str, Path] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for role, explicit in CANONICAL_ROLE_SPECS.items():
        candidates = discover_role_candidates(repo_root, role, explicit)
        winner, rows = choose_canonical(candidates)
        for row in rows:
            row["source_role"] = role
        duplicate_rows.extend(rows)
        if winner:
            selected_paths[role] = winner

    duplicate_resolution = pd.DataFrame(duplicate_rows)
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    for role in CANONICAL_ROLE_SPECS:
        path = selected_paths.get(role, repo_root / "__missing__")
        info, df = artifact_info(path, args.max_read_mb, args.sample_rows)
        frames[role] = df
        manifest_rows.append({
            "source_role": role,
            "selected_path": path.relative_to(repo_root).as_posix() if path.exists() else "",
            "exists": info.exists,
            "size_mb": info.size_bytes / (1024 * 1024),
            "sha256": info.sha256,
            "rows_read": info.rows,
            "column_count": len(info.columns),
            "read_status": info.read_status,
            "code_commit": git_commit(repo_root),
            "script_version": SCRIPT_VERSION,
        })
    canonical_manifest = pd.DataFrame(manifest_rows)

    registry = canonical_registry_frame(frames)
    feature_df = frames.get("canonical_feature_table", pd.DataFrame())
    feature_manifest = frames.get("canonical_feature_manifest", pd.DataFrame())

    record_map = build_record_source_map(registry, frames, selected_paths)
    crosswalk = build_key_crosswalk(feature_df, frames, selected_paths)
    carrier_resolution = build_carrier_resolution(registry, feature_df, feature_manifest)
    cluster_hierarchy = build_cluster_hierarchy(feature_df, args.partition_seed)
    partition_balance = build_partition_balance(
        registry, feature_df, cluster_hierarchy, args.partition_seed, args.min_per_class_clusters
    )
    support_resolution = build_support_resolution(frames, selected_paths)
    field_roles = build_field_roles(frames)

    c2 = record_map[record_map["confirmation_eligibility"] == "fl3_confirmation_eligible"] if not record_map.empty else pd.DataFrame()
    c2_resolved = int((c2["mapping_status"] != "unresolved").sum()) if not c2.empty else 0
    all_c2_resolved = bool(len(c2) > 0 and c2_resolved == len(c2))
    all_carriers_resolved = bool(not carrier_resolution.empty and (carrier_resolution["carrier_resolution_status"] == "resolved").all())
    overlap_bridges = int((crosswalk["schema_bridge_status"] == "value_overlap_supported").sum()) if not crosswalk.empty else 0
    balanced_rows = int(partition_balance["three_way_balance_candidate"].sum()) if not partition_balance.empty else 0
    canonical_required = canonical_manifest[canonical_manifest["source_role"].isin({
        "canonical_feature_table", "registry", "obs083_subclasses", "pairwise_predictions"
    })]
    required_present = bool(not canonical_required.empty and canonical_required["exists"].all())

    if required_present and all_c2_resolved and all_carriers_resolved and overlap_bridges > 0 and balanced_rows > 0:
        overall = "lineage_bridge_ready_for_reviewed_candidate_discovery_design"
    elif required_present and c2_resolved > 0:
        overall = "lineage_bridge_partially_ready_with_manual_resolution_required"
    else:
        overall = "lineage_bridge_blocked"

    summary = pd.DataFrame([{
        "overall_status": overall,
        "script_version": SCRIPT_VERSION,
        "code_commit": git_commit(repo_root),
        "canonical_sources_present": int(canonical_manifest["exists"].sum()),
        "canonical_source_roles": len(canonical_manifest),
        "registry_records": len(record_map),
        "c2_records": len(c2),
        "c2_sources_resolved_or_candidate": c2_resolved,
        "all_carriers_resolved": all_carriers_resolved,
        "value_overlap_key_bridges": overlap_bridges,
        "three_way_balance_candidate_rows": balanced_rows,
        "support_families_available": int(support_resolution["available"].sum()) if not support_resolution.empty else 0,
        "forbidden_predictive_fields": int((field_roles["primary_field_role"] == "forbidden_predictive_leakage").sum()) if not field_roles.empty else 0,
        "claim_scope": "semantic_lineage_feasibility_only",
    }])

    outputs = {
        "obs084a_canonical_source_manifest.csv": canonical_manifest,
        "obs084a_artifact_duplicate_resolution.csv": duplicate_resolution,
        "obs084a_registry_record_to_prediction_source.csv": record_map,
        "obs084a_observation_key_crosswalk.csv": crosswalk,
        "obs084a_carrier_feature_resolution.csv": carrier_resolution,
        "obs084a_cluster_hierarchy.csv": cluster_hierarchy,
        "obs084a_partition_balance_by_record.csv": partition_balance,
        "obs084a_support_family_resolution.csv": support_resolution,
        "obs084a_field_role_classification.csv": field_roles,
        "obs084a_lineage_bridge_summary.csv": summary,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)

    write_report(
        output_dir / "obs084a_lineage_bridge_report.md",
        canonical_manifest, duplicate_resolution, record_map, crosswalk,
        carrier_resolution, cluster_hierarchy, partition_balance,
        support_resolution, field_roles, summary,
    )

    print(f"OBS-084a lineage bridge complete: {overall}")
    print(f"Canonical source roles: {len(canonical_manifest)}")
    print(f"Registry records / C2 records: {len(record_map)} / {len(c2)}")
    print(f"C2 sources resolved or candidate: {c2_resolved}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
