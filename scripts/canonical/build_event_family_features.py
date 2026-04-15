#!/usr/bin/env python3
"""
Build the canonical event-level family feature table.

This script consolidates heterogeneous gateway/event rows into a single
canonical artifact:

    outputs/canonical/event_family_features.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/event_family_features_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- includes only gateway-relevant rows with event types:
    * core_internal
    * core_to_escape
- normalizes five primary source row classes:
    * gateway_instance      (OBS-035c)
    * gateway_refined       (OBS-036)
    * gateway_history       (OBS-037)
    * gateway_pre_second_step (OBS-037b)
    * gateway_context       (OBS-039)
- uses canonical family-level tables as family-mapped enrichments
- does not yet attempt aggressive trajectory linkage
- leaves trajectory_id null unless a justified linkage exists

Expected inputs
---------------
Primary event sources:
- outputs/obs035c_instance_level_gateway_predictor/gateway_crossing_instance_dataset.csv
- outputs/obs036_gateway_state_refinement/gateway_state_refined_dataset.csv
- outputs/obs037_history_aware_gateway_predictor/gateway_crossing_history_dataset.csv
- outputs/obs037b_pre_second_step_gateway_predictor/gateway_crossing_pre_second_step_dataset.csv
- outputs/obs039_reorganization_heavy_path_context/reorg_path_context_dataset.csv

Family-level enrichments:
- outputs/canonical/gateway_comparison.csv
- outputs/canonical/temporal_depth_summary.csv
- outputs/canonical/memory_compression_summary.csv
"""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import sys
from typing import Iterable, List

import pandas as pd


CANONICAL_ROUTE_CLASSES = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

SUPPORTED_SOURCE_ROW_TYPES = [
    "gateway_instance",
    "gateway_refined",
    "gateway_history",
    "gateway_pre_second_step",
    "gateway_context",
]

SUPPORTED_EVENT_TYPES = [
    "core_internal",
    "core_to_escape",
]

NORMALIZATION_VERSION = "event_family_features_v1"


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def normalize_event_type(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if s in SUPPORTED_EVENT_TYPES:
        return s
    return pd.NA


def stable_hash(parts: Iterable[object]) -> str:
    payload = "||".join("" if pd.isna(x) else str(x) for x in parts)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def synthesize_event_id(row: pd.Series) -> str:
    row_type = row["source_row_type"]
    src_key = row.get("source_key", "")
    route_class = row.get("route_class", "")
    event_type = row.get("event_type", "")
    signature = [
        row_type,
        src_key,
        route_class,
        event_type,
        row.get("composition_typed", ""),
        row.get("launch_generator", ""),
        row.get("prev_generator", ""),
        row.get("prelaunch_word", ""),
        row.get("history_word", ""),
        row.get("generator_word", ""),
        row.get("mean_relational", ""),
        row.get("mean_anisotropy", ""),
        row.get("mean_distance", ""),
    ]
    return f"{row_type}_{stable_hash(signature)}"


def base_frame(df: pd.DataFrame, source_row_type: str, source_table: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # identity / source
    out["source_row_type"] = source_row_type
    out["source_table"] = source_table
    out["source_key"] = [f"{source_row_type}:{i}" for i in range(len(df))]

    # common fields initialized to NA unless filled below
    cols = [
        "route_class",
        "event_type",
        "y_cross",
        "composition_typed",
        "launch_generator",
        "next_generator",
        "prev_generator",
        "launch_state",
        "launch_target",
        "prev_state",
        "prev_target",
        "history_word",
        "generator_word",
        "prelaunch_word",
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
        "trajectory_id",
        # OBS-039 context fields
        "cum_core_before",
        "cum_escape_before",
        "core_share_before",
        "escape_share_before",
        "escape_touched_before",
        "recent_sector_entropy",
        "recent_hotspot_entropy",
        "recent_shared_count",
        "recent_non_hotspot_count",
        "runlen_core_before",
        "runlen_escape_before",
    ]
    for c in cols:
        out[c] = pd.NA

    return out


def normalize_obs035c(df: pd.DataFrame, path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        ["route_class", "crossing_type", "y_cross", "launch_generator", "launch_state",
         "launch_target", "mean_relational", "mean_anisotropy", "mean_distance", "composition_typed"],
        str(path),
    )
    out = base_frame(df, "gateway_instance", repo_relative(path, repo_root))
    out["route_class"] = df["route_class"]
    out["event_type"] = df["crossing_type"].map(normalize_event_type)
    out["y_cross"] = df["y_cross"]
    out["composition_typed"] = df["composition_typed"]
    out["launch_generator"] = df["launch_generator"]
    out["launch_state"] = df["launch_state"]
    out["launch_target"] = df["launch_target"]
    out["mean_relational"] = df["mean_relational"]
    out["mean_anisotropy"] = df["mean_anisotropy"]
    out["mean_distance"] = df["mean_distance"]
    return out


def normalize_obs036(df: pd.DataFrame, path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        ["route_class", "crossing_type", "y_cross", "launch_generator", "launch_state",
         "launch_target", "mean_relational", "mean_anisotropy", "mean_distance", "composition_typed"],
        str(path),
    )
    out = base_frame(df, "gateway_refined", repo_relative(path, repo_root))
    out["route_class"] = df["route_class"]
    out["event_type"] = df["crossing_type"].map(normalize_event_type)
    out["y_cross"] = df["y_cross"]
    out["composition_typed"] = df["composition_typed"]
    out["launch_generator"] = df["launch_generator"]
    out["launch_state"] = df["launch_state"]
    out["launch_target"] = df["launch_target"]
    out["mean_relational"] = df["mean_relational"]
    out["mean_anisotropy"] = df["mean_anisotropy"]
    out["mean_distance"] = df["mean_distance"]
    return out


def normalize_obs037(df: pd.DataFrame, path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        ["route_class", "crossing_type", "y_cross", "prev_generator", "launch_generator",
         "prev_state", "prev_target", "launch_state", "launch_target", "history_word",
         "generator_word", "mean_relational", "mean_anisotropy", "mean_distance", "composition_typed"],
        str(path),
    )
    out = base_frame(df, "gateway_history", repo_relative(path, repo_root))
    out["route_class"] = df["route_class"]
    out["event_type"] = df["crossing_type"].map(normalize_event_type)
    out["y_cross"] = df["y_cross"]
    out["composition_typed"] = df["composition_typed"]
    out["launch_generator"] = df["launch_generator"]
    out["prev_generator"] = df["prev_generator"]
    out["launch_state"] = df["launch_state"]
    out["launch_target"] = df["launch_target"]
    out["prev_state"] = df["prev_state"]
    out["prev_target"] = df["prev_target"]
    out["history_word"] = df["history_word"]
    out["generator_word"] = df["generator_word"]
    out["mean_relational"] = df["mean_relational"]
    out["mean_anisotropy"] = df["mean_anisotropy"]
    out["mean_distance"] = df["mean_distance"]
    return out


def normalize_obs037b(df: pd.DataFrame, path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        ["route_class", "crossing_type", "y_cross", "prev_generator", "prev_state",
         "prev_target", "prelaunch_word", "mean_relational", "mean_anisotropy",
         "mean_distance", "composition_typed"],
        str(path),
    )
    out = base_frame(df, "gateway_pre_second_step", repo_relative(path, repo_root))
    out["route_class"] = df["route_class"]
    out["event_type"] = df["crossing_type"].map(normalize_event_type)
    out["y_cross"] = df["y_cross"]
    out["composition_typed"] = df["composition_typed"]
    out["prev_generator"] = df["prev_generator"]
    out["prev_state"] = df["prev_state"]
    out["prev_target"] = df["prev_target"]
    out["prelaunch_word"] = df["prelaunch_word"]
    out["mean_relational"] = df["mean_relational"]
    out["mean_anisotropy"] = df["mean_anisotropy"]
    out["mean_distance"] = df["mean_distance"]
    return out


def normalize_obs039(df: pd.DataFrame, path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        ["route_class", "crossing_type", "y_cross", "prev_generator", "prev_state",
         "prev_target", "mean_relational", "mean_anisotropy", "mean_distance",
         "cum_core_before", "cum_escape_before", "core_share_before", "escape_share_before",
         "escape_touched_before", "recent_sector_entropy", "recent_hotspot_entropy",
         "recent_shared_count", "recent_non_hotspot_count", "runlen_core_before",
         "runlen_escape_before"],
        str(path),
    )
    out = base_frame(df, "gateway_context", repo_relative(path, repo_root))
    out["route_class"] = df["route_class"]
    out["event_type"] = df["crossing_type"].map(normalize_event_type)
    out["y_cross"] = df["y_cross"]
    out["prev_generator"] = df["prev_generator"]
    out["prev_state"] = df["prev_state"]
    out["prev_target"] = df["prev_target"]
    out["mean_relational"] = df["mean_relational"]
    out["mean_anisotropy"] = df["mean_anisotropy"]
    out["mean_distance"] = df["mean_distance"]
    out["cum_core_before"] = df["cum_core_before"]
    out["cum_escape_before"] = df["cum_escape_before"]
    out["core_share_before"] = df["core_share_before"]
    out["escape_share_before"] = df["escape_share_before"]
    out["escape_touched_before"] = df["escape_touched_before"]
    out["recent_sector_entropy"] = df["recent_sector_entropy"]
    out["recent_hotspot_entropy"] = df["recent_hotspot_entropy"]
    out["recent_shared_count"] = df["recent_shared_count"]
    out["recent_non_hotspot_count"] = df["recent_non_hotspot_count"]
    out["runlen_core_before"] = df["runlen_core_before"]
    out["runlen_escape_before"] = df["runlen_escape_before"]
    return out


def attach_family_enrichments(
    df: pd.DataFrame,
    gateway_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    compression_df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    gateway_keep = gateway_df[["route_class", "local_only_metric", "crossing_rate"]].copy()
    gateway_keep = gateway_keep.rename(columns={"local_only_metric": "local_gateway_strength"})

    temporal_keep = temporal_df[["route_class", "best_horizon_k", "memory_regime_label"]].copy()
    temporal_keep = temporal_keep.rename(columns={"best_horizon_k": "effective_temporal_depth"})

    compression_keep = compression_df[["route_class", "forgetting_share", "dominant_compression_state"]].copy()

    out = out.merge(gateway_keep, on="route_class", how="left", validate="many_to_one")
    out = out.merge(temporal_keep, on="route_class", how="left", validate="many_to_one")
    out = out.merge(compression_keep, on="route_class", how="left", validate="many_to_one")

    # provenance for mapped enrichments
    out["provenance_class_local_gateway_strength"] = "family_mapped"
    out["provenance_class_effective_temporal_depth"] = "family_mapped"
    out["provenance_class_forgetting_share"] = "family_mapped"

    out["projection_flag_local_gateway_strength"] = True
    out["projection_flag_effective_temporal_depth"] = True
    out["projection_flag_forgetting_share"] = True

    return out


def build_event_family_features(
    obs035c: pd.DataFrame,
    obs036: pd.DataFrame,
    obs037: pd.DataFrame,
    obs037b: pd.DataFrame,
    obs039: pd.DataFrame,
    gateway_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    compression_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.concat([obs035c, obs036, obs037, obs037b, obs039], ignore_index=True, sort=False)

    # keep only supported primary event types
    df = df[df["event_type"].isin(SUPPORTED_EVENT_TYPES)].copy()

    # synthesize deterministic event ids
    df["event_id"] = df.apply(synthesize_event_id, axis=1)

    # no clean trajectory linkage in v1
    if "trajectory_id" not in df.columns:
        df["trajectory_id"] = pd.NA

    df = attach_family_enrichments(df, gateway_df, temporal_df, compression_df)

    df["normalization_version"] = NORMALIZATION_VERSION
    df["provisional_flag"] = True

    # canonical ordering
    df["route_class"] = pd.Categorical(
        df["route_class"],
        categories=CANONICAL_ROUTE_CLASSES,
        ordered=True,
    )
    df = df.sort_values(["route_class", "source_row_type", "event_type", "event_id"]).reset_index(drop=True)

    preferred_order = [
        "event_id",
        "source_row_type",
        "source_table",
        "source_key",
        "route_class",
        "event_type",
        "y_cross",
        "composition_typed",
        "launch_generator",
        "next_generator",
        "prev_generator",
        "launch_state",
        "launch_target",
        "prev_state",
        "prev_target",
        "history_word",
        "generator_word",
        "prelaunch_word",
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
        "trajectory_id",
        "cum_core_before",
        "cum_escape_before",
        "core_share_before",
        "escape_share_before",
        "escape_touched_before",
        "recent_sector_entropy",
        "recent_hotspot_entropy",
        "recent_shared_count",
        "recent_non_hotspot_count",
        "runlen_core_before",
        "runlen_escape_before",
        "local_gateway_strength",
        "crossing_rate",
        "effective_temporal_depth",
        "memory_regime_label",
        "forgetting_share",
        "dominant_compression_state",
        "provenance_class_local_gateway_strength",
        "provenance_class_effective_temporal_depth",
        "provenance_class_forgetting_share",
        "projection_flag_local_gateway_strength",
        "projection_flag_effective_temporal_depth",
        "projection_flag_forgetting_share",
        "normalization_version",
        "provisional_flag",
    ]
    actual_order = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[actual_order]


def validate_event_family_features(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    required_cols = [
        "event_id",
        "source_row_type",
        "source_table",
        "source_key",
        "route_class",
        "event_type",
        "y_cross",
        "local_gateway_strength",
        "effective_temporal_depth",
        "forgetting_share",
        "normalization_version",
        "provisional_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"event_family_features is missing required columns: {missing}")

    if df["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id values found in event_family_features.")

    bad_route_classes = sorted(set(df["route_class"].dropna().astype(str)) - set(CANONICAL_ROUTE_CLASSES))
    if bad_route_classes:
        warnings.append(f"Unexpected non-canonical route_class values present: {bad_route_classes}")

    bad_event_types = sorted(set(df["event_type"].dropna().astype(str)) - set(SUPPORTED_EVENT_TYPES))
    if bad_event_types:
        warnings.append(f"Unexpected event_type values present: {bad_event_types}")

    bad_row_types = sorted(set(df["source_row_type"].dropna().astype(str)) - set(SUPPORTED_SOURCE_ROW_TYPES))
    if bad_row_types:
        warnings.append(f"Unexpected source_row_type values present: {bad_row_types}")

    represented_row_types = set(df["source_row_type"].dropna().astype(str))
    missing_row_types = [x for x in SUPPORTED_SOURCE_ROW_TYPES if x not in represented_row_types]
    if missing_row_types:
        warnings.append(f"Some supported source_row_types are not represented: {missing_row_types}")

    for col in ["local_gateway_strength", "effective_temporal_depth", "forgetting_share"]:
        if df[col].isna().any():
            warnings.append(f"Some rows have null {col} values.")

    return warnings


def write_validation_report(path: Path, df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("event_family_features validation report")
    lines.append(f"normalization_version: {NORMALIZATION_VERSION}")
    lines.append(f"n_rows: {len(df)}")
    lines.append(f"route_classes: {', '.join(sorted(df['route_class'].dropna().astype(str).unique().tolist()))}")
    lines.append(f"event_types: {', '.join(sorted(df['event_type'].dropna().astype(str).unique().tolist()))}")
    lines.append(f"source_row_types: {', '.join(sorted(df['source_row_type'].dropna().astype(str).unique().tolist()))}")
    lines.append("")

    if warnings:
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("warnings:")
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    # primary event sources
    path_035c = repo_root / "outputs" / "obs035c_instance_level_gateway_predictor" / "gateway_crossing_instance_dataset.csv"
    path_036 = repo_root / "outputs" / "obs036_gateway_state_refinement" / "gateway_state_refined_dataset.csv"
    path_037 = repo_root / "outputs" / "obs037_history_aware_gateway_predictor" / "gateway_crossing_history_dataset.csv"
    path_037b = repo_root / "outputs" / "obs037b_pre_second_step_gateway_predictor" / "gateway_crossing_pre_second_step_dataset.csv"
    path_039 = repo_root / "outputs" / "obs039_reorganization_heavy_path_context" / "reorg_path_context_dataset.csv"

    # canonical enrichments
    path_gateway = repo_root / "outputs" / "canonical" / "gateway_comparison.csv"
    path_temporal = repo_root / "outputs" / "canonical" / "temporal_depth_summary.csv"
    path_compression = repo_root / "outputs" / "canonical" / "memory_compression_summary.csv"

    out_csv = repo_root / "outputs" / "canonical" / "event_family_features.csv"
    out_report = repo_root / "outputs" / "canonical" / "validation" / "event_family_features_validation.txt"

    try:
        df_035c = normalize_obs035c(load_csv(path_035c), path_035c, repo_root)
        df_036 = normalize_obs036(load_csv(path_036), path_036, repo_root)
        df_037 = normalize_obs037(load_csv(path_037), path_037, repo_root)
        df_037b = normalize_obs037b(load_csv(path_037b), path_037b, repo_root)
        df_039 = normalize_obs039(load_csv(path_039), path_039, repo_root)

        gateway_df = load_csv(path_gateway)
        temporal_df = load_csv(path_temporal)
        compression_df = load_csv(path_compression)

        out_df = build_event_family_features(
            df_035c, df_036, df_037, df_037b, df_039,
            gateway_df, temporal_df, compression_df
        )
        warnings = validate_event_family_features(out_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, out_df, warnings)

        print(f"Wrote: {out_csv}")
        print(f"Wrote: {out_report}")
        if warnings:
            print("Validation warnings:")
            for w in warnings:
                print(f" - {w}")
        else:
            print("Validation warnings: none")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
