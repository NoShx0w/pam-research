#!/usr/bin/env python3
"""
Validate the integrated canonical family layer.

This script validates the current canonical family/gateway layer across:

- gateway_comparison
- temporal_depth_summary
- memory_compression_summary
- event_family_features
- event_family_assignment
- family_aggregation

Outputs:
- outputs/canonical/validation/family_layer_validation.txt
- outputs/canonical/validation/family_layer_warnings.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


CANONICAL_ROUTE_CLASSES = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

SUPPORTED_EVENT_TYPES = [
    "core_internal",
    "core_to_escape",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def add_warning(warnings: List[dict], level: str, check: str, message: str) -> None:
    warnings.append(
        {
            "level": level,
            "check": check,
            "message": message,
        }
    )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    table_name: str,
    warnings: List[dict],
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_warning(
            warnings,
            "ERROR",
            f"{table_name}.required_columns",
            f"Missing required columns: {missing}",
        )


def check_duplicates(
    df: pd.DataFrame,
    key_cols: list[str],
    table_name: str,
    warnings: List[dict],
) -> None:
    if not all(c in df.columns for c in key_cols):
        return
    if df.duplicated(subset=key_cols).any():
        add_warning(
            warnings,
            "ERROR",
            f"{table_name}.duplicate_keys",
            f"Duplicate keys found for columns: {key_cols}",
        )


def check_route_classes(
    df: pd.DataFrame,
    col: str,
    table_name: str,
    warnings: List[dict],
) -> None:
    if col not in df.columns:
        return
    observed = set(df[col].dropna().astype(str))
    missing = [c for c in CANONICAL_ROUTE_CLASSES if c not in observed]
    extra = [c for c in observed if c not in CANONICAL_ROUTE_CLASSES]
    if missing:
        add_warning(
            warnings,
            "WARNING",
            f"{table_name}.missing_route_classes",
            f"Missing canonical route classes: {missing}",
        )
    if extra:
        add_warning(
            warnings,
            "WARNING",
            f"{table_name}.extra_route_classes",
            f"Unexpected route classes: {extra}",
        )


def check_event_types(
    df: pd.DataFrame,
    col: str,
    table_name: str,
    warnings: List[dict],
) -> None:
    if col not in df.columns:
        return
    observed = set(df[col].dropna().astype(str))
    extra = [c for c in observed if c not in SUPPORTED_EVENT_TYPES]
    if extra:
        add_warning(
            warnings,
            "WARNING",
            f"{table_name}.extra_event_types",
            f"Unexpected event types: {extra}",
        )


def safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def validate_files_exist(paths: Dict[str, Path], warnings: List[dict]) -> dict[str, pd.DataFrame]:
    loaded = {}
    for name, path in paths.items():
        try:
            loaded[name] = load_csv(path)
        except Exception as exc:
            add_warning(warnings, "ERROR", f"{name}.load", str(exc))
    return loaded


def validate_schemas(dfs: dict[str, pd.DataFrame], warnings: List[dict]) -> None:
    if "gateway_comparison" in dfs:
        df = dfs["gateway_comparison"]
        require_columns(
            df,
            ["route_class", "local_only_metric", "crossing_rate", "source_version"],
            "gateway_comparison",
            warnings,
        )
        check_duplicates(df, ["route_class"], "gateway_comparison", warnings)
        check_route_classes(df, "route_class", "gateway_comparison", warnings)

    if "temporal_depth_summary" in dfs:
        df = dfs["temporal_depth_summary"]
        require_columns(
            df,
            ["route_class", "best_horizon_k", "memory_regime_label", "source_version"],
            "temporal_depth_summary",
            warnings,
        )
        check_duplicates(df, ["route_class"], "temporal_depth_summary", warnings)
        check_route_classes(df, "route_class", "temporal_depth_summary", warnings)

    if "memory_compression_summary" in dfs:
        df = dfs["memory_compression_summary"]
        require_columns(
            df,
            ["route_class", "forgetting_share", "dominant_compression_state", "source_version"],
            "memory_compression_summary",
            warnings,
        )
        check_duplicates(df, ["route_class"], "memory_compression_summary", warnings)
        check_route_classes(df, "route_class", "memory_compression_summary", warnings)

    if "event_family_features" in dfs:
        df = dfs["event_family_features"]
        require_columns(
            df,
            [
                "event_id",
                "route_class",
                "event_type",
                "local_gateway_strength",
                "effective_temporal_depth",
                "forgetting_share",
                "normalization_version",
                "provenance_class_local_gateway_strength",
                "projection_flag_local_gateway_strength",
            ],
            "event_family_features",
            warnings,
        )
        check_duplicates(df, ["event_id"], "event_family_features", warnings)
        check_route_classes(df, "route_class", "event_family_features", warnings)
        check_event_types(df, "event_type", "event_family_features", warnings)

    if "event_family_assignment" in dfs:
        df = dfs["event_family_assignment"]
        require_columns(
            df,
            [
                "event_id",
                "assigned_family",
                "assignment_confidence",
                "assignment_version",
                "source_feature_version",
            ],
            "event_family_assignment",
            warnings,
        )
        check_duplicates(df, ["event_id"], "event_family_assignment", warnings)
        check_route_classes(df, "assigned_family", "event_family_assignment", warnings)

    if "family_aggregation" in dfs:
        df = dfs["family_aggregation"]
        require_columns(
            df,
            [
                "route_class",
                "n_events",
                "mean_assignment_confidence",
                "mean_forgetting_share",
                "aggregation_version",
                "source_feature_version",
                "source_assignment_version",
            ],
            "family_aggregation",
            warnings,
        )
        check_duplicates(df, ["route_class"], "family_aggregation", warnings)
        check_route_classes(df, "route_class", "family_aggregation", warnings)


def validate_cross_table_consistency(dfs: dict[str, pd.DataFrame], warnings: List[dict]) -> None:
    if "event_family_features" in dfs and "event_family_assignment" in dfs:
        feat = dfs["event_family_features"]
        assign = dfs["event_family_assignment"]

        feat_ids = set(feat["event_id"].astype(str))
        assign_ids = set(assign["event_id"].astype(str))

        missing_from_feat = assign_ids - feat_ids
        missing_from_assign = feat_ids - assign_ids

        if missing_from_feat:
            add_warning(
                warnings,
                "ERROR",
                "assignment.feature_join",
                f"Assignment rows missing from features: {len(missing_from_feat)}",
            )
        if missing_from_assign:
            add_warning(
                warnings,
                "WARNING",
                "feature.assignment_join",
                f"Feature rows missing from assignment: {len(missing_from_assign)}",
            )

        joined = assign.merge(
            feat[["event_id", "route_class"]],
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        mismatch = joined["assigned_family"].astype(str) != joined["route_class"].astype(str)
        if mismatch.any():
            add_warning(
                warnings,
                "WARNING",
                "assignment.route_class_match",
                f"Assigned family differs from observed route_class in {int(mismatch.sum())} rows.",
            )

    if "family_aggregation" in dfs and "event_family_assignment" in dfs:
        agg = dfs["family_aggregation"]
        assign = dfs["event_family_assignment"]

        expected_counts = (
            assign.groupby("assigned_family", dropna=False)
            .size()
            .rename("expected_n_events")
            .reset_index()
            .rename(columns={"assigned_family": "route_class"})
        )
        merged = agg.merge(expected_counts, on="route_class", how="left", validate="one_to_one")
        bad = pd.to_numeric(merged["n_events"], errors="coerce") != pd.to_numeric(
            merged["expected_n_events"], errors="coerce"
        )
        if bad.any():
            add_warning(
                warnings,
                "ERROR",
                "family_aggregation.count_match",
                f"Aggregation counts mismatch in {int(bad.sum())} families.",
            )

    if "family_aggregation" in dfs and "event_family_features" in dfs and "event_family_assignment" in dfs:
        agg = dfs["family_aggregation"]
        feat = dfs["event_family_features"]
        assign = dfs["event_family_assignment"]

        feat_ver = sorted(feat["normalization_version"].dropna().astype(str).unique().tolist())
        assign_feat_ver = sorted(assign["source_feature_version"].dropna().astype(str).unique().tolist())
        agg_feat_ver = sorted(agg["source_feature_version"].dropna().astype(str).unique().tolist())
        agg_assign_ver = sorted(agg["source_assignment_version"].dropna().astype(str).unique().tolist())
        assign_ver = sorted(assign["assignment_version"].dropna().astype(str).unique().tolist())

        if feat_ver != assign_feat_ver:
            add_warning(
                warnings,
                "WARNING",
                "version.feature_vs_assignment",
                f"Feature version mismatch between features and assignment: {feat_ver} vs {assign_feat_ver}",
            )
        if feat_ver != agg_feat_ver:
            add_warning(
                warnings,
                "WARNING",
                "version.feature_vs_aggregation",
                f"Feature version mismatch between features and aggregation: {feat_ver} vs {agg_feat_ver}",
            )
        if assign_ver != agg_assign_ver:
            add_warning(
                warnings,
                "WARNING",
                "version.assignment_vs_aggregation",
                f"Assignment version mismatch between assignment and aggregation: {assign_ver} vs {agg_assign_ver}",
            )


def validate_scientific_coherence(dfs: dict[str, pd.DataFrame], warnings: List[dict]) -> None:
    if "family_aggregation" not in dfs:
        return

    agg = dfs["family_aggregation"].set_index("route_class")

    try:
        be = float(agg.loc["branch_exit", "mean_forgetting_share"])
        co = float(agg.loc["stable_seam_corridor", "mean_forgetting_share"])
        rh = float(agg.loc["reorganization_heavy", "mean_forgetting_share"])
        if not (be < co < rh):
            add_warning(
                warnings,
                "WARNING",
                "science.forgetting_order",
                "Expected forgetting ordering branch_exit < stable_seam_corridor < reorganization_heavy is violated.",
            )
    except Exception:
        add_warning(
            warnings,
            "WARNING",
            "science.forgetting_order",
            "Could not verify forgetting ordering.",
        )

    try:
        be = float(agg.loc["branch_exit", "mean_effective_temporal_depth"])
        co = float(agg.loc["stable_seam_corridor", "mean_effective_temporal_depth"])
        rh = float(agg.loc["reorganization_heavy", "mean_effective_temporal_depth"])
        if not (be < co < rh):
            add_warning(
                warnings,
                "WARNING",
                "science.temporal_depth_order",
                "Expected temporal-depth ordering branch_exit < stable_seam_corridor < reorganization_heavy is violated.",
            )
    except Exception:
        add_warning(
            warnings,
            "WARNING",
            "science.temporal_depth_order",
            "Could not verify temporal-depth ordering.",
        )

    try:
        co = float(agg.loc["stable_seam_corridor", "mean_local_gateway_strength"])
        be = float(agg.loc["branch_exit", "mean_local_gateway_strength"])
        rh = float(agg.loc["reorganization_heavy", "mean_local_gateway_strength"])
        if not (co > be and co > rh):
            add_warning(
                warnings,
                "WARNING",
                "science.local_gateway_strength",
                "Expected stable_seam_corridor to have strongest local gateway strength is violated.",
            )
    except Exception:
        add_warning(
            warnings,
            "WARNING",
            "science.local_gateway_strength",
            "Could not verify local gateway strength ordering.",
        )

    for family in CANONICAL_ROUTE_CLASSES:
        try:
            ci = float(agg.loc[family, "core_internal_share"])
            ce = float(agg.loc[family, "core_to_escape_share"])
            total = ci + ce
            if abs(total - 1.0) > 1e-6:
                add_warning(
                    warnings,
                    "WARNING",
                    "science.event_type_share_sum",
                    f"{family} event-type shares sum to {total:.6f}, expected ~1.0.",
                )
        except Exception:
            add_warning(
                warnings,
                "WARNING",
                "science.event_type_share_sum",
                f"Could not verify event-type shares for {family}.",
            )


def validate_provenance_sanity(dfs: dict[str, pd.DataFrame], warnings: List[dict]) -> None:
    if "event_family_features" not in dfs:
        return

    feat = dfs["event_family_features"]

    required_prov_cols = [
        "provenance_class_local_gateway_strength",
        "provenance_class_effective_temporal_depth",
        "provenance_class_forgetting_share",
        "projection_flag_local_gateway_strength",
        "projection_flag_effective_temporal_depth",
        "projection_flag_forgetting_share",
    ]
    missing = [c for c in required_prov_cols if c not in feat.columns]
    if missing:
        add_warning(
            warnings,
            "WARNING",
            "provenance.columns",
            f"Missing provenance columns: {missing}",
        )
        return

    for col in [
        "provenance_class_local_gateway_strength",
        "provenance_class_effective_temporal_depth",
        "provenance_class_forgetting_share",
    ]:
        observed = set(feat[col].dropna().astype(str))
        if observed != {"family_mapped"}:
            add_warning(
                warnings,
                "WARNING",
                f"provenance.{col}",
                f"Unexpected provenance values in {col}: {sorted(observed)}",
            )

    for col in [
        "projection_flag_local_gateway_strength",
        "projection_flag_effective_temporal_depth",
        "projection_flag_forgetting_share",
    ]:
        if not feat[col].fillna(False).astype(bool).all():
            add_warning(
                warnings,
                "WARNING",
                f"provenance.{col}",
                f"Some rows are not flagged as projected in {col}.",
            )


def write_validation_report(path: Path, warnings: List[dict], dfs: dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("family_layer validation report")
    lines.append("")

    for name, df in dfs.items():
        lines.append(f"{name}: rows={len(df)}")
    lines.append("")

    error_count = sum(1 for w in warnings if w["level"] == "ERROR")
    warning_count = sum(1 for w in warnings if w["level"] == "WARNING")

    lines.append(f"errors: {error_count}")
    lines.append(f"warnings: {warning_count}")
    lines.append("")

    if warnings:
        lines.append("issues:")
        for w in warnings:
            lines.append(f"- [{w['level']}] {w['check']}: {w['message']}")
    else:
        lines.append("issues:")
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_warnings_csv(path: Path, warnings: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(warnings, columns=["level", "check", "message"])
    df.to_csv(path, index=False)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    paths = {
        "gateway_comparison": repo_root / "outputs" / "canonical" / "gateway_comparison.csv",
        "temporal_depth_summary": repo_root / "outputs" / "canonical" / "temporal_depth_summary.csv",
        "memory_compression_summary": repo_root / "outputs" / "canonical" / "memory_compression_summary.csv",
        "event_family_features": repo_root / "outputs" / "canonical" / "event_family_features.csv",
        "event_family_assignment": repo_root / "outputs" / "canonical" / "event_family_assignment.csv",
        "family_aggregation": repo_root / "outputs" / "canonical" / "family_aggregation.csv",
    }

    out_report = repo_root / "outputs" / "canonical" / "validation" / "family_layer_validation.txt"
    out_warnings = repo_root / "outputs" / "canonical" / "validation" / "family_layer_warnings.csv"

    warnings: List[dict] = []

    try:
        dfs = validate_files_exist(paths, warnings)

        if not dfs:
            raise RuntimeError("No canonical files could be loaded.")

        validate_schemas(dfs, warnings)
        validate_cross_table_consistency(dfs, warnings)
        validate_scientific_coherence(dfs, warnings)
        validate_provenance_sanity(dfs, warnings)

        write_validation_report(out_report, warnings, dfs)
        write_warnings_csv(out_warnings, warnings)

        print(f"Wrote: {out_report}")
        print(f"Wrote: {out_warnings}")

        error_count = sum(1 for w in warnings if w["level"] == "ERROR")
        warning_count = sum(1 for w in warnings if w["level"] == "WARNING")
        print(f"Validation errors: {error_count}")
        print(f"Validation warnings: {warning_count}")

        return 1 if error_count > 0 else 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
