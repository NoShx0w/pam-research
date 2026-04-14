#!/usr/bin/env python3
"""
Build the first canonical family-level gateway comparison table.

This script consolidates family-level gateway outputs from OBS-038 into a
single canonical artifact:

    outputs/canonical/gateway_comparison.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/gateway_comparison_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- uses OBS-038 as the primary source
- does not attempt aggressive cross-study joins
- keeps provenance explicit
- prefers a small honest table over inferred completeness

Expected inputs
---------------
- outputs/obs038_family_specific_gateway_laws/family_specific_gateway_metrics.csv
- outputs/obs038_family_specific_gateway_laws/family_specific_gateway_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import pandas as pd


CANONICAL_ROUTE_CLASSES = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

SOURCE_VERSION = "gateway_comparison_v1"


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


def normalize_metrics(df: pd.DataFrame, metrics_path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        required=["route_class", "n_rows", "auc"],
        name=str(metrics_path),
    )

    out = df.copy()

    out = out.rename(
        columns={
            "auc": "local_only_metric",
        }
    )

    keep_cols = [
        c
        for c in [
            "route_class",
            "n_rows",
            "local_only_metric",
        ]
        if c in out.columns
    ]
    out = out[keep_cols].copy()

    # Semantic cleanup.
    if "n_rows" in out.columns:
        out["n_rows"] = pd.to_numeric(out["n_rows"], errors="coerce").astype("Int64")

    out["metric_name"] = "auc"
    out["source_metrics_table"] = repo_relative(metrics_path, repo_root)
    return out


def normalize_summary(df: pd.DataFrame, summary_path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        required=["route_class", "crossing_rate"],
        name=str(summary_path),
    )

    desired_cols = [
        "route_class",
        "crossing_rate",
        "mean_relational_cross",
        "mean_relational_internal",
        "mean_anisotropy_cross",
        "mean_anisotropy_internal",
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
    ]

    keep_cols = [c for c in desired_cols if c in df.columns]
    out = df[keep_cols].copy()
    out["source_summary_table"] = repo_relative(summary_path, repo_root)
    return out


def build_gateway_comparison(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.merge(
        metrics_df,
        summary_df,
        on="route_class",
        how="outer",
        validate="one_to_one",
    )

    df["route_class"] = pd.Categorical(
        df["route_class"],
        categories=CANONICAL_ROUTE_CLASSES,
        ordered=True,
    )
    df = df.sort_values("route_class").reset_index(drop=True)

    df["source_version"] = SOURCE_VERSION
    df["provisional_flag"] = True

    preferred_order = [
        "route_class",
        "crossing_rate",
        "local_only_metric",
        "metric_name",
        "n_rows",
        "mean_relational_cross",
        "mean_relational_internal",
        "mean_anisotropy_cross",
        "mean_anisotropy_internal",
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
        "source_metrics_table",
        "source_summary_table",
        "source_version",
        "provisional_flag",
    ]
    actual_order = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    return df[actual_order]


def validate_gateway_comparison(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    required_cols = [
        "route_class",
        "crossing_rate",
        "local_only_metric",
        "metric_name",
        "n_rows",
        "source_version",
        "provisional_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"gateway_comparison is missing required columns: {missing}")

    if df["route_class"].isna().any():
        warnings.append("Some rows have null route_class values.")

    if df["route_class"].duplicated().any():
        raise ValueError("Duplicate route_class rows found in gateway_comparison.")

    observed_route_classes = [x for x in df["route_class"].astype(str).tolist()]
    missing_classes = [c for c in CANONICAL_ROUTE_CLASSES if c not in observed_route_classes]
    extra_classes = [c for c in observed_route_classes if c not in CANONICAL_ROUTE_CLASSES]

    if missing_classes:
        warnings.append(f"Missing canonical route classes: {missing_classes}")
    if extra_classes:
        warnings.append(f"Unexpected non-canonical route classes present: {extra_classes}")

    metric_names = set(df["metric_name"].dropna().astype(str))
    if len(metric_names) != 1:
        warnings.append(f"Multiple metric_name values found: {sorted(metric_names)}")

    try:
        by_class = df.set_index("route_class")
        corridor_auc = float(by_class.loc["stable_seam_corridor", "local_only_metric"])
        reorg_auc = float(by_class.loc["reorganization_heavy", "local_only_metric"])
        if corridor_auc < reorg_auc:
            warnings.append(
                "Unexpected ordering: stable_seam_corridor local_only_metric "
                "is lower than reorganization_heavy."
            )
    except Exception:
        warnings.append(
            "Could not verify local_only_metric ordering between "
            "stable_seam_corridor and reorganization_heavy."
        )

    try:
        _ = float(by_class.loc["branch_exit", "local_only_metric"])
    except Exception:
        warnings.append("branch_exit local_only_metric is missing or non-numeric.")

    if df["crossing_rate"].isna().any():
        warnings.append("Some rows have null crossing_rate values.")

    if df["local_only_metric"].isna().any():
        warnings.append("Some rows have null local_only_metric values.")

    return warnings


def write_validation_report(path: Path, df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("gateway_comparison validation report")
    lines.append(f"source_version: {SOURCE_VERSION}")
    lines.append(f"n_rows: {len(df)}")

    metric_names = sorted(set(df["metric_name"].dropna().astype(str)))
    lines.append(f"metric_name: {', '.join(metric_names) if metric_names else 'unknown'}")

    lines.append(f"route_classes: {', '.join(df['route_class'].astype(str).tolist())}")
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

    metrics_path = (
        repo_root
        / "outputs"
        / "obs038_family_specific_gateway_laws"
        / "family_specific_gateway_metrics.csv"
    )
    summary_path = (
        repo_root
        / "outputs"
        / "obs038_family_specific_gateway_laws"
        / "family_specific_gateway_summary.csv"
    )

    out_csv = repo_root / "outputs" / "canonical" / "gateway_comparison.csv"
    out_report = (
        repo_root
        / "outputs"
        / "canonical"
        / "validation"
        / "gateway_comparison_validation.txt"
    )

    try:
        metrics_raw = load_csv(metrics_path)
        summary_raw = load_csv(summary_path)

        metrics = normalize_metrics(metrics_raw, metrics_path, repo_root)
        summary = normalize_summary(summary_raw, summary_path, repo_root)

        gateway_df = build_gateway_comparison(metrics, summary)
        warnings = validate_gateway_comparison(gateway_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        gateway_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, gateway_df, warnings)

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
