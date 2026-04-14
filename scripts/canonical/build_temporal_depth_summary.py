#!/usr/bin/env python3
"""
Build the canonical family-level temporal depth summary table.

This script consolidates family-level temporal depth outputs from OBS-040,
OBS-040b, and OBS-042 into a single canonical artifact:

    outputs/canonical/temporal_depth_summary.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/temporal_depth_summary_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- uses OBS-040 for raw horizon metrics
- uses OBS-042 for canonical regime interpretation
- uses OBS-040b only for light stress-test support metadata
- does not attempt to expose full encoding-level stress-test detail yet

Expected inputs
---------------
- outputs/obs040_variable_horizon_gateway_models/family_horizon_summary.csv
- outputs/obs040b_memory_scale_stress_test/stress_test_metrics.csv
- outputs/obs042_family_temporal_regimes_synthesis/family_temporal_regimes_summary.csv
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

SOURCE_VERSION = "temporal_depth_summary_v1"


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


def normalize_horizon_summary(
    df: pd.DataFrame, horizon_path: Path, repo_root: Path
) -> pd.DataFrame:
    require_columns(
        df,
        required=["route_class", "best_k", "best_auc", "k0_auc", "delta_best_vs_k0", "saturation_k"],
        name=str(horizon_path),
    )

    out = df.copy().rename(
        columns={
            "best_k": "raw_best_k",
            "best_auc": "raw_best_auc",
        }
    )

    keep_cols = [
        "route_class",
        "raw_best_k",
        "raw_best_auc",
        "k0_auc",
        "delta_best_vs_k0",
        "saturation_k",
    ]
    out = out[keep_cols].copy()

    out["source_horizon_table"] = repo_relative(horizon_path, repo_root)
    return out


def normalize_synthesis_summary(
    df: pd.DataFrame, synthesis_path: Path, repo_root: Path
) -> pd.DataFrame:
    require_columns(
        df,
        required=[
            "route_class",
            "canonical_regime",
            "predictive_locus",
            "best_horizon_k",
            "best_horizon_auc",
            "memory_interpretation",
        ],
        name=str(synthesis_path),
    )

    out = df.copy().rename(
        columns={
            "canonical_regime": "memory_regime_label",
            "memory_interpretation": "memory_interpretation",
            "canonical_interpretation": "canonical_interpretation",
        }
    )

    keep_cols = [
        "route_class",
        "memory_regime_label",
        "predictive_locus",
        "best_horizon_k",
        "best_horizon_auc",
        "memory_interpretation",
        "canonical_interpretation",
    ]
    out = out[keep_cols].copy()

    out["source_synthesis_table"] = repo_relative(synthesis_path, repo_root)
    return out


def derive_stress_support(
    df: pd.DataFrame, stress_path: Path, repo_root: Path
) -> pd.DataFrame:
    require_columns(
        df,
        required=["route_class"],
        name=str(stress_path),
    )

    grouped = (
        df.groupby("route_class", dropna=False)
        .agg(
            n_stress_rows=("route_class", "size"),
            n_encodings=("encoding", "nunique") if "encoding" in df.columns else ("route_class", "size"),
        )
        .reset_index()
    )

    grouped["stress_test_supported"] = True
    grouped["stress_test_note"] = grouped.apply(
        lambda row: f"stress-test rows={int(row['n_stress_rows'])}; encodings={int(row['n_encodings'])}",
        axis=1,
    )
    grouped["source_stress_table"] = repo_relative(stress_path, repo_root)

    keep_cols = [
        "route_class",
        "stress_test_supported",
        "stress_test_note",
        "source_stress_table",
    ]
    return grouped[keep_cols].copy()


def build_temporal_depth_summary(
    horizon_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    synthesis_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.merge(
        horizon_df,
        synthesis_df,
        on="route_class",
        how="outer",
        validate="one_to_one",
    )
    df = pd.merge(
        df,
        stress_df,
        on="route_class",
        how="left",
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
        "best_horizon_k",
        "best_horizon_auc",
        "raw_best_k",
        "raw_best_auc",
        "k0_auc",
        "delta_best_vs_k0",
        "saturation_k",
        "memory_regime_label",
        "predictive_locus",
        "memory_interpretation",
        "canonical_interpretation",
        "stress_test_supported",
        "stress_test_note",
        "source_horizon_table",
        "source_stress_table",
        "source_synthesis_table",
        "source_version",
        "provisional_flag",
    ]
    actual_order = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    return df[actual_order]


def validate_temporal_depth_summary(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    required_cols = [
        "route_class",
        "best_horizon_k",
        "best_horizon_auc",
        "k0_auc",
        "delta_best_vs_k0",
        "saturation_k",
        "memory_regime_label",
        "predictive_locus",
        "source_version",
        "provisional_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"temporal_depth_summary is missing required columns: {missing}")

    if df["route_class"].isna().any():
        warnings.append("Some rows have null route_class values.")

    if df["route_class"].duplicated().any():
        raise ValueError("Duplicate route_class rows found in temporal_depth_summary.")

    observed_route_classes = [x for x in df["route_class"].astype(str).tolist()]
    missing_classes = [c for c in CANONICAL_ROUTE_CLASSES if c not in observed_route_classes]
    extra_classes = [c for c in observed_route_classes if c not in CANONICAL_ROUTE_CLASSES]

    if missing_classes:
        warnings.append(f"Missing canonical route classes: {missing_classes}")
    if extra_classes:
        warnings.append(f"Unexpected non-canonical route classes present: {extra_classes}")

    by_class = df.set_index("route_class")

    try:
        if str(by_class.loc["branch_exit", "memory_regime_label"]) != "directed/downstream":
            warnings.append(
                "branch_exit memory_regime_label is not 'directed/downstream' as expected from OBS-042."
            )
    except Exception:
        warnings.append("Could not verify branch_exit memory_regime_label.")

    try:
        corridor_label = str(by_class.loc["stable_seam_corridor", "memory_regime_label"])
        if corridor_label != "local gateway":
            warnings.append(
                f"stable_seam_corridor memory_regime_label is '{corridor_label}', expected 'local gateway'."
            )
    except Exception:
        warnings.append("Could not verify stable_seam_corridor memory_regime_label.")

    try:
        reorg_label = str(by_class.loc["reorganization_heavy", "memory_regime_label"])
        if reorg_label != "path-context":
            warnings.append(
                f"reorganization_heavy memory_regime_label is '{reorg_label}', expected 'path-context'."
            )
    except Exception:
        warnings.append("Could not verify reorganization_heavy memory_regime_label.")

    # Light numeric sanity.
    for _, row in df.iterrows():
        rc = row["route_class"]
        try:
            if pd.notna(row["best_horizon_auc"]) and pd.notna(row["k0_auc"]):
                if float(row["best_horizon_auc"]) < float(row["k0_auc"]):
                    warnings.append(
                        f"{rc}: best_horizon_auc is lower than k0_auc."
                    )
        except Exception:
            warnings.append(f"{rc}: could not compare best_horizon_auc and k0_auc.")

        try:
            if pd.notna(row["saturation_k"]) and pd.notna(row["raw_best_k"]):
                if float(row["saturation_k"]) > float(row["raw_best_k"]):
                    warnings.append(
                        f"{rc}: saturation_k is greater than raw_best_k."
                    )
        except Exception:
            warnings.append(f"{rc}: could not compare saturation_k and raw_best_k.")

    return warnings


def write_validation_report(path: Path, df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("temporal_depth_summary validation report")
    lines.append(f"source_version: {SOURCE_VERSION}")
    lines.append(f"n_rows: {len(df)}")
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

    horizon_path = (
        repo_root
        / "outputs"
        / "obs040_variable_horizon_gateway_models"
        / "family_horizon_summary.csv"
    )
    stress_path = (
        repo_root
        / "outputs"
        / "obs040b_memory_scale_stress_test"
        / "stress_test_metrics.csv"
    )
    synthesis_path = (
        repo_root
        / "outputs"
        / "obs042_family_temporal_regimes_synthesis"
        / "family_temporal_regimes_summary.csv"
    )

    out_csv = repo_root / "outputs" / "canonical" / "temporal_depth_summary.csv"
    out_report = (
        repo_root
        / "outputs"
        / "canonical"
        / "validation"
        / "temporal_depth_summary_validation.txt"
    )

    try:
        horizon_raw = load_csv(horizon_path)
        stress_raw = load_csv(stress_path)
        synthesis_raw = load_csv(synthesis_path)

        horizon = normalize_horizon_summary(horizon_raw, horizon_path, repo_root)
        stress = derive_stress_support(stress_raw, stress_path, repo_root)
        synthesis = normalize_synthesis_summary(synthesis_raw, synthesis_path, repo_root)

        temporal_df = build_temporal_depth_summary(horizon, stress, synthesis)
        warnings = validate_temporal_depth_summary(temporal_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        temporal_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, temporal_df, warnings)

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
