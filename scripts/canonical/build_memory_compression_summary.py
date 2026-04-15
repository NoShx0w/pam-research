#!/usr/bin/env python3
"""
Build the canonical family-level memory compression summary table.

This script consolidates family-level compression outputs from OBS-041 into a
single canonical artifact:

    outputs/canonical/memory_compression_summary.csv

It also writes a small human-readable validation report:

    outputs/canonical/validation/memory_compression_summary_validation.txt

Current scope
-------------
Version 1 is intentionally conservative:
- uses OBS-041 as the primary source
- preserves raw dominant compression state
- derives a cleaned structural compression state where possible
- flags likely padding contamination
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

STRUCTURAL_STATES = {"core", "escape"}
NON_STRUCTURAL_STATES = {"NONE", "nan", "", None}

SOURCE_VERSION = "memory_compression_summary_v1"


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


def normalize_memory_summary(df: pd.DataFrame, summary_path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        required=[
            "route_class",
            "forgetting_share",
            "mean_gain_over_suffix",
            "top_middle_state",
            "top_middle_state_count",
        ],
        name=str(summary_path),
    )

    out = df.copy().rename(
        columns={
            "top_middle_state": "raw_dominant_compression_state",
            "top_middle_state_count": "raw_dominant_compression_state_support",
        }
    )

    keep_cols = [
        "route_class",
        "forgetting_share",
        "mean_gain_over_suffix",
        "raw_dominant_compression_state",
        "raw_dominant_compression_state_support",
    ]
    out = out[keep_cols].copy()
    out["source_summary_table"] = repo_relative(summary_path, repo_root)
    return out


def normalize_candidate_states(df: pd.DataFrame, candidates_path: Path, repo_root: Path) -> pd.DataFrame:
    require_columns(
        df,
        required=[
            "route_class",
            "middle_state",
            "n_candidate_words",
            "total_count",
        ],
        name=str(candidates_path),
    )

    out = df.copy()
    out["middle_state"] = out["middle_state"].astype(str)
    out = out[out["middle_state"].isin(STRUCTURAL_STATES)].copy()

    if out.empty:
        return pd.DataFrame(
            columns=[
                "route_class",
                "_candidate_dominant_compression_state",
                "_candidate_dominant_compression_state_support",
                "secondary_compression_state",
                "secondary_compression_state_support",
                "source_candidates_table",
            ]
        )

    grouped = (
        out.groupby(["route_class", "middle_state"], dropna=False)
        .agg(
            support_n_candidate_words=("n_candidate_words", "sum"),
            support_total_count=("total_count", "sum"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        ["route_class", "support_total_count", "support_n_candidate_words", "middle_state"],
        ascending=[True, False, False, True],
    )

    rows = []
    for route_class, sub in grouped.groupby("route_class", dropna=False):
        top = sub.iloc[0]
        second = sub.iloc[1] if len(sub) > 1 else None
        rows.append(
            {
                "route_class": route_class,
                "_candidate_dominant_compression_state": top["middle_state"],
                "_candidate_dominant_compression_state_support": int(top["support_total_count"]),
                "secondary_compression_state": None if second is None else second["middle_state"],
                "secondary_compression_state_support": None
                if second is None
                else int(second["support_total_count"]),
                "source_candidates_table": repo_relative(candidates_path, repo_root),
            }
        )

    return pd.DataFrame(rows)


def resolve_structural_state(summary_df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        summary_df,
        candidates_df,
        on="route_class",
        how="left",
        validate="one_to_one",
    )

    def is_structural(value: object) -> bool:
        if pd.isna(value):
            return False
        return str(value) in STRUCTURAL_STATES

    def is_non_structural(value: object) -> bool:
        if pd.isna(value):
            return True
        return str(value) in NON_STRUCTURAL_STATES

    cleaned_states = []
    cleaned_supports = []
    padding_flags = []

    for _, row in df.iterrows():
        raw_state = row.get("raw_dominant_compression_state")
        raw_support = row.get("raw_dominant_compression_state_support")
        fallback_state = row.get("_candidate_dominant_compression_state")
        fallback_support = row.get("_candidate_dominant_compression_state_support")

        if is_structural(raw_state):
            cleaned_states.append(str(raw_state))
            cleaned_supports.append(raw_support)
            padding_flags.append(False)
        elif is_non_structural(raw_state):
            if pd.notna(fallback_state):
                cleaned_states.append(fallback_state)
                cleaned_supports.append(fallback_support)
            else:
                cleaned_states.append(pd.NA)
                cleaned_supports.append(pd.NA)
            padding_flags.append(True)
        else:
            cleaned_states.append(pd.NA)
            cleaned_supports.append(pd.NA)
            padding_flags.append(True)

    df["dominant_compression_state"] = cleaned_states
    df["dominant_compression_state_support"] = cleaned_supports
    df["padding_contaminated_flag"] = padding_flags

    df = df.drop(
        columns=[
            "_candidate_dominant_compression_state",
            "_candidate_dominant_compression_state_support",
        ],
        errors="ignore",
    )

    return df


def derive_compression_regime_label(df: pd.DataFrame) -> pd.DataFrame:
    regime_map = {
        "branch_exit": "weak",
        "stable_seam_corridor": "rapid",
        "reorganization_heavy": "strong",
    }
    df = df.copy()
    df["compression_regime_label"] = df["route_class"].map(regime_map)
    return df


def build_memory_compression_summary(summary_df: pd.DataFrame, candidates_df: pd.DataFrame) -> pd.DataFrame:
    df = resolve_structural_state(summary_df, candidates_df)
    df = derive_compression_regime_label(df)

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
        "forgetting_share",
        "mean_gain_over_suffix",
        "raw_dominant_compression_state",
        "raw_dominant_compression_state_support",
        "dominant_compression_state",
        "dominant_compression_state_support",
        "secondary_compression_state",
        "secondary_compression_state_support",
        "compression_regime_label",
        "padding_contaminated_flag",
        "source_summary_table",
        "source_candidates_table",
        "source_version",
        "provisional_flag",
    ]
    actual_order = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    return df[actual_order]


def validate_memory_compression_summary(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    required_cols = [
        "route_class",
        "forgetting_share",
        "mean_gain_over_suffix",
        "raw_dominant_compression_state",
        "dominant_compression_state",
        "compression_regime_label",
        "padding_contaminated_flag",
        "source_version",
        "provisional_flag",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"memory_compression_summary is missing required columns: {missing}")

    if df["route_class"].isna().any():
        warnings.append("Some rows have null route_class values.")

    if df["route_class"].duplicated().any():
        raise ValueError("Duplicate route_class rows found in memory_compression_summary.")

    observed_route_classes = [x for x in df["route_class"].astype(str).tolist()]
    missing_classes = [c for c in CANONICAL_ROUTE_CLASSES if c not in observed_route_classes]
    extra_classes = [c for c in observed_route_classes if c not in CANONICAL_ROUTE_CLASSES]

    if missing_classes:
        warnings.append(f"Missing canonical route classes: {missing_classes}")
    if extra_classes:
        warnings.append(f"Unexpected non-canonical route classes present: {extra_classes}")

    by_class = df.set_index("route_class")

    try:
        be = float(by_class.loc["branch_exit", "forgetting_share"])
        co = float(by_class.loc["stable_seam_corridor", "forgetting_share"])
        rh = float(by_class.loc["reorganization_heavy", "forgetting_share"])
        if not (be < co < rh):
            warnings.append(
                "Forgetting-share ordering is not strictly branch_exit < stable_seam_corridor < reorganization_heavy."
            )
    except Exception:
        warnings.append("Could not verify forgetting_share ordering.")

    for _, row in df.iterrows():
        raw_state = row["raw_dominant_compression_state"]
        resolved_state = row["dominant_compression_state"]
        padding_flag = bool(row["padding_contaminated_flag"])

        if str(raw_state) == "NONE" and not padding_flag:
            warnings.append(
                f"{row['route_class']}: raw_dominant_compression_state is NONE but padding_contaminated_flag is False."
            )

        if pd.notna(resolved_state) and str(resolved_state) not in STRUCTURAL_STATES:
            warnings.append(
                f"{row['route_class']}: dominant_compression_state is non-structural ({resolved_state})."
            )

    return warnings


def write_validation_report(path: Path, df: pd.DataFrame, warnings: List[str]) -> None:
    lines: list[str] = []
    lines.append("memory_compression_summary validation report")
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

    summary_path = (
        repo_root
        / "outputs"
        / "obs041_forgetting_nodes_and_memory_compression"
        / "memory_compression_summary.csv"
    )
    candidates_path = (
        repo_root
        / "outputs"
        / "obs041_forgetting_nodes_and_memory_compression"
        / "forgetting_node_candidates.csv"
    )

    out_csv = repo_root / "outputs" / "canonical" / "memory_compression_summary.csv"
    out_report = (
        repo_root
        / "outputs"
        / "canonical"
        / "validation"
        / "memory_compression_summary_validation.txt"
    )

    try:
        summary_raw = load_csv(summary_path)
        candidates_raw = load_csv(candidates_path)

        summary = normalize_memory_summary(summary_raw, summary_path, repo_root)
        candidates = normalize_candidate_states(candidates_raw, candidates_path, repo_root)

        compression_df = build_memory_compression_summary(summary, candidates)
        warnings = validate_memory_compression_summary(compression_df)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        compression_df.to_csv(out_csv, index=False)
        write_validation_report(out_report, compression_df, warnings)

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
