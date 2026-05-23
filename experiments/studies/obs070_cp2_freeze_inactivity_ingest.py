#!/usr/bin/env python3
"""
OBS-070 — Cp2 freeze-inactivity downstream ingest preparation.

Purpose
-------
Prepare the completed Cp2 full_v2 campaign for downstream PAM analyses while
preserving the scientific meaning of freeze-derived NaNs.

OBS-070 established that Cp2 full_v2 completed cleanly, but the freeze
macrostate was inactive across the full grid:

    piF_mean = 0.0
    piF_tail = 0.0
    corr0 = NaN
    best_corr = NaN
    delta_r2_freeze = NaN

Those NaNs are not pipeline failures. They are mathematically undefined
freeze-coupling metrics caused by absence of freeze-state variation.

This script creates a downstream-safe index with explicit status/defined flags
and finite companion columns for algorithms that require numeric arrays.

Guardrail
---------
Do not interpret *_value = 0.0 for undefined freeze rows as measured zero
correlation or measured zero predictive power. The authoritative interpretation
is carried by freeze_metric_status and *_defined columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FREEZE_METRIC_COLS = [
    "corr0",
    "best_corr",
    "delta_r2_freeze",
]

ENTROPY_METRIC_COLS = [
    "H_joint_mean",
    "var_H_joint",
    "delta_r2_entropy",
]

REQUIRED_COLS = [
    "filename",
    "corpus",
    "alpha",
    "r",
    "iters",
    "W",
    "seed",
    "piF_mean",
    "piF_tail",
    "H_joint_mean",
    "var_H_joint",
    "corr0",
    "delta_r2_freeze",
    "delta_r2_entropy",
    "best_lag",
    "best_corr",
]


def finite_float(value) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def all_nan(row: pd.Series, cols: Iterable[str]) -> bool:
    return all(pd.isna(row.get(c)) for c in cols)


def any_nan(row: pd.Series, cols: Iterable[str]) -> bool:
    return any(pd.isna(row.get(c)) for c in cols)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input index is missing required columns: "
            + ", ".join(missing)
        )


def classify_freeze_metric_status(row: pd.Series) -> str:
    """
    Classify the reason freeze-derived metrics are or are not usable.

    Status meanings:
    - ok:
        Freeze channel is active and freeze-derived metrics are finite.
    - undefined_no_freeze_variation:
        Freeze channel is inactive; freeze-derived metrics are undefined by
        measurement geometry, not by pipeline failure.
    - undefined_constant_freeze:
        Freeze appears constantly active. This is also a no-variation case,
        but not expected for Cp2 full_v2.
    - partial_freeze_metric_nan:
        Some freeze metrics are missing while others are present.
    - missing_pipeline_or_schema_error:
        Freeze activity fields themselves are missing/invalid.
    """
    piF_mean = finite_float(row.get("piF_mean"))
    piF_tail = finite_float(row.get("piF_tail"))

    freeze_metrics_all_nan = all_nan(row, FREEZE_METRIC_COLS)
    freeze_metrics_any_nan = any_nan(row, FREEZE_METRIC_COLS)

    if not np.isfinite(piF_mean) or not np.isfinite(piF_tail):
        return "missing_pipeline_or_schema_error"

    if piF_mean == 0.0 and piF_tail == 0.0 and freeze_metrics_all_nan:
        return "undefined_no_freeze_variation"

    if piF_mean == 1.0 and piF_tail == 1.0 and freeze_metrics_all_nan:
        return "undefined_constant_freeze"

    if freeze_metrics_any_nan:
        return "partial_freeze_metric_nan"

    return "ok"


def add_downstream_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["freeze_metric_status"] = out.apply(classify_freeze_metric_status, axis=1)

    piF_mean = pd.to_numeric(out["piF_mean"], errors="coerce")
    piF_tail = pd.to_numeric(out["piF_tail"], errors="coerce")

    out["freeze_active"] = (piF_mean > 0.0).astype(int)
    out["freeze_tail_active"] = (piF_tail > 0.0).astype(int)

    out["freeze_variance_available"] = (
        out["freeze_metric_status"].eq("ok")
        | out["freeze_metric_status"].eq("partial_freeze_metric_nan")
    ).astype(int)

    for col in FREEZE_METRIC_COLS:
        defined_col = f"{col}_defined"
        value_col = f"{col}_value"

        out[defined_col] = out[col].notna().astype(int)

        # Numeric compatibility column. Interpretation must use *_defined and
        # freeze_metric_status.
        out[value_col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["freeze_correlation_defined"] = out["corr0_defined"]
    out["freeze_prediction_defined"] = out["delta_r2_freeze_defined"]

    # Helpful explicit reason fields for downstream joins/reports.
    out["freeze_nan_semantics"] = np.where(
        out["freeze_metric_status"].eq("undefined_no_freeze_variation"),
        "structural_undefined_not_failure",
        np.where(
            out["freeze_metric_status"].eq("ok"),
            "measured",
            "requires_review",
        ),
    )

    return out


def build_grid_summary(safe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        safe.groupby(["r", "alpha"], dropna=False)
        .agg(
            n=("seed", "count"),
            freeze_active_count=("freeze_active", "sum"),
            freeze_tail_active_count=("freeze_tail_active", "sum"),
            freeze_metric_status_nunique=("freeze_metric_status", "nunique"),
            undefined_no_freeze_variation=(
                "freeze_metric_status",
                lambda x: int((x == "undefined_no_freeze_variation").sum()),
            ),
            ok_freeze_metrics=(
                "freeze_metric_status",
                lambda x: int((x == "ok").sum()),
            ),
            partial_freeze_metric_nan=(
                "freeze_metric_status",
                lambda x: int((x == "partial_freeze_metric_nan").sum()),
            ),
            piF_mean=("piF_mean", "mean"),
            piF_tail=("piF_tail", "mean"),
            H_joint_mean=("H_joint_mean", "mean"),
            var_H_joint=("var_H_joint", "mean"),
            delta_r2_entropy=("delta_r2_entropy", "mean"),
            corr0_defined_count=("corr0_defined", "sum"),
            best_corr_defined_count=("best_corr_defined", "sum"),
            delta_r2_freeze_defined_count=("delta_r2_freeze_defined", "sum"),
        )
        .reset_index()
        .sort_values(["r", "alpha"])
    )
    return summary


def build_status_summary(safe: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for status, sub in safe.groupby("freeze_metric_status", dropna=False):
        rows.append(
            {
                "freeze_metric_status": status,
                "rows": len(sub),
                "mean_piF_mean": pd.to_numeric(
                    sub["piF_mean"], errors="coerce"
                ).mean(),
                "mean_piF_tail": pd.to_numeric(
                    sub["piF_tail"], errors="coerce"
                ).mean(),
                "mean_H_joint_mean": pd.to_numeric(
                    sub["H_joint_mean"], errors="coerce"
                ).mean(),
                "mean_var_H_joint": pd.to_numeric(
                    sub["var_H_joint"], errors="coerce"
                ).mean(),
                "corr0_defined_count": int(sub["corr0_defined"].sum()),
                "best_corr_defined_count": int(sub["best_corr_defined"].sum()),
                "delta_r2_freeze_defined_count": int(
                    sub["delta_r2_freeze_defined"].sum()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("freeze_metric_status")


def write_report(
    *,
    index_path: Path,
    out_dir: Path,
    raw: pd.DataFrame,
    safe: pd.DataFrame,
    grid_summary: pd.DataFrame,
    status_summary: pd.DataFrame,
) -> None:
    report_path = out_dir / "cp2_downstream_ingest_report.md"

    status_counts = safe["freeze_metric_status"].value_counts(dropna=False)
    corpus_counts = safe["corpus"].value_counts(dropna=False)

    trajectory_dir = index_path.parent / "trajectories"
    trajectory_count = len(list(trajectory_dir.glob("*.npz"))) if trajectory_dir.exists() else None

    unique_jobs = (
        safe[["corpus", "r", "alpha", "iters", "W", "seed"]]
        .drop_duplicates()
        .shape[0]
    )

    lines: list[str] = [
        "# OBS-070 — Cp2 downstream ingestibility report",
        "",
        "## Purpose",
        "",
        "Prepare the completed Cp2 campaign for downstream PAM analyses while preserving the scientific meaning of freeze-derived NaNs.",
        "",
        "## Input",
        "",
        f"- Index: `{index_path}`",
        f"- Rows: `{len(raw)}`",
        f"- Unique jobs: `{unique_jobs}`",
        f"- Trajectory files: `{trajectory_count}`",
        "",
        "## Outputs",
        "",
        f"- Downstream-safe index: `{out_dir / 'index_downstream_safe.csv'}`",
        f"- Grid diagnostic: `{out_dir / 'cp2_freeze_inactivity_diagnostic.csv'}`",
        f"- Status summary: `{out_dir / 'cp2_freeze_metric_status_summary.csv'}`",
        f"- JSON metadata: `{out_dir / 'cp2_downstream_ingest_metadata.json'}`",
        "",
        "## Corpus rows",
        "",
    ]

    for corpus, count in corpus_counts.items():
        lines.append(f"- `{corpus}`: `{int(count)}`")

    lines.extend(
        [
            "",
            "## Freeze metric status",
            "",
        ]
    )

    for status, count in status_counts.items():
        lines.append(f"- `{status}`: `{int(count)}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows with `freeze_metric_status = undefined_no_freeze_variation` are not treated as pipeline failures.",
            "",
            "Their freeze-correlation and freeze-prediction metrics are mathematically undefined because the freeze channel has no variation under the current macrostate definition.",
            "",
            "Entropy-derived summaries remain finite and should be used for Cp2 comparisons alongside transition, response, route, scale-space, or revised macrostate diagnostics.",
            "",
            "## Downstream guardrail",
            "",
            "The `_value` companion columns replace undefined freeze metrics with `0.0` only to support finite numeric ingestion by downstream scripts.",
            "",
            "Interpretive code must use the corresponding `_defined` flags and `freeze_metric_status`.",
            "",
            "Do **not** interpret `corr0_value = 0.0`, `best_corr_value = 0.0`, or `delta_r2_freeze_value = 0.0` on an undefined row as a measured zero effect.",
            "",
            "## Status summary",
            "",
            status_summary.to_markdown(index=False),
            "",
            "## By-grid summary",
            "",
            grid_summary.to_markdown(index=False),
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    *,
    index_path: Path,
    out_dir: Path,
    safe: pd.DataFrame,
    grid_summary: pd.DataFrame,
    status_summary: pd.DataFrame,
) -> None:
    meta = {
        "obs_id": "OBS-070",
        "purpose": "Cp2 freeze-inactivity downstream ingest preparation",
        "input_index_csv": str(index_path),
        "outputs": {
            "index_downstream_safe": str(out_dir / "index_downstream_safe.csv"),
            "freeze_inactivity_diagnostic": str(
                out_dir / "cp2_freeze_inactivity_diagnostic.csv"
            ),
            "freeze_metric_status_summary": str(
                out_dir / "cp2_freeze_metric_status_summary.csv"
            ),
            "report": str(out_dir / "cp2_downstream_ingest_report.md"),
        },
        "rows": int(len(safe)),
        "grid_rows": int(len(grid_summary)),
        "status_rows": int(len(status_summary)),
        "freeze_metric_status_counts": {
            str(k): int(v)
            for k, v in safe["freeze_metric_status"]
            .value_counts(dropna=False)
            .to_dict()
            .items()
        },
        "guardrail": (
            "Undefined freeze metrics are structural undefined values caused by "
            "absence of freeze-state variation. Numeric *_value columns are "
            "compatibility fields and must not be interpreted without *_defined "
            "flags and freeze_metric_status."
        ),
    }

    (out_dir / "cp2_downstream_ingest_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index-csv",
        default="outputs/corpora/Cp2/campaigns/full_v2/index.csv",
        help="Campaign index CSV to prepare for downstream ingestion.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Defaults to <campaign root>/downstream "
            "beside the input index.csv."
        ),
    )
    ap.add_argument(
        "--allow-non-cp2",
        action="store_true",
        help="Allow processing indexes whose corpus column is not exclusively Cp2.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.index_csv)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    out_dir = Path(args.out_dir) if args.out_dir else index_path.parent / "downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(index_path)
    validate_required_columns(raw)

    corpus_values = set(raw["corpus"].dropna().astype(str).unique())
    if not args.allow_non_cp2 and corpus_values != {"Cp2"}:
        raise ValueError(
            "This OBS-070 ingest script is scoped to Cp2 by default. "
            f"Observed corpus values: {sorted(corpus_values)}. "
            "Pass --allow-non-cp2 only if you intentionally want generic use."
        )

    safe = add_downstream_columns(raw)
    grid_summary = build_grid_summary(safe)
    status_summary = build_status_summary(safe)

    safe_path = out_dir / "index_downstream_safe.csv"
    diag_path = out_dir / "cp2_freeze_inactivity_diagnostic.csv"
    status_path = out_dir / "cp2_freeze_metric_status_summary.csv"

    safe.to_csv(safe_path, index=False)
    grid_summary.to_csv(diag_path, index=False)
    status_summary.to_csv(status_path, index=False)

    write_report(
        index_path=index_path,
        out_dir=out_dir,
        raw=raw,
        safe=safe,
        grid_summary=grid_summary,
        status_summary=status_summary,
    )
    write_metadata(
        index_path=index_path,
        out_dir=out_dir,
        safe=safe,
        grid_summary=grid_summary,
        status_summary=status_summary,
    )

    print("wrote", safe_path)
    print("wrote", diag_path)
    print("wrote", status_path)
    print("wrote", out_dir / "cp2_downstream_ingest_report.md")
    print("wrote", out_dir / "cp2_downstream_ingest_metadata.json")
    print()
    print("freeze_metric_status:")
    print(safe["freeze_metric_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
