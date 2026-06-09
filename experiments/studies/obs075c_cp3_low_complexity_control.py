#!/usr/bin/env python3
"""
OBS-075c — Cp3 low-complexity model control

Purpose
-------
OBS-075c compares OBS-075b-style endpoint/velocity/tortuosity ablation
results across model families / capacities.

It asks whether Cp3 directional asymmetry survives when the model boundary is
made simpler:

    - full Random Forest
    - shallow Random Forest, max_depth=2
    - shallow Random Forest, max_depth=4
    - logistic regression

This script is artifact-first. It consumes already-produced OBS-075b
model-score artifacts and does not recompute geometry, path families,
coupling, outcomes, recovery channels, or model predictions.

Recommended prior runs
----------------------
Example OBS-075b runs to feed this script:

    PYTHONPATH=src .venv/bin/python experiments/studies/obs075b_cp3_endpoint_velocity_ablation.py \\
      --model-family rf \\
      --outdir outputs/comparisons/obs075b_rf_full

    PYTHONPATH=src .venv/bin/python experiments/studies/obs075b_cp3_endpoint_velocity_ablation.py \\
      --model-family rf --max-depth 2 \\
      --outdir outputs/comparisons/obs075b_rf_depth2

    PYTHONPATH=src .venv/bin/python experiments/studies/obs075b_cp3_endpoint_velocity_ablation.py \\
      --model-family rf --max-depth 4 \\
      --outdir outputs/comparisons/obs075b_rf_depth4

    PYTHONPATH=src .venv/bin/python experiments/studies/obs075b_cp3_endpoint_velocity_ablation.py \\
      --model-family logreg \\
      --outdir outputs/comparisons/obs075b_logreg

Then:

    PYTHONPATH=src .venv/bin/python experiments/studies/obs075c_cp3_low_complexity_control.py \\
      --run rf_full=outputs/comparisons/obs075b_rf_full \\
      --run rf_depth2=outputs/comparisons/obs075b_rf_depth2 \\
      --run rf_depth4=outputs/comparisons/obs075b_rf_depth4 \\
      --run logreg=outputs/comparisons/obs075b_logreg \\
      --outdir outputs/comparisons/obs075c_cp3_low_complexity_control
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_RUNS = [
    "rf_full=outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_v2_smoke",
]

STRICT_FEATURE_SETS = {
    "no_direct_seam_no_grid_no_endpoint_velocity",
    "no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity",
    "holonomy_criticality_shape_only",
}

FOCUS_TARGETS = {
    "coupled_outcome_group_no_direct_seam_no_grid",
    "recovery_channel_no_direct_seam_no_grid",
    "recovery_channel_boundedness_strict",
    "coupling_class_no_direct_seam_no_grid",
    "outcome_group_no_direct_seam_no_grid",
    "path_family_no_direct_seam_no_grid",
}

PRIMARY_TARGETS = [
    "coupled_outcome_group_no_direct_seam_no_grid",
    "recovery_channel_no_direct_seam_no_grid",
    "recovery_channel_boundedness_strict",
    "coupling_class_no_direct_seam_no_grid",
    "outcome_group_no_direct_seam_no_grid",
    "path_family_no_direct_seam_no_grid",
]

PRIMARY_FEATURE_SETS = [
    "no_direct_seam_no_grid",
    "no_direct_seam_no_grid_no_endpoint_velocity",
    "no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity",
    "holonomy_criticality_shape_only",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path
    scores_csv: Path
    specificity_csv: Path | None
    summary_md: Path | None


def parse_run_spec(raw: str) -> RunSpec:
    if "=" not in raw:
        raise ValueError(
            f"--run must have form label=outdir, got: {raw!r}"
        )

    label, root_s = raw.split("=", 1)
    label = label.strip()
    root = Path(root_s.strip())

    if not label:
        raise ValueError(f"Empty run label in --run {raw!r}")

    scores_csv = root / "obs075b_model_scores.csv"
    specificity_csv = root / "obs075b_asymmetry_specificity.csv"
    summary_md = root / "obs075b_summary.md"

    return RunSpec(
        label=label,
        root=root,
        scores_csv=scores_csv,
        specificity_csv=specificity_csv if specificity_csv.exists() else None,
        summary_md=summary_md if summary_md.exists() else None,
    )


def safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except TypeError:
        pass
    try:
        return float(x)
    except Exception:
        return None


def format_float(x, digits: int = 4) -> str:
    val = safe_float(x)
    if val is None or not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def read_scores(spec: RunSpec) -> pd.DataFrame:
    if not spec.scores_csv.exists():
        raise FileNotFoundError(f"Missing model scores for {spec.label}: {spec.scores_csv}")

    df = pd.read_csv(spec.scores_csv)
    df["run_label"] = spec.label
    df["run_root"] = str(spec.root)

    required = {
        "pair",
        "target",
        "feature_set",
        "train_corpus",
        "test_corpus",
        "status",
        "balanced_accuracy",
        "macro_f1",
        "n_train",
        "n_test",
        "feature_count",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{spec.scores_csv} missing required columns: {missing}")

    if "model_family" not in df.columns:
        df["model_family"] = "unknown"
    if "max_depth" not in df.columns:
        df["max_depth"] = pd.NA
    if "status_detail" not in df.columns:
        df["status_detail"] = df["status"].astype(str)

    return df


def status_ok(status: str) -> bool:
    return str(status) in {"ok", "ok_warn_small_class"}


def get_directional_row(
    df: pd.DataFrame,
    *,
    pair: str,
    target: str,
    feature_set: str,
    train_corpus: str,
    test_corpus: str,
) -> pd.Series | None:
    sub = df[
        (df["pair"].astype(str) == pair)
        & (df["target"].astype(str) == target)
        & (df["feature_set"].astype(str) == feature_set)
        & (df["train_corpus"].astype(str) == train_corpus)
        & (df["test_corpus"].astype(str) == test_corpus)
    ]

    if sub.empty:
        return None

    # In normal OBS-075b output there should be exactly one row.
    return sub.iloc[0]


def metric_from_row(row: pd.Series | None, col: str) -> float | None:
    if row is None:
        return None
    if col not in row.index:
        return None
    return safe_float(row[col])


def status_from_row(row: pd.Series | None) -> str:
    if row is None:
        return "missing"
    return str(row.get("status", "missing"))


def build_pair_asymmetry(
    df: pd.DataFrame,
    *,
    run_label: str,
    pair: str,
    baseline: str,
    cp3: str = "Cp3",
) -> pd.DataFrame:
    rows: list[dict] = []

    targets = sorted(df.loc[df["pair"].astype(str) == pair, "target"].dropna().astype(str).unique())
    feature_sets = sorted(df.loc[df["pair"].astype(str) == pair, "feature_set"].dropna().astype(str).unique())

    for target in targets:
        for feature_set in feature_sets:
            a = get_directional_row(
                df,
                pair=pair,
                target=target,
                feature_set=feature_set,
                train_corpus=baseline,
                test_corpus=cp3,
            )
            b = get_directional_row(
                df,
                pair=pair,
                target=target,
                feature_set=feature_set,
                train_corpus=cp3,
                test_corpus=baseline,
            )

            ba_a = metric_from_row(a, "balanced_accuracy")
            ba_b = metric_from_row(b, "balanced_accuracy")
            f1_a = metric_from_row(a, "macro_f1")
            f1_b = metric_from_row(b, "macro_f1")

            asym = None
            if ba_a is not None and ba_b is not None:
                asym = ba_b - ba_a

            status_a = status_from_row(a)
            status_b = status_from_row(b)
            asym_status = "ok" if status_ok(status_a) and status_ok(status_b) else "partial_or_unavailable"

            model_family = None
            max_depth = None
            feature_count = None
            n_train_a = None
            n_test_a = None
            n_train_b = None
            n_test_b = None

            for row in [a, b]:
                if row is None:
                    continue
                if model_family is None:
                    model_family = row.get("model_family", pd.NA)
                if max_depth is None:
                    max_depth = row.get("max_depth", pd.NA)
                if feature_count is None:
                    feature_count = row.get("feature_count", pd.NA)

            if a is not None:
                n_train_a = a.get("n_train", pd.NA)
                n_test_a = a.get("n_test", pd.NA)
            if b is not None:
                n_train_b = b.get("n_train", pd.NA)
                n_test_b = b.get("n_test", pd.NA)

            rows.append(
                {
                    "run_label": run_label,
                    "model_family": model_family,
                    "max_depth": max_depth,
                    "pair": pair,
                    "baseline_corpus": baseline,
                    "cp3_corpus": cp3,
                    "target": target,
                    "feature_set": feature_set,
                    f"ba_{baseline.lower()}_to_cp3": ba_a,
                    f"ba_cp3_to_{baseline.lower()}": ba_b,
                    f"macro_f1_{baseline.lower()}_to_cp3": f1_a,
                    f"macro_f1_cp3_to_{baseline.lower()}": f1_b,
                    f"asymmetry_cp3_minus_{baseline.lower()}": asym,
                    f"status_{baseline.lower()}_to_cp3": status_a,
                    f"status_cp3_to_{baseline.lower()}": status_b,
                    "asymmetry_status": asym_status,
                    "feature_count": feature_count,
                    f"n_train_{baseline.lower()}_to_cp3": n_train_a,
                    f"n_test_{baseline.lower()}_to_cp3": n_test_a,
                    f"n_train_cp3_to_{baseline.lower()}": n_train_b,
                    f"n_test_cp3_to_{baseline.lower()}": n_test_b,
                    "is_strict_feature_set": feature_set in STRICT_FEATURE_SETS,
                    "is_focus_target": target in FOCUS_TARGETS,
                }
            )

    return pd.DataFrame(rows)


def build_specificity(
    cp2_asym: pd.DataFrame,
    cp_asym: pd.DataFrame,
    c_asym: pd.DataFrame,
) -> pd.DataFrame:
    key = ["run_label", "target", "feature_set"]

    cp2_cols = [
        "run_label",
        "model_family",
        "max_depth",
        "target",
        "feature_set",
        "feature_count",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "macro_f1_cp2_to_cp3",
        "macro_f1_cp3_to_cp2",
        "asymmetry_cp3_minus_cp2",
        "status_cp2_to_cp3",
        "status_cp3_to_cp2",
        "asymmetry_status",
        "is_strict_feature_set",
        "is_focus_target",
    ]
    cp_cols = [
        "run_label",
        "target",
        "feature_set",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "macro_f1_cp_to_cp3",
        "macro_f1_cp3_to_cp",
        "asymmetry_cp3_minus_cp",
        "status_cp_to_cp3",
        "status_cp3_to_cp",
        "asymmetry_status",
    ]
    c_cols = [
        "run_label",
        "target",
        "feature_set",
        "ba_c_to_cp3",
        "ba_cp3_to_c",
        "macro_f1_c_to_cp3",
        "macro_f1_cp3_to_c",
        "asymmetry_cp3_minus_c",
        "status_c_to_cp3",
        "status_cp3_to_c",
        "asymmetry_status",
    ]

    out = cp2_asym[[c for c in cp2_cols if c in cp2_asym.columns]].copy()
    out = out.merge(
        cp_asym[[c for c in cp_cols if c in cp_asym.columns]].copy(),
        on=key,
        how="left",
        suffixes=("", "_cp_control"),
    )
    out = out.merge(
        c_asym[[c for c in c_cols if c in c_asym.columns]].copy(),
        on=key,
        how="left",
        suffixes=("", "_c_control"),
    )

    for col in [
        "asymmetry_cp3_minus_cp2",
        "asymmetry_cp3_minus_cp",
        "asymmetry_cp3_minus_c",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["specificity_vs_cp"] = out["asymmetry_cp3_minus_cp2"] - out["asymmetry_cp3_minus_cp"]
    out["specificity_vs_c"] = out["asymmetry_cp3_minus_cp2"] - out["asymmetry_cp3_minus_c"]

    out["abs_specificity_vs_cp"] = out["specificity_vs_cp"].abs()
    out["abs_specificity_vs_c"] = out["specificity_vs_c"].abs()

    def row_status(row: pd.Series) -> str:
        cp2_ok = status_ok(row.get("status_cp2_to_cp3")) and status_ok(row.get("status_cp3_to_cp2"))
        cp_ok = status_ok(row.get("status_cp_to_cp3")) and status_ok(row.get("status_cp3_to_cp"))
        c_ok = status_ok(row.get("status_c_to_cp3")) and status_ok(row.get("status_cp3_to_c"))

        if cp2_ok and cp_ok and c_ok:
            return "ok"
        if cp2_ok and cp_ok:
            return "ok_vs_cp_only"
        if cp2_ok:
            return "cp2_available_only"
        return "partial_or_unavailable"

    out["specificity_status"] = out.apply(row_status, axis=1)

    target_priority = {
        "coupled_outcome_group_no_direct_seam_no_grid": 1,
        "recovery_channel_no_direct_seam_no_grid": 2,
        "recovery_channel_boundedness_strict": 3,
        "coupling_class_no_direct_seam_no_grid": 4,
        "outcome_group_no_direct_seam_no_grid": 5,
        "path_family_no_direct_seam_no_grid": 6,
    }
    out["focus_priority"] = out["target"].map(target_priority).fillna(99).astype(int)

    return out


def classify_survival(row: pd.Series) -> str:
    """
    Conservative symbolic read for one row.

    This is intentionally simple and descriptive; the CSV preserves the actual
    numeric metrics.
    """
    status = str(row.get("specificity_status", ""))
    if not status.startswith("ok"):
        return "unavailable_or_partial"

    asym = safe_float(row.get("asymmetry_cp3_minus_cp2"))
    spec_cp = safe_float(row.get("specificity_vs_cp"))
    spec_c = safe_float(row.get("specificity_vs_c"))

    if asym is None:
        return "unavailable_or_partial"

    if asym <= 0.02:
        return "collapsed_or_near_zero"

    cp_positive = spec_cp is not None and spec_cp > 0.02
    c_positive = spec_c is not None and spec_c > 0.02

    if cp_positive and c_positive:
        return "survives_vs_cp_and_c"
    if cp_positive:
        return "survives_vs_cp_only"
    if c_positive:
        return "survives_vs_c_only"
    return "directional_without_specificity"


def build_survival_table(specificity: pd.DataFrame) -> pd.DataFrame:
    rows = specificity.copy()
    rows = rows[
        rows["target"].isin(PRIMARY_TARGETS)
        & rows["feature_set"].isin(PRIMARY_FEATURE_SETS)
    ].copy()

    rows["survival_read"] = rows.apply(classify_survival, axis=1)

    keep = [
        "run_label",
        "model_family",
        "max_depth",
        "target",
        "feature_set",
        "feature_count",
        "asymmetry_cp3_minus_cp2",
        "specificity_vs_cp",
        "specificity_vs_c",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "ba_c_to_cp3",
        "ba_cp3_to_c",
        "status_cp2_to_cp3",
        "status_cp3_to_cp2",
        "status_cp_to_cp3",
        "status_cp3_to_cp",
        "status_c_to_cp3",
        "status_cp3_to_c",
        "specificity_status",
        "survival_read",
        "focus_priority",
    ]
    rows = rows[[c for c in keep if c in rows.columns]].copy()

    rows = rows.sort_values(
        [
            "focus_priority",
            "feature_set",
            "run_label",
        ],
        kind="mergesort",
    )

    return rows


def build_model_capacity_matrix(survival: pd.DataFrame) -> pd.DataFrame:
    """
    Wide summary: one row per target/feature_set and one compact read per run.
    """
    if survival.empty:
        return pd.DataFrame()

    tmp = survival.copy()

    def compact(row: pd.Series) -> str:
        asym = format_float(row.get("asymmetry_cp3_minus_cp2"))
        sp = format_float(row.get("specificity_vs_cp"))
        sc = format_float(row.get("specificity_vs_c"))
        read = str(row.get("survival_read", ""))
        return f"{read}; asym={asym}; spec_cp={sp}; spec_c={sc}"

    tmp["compact_read"] = tmp.apply(compact, axis=1)

    wide = tmp.pivot_table(
        index=["target", "feature_set"],
        columns="run_label",
        values="compact_read",
        aggfunc="first",
    ).reset_index()

    wide.columns = [str(c) for c in wide.columns]
    wide["focus_priority"] = wide["target"].map(
        {
            "coupled_outcome_group_no_direct_seam_no_grid": 1,
            "recovery_channel_no_direct_seam_no_grid": 2,
            "recovery_channel_boundedness_strict": 3,
            "coupling_class_no_direct_seam_no_grid": 4,
            "outcome_group_no_direct_seam_no_grid": 5,
            "path_family_no_direct_seam_no_grid": 6,
        }
    ).fillna(99).astype(int)

    return wide.sort_values(["focus_priority", "feature_set"]).drop(columns=["focus_priority"])


def summarize_by_run(specificity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_label, g in specificity.groupby("run_label", dropna=False):
        strict = g[g["feature_set"].isin(STRICT_FEATURE_SETS)]
        focus = strict[strict["target"].isin(FOCUS_TARGETS)]

        def count_read(frame: pd.DataFrame, read: str) -> int:
            if frame.empty:
                return 0
            return int((frame.apply(classify_survival, axis=1) == read).sum())

        rows.append(
            {
                "run_label": run_label,
                "n_rows": len(g),
                "n_strict_rows": len(strict),
                "n_focus_strict_rows": len(focus),
                "n_focus_survives_vs_cp_and_c": count_read(focus, "survives_vs_cp_and_c"),
                "n_focus_survives_vs_cp_only": count_read(focus, "survives_vs_cp_only"),
                "n_focus_directional_without_specificity": count_read(focus, "directional_without_specificity"),
                "n_focus_collapsed_or_near_zero": count_read(focus, "collapsed_or_near_zero"),
                "n_focus_unavailable_or_partial": count_read(focus, "unavailable_or_partial"),
                "mean_focus_strict_asymmetry_cp3_minus_cp2": pd.to_numeric(
                    focus.get("asymmetry_cp3_minus_cp2", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean(),
                "mean_focus_strict_specificity_vs_cp": pd.to_numeric(
                    focus.get("specificity_vs_cp", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean(),
                "mean_focus_strict_specificity_vs_c": pd.to_numeric(
                    focus.get("specificity_vs_c", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("run_label")


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "No rows."

    view = df.copy()
    if columns is not None:
        view = view[[c for c in columns if c in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)

    def fmt(v) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "NA"
            return f"{v:.4f}"
        if pd.isna(v):
            return "NA"
        return str(v)

    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def write_summary(
    outdir: Path,
    *,
    specs: list[RunSpec],
    run_summary: pd.DataFrame,
    survival: pd.DataFrame,
    capacity_matrix: pd.DataFrame,
    specificity: pd.DataFrame,
) -> None:
    lines: list[str] = []

    lines.append("# OBS-075c — Cp3 low-complexity model control")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "OBS-075c compares already-produced OBS-075b endpoint/velocity ablation "
        "runs across model capacity. It does not recompute geometry, path families, "
        "coupling, outcomes, recovery channels, or model predictions."
    )
    lines.append("")
    lines.append("The question is whether Cp3 directional asymmetry survives when the classifier boundary is made simpler.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for spec in specs:
        lines.append(f"- `{spec.label}`: `{spec.root}`")
    lines.append("")
    lines.append("## Survival read definitions")
    lines.append("")
    lines.append("- `survives_vs_cp_and_c`: Cp3→Cp2 asymmetry is positive and Cp2 specificity is positive against both Cp and C controls.")
    lines.append("- `survives_vs_cp_only`: Cp3→Cp2 asymmetry is positive and Cp2 specificity is positive against Cp, but not C.")
    lines.append("- `survives_vs_c_only`: Cp3→Cp2 asymmetry is positive and Cp2 specificity is positive against C, but not Cp.")
    lines.append("- `directional_without_specificity`: Cp3→Cp2 asymmetry is positive but not specifically stronger than controls.")
    lines.append("- `collapsed_or_near_zero`: Cp3→Cp2 asymmetry is not meaningfully positive under this threshold.")
    lines.append("- `unavailable_or_partial`: at least one required directional/control row is missing or below class-count availability.")
    lines.append("")
    lines.append("## Run-level summary")
    lines.append("")
    lines.append(markdown_table(run_summary))
    lines.append("")
    lines.append("## Primary survival table")
    lines.append("")
    primary_cols = [
        "run_label",
        "target",
        "feature_set",
        "asymmetry_cp3_minus_cp2",
        "specificity_vs_cp",
        "specificity_vs_c",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "specificity_status",
        "survival_read",
    ]
    lines.append(markdown_table(survival, columns=primary_cols, max_rows=80))
    lines.append("")
    lines.append("## Model-capacity matrix")
    lines.append("")
    lines.append(markdown_table(capacity_matrix, max_rows=80))
    lines.append("")
    lines.append("## Highest strict Cp-specific rows")
    lines.append("")
    strict = specificity[
        specificity["feature_set"].isin(STRICT_FEATURE_SETS)
        & specificity["target"].isin(FOCUS_TARGETS)
    ].copy()
    strict = strict.sort_values(
        ["specificity_vs_cp", "specificity_vs_c", "asymmetry_cp3_minus_cp2"],
        ascending=[False, False, False],
        na_position="last",
    )
    high_cols = [
        "run_label",
        "target",
        "feature_set",
        "asymmetry_cp3_minus_cp2",
        "specificity_vs_cp",
        "specificity_vs_c",
        "ba_cp2_to_cp3",
        "ba_cp3_to_cp2",
        "ba_cp_to_cp3",
        "ba_cp3_to_cp",
        "ba_c_to_cp3",
        "ba_cp3_to_c",
        "specificity_status",
    ]
    lines.append(markdown_table(strict, columns=high_cols, max_rows=40))
    lines.append("")
    lines.append("## Provisional interpretation")
    lines.append("")
    lines.append(
        "If a row survives in logistic regression or shallow forests, the broad-boundary "
        "critique is weakened for that target/feature set. If it survives only in full "
        "Random Forests, the safer interpretation is flexible-boundary or distribution-mediated. "
        "If coupled/recovery rows collapse under low-complexity models, OBS-075 should remain "
        "downgraded for those targets."
    )
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append("- OBS-075c is a cross-run comparison; it is only as valid as the OBS-075b runs supplied.")
    lines.append("- Missing or partial rows are not evidence of absence; they indicate unavailable class support or missing artifacts.")
    lines.append("- Positive specificity versus Cp but not C suggests a corpus-family interaction rather than strict Cp2 specificity.")
    lines.append("- Logistic regression survival is stronger evidence for a smooth low-complexity field separation.")
    lines.append("- Shallow RF survival indicates low-depth nonlinear separation.")
    lines.append("- Full-RF-only survival remains compatible with a flexible-boundary artifact.")
    lines.append("- Path-level lexical controls remain outside this script.")

    (outdir / "obs075c_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OBS-075c low-complexity control over OBS-075b artifacts."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run spec in form label=outdir. May be repeated. "
            "Each outdir must contain obs075b_model_scores.csv."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="outputs/comparisons/obs075c_cp3_low_complexity_control",
        help="Output directory for OBS-075c summary artifacts.",
    )
    args = parser.parse_args()

    raw_runs = args.run or DEFAULT_RUNS
    specs = [parse_run_spec(r) for r in raw_runs]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_scores = []
    all_pair_asym = []
    all_specificity = []

    for spec in specs:
        scores = read_scores(spec)
        all_scores.append(scores)

        cp2_asym = build_pair_asymmetry(
            scores,
            run_label=spec.label,
            pair="Cp2_vs_Cp3",
            baseline="Cp2",
        )
        cp_asym = build_pair_asymmetry(
            scores,
            run_label=spec.label,
            pair="Cp_vs_Cp3",
            baseline="Cp",
        )
        c_asym = build_pair_asymmetry(
            scores,
            run_label=spec.label,
            pair="C_vs_Cp3",
            baseline="C",
        )

        all_pair_asym.extend([cp2_asym, cp_asym, c_asym])
        all_specificity.append(build_specificity(cp2_asym, cp_asym, c_asym))

    scores_df = pd.concat(all_scores, ignore_index=True)
    pair_asym_df = pd.concat(all_pair_asym, ignore_index=True)
    specificity_df = pd.concat(all_specificity, ignore_index=True)

    survival_df = build_survival_table(specificity_df)
    capacity_matrix_df = build_model_capacity_matrix(survival_df)
    run_summary_df = summarize_by_run(specificity_df)

    scores_df.to_csv(outdir / "obs075c_model_scores_combined.csv", index=False)
    pair_asym_df.to_csv(outdir / "obs075c_pair_asymmetry.csv", index=False)
    specificity_df.to_csv(outdir / "obs075c_specificity.csv", index=False)
    survival_df.to_csv(outdir / "obs075c_survival_table.csv", index=False)
    capacity_matrix_df.to_csv(outdir / "obs075c_model_capacity_matrix.csv", index=False)
    run_summary_df.to_csv(outdir / "obs075c_run_summary.csv", index=False)

    write_summary(
        outdir,
        specs=specs,
        run_summary=run_summary_df,
        survival=survival_df,
        capacity_matrix=capacity_matrix_df,
        specificity=specificity_df,
    )

    print(outdir / "obs075c_summary.md")
    print(outdir / "obs075c_model_scores_combined.csv")
    print(outdir / "obs075c_pair_asymmetry.csv")
    print(outdir / "obs075c_specificity.csv")
    print(outdir / "obs075c_survival_table.csv")
    print(outdir / "obs075c_model_capacity_matrix.csv")
    print(outdir / "obs075c_run_summary.csv")


if __name__ == "__main__":
    main()
