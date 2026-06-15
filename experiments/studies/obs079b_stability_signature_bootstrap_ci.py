#!/usr/bin/env python3
"""
obs079b_stability_signature_bootstrap_ci.py

OBS-079b — Stability Signature Bootstrap Confidence Intervals.

Purpose
-------
OBS-078 found that the C / Cp2 / Cp3 distinction compresses to a small
window-local stability signature:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-079a showed that this signature survives leave-structure-out validation.

OBS-079b asks:

    Are the measured stability coordinates themselves stable under resampling?

This script bootstraps confidence intervals for the stability signature by:

    case
    case × object
    case × cohort
    case × transition
    case × object × cohort

It also bootstraps pairwise case contrasts.

Inputs
------
Default:

    outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
      obs078a_feature_table.csv

Outputs
-------
    obs079b_input_manifest.csv
    obs079b_zscore_stats.csv
    obs079b_feature_zscores.csv
    obs079b_group_inventory.csv
    obs079b_bootstrap_ci_by_case.csv
    obs079b_bootstrap_ci_by_object.csv
    obs079b_bootstrap_ci_by_cohort.csv
    obs079b_bootstrap_ci_by_transition.csv
    obs079b_bootstrap_ci_by_object_cohort.csv
    obs079b_bootstrap_pairwise_case_contrasts.csv
    obs079b_bootstrap_pairwise_case_contrasts_by_object.csv
    obs079b_bootstrap_pairwise_case_contrasts_by_cohort.csv
    obs079b_report.md

Scientific guardrail
--------------------
OBS-079b is a measurement-robustness diagnostic. It does not establish causality.

The key supported statement is:

    The OBS-078 local stability coordinates are stable under row resampling
    if bootstrap intervals preserve the C vs Cp2/Cp3 separation.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_RANDOM_STATE = 79002
EPS = 1e-12

STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

BASE_KEY_COLS = [
    "case",
    "object",
    "cohort",
    "scale_index_from",
    "scale_index_to",
]

DERIVED_COORDS = [
    "lambda_z_global",
    "delta_d_z_global",
    "bounded_z_global",
    "divergence_mean_z",
    "unboundedness_z",
    "instability_signature_z",
    "bounded_stability_signature_z",
]

BOOTSTRAP_METRICS = STABILITY_FEATURES + DERIVED_COORDS


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def add_transition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "transition" not in out.columns:
        require_columns(out, ["scale_index_from", "scale_index_to"], "feature table")
        out["transition"] = out.apply(transition_label, axis=1)
    return out


def quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.quantile(vals, q))


def ci_excludes_zero(lo: float, hi: float) -> bool:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return False
    return bool((lo > 0) or (hi < 0))


# -----------------------------------------------------------------------------
# Stability coordinates
# -----------------------------------------------------------------------------

def add_stability_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add global z-score coordinates for OBS-078 stability features.

    Derived:
      divergence_mean_z:
        mean(lambda_z, delta_d_z)

      unboundedness_z:
        -bounded_z

      instability_signature_z:
        mean(lambda_z, delta_d_z, -bounded_z)

      bounded_stability_signature_z:
        bounded_z - mean(lambda_z, delta_d_z)
    """
    out = df.copy()
    rows = []

    for feature in STABILITY_FEATURES:
        vals = pd.to_numeric(out[feature], errors="coerce")
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1))

        z_col = feature.replace("_mean", "") + "_z_global"

        if not np.isfinite(sd) or sd <= EPS:
            out[z_col] = 0.0
        else:
            out[z_col] = (vals - mu) / sd

        rows.append(
            {
                "feature": feature,
                "mean": mu,
                "std": sd,
                "z_col": z_col,
                "n_non_null": int(vals.notna().sum()),
            }
        )

    out["lambda_z_global"] = out["mean_lambda_local_z_global"]
    out["delta_d_z_global"] = out["mean_delta_d_z_global"]
    out["bounded_z_global"] = out["bounded_share_z_global"]

    out["divergence_mean_z"] = out[["lambda_z_global", "delta_d_z_global"]].mean(axis=1)
    out["unboundedness_z"] = -out["bounded_z_global"]
    out["instability_signature_z"] = out[
        ["lambda_z_global", "delta_d_z_global", "unboundedness_z"]
    ].mean(axis=1)
    out["bounded_stability_signature_z"] = (
        out["bounded_z_global"] - out[["lambda_z_global", "delta_d_z_global"]].mean(axis=1)
    )

    return out, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Bootstrap CIs
# -----------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: pd.Series,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_low: float,
    ci_high: float,
) -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(vals)

    if n == 0:
        return {
            "n": 0,
            "observed_mean": np.nan,
            "bootstrap_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_std": np.nan,
        }

    observed = float(np.mean(vals))

    if n == 1:
        return {
            "n": 1,
            "observed_mean": observed,
            "bootstrap_mean": observed,
            "ci_low": observed,
            "ci_high": observed,
            "bootstrap_std": 0.0,
        }

    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = np.mean(vals[idx])

    return {
        "n": int(n),
        "observed_mean": observed,
        "bootstrap_mean": float(np.mean(samples)),
        "ci_low": quantile(samples, ci_low),
        "ci_high": quantile(samples, ci_high),
        "bootstrap_std": float(np.std(samples, ddof=1)) if n_bootstrap > 1 else np.nan,
    }


def bootstrap_group_cis(
    df: pd.DataFrame,
    group_cols: list[str],
    n_bootstrap: int,
    seed: int,
    ci_low: float,
    ci_high: float,
    min_n: int,
) -> pd.DataFrame:
    require_columns(df, group_cols, "bootstrap group input")

    rows = []
    rng = np.random.default_rng(seed)

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        group_base = {col: val for col, val in zip(group_cols, keys)}
        group_base["n_rows"] = int(len(g))
        group_base["valid_min_n"] = bool(len(g) >= min_n)

        for metric in BOOTSTRAP_METRICS:
            stats = bootstrap_mean_ci(
                g[metric],
                n_bootstrap=n_bootstrap,
                rng=rng,
                ci_low=ci_low,
                ci_high=ci_high,
            )

            row = dict(group_base)
            row["metric"] = metric
            row.update(stats)
            row["ci_width"] = (
                row["ci_high"] - row["ci_low"]
                if np.isfinite(row["ci_high"]) and np.isfinite(row["ci_low"])
                else np.nan
            )
            row["ci_excludes_zero"] = ci_excludes_zero(row["ci_low"], row["ci_high"])
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols + ["metric"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Pairwise contrast bootstrap
# -----------------------------------------------------------------------------

def bootstrap_pairwise_diff_ci(
    a: pd.Series,
    b: pd.Series,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci_low: float,
    ci_high: float,
) -> dict[str, float]:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)

    nx, ny = len(x), len(y)

    if nx == 0 or ny == 0:
        return {
            "n_a": int(nx),
            "n_b": int(ny),
            "observed_diff_a_minus_b": np.nan,
            "bootstrap_mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_std": np.nan,
        }

    observed = float(np.mean(x) - np.mean(y))

    if nx == 1 and ny == 1:
        return {
            "n_a": int(nx),
            "n_b": int(ny),
            "observed_diff_a_minus_b": observed,
            "bootstrap_mean_diff": observed,
            "ci_low": observed,
            "ci_high": observed,
            "bootstrap_std": 0.0,
        }

    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        ix = rng.integers(0, nx, size=nx)
        iy = rng.integers(0, ny, size=ny)
        samples[i] = np.mean(x[ix]) - np.mean(y[iy])

    return {
        "n_a": int(nx),
        "n_b": int(ny),
        "observed_diff_a_minus_b": observed,
        "bootstrap_mean_diff": float(np.mean(samples)),
        "ci_low": quantile(samples, ci_low),
        "ci_high": quantile(samples, ci_high),
        "bootstrap_std": float(np.std(samples, ddof=1)) if n_bootstrap > 1 else np.nan,
    }


def bootstrap_pairwise_case_contrasts(
    df: pd.DataFrame,
    group_cols: list[str] | None,
    n_bootstrap: int,
    seed: int,
    ci_low: float,
    ci_high: float,
    min_n_per_case: int,
) -> pd.DataFrame:
    group_cols = group_cols or []
    require_columns(df, ["case"] + group_cols, "pairwise contrast input")

    rng = np.random.default_rng(seed)
    rows = []

    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for group_key, g in grouped:
        if group_cols and not isinstance(group_key, tuple):
            group_key = (group_key,)
        elif not group_cols:
            group_key = ()

        cases = sorted(g["case"].dropna().astype(str).unique())
        if len(cases) < 2:
            continue

        for case_a, case_b in combinations(cases, 2):
            ga = g[g["case"].astype(str) == case_a]
            gb = g[g["case"].astype(str) == case_b]

            base = {col: val for col, val in zip(group_cols, group_key)}
            base["case_a"] = case_a
            base["case_b"] = case_b

            for metric in BOOTSTRAP_METRICS:
                stats = bootstrap_pairwise_diff_ci(
                    ga[metric],
                    gb[metric],
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )

                row = dict(base)
                row["metric"] = metric
                row.update(stats)
                row["valid_min_n_per_case"] = bool(
                    row["n_a"] >= min_n_per_case and row["n_b"] >= min_n_per_case
                )
                row["ci_width"] = (
                    row["ci_high"] - row["ci_low"]
                    if np.isfinite(row["ci_high"]) and np.isfinite(row["ci_low"])
                    else np.nan
                )
                row["ci_excludes_zero"] = ci_excludes_zero(row["ci_low"], row["ci_high"])
                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = group_cols + ["case_a", "case_b", "metric"]
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Inventory / reporting
# -----------------------------------------------------------------------------

def group_inventory(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("case", ["case"]),
        ("case_object", ["case", "object"]),
        ("case_cohort", ["case", "cohort"]),
        ("case_transition", ["case", "transition"]),
        ("case_object_cohort", ["case", "object", "cohort"]),
    ]

    rows = []
    for group_type, cols in specs:
        for keys, g in df.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"group_type": group_type}
            row.update({col: val for col, val in zip(cols, keys)})
            row["n_rows"] = int(len(g))
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group_type", "n_rows"], ascending=[True, False]).reset_index(drop=True)
    return out


def write_input_manifest(outdir: Path, feature_table_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "artifact": "feature_table",
            "path": str(feature_table_path),
            "status": "ok" if feature_table_path.exists() else "missing",
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
        }
    ]
    out = pd.DataFrame(rows)
    write_csv(out, outdir / "obs079b_input_manifest.csv")
    return out


def select_metric_rows(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["metric"].isin(metrics)].copy()


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    zstats: pd.DataFrame,
    inventory: pd.DataFrame,
    ci_case: pd.DataFrame,
    pairwise_global: pd.DataFrame,
    pairwise_by_object: pd.DataFrame,
    pairwise_by_cohort: pd.DataFrame,
    n_bootstrap: int,
    ci_low: float,
    ci_high: float,
) -> None:
    lines = []

    lines.append("# OBS-079b — Stability Signature Bootstrap Confidence Intervals")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-079b tests whether the OBS-078 local stability coordinates are stable "
        "under row resampling."
    )
    lines.append("")
    lines.append("Stability core:")
    lines.append("")
    lines.append("```text")
    for f in STABILITY_FEATURES:
        lines.append(f)
    lines.append("```")
    lines.append("")
    lines.append("Derived coordinates:")
    lines.append("")
    lines.append("```text")
    lines.append("divergence_mean_z = mean(lambda_z, delta_d_z)")
    lines.append("unboundedness_z = -bounded_z")
    lines.append("instability_signature_z = mean(lambda_z, delta_d_z, -bounded_z)")
    lines.append("bounded_stability_signature_z = bounded_z - mean(lambda_z, delta_d_z)")
    lines.append("```")
    lines.append("")
    lines.append("Bootstrap settings:")
    lines.append("")
    lines.append("```text")
    lines.append(f"n_bootstrap = {n_bootstrap}")
    lines.append(f"ci_low      = {ci_low}")
    lines.append(f"ci_high     = {ci_high}")
    lines.append("```")
    lines.append("")
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Z-score statistics")
    lines.append("")
    lines.append(markdown_table(zstats))
    lines.append("")
    lines.append("## Group inventory")
    lines.append("")
    inv_summary = (
        inventory.groupby("group_type", dropna=False)
        .agg(
            n_groups=("n_rows", "count"),
            min_rows=("n_rows", "min"),
            mean_rows=("n_rows", "mean"),
            max_rows=("n_rows", "max"),
        )
        .reset_index()
        .sort_values("group_type")
    )
    lines.append(markdown_table(inv_summary, max_rows=50))
    lines.append("")
    lines.append("## Bootstrap CIs by case")
    lines.append("")
    case_display_metrics = [
        "mean_lambda_local_mean",
        "mean_delta_d_mean",
        "bounded_share_mean",
        "instability_signature_z",
        "bounded_stability_signature_z",
    ]
    case_disp = select_metric_rows(ci_case, case_display_metrics)
    case_cols = [
        "case",
        "metric",
        "n",
        "observed_mean",
        "ci_low",
        "ci_high",
        "ci_width",
        "ci_excludes_zero",
    ]
    lines.append(markdown_table(case_disp[[c for c in case_cols if c in case_disp.columns]], max_rows=80))
    lines.append("")
    lines.append("## Global pairwise case contrasts")
    lines.append("")
    pair_display_metrics = [
        "instability_signature_z",
        "bounded_stability_signature_z",
        "mean_lambda_local_mean",
        "mean_delta_d_mean",
        "bounded_share_mean",
    ]
    pair_disp = select_metric_rows(pairwise_global, pair_display_metrics)
    pair_cols = [
        "case_a",
        "case_b",
        "metric",
        "n_a",
        "n_b",
        "observed_diff_a_minus_b",
        "ci_low",
        "ci_high",
        "ci_width",
        "ci_excludes_zero",
    ]
    lines.append(markdown_table(pair_disp[[c for c in pair_cols if c in pair_disp.columns]], max_rows=120))
    lines.append("")
    lines.append("## Object-level pairwise contrasts")
    lines.append("")
    obj_disp = select_metric_rows(pairwise_by_object, [
        "instability_signature_z",
        "bounded_stability_signature_z",
    ])
    if not obj_disp.empty:
        obj_disp = obj_disp.sort_values(
            ["ci_excludes_zero", "object", "case_a", "case_b", "metric"],
            ascending=[False, True, True, True, True],
        )
    obj_cols = [
        "object",
        "case_a",
        "case_b",
        "metric",
        "n_a",
        "n_b",
        "observed_diff_a_minus_b",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
    ]
    lines.append(markdown_table(obj_disp[[c for c in obj_cols if c in obj_disp.columns]], max_rows=80))
    lines.append("")
    lines.append("## Cohort-level pairwise contrasts")
    lines.append("")
    coh_disp = select_metric_rows(pairwise_by_cohort, [
        "instability_signature_z",
        "bounded_stability_signature_z",
    ])
    if not coh_disp.empty:
        coh_disp = coh_disp.sort_values(
            ["ci_excludes_zero", "cohort", "case_a", "case_b", "metric"],
            ascending=[False, True, True, True, True],
        )
    coh_cols = [
        "cohort",
        "case_a",
        "case_b",
        "metric",
        "n_a",
        "n_b",
        "observed_diff_a_minus_b",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
    ]
    lines.append(markdown_table(coh_disp[[c for c in coh_cols if c in coh_disp.columns]], max_rows=80))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Expected robust pattern:")
    lines.append("")
    lines.append("```text")
    lines.append("C:")
    lines.append("  bounded_stability_signature_z positive")
    lines.append("  instability_signature_z negative")
    lines.append("")
    lines.append("Cp2 / Cp3:")
    lines.append("  bounded_stability_signature_z negative")
    lines.append("  instability_signature_z positive")
    lines.append("")
    lines.append("C vs Cp2 and C vs Cp3:")
    lines.append("  pairwise CIs exclude zero")
    lines.append("")
    lines.append("Cp2 vs Cp3:")
    lines.append("  weaker; CIs may overlap or include zero")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-079b is a measurement-robustness diagnostic, not causal proof.")
    lines.append("Small object/cohort groups should be interpreted cautiously.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs079b_input_manifest.csv")
    lines.append("obs079b_zscore_stats.csv")
    lines.append("obs079b_feature_zscores.csv")
    lines.append("obs079b_group_inventory.csv")
    lines.append("obs079b_bootstrap_ci_by_case.csv")
    lines.append("obs079b_bootstrap_ci_by_object.csv")
    lines.append("obs079b_bootstrap_ci_by_cohort.csv")
    lines.append("obs079b_bootstrap_ci_by_transition.csv")
    lines.append("obs079b_bootstrap_ci_by_object_cohort.csv")
    lines.append("obs079b_bootstrap_pairwise_case_contrasts.csv")
    lines.append("obs079b_bootstrap_pairwise_case_contrasts_by_object.csv")
    lines.append("obs079b_bootstrap_pairwise_case_contrasts_by_cohort.csv")
    lines.append("obs079b_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-079b")

    (outdir / "obs079b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-079b stability signature bootstrap CIs.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs079b_stability_signature_bootstrap_ci",
    )
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--ci-low", type=float, default=0.025)
    ap.add_argument("--ci-high", type=float, default=0.975)
    ap.add_argument("--seed", type=int, default=MODEL_RANDOM_STATE)
    ap.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum group size flag for group CI validity.",
    )
    ap.add_argument(
        "--min-n-per-case",
        type=int,
        default=3,
        help="Minimum per-case count flag for pairwise contrast validity.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    outdir = ensure_outdir(args.outdir)
    feature_table_path = Path(args.feature_table)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)

    df = pd.read_csv(feature_table_path)
    require_columns(df, ["case", "object", "cohort"] + STABILITY_FEATURES, "feature table")

    df = add_transition(df)
    df = safe_numeric(df, STABILITY_FEATURES)

    input_manifest = write_input_manifest(outdir, feature_table_path, df)

    zdf, zstats = add_stability_coordinates(df)

    write_csv(zdf, outdir / "obs079b_feature_zscores.csv")
    write_csv(zstats, outdir / "obs079b_zscore_stats.csv")

    inventory = group_inventory(zdf)
    write_csv(inventory, outdir / "obs079b_group_inventory.csv")

    # Group CIs
    ci_case = bootstrap_group_cis(
        zdf,
        group_cols=["case"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n=args.min_n,
    )

    ci_object = bootstrap_group_cis(
        zdf,
        group_cols=["case", "object"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 2,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n=args.min_n,
    )

    ci_cohort = bootstrap_group_cis(
        zdf,
        group_cols=["case", "cohort"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 3,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n=args.min_n,
    )

    ci_transition = bootstrap_group_cis(
        zdf,
        group_cols=["case", "transition"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 4,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n=args.min_n,
    )

    ci_object_cohort = bootstrap_group_cis(
        zdf,
        group_cols=["case", "object", "cohort"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 5,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n=args.min_n,
    )

    write_csv(ci_case, outdir / "obs079b_bootstrap_ci_by_case.csv")
    write_csv(ci_object, outdir / "obs079b_bootstrap_ci_by_object.csv")
    write_csv(ci_cohort, outdir / "obs079b_bootstrap_ci_by_cohort.csv")
    write_csv(ci_transition, outdir / "obs079b_bootstrap_ci_by_transition.csv")
    write_csv(ci_object_cohort, outdir / "obs079b_bootstrap_ci_by_object_cohort.csv")

    # Pairwise contrasts
    pairwise_global = bootstrap_pairwise_case_contrasts(
        zdf,
        group_cols=[],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 11,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n_per_case=args.min_n_per_case,
    )

    pairwise_by_object = bootstrap_pairwise_case_contrasts(
        zdf,
        group_cols=["object"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 12,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n_per_case=args.min_n_per_case,
    )

    pairwise_by_cohort = bootstrap_pairwise_case_contrasts(
        zdf,
        group_cols=["cohort"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 13,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n_per_case=args.min_n_per_case,
    )

    pairwise_by_transition = bootstrap_pairwise_case_contrasts(
        zdf,
        group_cols=["transition"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 14,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
        min_n_per_case=args.min_n_per_case,
    )

    write_csv(pairwise_global, outdir / "obs079b_bootstrap_pairwise_case_contrasts.csv")
    write_csv(pairwise_by_object, outdir / "obs079b_bootstrap_pairwise_case_contrasts_by_object.csv")
    write_csv(pairwise_by_cohort, outdir / "obs079b_bootstrap_pairwise_case_contrasts_by_cohort.csv")
    write_csv(pairwise_by_transition, outdir / "obs079b_bootstrap_pairwise_case_contrasts_by_transition.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        zstats=zstats,
        inventory=inventory,
        ci_case=ci_case,
        pairwise_global=pairwise_global,
        pairwise_by_object=pairwise_by_object,
        pairwise_by_cohort=pairwise_by_cohort,
        n_bootstrap=args.n_bootstrap,
        ci_low=args.ci_low,
        ci_high=args.ci_high,
    )

    print(f"[OBS-079b] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
