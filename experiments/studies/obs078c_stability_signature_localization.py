#!/usr/bin/env python3
"""
obs078c_stability_signature_localization.py

OBS-078c — Stability Signature Localization.

Purpose
-------
OBS-078a showed that OBS-077-derived mechanistic features classify C / Cp2 / Cp3.
OBS-078b showed that a minimal 3-feature local stability signature already
carries most of the separable signal:

    mean_lambda_local_mean
    mean_delta_d_mean
    bounded_share_mean

OBS-078c asks:

    Where does that minimal stability signature live?

It localizes the 3-feature signature over:

    case
    object
    cohort
    scale transition

and ranks the objects/cohorts/transitions that most separate C / Cp2 / Cp3.

Inputs
------
Default input is the OBS-078a v2 feature table:

    outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
      obs078a_feature_table.csv

Outputs
-------
    obs078c_input_manifest.csv
    obs078c_feature_zscores.csv
    obs078c_signature_by_case.csv
    obs078c_signature_by_object.csv
    obs078c_signature_by_cohort.csv
    obs078c_signature_by_object_cohort.csv
    obs078c_signature_by_transition.csv
    obs078c_pairwise_case_contrasts.csv
    obs078c_top_separating_groups.csv
    obs078c_case_object_matrix.csv
    obs078c_case_cohort_matrix.csv
    obs078c_report.md

Scientific guardrail
--------------------
OBS-078c is a localization/contrast diagnostic. It does not establish causality.

The key claim it can support is:

    The minimal OBS-078b stability signature localizes back onto the
    same transition/cohort structure interpreted in OBS-077.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


STABILITY_FEATURES = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
]

KEY_COLS = [
    "case",
    "candidate_rank",
    "object",
    "scale_index_from",
    "scale_index_to",
    "cohort",
]

EPS = 1e-12


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


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def transition_label(row: pd.Series) -> str:
    return f"{row.get('scale_index_from')}→{row.get('scale_index_to')}"


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)

    if pooled <= EPS:
        return np.nan

    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def signed_mean_difference(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if x.empty or y.empty:
        return np.nan
    return float(x.mean() - y.mean())


# -----------------------------------------------------------------------------
# Core feature construction
# -----------------------------------------------------------------------------

def add_stability_zscores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add global z-scores for the 3 OBS-078b stability features.

    Higher lambda and delta are interpreted as more locally divergent/displacing.
    Higher bounded_share is interpreted as more locally bounded/contained.

    Derived coordinates:
      divergence_mean_z:
        average of lambda_z and delta_z

      unboundedness_z:
        -bounded_share_z

      instability_signature_z:
        average of lambda_z, delta_z, and -bounded_share_z

      bounded_stability_signature_z:
        bounded_share_z - average(lambda_z, delta_z)

    These coordinates are descriptive; the primary outputs keep individual
    feature z-scores as well.
    """
    out = df.copy()
    stats_rows = []

    for col in STABILITY_FEATURES:
        vals = pd.to_numeric(out[col], errors="coerce")
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1))
        if not np.isfinite(sd) or sd <= EPS:
            z = pd.Series(0.0, index=out.index)
        else:
            z = (vals - mu) / sd

        z_col = col.replace("_mean", "") + "_z_global"
        out[z_col] = z

        stats_rows.append(
            {
                "feature": col,
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

    return out, pd.DataFrame(stats_rows)


SIGNATURE_NUMERIC = [
    "mean_lambda_local_mean",
    "mean_delta_d_mean",
    "bounded_share_mean",
    "lambda_z_global",
    "delta_d_z_global",
    "bounded_z_global",
    "divergence_mean_z",
    "unboundedness_z",
    "instability_signature_z",
    "bounded_stability_signature_z",
]


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    require_columns(df, group_cols, "summarize_group input")

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_rows"] = int(len(g))

        for col in SIGNATURE_NUMERIC:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}__mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}__median"] = float(vals.median()) if vals.notna().any() else np.nan
            row[f"{col}__std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = [c for c in group_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def build_case_object_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Wide object × case matrix for quick inspection.
    """
    if summary.empty:
        return pd.DataFrame()

    metric_cols = [
        "instability_signature_z__mean",
        "bounded_stability_signature_z__mean",
        "divergence_mean_z__mean",
        "bounded_z_global__mean",
    ]

    pieces = []
    for metric in metric_cols:
        if metric not in summary.columns:
            continue

        piv = summary.pivot_table(
            index="object",
            columns="case",
            values=metric,
            aggfunc="mean",
        ).reset_index()

        piv.columns = [
            c if c == "object" else f"{metric}__{c}"
            for c in piv.columns
        ]
        pieces.append(piv)

    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="object", how="outer")

    return out


def build_case_cohort_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    metric_cols = [
        "instability_signature_z__mean",
        "bounded_stability_signature_z__mean",
        "divergence_mean_z__mean",
        "bounded_z_global__mean",
    ]

    pieces = []
    for metric in metric_cols:
        if metric not in summary.columns:
            continue

        piv = summary.pivot_table(
            index="cohort",
            columns="case",
            values=metric,
            aggfunc="mean",
        ).reset_index()

        piv.columns = [
            c if c == "cohort" else f"{metric}__{c}"
            for c in piv.columns
        ]
        pieces.append(piv)

    if not pieces:
        return pd.DataFrame()

    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="cohort", how="outer")

    return out


# -----------------------------------------------------------------------------
# Pairwise contrasts and top separating groups
# -----------------------------------------------------------------------------

def pairwise_case_contrasts(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Compute pairwise case contrasts for the stability signature.

    If group_cols is None or empty, contrasts are global by case.
    If group_cols is provided, contrasts are computed within matching group values.
    """
    group_cols = group_cols or []
    require_columns(df, ["case"] + group_cols, "pairwise contrast input")

    rows = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [((), df)]

    for group_key, g in grouped:
        if group_cols and not isinstance(group_key, tuple):
            group_key = (group_key,)
        elif not group_cols:
            group_key = ()

        cases = sorted(g["case"].dropna().astype(str).unique())
        if len(cases) < 2:
            continue

        for a, b in combinations(cases, 2):
            ga = g[g["case"].astype(str) == a]
            gb = g[g["case"].astype(str) == b]

            base = {col: val for col, val in zip(group_cols, group_key)}
            base["case_a"] = a
            base["case_b"] = b
            base["n_a"] = int(len(ga))
            base["n_b"] = int(len(gb))

            for col in SIGNATURE_NUMERIC:
                base[f"{col}__mean_a"] = float(pd.to_numeric(ga[col], errors="coerce").mean())
                base[f"{col}__mean_b"] = float(pd.to_numeric(gb[col], errors="coerce").mean())
                base[f"{col}__diff_a_minus_b"] = signed_mean_difference(ga[col], gb[col])
                base[f"{col}__cohen_d_a_minus_b"] = cohen_d(ga[col], gb[col])

            # Primary separation score: average absolute Cohen d over the compact
            # stability coordinates, falling back to absolute mean differences if
            # Cohen d is unavailable.
            d_cols = [
                "lambda_z_global__cohen_d_a_minus_b",
                "delta_d_z_global__cohen_d_a_minus_b",
                "bounded_z_global__cohen_d_a_minus_b",
                "instability_signature_z__cohen_d_a_minus_b",
                "bounded_stability_signature_z__cohen_d_a_minus_b",
            ]

            ds = [abs(base[c]) for c in d_cols if c in base and np.isfinite(base[c])]
            if ds:
                base["separation_score"] = float(np.mean(ds))
            else:
                diff_cols = [
                    "lambda_z_global__diff_a_minus_b",
                    "delta_d_z_global__diff_a_minus_b",
                    "bounded_z_global__diff_a_minus_b",
                    "instability_signature_z__diff_a_minus_b",
                    "bounded_stability_signature_z__diff_a_minus_b",
                ]
                diffs = [abs(base[c]) for c in diff_cols if c in base and np.isfinite(base[c])]
                base["separation_score"] = float(np.mean(diffs)) if diffs else np.nan

            rows.append(base)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("separation_score", ascending=False, na_position="last").reset_index(drop=True)
    return out


def top_separating_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank object/cohort/object+cohort/transition groups by cross-case separation.
    """
    specs = [
        ("object", ["object"]),
        ("cohort", ["cohort"]),
        ("object_cohort", ["object", "cohort"]),
        ("transition", ["transition"]),
        ("object_transition", ["object", "transition"]),
    ]

    pieces = []
    for group_type, cols in specs:
        available = [c for c in cols if c in df.columns]
        if len(available) != len(cols):
            continue

        pc = pairwise_case_contrasts(df, group_cols=cols)
        if not pc.empty:
            pc.insert(0, "group_type", group_type)
            pieces.append(pc)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True, sort=False)
    out = out.sort_values("separation_score", ascending=False, na_position="last").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

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
    write_csv(out, outdir / "obs078c_input_manifest.csv")
    return out


def write_report(
    outdir: Path,
    input_manifest: pd.DataFrame,
    zstats: pd.DataFrame,
    by_case: pd.DataFrame,
    by_object: pd.DataFrame,
    by_cohort: pd.DataFrame,
    by_object_cohort: pd.DataFrame,
    by_transition: pd.DataFrame,
    pairwise_global: pd.DataFrame,
    top_groups: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# OBS-078c — Stability Signature Localization")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "OBS-078c localizes the minimal OBS-078b stability signature back onto "
        "case, object, cohort, and transition structure."
    )
    lines.append("")
    lines.append("Minimal stability features:")
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
    lines.append("## Input manifest")
    lines.append("")
    lines.append(markdown_table(input_manifest))
    lines.append("")
    lines.append("## Z-score statistics")
    lines.append("")
    lines.append(markdown_table(zstats))
    lines.append("")
    lines.append("## Signature by case")
    lines.append("")
    case_cols = [
        "case",
        "n_rows",
        "mean_lambda_local_mean__mean",
        "mean_delta_d_mean__mean",
        "bounded_share_mean__mean",
        "divergence_mean_z__mean",
        "instability_signature_z__mean",
        "bounded_stability_signature_z__mean",
    ]
    lines.append(markdown_table(by_case[[c for c in case_cols if c in by_case.columns]], max_rows=20))
    lines.append("")
    lines.append("## Global pairwise case contrasts")
    lines.append("")
    contrast_cols = [
        "case_a",
        "case_b",
        "n_a",
        "n_b",
        "lambda_z_global__diff_a_minus_b",
        "delta_d_z_global__diff_a_minus_b",
        "bounded_z_global__diff_a_minus_b",
        "instability_signature_z__diff_a_minus_b",
        "bounded_stability_signature_z__diff_a_minus_b",
        "separation_score",
    ]
    lines.append(markdown_table(pairwise_global[[c for c in contrast_cols if c in pairwise_global.columns]], max_rows=20))
    lines.append("")
    lines.append("## Signature by object")
    lines.append("")
    object_cols = [
        "case",
        "object",
        "n_rows",
        "divergence_mean_z__mean",
        "instability_signature_z__mean",
        "bounded_stability_signature_z__mean",
    ]
    bo = by_object.sort_values(
        ["case", "instability_signature_z__mean"],
        ascending=[True, False],
        na_position="last",
    )
    lines.append(markdown_table(bo[[c for c in object_cols if c in bo.columns]], max_rows=40))
    lines.append("")
    lines.append("## Signature by cohort")
    lines.append("")
    cohort_cols = [
        "case",
        "cohort",
        "n_rows",
        "divergence_mean_z__mean",
        "instability_signature_z__mean",
        "bounded_stability_signature_z__mean",
    ]
    bc = by_cohort.sort_values(
        ["case", "instability_signature_z__mean"],
        ascending=[True, False],
        na_position="last",
    )
    lines.append(markdown_table(bc[[c for c in cohort_cols if c in bc.columns]], max_rows=40))
    lines.append("")
    lines.append("## Top separating groups")
    lines.append("")
    top_cols = [
        "group_type",
        "object",
        "cohort",
        "transition",
        "case_a",
        "case_b",
        "n_a",
        "n_b",
        "instability_signature_z__diff_a_minus_b",
        "bounded_stability_signature_z__diff_a_minus_b",
        "separation_score",
    ]
    lines.append(markdown_table(top_groups[[c for c in top_cols if c in top_groups.columns]], max_rows=50))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("OBS-078c succeeds if the 3-feature stability signature localizes back onto the same mechanism identified in OBS-077:")
    lines.append("")
    lines.append("```text")
    lines.append("C:")
    lines.append("  bounded recovery recruitment")
    lines.append("")
    lines.append("Cp2:")
    lines.append("  high-divergence recovery sorting")
    lines.append("")
    lines.append("Cp3:")
    lines.append("  earlier divergence, later nonrecovering settlement")
    lines.append("```")
    lines.append("")
    lines.append("Useful localization evidence looks like:")
    lines.append("")
    lines.append("```text")
    lines.append("C objects/cohorts show higher bounded_stability_signature_z")
    lines.append("Cp2 response/entry-like cohorts show higher instability/divergence")
    lines.append("Cp3 energy/seam/coupling cohorts show shifted stability/divergence placement")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-078c is a localization diagnostic, not causal proof.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs078c_input_manifest.csv")
    lines.append("obs078c_feature_zscores.csv")
    lines.append("obs078c_signature_by_case.csv")
    lines.append("obs078c_signature_by_object.csv")
    lines.append("obs078c_signature_by_cohort.csv")
    lines.append("obs078c_signature_by_object_cohort.csv")
    lines.append("obs078c_signature_by_transition.csv")
    lines.append("obs078c_pairwise_case_contrasts.csv")
    lines.append("obs078c_top_separating_groups.csv")
    lines.append("obs078c_case_object_matrix.csv")
    lines.append("obs078c_case_cohort_matrix.csv")
    lines.append("obs078c_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-078c")

    (outdir / "obs078c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OBS-078c stability signature localization.")
    ap.add_argument(
        "--feature-table",
        default=(
            "outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/"
            "obs078a_feature_table.csv"
        ),
        help="OBS-078a v2 feature table.",
    )
    ap.add_argument(
        "--outdir",
        default="outputs/comparisons/obs078c_stability_signature_localization",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    feature_table_path = Path(args.feature_table)
    outdir = ensure_outdir(args.outdir)

    if not feature_table_path.exists():
        raise FileNotFoundError(feature_table_path)

    df = pd.read_csv(feature_table_path)
    require_columns(df, ["case", "object", "cohort"] + STABILITY_FEATURES, "feature table")
    df = safe_numeric(df, STABILITY_FEATURES)

    if "transition" not in df.columns:
        require_columns(df, ["scale_index_from", "scale_index_to"], "feature table transition columns")
        df["transition"] = df.apply(transition_label, axis=1)

    input_manifest = write_input_manifest(outdir, feature_table_path, df)

    zdf, zstats = add_stability_zscores(df)

    write_csv(zdf, outdir / "obs078c_feature_zscores.csv")
    write_csv(zstats, outdir / "obs078c_zscore_stats.csv")

    by_case = summarize_group(zdf, ["case"])
    by_object = summarize_group(zdf, ["case", "object"])
    by_cohort = summarize_group(zdf, ["case", "cohort"])
    by_object_cohort = summarize_group(zdf, ["case", "object", "cohort"])
    by_transition = summarize_group(zdf, ["case", "transition"])
    by_object_transition = summarize_group(zdf, ["case", "object", "transition"])

    write_csv(by_case, outdir / "obs078c_signature_by_case.csv")
    write_csv(by_object, outdir / "obs078c_signature_by_object.csv")
    write_csv(by_cohort, outdir / "obs078c_signature_by_cohort.csv")
    write_csv(by_object_cohort, outdir / "obs078c_signature_by_object_cohort.csv")
    write_csv(by_transition, outdir / "obs078c_signature_by_transition.csv")
    write_csv(by_object_transition, outdir / "obs078c_signature_by_object_transition.csv")

    pairwise_global = pairwise_case_contrasts(zdf)
    pairwise_by_object = pairwise_case_contrasts(zdf, group_cols=["object"])
    pairwise_by_cohort = pairwise_case_contrasts(zdf, group_cols=["cohort"])
    pairwise_by_object_cohort = pairwise_case_contrasts(zdf, group_cols=["object", "cohort"])
    pairwise_by_transition = pairwise_case_contrasts(zdf, group_cols=["transition"])

    pairwise_all = pd.concat(
        [
            pairwise_global.assign(group_type="global"),
            pairwise_by_object.assign(group_type="object"),
            pairwise_by_cohort.assign(group_type="cohort"),
            pairwise_by_object_cohort.assign(group_type="object_cohort"),
            pairwise_by_transition.assign(group_type="transition"),
        ],
        ignore_index=True,
        sort=False,
    )

    if not pairwise_all.empty:
        pairwise_all = pairwise_all.sort_values(
            "separation_score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    write_csv(pairwise_all, outdir / "obs078c_pairwise_case_contrasts.csv")

    top_groups = top_separating_groups(zdf)
    write_csv(top_groups, outdir / "obs078c_top_separating_groups.csv")

    case_object_matrix = build_case_object_matrix(by_object)
    case_cohort_matrix = build_case_cohort_matrix(by_cohort)

    write_csv(case_object_matrix, outdir / "obs078c_case_object_matrix.csv")
    write_csv(case_cohort_matrix, outdir / "obs078c_case_cohort_matrix.csv")

    write_report(
        outdir=outdir,
        input_manifest=input_manifest,
        zstats=zstats,
        by_case=by_case,
        by_object=by_object,
        by_cohort=by_cohort,
        by_object_cohort=by_object_cohort,
        by_transition=by_transition,
        pairwise_global=pairwise_global,
        top_groups=top_groups,
    )

    print(f"[OBS-078c] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
