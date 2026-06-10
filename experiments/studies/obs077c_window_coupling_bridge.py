#!/usr/bin/env python3
"""
obs077c_window_coupling_bridge.py

OBS-077c v2 — Window-local divergence / boundedness bridge.

Purpose
-------
Bridge OBS-077b scale-space path-object support memberships and OBS-077a
pinch cohorts to OBS-051 window-level local divergence diagnostics.

v2 patch
--------
1. Detect degenerate categorical labels, e.g. coupling_class == coupled for all
   joined OBS-051 window paths.
2. Keep categorical enrichment outputs for provenance, but exclude degenerate
   label columns from the primary report.
3. Add global numeric baselines for OBS-051 window fields.
4. Add support-level numeric contrasts against global baselines.
5. Add pinch-cohort numeric contrasts against global baselines.
6. Add top-pinch focused numeric contrast table.
7. Reframe OBS-077c as a local-divergence / boundedness bridge over coupled
   windows, not as a categorical coupling_class enrichment test when the
   categorical field is constant.

Core question
-------------
Do paths occupying OBS-076/077 scale-space supports and pinch cohorts show
distinct OBS-051 window-local divergence / boundedness structure?

Inputs
------
Required:
  --case CASE
  --membership obs077b_path_object_membership_path_weighted.csv
  --windows obs051_window_divergence_all.csv
  --outdir OUTDIR

Optional:
  --pinch-candidates obs077a_pinch_point_candidates.csv
  --top-k-pinch 10
  --min-report-paths 25

Outputs
-------
  obs077c_input_manifest.csv
  obs077c_join_audit.csv
  obs077c_label_degeneracy_audit.csv
  obs077c_global_numeric_baseline.csv

  obs077c_support_window_coupling_summary.csv
  obs077c_support_numeric_contrast.csv
  obs077c_support_coupling_class_enrichment.csv
  obs077c_support_seam_band_enrichment.csv
  obs077c_support_path_family_enrichment.csv
  obs077c_support_outcome_group_enrichment.csv

  obs077c_pinch_cohort_window_coupling_summary.csv
  obs077c_pinch_cohort_numeric_contrast.csv
  obs077c_top_pinch_numeric_contrast.csv
  obs077c_pinch_cohort_coupling_class_enrichment.csv
  obs077c_pinch_cohort_seam_band_enrichment.csv
  obs077c_pinch_cohort_path_family_enrichment.csv
  obs077c_pinch_cohort_outcome_group_enrichment.csv

  obs077c_report.md

Scientific guardrail
--------------------
This is not a direct coupled_outcome_group test unless that label is supplied
as a path/window-joinable column. Current bridge uses available OBS-051 fields:
  coupling_class
  seam_band
  mean_lambda_local
  mean_delta_d
  bounded_share

If coupling_class is degenerate, the script reports it as such and treats the
scientific read as numeric local-divergence / boundedness contrast.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


WINDOW_NUMERIC_CANDIDATES = [
    "mean_lambda_local",
    "median_lambda_local",
    "max_lambda_local",
    "mean_delta_d",
    "bounded_share",
    "mean_d_start",
    "mean_d_end",
    "n_neighbors",
]

WINDOW_CATEGORICAL_CANDIDATES = [
    "coupling_class",
    "seam_band",
    "path_family",
    "outcome_group",
]


# -----------------------------
# General utilities
# -----------------------------


def ensure_outdir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")
    return pd.read_csv(p)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def present_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_mean(s: pd.Series) -> float:
    x = safe_numeric(s)
    if x.notna().sum() == 0:
        return float("nan")
    return float(x.mean())


def safe_median(s: pd.Series) -> float:
    x = safe_numeric(s)
    if x.notna().sum() == 0:
        return float("nan")
    return float(x.median())


def safe_std(s: pd.Series) -> float:
    x = safe_numeric(s)
    if x.notna().sum() <= 1:
        return float("nan")
    return float(x.std(ddof=1))


def safe_min(s: pd.Series) -> float:
    x = safe_numeric(s)
    if x.notna().sum() == 0:
        return float("nan")
    return float(x.min())


def safe_max(s: pd.Series) -> float:
    x = safe_numeric(s)
    if x.notna().sum() == 0:
        return float("nan")
    return float(x.max())


def log2_enrichment(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(math.log(value, 2))


def z_score(value: float, mean: float, std: float) -> float:
    if not np.isfinite(value) or not np.isfinite(mean) or not np.isfinite(std) or abs(std) < EPS:
        return float("nan")
    return float((value - mean) / std)


def df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def sort_existing(df: pd.DataFrame, by: list[str], ascending=True) -> pd.DataFrame:
    keys = [c for c in by if c in df.columns]
    if not keys:
        return df
    return df.sort_values(keys, ascending=ascending).reset_index(drop=True)


# -----------------------------
# Input preparation
# -----------------------------


def validate_membership(df: pd.DataFrame) -> None:
    required = {"path_id", "scale_index", "object"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Membership missing required columns: {sorted(missing)}")


def validate_windows(df: pd.DataFrame) -> None:
    required = {"path_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Windows missing required columns: {sorted(missing)}")


def prepare_membership(df: pd.DataFrame) -> pd.DataFrame:
    validate_membership(df)
    out = df.copy()
    out["path_id"] = out["path_id"].astype(str)
    out["object"] = out["object"].astype(str)
    out["scale_index"] = pd.to_numeric(out["scale_index"], errors="coerce").astype("Int64")

    keep = ["path_id", "scale_index", "object"]
    extra = [c for c in ["path_family", "outcome_group", "seam_class"] if c in out.columns]

    out = out[keep + extra].drop_duplicates(subset=keep)
    return out


def prepare_windows(df: pd.DataFrame) -> pd.DataFrame:
    validate_windows(df)
    out = df.copy()
    out["path_id"] = out["path_id"].astype(str)

    for c in WINDOW_NUMERIC_CANDIDATES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in WINDOW_CATEGORICAL_CANDIDATES:
        if c in out.columns:
            out[c] = out[c].astype(str).fillna("NA")

    return out


# -----------------------------
# Manifest / audit
# -----------------------------


def write_manifest(
    outdir: Path,
    case: str,
    membership_path: str,
    windows_path: str,
    pinch_path: str | None,
    membership: pd.DataFrame,
    windows: pd.DataFrame,
    pinch: pd.DataFrame | None,
) -> None:
    rows = [
        {
            "artifact": "case",
            "status": "ok",
            "rows": 1,
            "details": case,
            "path": "",
        },
        {
            "artifact": "membership",
            "status": "ok",
            "rows": len(membership),
            "details": f"cols={len(membership.columns)}",
            "path": membership_path,
        },
        {
            "artifact": "windows",
            "status": "ok",
            "rows": len(windows),
            "details": f"cols={len(windows.columns)}",
            "path": windows_path,
        },
        {
            "artifact": "pinch_candidates",
            "status": "ok" if pinch_path and pinch is not None else ("not_provided" if not pinch_path else "missing"),
            "rows": 0 if pinch is None else len(pinch),
            "details": "",
            "path": pinch_path or "",
        },
    ]
    write_csv(pd.DataFrame(rows), outdir / "obs077c_input_manifest.csv")


def write_join_audit(
    outdir: Path,
    membership: pd.DataFrame,
    windows: pd.DataFrame,
    support_joined: pd.DataFrame,
    pinch_membership: pd.DataFrame,
    pinch_summary: pd.DataFrame,
) -> pd.DataFrame:
    mem_paths = set(membership["path_id"].astype(str))
    win_paths = set(windows["path_id"].astype(str))
    overlap = mem_paths & win_paths

    rows = [
        {
            "audit": "membership_paths",
            "value": len(mem_paths),
            "details": "",
        },
        {
            "audit": "window_paths",
            "value": len(win_paths),
            "details": "",
        },
        {
            "audit": "path_id_overlap",
            "value": len(overlap),
            "details": f"membership_only={len(mem_paths - win_paths)};windows_only={len(win_paths - mem_paths)}",
        },
        {
            "audit": "support_membership_rows",
            "value": len(membership),
            "details": "",
        },
        {
            "audit": "window_rows",
            "value": len(windows),
            "details": "",
        },
        {
            "audit": "support_joined_rows",
            "value": len(support_joined),
            "details": f"support_joined_paths={support_joined['path_id'].nunique() if not support_joined.empty else 0}",
        },
        {
            "audit": "pinch_cohort_membership_rows",
            "value": len(pinch_membership),
            "details": f"paths={pinch_membership['path_id'].nunique() if not pinch_membership.empty else 0}",
        },
        {
            "audit": "pinch_summary_rows",
            "value": len(pinch_summary),
            "details": "",
        },
        {
            "audit": "numeric_cols_used",
            "value": len(present_cols(windows, WINDOW_NUMERIC_CANDIDATES)),
            "details": ",".join(present_cols(windows, WINDOW_NUMERIC_CANDIDATES)),
        },
        {
            "audit": "categorical_cols_used",
            "value": len(present_cols(windows, WINDOW_CATEGORICAL_CANDIDATES)),
            "details": ",".join(present_cols(windows, WINDOW_CATEGORICAL_CANDIDATES)),
        },
    ]

    audit = pd.DataFrame(rows)
    write_csv(audit, outdir / "obs077c_join_audit.csv")
    return audit


# -----------------------------
# Degeneracy and baselines
# -----------------------------


def label_degeneracy_audit(windows: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    total_paths = windows["path_id"].nunique()
    total_windows = len(windows)

    for col in label_cols:
        g = windows.groupby(col, dropna=False)
        n_values = int(g.ngroups)
        top_value = None
        top_windows = 0
        top_paths = 0

        for val, gg in g:
            nwin = len(gg)
            if nwin > top_windows:
                top_value = val
                top_windows = nwin
                top_paths = gg["path_id"].nunique()

        top_window_share = top_windows / max(total_windows, 1)
        top_path_share = top_paths / max(total_paths, 1)

        rows.append(
            {
                "label_col": col,
                "n_values": n_values,
                "top_value": top_value,
                "top_windows": int(top_windows),
                "top_window_share": float(top_window_share),
                "top_paths": int(top_paths),
                "top_path_share": float(top_path_share),
                "is_degenerate_window": bool(n_values <= 1 or top_window_share >= 0.999999),
                "is_degenerate_path": bool(n_values <= 1 or top_path_share >= 0.999999),
                "use_in_primary_report": bool(not (n_values <= 1 or top_path_share >= 0.999999)),
            }
        )

    return pd.DataFrame(rows)


def global_numeric_baseline(windows: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        rows.append(
            {
                "metric": col,
                "global_n_windows": int(safe_numeric(windows[col]).notna().sum()),
                "global_mean": safe_mean(windows[col]),
                "global_median": safe_median(windows[col]),
                "global_std": safe_std(windows[col]),
                "global_min": safe_min(windows[col]),
                "global_max": safe_max(windows[col]),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Summary and enrichment
# -----------------------------


def join_membership_windows(membership: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    return membership.merge(
        windows,
        on="path_id",
        how="inner",
        suffixes=("_support", "_window"),
    )


def summarize_group_windows(joined: pd.DataFrame, group_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    if joined.empty:
        return pd.DataFrame()

    for keys, g in joined.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["n_paths"] = int(g["path_id"].nunique())
        row["n_windows"] = int(len(g))

        if "segment_id" in g.columns:
            row["n_segments"] = int(g["segment_id"].nunique())

        for col in numeric_cols:
            row[f"{col}_mean"] = safe_mean(g[col])
            row[f"{col}_median"] = safe_median(g[col])
            row[f"{col}_std"] = safe_std(g[col])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def add_numeric_contrasts(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    group_cols: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    if summary.empty or baseline.empty:
        return pd.DataFrame()

    base_by_metric = baseline.set_index("metric").to_dict(orient="index")
    rows = []

    for _, row in summary.iterrows():
        out = {c: row[c] for c in group_cols if c in row.index}
        out["n_paths"] = int(row.get("n_paths", 0))
        out["n_windows"] = int(row.get("n_windows", 0))

        for metric in numeric_cols:
            mean_col = f"{metric}_mean"
            if mean_col not in summary.columns:
                continue

            val = float(row[mean_col]) if pd.notna(row[mean_col]) else float("nan")
            b = base_by_metric.get(metric, {})
            gmean = float(b.get("global_mean", np.nan))
            gstd = float(b.get("global_std", np.nan))
            gmedian = float(b.get("global_median", np.nan))

            out[f"{metric}_mean"] = val
            out[f"{metric}_global_mean"] = gmean
            out[f"{metric}_delta"] = val - gmean if np.isfinite(val) and np.isfinite(gmean) else float("nan")
            out[f"{metric}_z"] = z_score(val, gmean, gstd)
            out[f"{metric}_global_median"] = gmedian

        rows.append(out)

    result = pd.DataFrame(rows)

    # Composite scores for useful sorting.
    for metric in ["mean_lambda_local", "mean_delta_d"]:
        zc = f"{metric}_z"
        if zc not in result.columns:
            result[zc] = np.nan

    if "bounded_share_z" not in result.columns:
        result["bounded_share_z"] = np.nan

    result["divergence_z_sum"] = (
        pd.to_numeric(result.get("mean_lambda_local_z"), errors="coerce").fillna(0)
        + pd.to_numeric(result.get("mean_delta_d_z"), errors="coerce").fillna(0)
        - pd.to_numeric(result.get("bounded_share_z"), errors="coerce").fillna(0)
    )

    return result


def global_label_distribution(windows: pd.DataFrame, label_col: str) -> pd.DataFrame:
    total_windows = len(windows)
    total_paths = windows["path_id"].nunique()

    rows = []
    for val, g in windows.groupby(label_col, dropna=False):
        rows.append(
            {
                "label_col": label_col,
                "label_value": val,
                "global_windows": int(len(g)),
                "global_window_share": float(len(g) / max(total_windows, 1)),
                "global_paths": int(g["path_id"].nunique()),
                "global_path_share": float(g["path_id"].nunique() / max(total_paths, 1)),
            }
        )

    return pd.DataFrame(rows)


def enrichment_by_group(
    joined: pd.DataFrame,
    windows: pd.DataFrame,
    group_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    if joined.empty or label_col not in joined.columns or label_col not in windows.columns:
        return pd.DataFrame()

    global_dist = global_label_distribution(windows, label_col)
    rows = []

    for keys, g in joined.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        denom_windows = len(g)
        denom_paths = g["path_id"].nunique()

        for val, gg in g.groupby(label_col, dropna=False):
            n_windows = len(gg)
            n_paths = gg["path_id"].nunique()

            gd = global_dist[global_dist["label_value"].astype(str) == str(val)]
            if gd.empty:
                global_window_share = np.nan
                global_path_share = np.nan
            else:
                global_window_share = float(gd["global_window_share"].iloc[0])
                global_path_share = float(gd["global_path_share"].iloc[0])

            window_share = n_windows / max(denom_windows, 1)
            path_share = n_paths / max(denom_paths, 1)

            window_enrich = window_share / max(global_window_share, EPS)
            path_enrich = path_share / max(global_path_share, EPS)

            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "label_col": label_col,
                    "label_value": val,
                    "n_windows": int(n_windows),
                    "denom_windows": int(denom_windows),
                    "window_share": float(window_share),
                    "global_window_share": global_window_share,
                    "window_enrichment": float(window_enrich),
                    "window_log2_enrichment": log2_enrichment(window_enrich),
                    "n_paths": int(n_paths),
                    "denom_paths": int(denom_paths),
                    "path_share": float(path_share),
                    "global_path_share": global_path_share,
                    "path_enrichment": float(path_enrich),
                    "path_log2_enrichment": log2_enrichment(path_enrich),
                }
            )
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(
        group_cols + ["path_log2_enrichment"],
        ascending=[True] * len(group_cols) + [False],
    ).reset_index(drop=True)


# -----------------------------
# Pinch cohorts
# -----------------------------


def prepare_pinch_candidates(pinch: pd.DataFrame, top_k: int) -> pd.DataFrame:
    required = {"object", "scale_index_from", "scale_index_to"}
    missing = required - set(pinch.columns)
    if missing:
        raise ValueError(f"Pinch candidate file missing required columns: {sorted(missing)}")

    out = pinch.copy()

    if "candidate_rank" not in out.columns:
        if "pinch_score_total" in out.columns:
            out = out.sort_values("pinch_score_total", ascending=False).reset_index(drop=True)
        out["candidate_rank"] = np.arange(1, len(out) + 1)

    out["candidate_rank"] = pd.to_numeric(out["candidate_rank"], errors="coerce").astype("Int64")
    out["scale_index_from"] = pd.to_numeric(out["scale_index_from"], errors="coerce").astype("Int64")
    out["scale_index_to"] = pd.to_numeric(out["scale_index_to"], errors="coerce").astype("Int64")

    return out.sort_values("candidate_rank").head(top_k).copy()


def build_candidate_cohort_membership(membership: pd.DataFrame, pinch: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, cand in pinch.iterrows():
        obj = str(cand["object"])
        s0 = int(cand["scale_index_from"])
        s1 = int(cand["scale_index_to"])
        rank = int(cand["candidate_rank"])

        before = membership[(membership["object"] == obj) & (membership["scale_index"] == s0)]
        after = membership[(membership["object"] == obj) & (membership["scale_index"] == s1)]

        before_paths = set(before["path_id"].astype(str))
        after_paths = set(after["path_id"].astype(str))

        cohorts = {
            "before": before_paths,
            "after": after_paths,
            "entered": after_paths - before_paths,
            "exited": before_paths - after_paths,
            "persisted": before_paths & after_paths,
            "union": before_paths | after_paths,
        }

        for cohort, paths in cohorts.items():
            for pid in paths:
                row = {
                    "candidate_rank": rank,
                    "object": obj,
                    "scale_index_from": s0,
                    "scale_index_to": s1,
                    "cohort": cohort,
                    "path_id": pid,
                }

                for optional in [
                    "dominant_family",
                    "dominant_reason",
                    "pinch_score_total",
                    "support_score",
                    "overlap_score",
                    "shape_score",
                    "id_score",
                ]:
                    if optional in cand.index:
                        row[optional] = cand[optional]

                rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Output builders
# -----------------------------


def write_enrichment_outputs(
    outdir: Path,
    prefix: str,
    joined: pd.DataFrame,
    windows: pd.DataFrame,
    group_cols: list[str],
    label_cols: list[str],
) -> dict[str, pd.DataFrame]:
    outputs = {}
    for label_col in label_cols:
        df = enrichment_by_group(joined, windows, group_cols, label_col)
        name = f"{prefix}_{label_col}_enrichment"
        outputs[name] = df
        write_csv(df, outdir / f"{name}.csv")
    return outputs


def build_support_outputs(
    outdir: Path,
    membership: pd.DataFrame,
    windows: pd.DataFrame,
    numeric_cols: list[str],
    label_cols: list[str],
    baseline: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    joined = join_membership_windows(membership, windows)

    group_cols = ["scale_index", "object"]
    summary = summarize_group_windows(joined, group_cols, numeric_cols)
    contrast = add_numeric_contrasts(summary, baseline, group_cols, numeric_cols)

    write_csv(summary, outdir / "obs077c_support_window_coupling_summary.csv")
    write_csv(contrast, outdir / "obs077c_support_numeric_contrast.csv")

    enrichments = write_enrichment_outputs(
        outdir=outdir,
        prefix="obs077c_support",
        joined=joined,
        windows=windows,
        group_cols=group_cols,
        label_cols=label_cols,
    )

    return {
        "support_joined": joined,
        "support_summary": summary,
        "support_numeric_contrast": contrast,
        **enrichments,
    }


def build_pinch_outputs(
    outdir: Path,
    membership: pd.DataFrame,
    windows: pd.DataFrame,
    pinch_raw: pd.DataFrame | None,
    numeric_cols: list[str],
    label_cols: list[str],
    baseline: pd.DataFrame,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    empty = {
        "pinch_membership": pd.DataFrame(),
        "pinch_joined": pd.DataFrame(),
        "pinch_summary": pd.DataFrame(),
        "pinch_numeric_contrast": pd.DataFrame(),
        "top_pinch_numeric_contrast": pd.DataFrame(),
    }

    if pinch_raw is None or pinch_raw.empty:
        write_csv(pd.DataFrame(), outdir / "obs077c_pinch_cohort_window_coupling_summary.csv")
        write_csv(pd.DataFrame(), outdir / "obs077c_pinch_cohort_numeric_contrast.csv")
        write_csv(pd.DataFrame(), outdir / "obs077c_top_pinch_numeric_contrast.csv")
        for label_col in label_cols:
            write_csv(pd.DataFrame(), outdir / f"obs077c_pinch_cohort_{label_col}_enrichment.csv")
        return empty

    pinch = prepare_pinch_candidates(pinch_raw, top_k)
    cohort_membership = build_candidate_cohort_membership(membership, pinch)

    if cohort_membership.empty:
        write_csv(pd.DataFrame(), outdir / "obs077c_pinch_cohort_window_coupling_summary.csv")
        write_csv(pd.DataFrame(), outdir / "obs077c_pinch_cohort_numeric_contrast.csv")
        write_csv(pd.DataFrame(), outdir / "obs077c_top_pinch_numeric_contrast.csv")
        for label_col in label_cols:
            write_csv(pd.DataFrame(), outdir / f"obs077c_pinch_cohort_{label_col}_enrichment.csv")
        return empty | {"pinch_membership": cohort_membership}

    joined = cohort_membership.merge(
        windows,
        on="path_id",
        how="inner",
        suffixes=("_cohort", "_window"),
    )

    group_cols = [
        "candidate_rank",
        "object",
        "scale_index_from",
        "scale_index_to",
        "cohort",
    ]
    if "dominant_family" in joined.columns:
        group_cols.append("dominant_family")

    summary = summarize_group_windows(joined, group_cols, numeric_cols)
    contrast = add_numeric_contrasts(summary, baseline, group_cols, numeric_cols)

    top_pinch = pd.DataFrame()
    if not contrast.empty and "candidate_rank" in contrast.columns:
        top_pinch = contrast[contrast["candidate_rank"] == 1].copy()

    write_csv(summary, outdir / "obs077c_pinch_cohort_window_coupling_summary.csv")
    write_csv(contrast, outdir / "obs077c_pinch_cohort_numeric_contrast.csv")
    write_csv(top_pinch, outdir / "obs077c_top_pinch_numeric_contrast.csv")

    enrichments = write_enrichment_outputs(
        outdir=outdir,
        prefix="obs077c_pinch_cohort",
        joined=joined,
        windows=windows,
        group_cols=group_cols,
        label_cols=label_cols,
    )

    return {
        "pinch_membership": cohort_membership,
        "pinch_joined": joined,
        "pinch_summary": summary,
        "pinch_numeric_contrast": contrast,
        "top_pinch_numeric_contrast": top_pinch,
        **enrichments,
    }


# -----------------------------
# Report helpers
# -----------------------------


def filter_reportable_enrichment(
    df: pd.DataFrame,
    min_paths: int,
    sort_col: str = "path_log2_enrichment",
    n: int = 15,
) -> pd.DataFrame:
    if df is None or df.empty or sort_col not in df.columns:
        return pd.DataFrame()

    q = df.copy()
    if "denom_paths" in q.columns:
        q = q[pd.to_numeric(q["denom_paths"], errors="coerce") >= min_paths]
    if q.empty:
        return pd.DataFrame()

    q = q.sort_values(sort_col, ascending=False)
    return q.head(n).reset_index(drop=True)


def filter_reportable_numeric(
    df: pd.DataFrame,
    min_paths: int,
    sort_col: str = "divergence_z_sum",
    n: int = 18,
) -> pd.DataFrame:
    if df is None or df.empty or sort_col not in df.columns:
        return pd.DataFrame()

    q = df.copy()
    if "n_paths" in q.columns:
        q = q[pd.to_numeric(q["n_paths"], errors="coerce") >= min_paths]
    if q.empty:
        return pd.DataFrame()

    q = q.sort_values(sort_col, ascending=False)
    return q.head(n).reset_index(drop=True)


def keep_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].copy()


def nondegenerate_label_cols(degeneracy: pd.DataFrame) -> list[str]:
    if degeneracy.empty:
        return []
    q = degeneracy[degeneracy["use_in_primary_report"] == True]
    return list(q["label_col"])


def write_report(
    outdir: Path,
    case: str,
    membership_path: str,
    windows_path: str,
    pinch_path: str | None,
    audit: pd.DataFrame,
    degeneracy: pd.DataFrame,
    baseline: pd.DataFrame,
    support_outputs: dict[str, pd.DataFrame],
    pinch_outputs: dict[str, pd.DataFrame],
    min_report_paths: int,
) -> None:
    reportable_labels = nondegenerate_label_cols(degeneracy)
    degenerate_labels = list(degeneracy[degeneracy["use_in_primary_report"] == False]["label_col"]) if not degeneracy.empty else []

    support_numeric = support_outputs.get("support_numeric_contrast", pd.DataFrame())
    pinch_numeric = pinch_outputs.get("pinch_numeric_contrast", pd.DataFrame())
    top_pinch_numeric = pinch_outputs.get("top_pinch_numeric_contrast", pd.DataFrame())

    numeric_cols_report = [
        "scale_index",
        "object",
        "n_paths",
        "n_windows",
        "mean_lambda_local_mean",
        "mean_lambda_local_z",
        "mean_delta_d_mean",
        "mean_delta_d_z",
        "bounded_share_mean",
        "bounded_share_z",
        "divergence_z_sum",
    ]

    pinch_numeric_cols_report = [
        "candidate_rank",
        "object",
        "scale_index_from",
        "scale_index_to",
        "cohort",
        "dominant_family",
        "n_paths",
        "n_windows",
        "mean_lambda_local_mean",
        "mean_lambda_local_z",
        "mean_delta_d_mean",
        "mean_delta_d_z",
        "bounded_share_mean",
        "bounded_share_z",
        "divergence_z_sum",
    ]

    support_numeric_top = keep_cols(
        filter_reportable_numeric(support_numeric, min_report_paths, n=15),
        numeric_cols_report,
    )
    pinch_numeric_top = keep_cols(
        filter_reportable_numeric(pinch_numeric, min_report_paths, n=18),
        pinch_numeric_cols_report,
    )
    top_pinch_numeric_view = keep_cols(
        sort_existing(top_pinch_numeric, ["cohort"], ascending=True),
        pinch_numeric_cols_report,
    )

    enrich_cols_support = [
        "scale_index",
        "object",
        "label_col",
        "label_value",
        "n_paths",
        "denom_paths",
        "path_share",
        "global_path_share",
        "path_enrichment",
        "path_log2_enrichment",
    ]
    enrich_cols_pinch = [
        "candidate_rank",
        "object",
        "scale_index_from",
        "scale_index_to",
        "cohort",
        "dominant_family",
        "label_col",
        "label_value",
        "n_paths",
        "denom_paths",
        "path_share",
        "global_path_share",
        "path_enrichment",
        "path_log2_enrichment",
    ]

    support_enrich_sections = []
    pinch_enrich_sections = []

    for label_col in reportable_labels:
        s_key = f"obs077c_support_{label_col}_enrichment"
        p_key = f"obs077c_pinch_cohort_{label_col}_enrichment"

        support_df = support_outputs.get(s_key, pd.DataFrame())
        pinch_df = pinch_outputs.get(p_key, pd.DataFrame())

        support_enrich_sections.append(
            (
                label_col,
                keep_cols(
                    filter_reportable_enrichment(support_df, min_report_paths, n=12),
                    enrich_cols_support,
                ),
            )
        )
        pinch_enrich_sections.append(
            (
                label_col,
                keep_cols(
                    filter_reportable_enrichment(pinch_df, min_report_paths, n=15),
                    enrich_cols_pinch,
                ),
            )
        )

    baseline_view = keep_cols(
        baseline,
        ["metric", "global_n_windows", "global_mean", "global_median", "global_std", "global_min", "global_max"],
    )
    degeneracy_view = keep_cols(
        degeneracy,
        [
            "label_col",
            "n_values",
            "top_value",
            "top_path_share",
            "top_window_share",
            "is_degenerate_path",
            "use_in_primary_report",
        ],
    )

    lines = []
    lines.append("# OBS-077c — Window-Local Divergence Bridge")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "OBS-077c joins OBS-077b scale-space support occupancy and OBS-077a "
        "pinch cohorts to OBS-051 window-local divergence / boundedness diagnostics."
    )
    lines.append("")
    lines.append(
        "v2 treats categorical coupling labels as provenance fields and detects "
        "when they are degenerate. The primary scientific read is numeric local "
        "divergence and boundedness contrast."
    )
    lines.append("")
    lines.append("## Case")
    lines.append("")
    lines.append(f"`{case}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("| artifact | path |")
    lines.append("| --- | --- |")
    lines.append(f"| membership | `{membership_path}` |")
    lines.append(f"| windows | `{windows_path}` |")
    lines.append(f"| pinch_candidates | `{pinch_path or ''}` |")
    lines.append("")
    lines.append("## Join audit")
    lines.append("")
    lines.append(df_to_markdown(audit))
    lines.append("")
    lines.append("## Label degeneracy audit")
    lines.append("")
    lines.append(df_to_markdown(degeneracy_view))
    lines.append("")
    if degenerate_labels:
        lines.append("Degenerate labels excluded from primary enrichment interpretation:")
        lines.append("")
        lines.append("```text")
        for label in degenerate_labels:
            lines.append(str(label))
        lines.append("```")
        lines.append("")
    lines.append("## Global numeric baseline")
    lines.append("")
    lines.append(df_to_markdown(baseline_view))
    lines.append("")
    lines.append("## Strongest support-level numeric contrasts")
    lines.append("")
    lines.append(df_to_markdown(support_numeric_top))
    lines.append("")
    lines.append("## Top-pinch focused numeric contrast")
    lines.append("")
    lines.append(df_to_markdown(top_pinch_numeric_view))
    lines.append("")
    lines.append("## Strongest pinch-cohort numeric contrasts")
    lines.append("")
    lines.append(df_to_markdown(pinch_numeric_top))
    lines.append("")

    lines.append("## Reportable support-level categorical enrichments")
    lines.append("")
    if not support_enrich_sections:
        lines.append("_No non-degenerate categorical labels available for primary support-level enrichment._")
        lines.append("")
    else:
        for label_col, table in support_enrich_sections:
            lines.append(f"### {label_col}")
            lines.append("")
            lines.append(df_to_markdown(table))
            lines.append("")

    lines.append("## Reportable pinch-cohort categorical enrichments")
    lines.append("")
    if not pinch_enrich_sections:
        lines.append("_No non-degenerate categorical labels available for primary pinch-cohort enrichment._")
        lines.append("")
    else:
        for label_col, table in pinch_enrich_sections:
            lines.append(f"### {label_col}")
            lines.append("")
            lines.append(df_to_markdown(table))
            lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("Primary numeric fields:")
    lines.append("")
    lines.append("```text")
    lines.append("mean_lambda_local:")
    lines.append("  local divergence / separation rate proxy")
    lines.append("")
    lines.append("mean_delta_d:")
    lines.append("  local change in distance over coupled windows")
    lines.append("")
    lines.append("bounded_share:")
    lines.append("  share of windows remaining bounded under the local divergence criterion")
    lines.append("")
    lines.append("divergence_z_sum:")
    lines.append("  mean_lambda_local_z + mean_delta_d_z - bounded_share_z")
    lines.append("```")
    lines.append("")
    lines.append("Guardrail:")
    lines.append("")
    lines.append("```text")
    lines.append("OBS-077c v2 is not a direct coupled_outcome_group test unless that")
    lines.append("label is supplied as a path/window-joinable column.")
    lines.append("")
    lines.append("If coupling_class is degenerate, OBS-077c should be read as a")
    lines.append("local-divergence / boundedness bridge over coupled windows.")
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("obs077c_input_manifest.csv")
    lines.append("obs077c_join_audit.csv")
    lines.append("obs077c_label_degeneracy_audit.csv")
    lines.append("obs077c_global_numeric_baseline.csv")
    lines.append("obs077c_support_window_coupling_summary.csv")
    lines.append("obs077c_support_numeric_contrast.csv")
    lines.append("obs077c_support_coupling_class_enrichment.csv")
    lines.append("obs077c_support_seam_band_enrichment.csv")
    lines.append("obs077c_support_path_family_enrichment.csv")
    lines.append("obs077c_support_outcome_group_enrichment.csv")
    lines.append("obs077c_pinch_cohort_window_coupling_summary.csv")
    lines.append("obs077c_pinch_cohort_numeric_contrast.csv")
    lines.append("obs077c_top_pinch_numeric_contrast.csv")
    lines.append("obs077c_pinch_cohort_coupling_class_enrichment.csv")
    lines.append("obs077c_pinch_cohort_seam_band_enrichment.csv")
    lines.append("obs077c_pinch_cohort_path_family_enrichment.csv")
    lines.append("obs077c_pinch_cohort_outcome_group_enrichment.csv")
    lines.append("obs077c_report.md")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("END OBS-077c v2")

    (outdir / "obs077c_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="OBS-077c v2 bridge from scale-space supports/pinch cohorts to OBS-051 window-local divergence."
    )
    ap.add_argument("--case", required=True, help="Case/corpus label, e.g. C, Cp2, Cp3.")
    ap.add_argument(
        "--membership",
        required=True,
        help="OBS-077b path-weighted path-object membership CSV.",
    )
    ap.add_argument(
        "--windows",
        required=True,
        help="OBS-051 window divergence CSV, preferably obs051_window_divergence_all.csv.",
    )
    ap.add_argument(
        "--pinch-candidates",
        default=None,
        help="Optional OBS-077a pinch candidates CSV.",
    )
    ap.add_argument(
        "--top-k-pinch",
        type=int,
        default=10,
        help="Number of top pinch candidates to include.",
    )
    ap.add_argument(
        "--min-report-paths",
        type=int,
        default=25,
        help="Minimum denominator path count for report tables.",
    )
    ap.add_argument("--outdir", required=True, help="Output directory.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    membership_raw = read_csv(args.membership)
    windows_raw = read_csv(args.windows)
    pinch_raw = read_csv(args.pinch_candidates) if args.pinch_candidates else None

    membership = prepare_membership(membership_raw)
    windows = prepare_windows(windows_raw)

    numeric_cols = present_cols(windows, WINDOW_NUMERIC_CANDIDATES)
    label_cols = present_cols(windows, WINDOW_CATEGORICAL_CANDIDATES)

    write_manifest(
        outdir=outdir,
        case=args.case,
        membership_path=args.membership,
        windows_path=args.windows,
        pinch_path=args.pinch_candidates,
        membership=membership,
        windows=windows,
        pinch=pinch_raw,
    )

    degeneracy = label_degeneracy_audit(windows, label_cols)
    baseline = global_numeric_baseline(windows, numeric_cols)

    write_csv(degeneracy, outdir / "obs077c_label_degeneracy_audit.csv")
    write_csv(baseline, outdir / "obs077c_global_numeric_baseline.csv")

    support_outputs = build_support_outputs(
        outdir=outdir,
        membership=membership,
        windows=windows,
        numeric_cols=numeric_cols,
        label_cols=label_cols,
        baseline=baseline,
    )

    pinch_outputs = build_pinch_outputs(
        outdir=outdir,
        membership=membership,
        windows=windows,
        pinch_raw=pinch_raw,
        numeric_cols=numeric_cols,
        label_cols=label_cols,
        baseline=baseline,
        top_k=args.top_k_pinch,
    )

    audit = write_join_audit(
        outdir=outdir,
        membership=membership,
        windows=windows,
        support_joined=support_outputs.get("support_joined", pd.DataFrame()),
        pinch_membership=pinch_outputs.get("pinch_membership", pd.DataFrame()),
        pinch_summary=pinch_outputs.get("pinch_summary", pd.DataFrame()),
    )

    write_report(
        outdir=outdir,
        case=args.case,
        membership_path=args.membership,
        windows_path=args.windows,
        pinch_path=args.pinch_candidates,
        audit=audit,
        degeneracy=degeneracy,
        baseline=baseline,
        support_outputs=support_outputs,
        pinch_outputs=pinch_outputs,
        min_report_paths=args.min_report_paths,
    )

    print(f"[OBS-077c v2] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
