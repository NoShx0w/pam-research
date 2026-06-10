#!/usr/bin/env python3
"""
obs077b_path_label_projection.py

OBS-077b — Path-label projection onto scale-space supports

Purpose
-------
OBS-077b projects instrument-level path labels onto OBS-076/077 scale-space
structural supports.

It asks:

    Which path-level labels occupy each structural object across diffusion scale?

    Do detected OBS-077a pinch-point candidates correspond to specific
    path-family / outcome / seam-class enrichments?

This script does NOT use generated text yet.
It uses existing path-level instrument labels.

Inputs
------
Required:
    --path-nodes
        path_nodes_for_family.csv
        Required columns:
            path_id, step, node_id

    --path-labels
        structural_coupling_path_summary.csv
        Required column:
            path_id
        Typical label columns:
            path_family, outcome_group, seam_class

    --objects
        obs076c_object_membership_by_scale.csv
        Required columns:
            node_id or id, scale_index, object

Optional:
    --pinch
        obs077a_pinch_point_candidates.csv

Outputs
-------
outdir/
    obs077b_input_manifest.csv
    obs077b_join_audit.csv
    obs077b_path_object_membership_step_weighted.csv
    obs077b_path_object_membership_path_weighted.csv
    obs077b_label_enrichment_by_object_scale.csv
    obs077b_numeric_summary_by_object_scale.csv
    obs077b_pinch_label_projection.csv
    obs077b_report.md

Definitions
-----------
step_weighted:
    every path-node-step visit counts.

path_weighted:
    each path_id contributes at most once per object/scale.

Enrichment:
    label share inside object/scale divided by global label share.

Guardrails
----------
- This is path-label projection, not text semantics.
- It does not prove linguistic categories.
- It measures association between existing instrument labels and scale-space
  structural supports.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class Config:
    case: str
    path_nodes: Path
    path_labels: Path
    objects: Path
    pinch: Path | None
    outdir: Path
    path_id_col: str
    node_id_col: str
    object_node_col: str | None
    label_cols: list[str]
    numeric_cols: list[str]
    min_count: int
    top_n: int


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="OBS-077b path-label projection onto scale-space supports."
    )

    p.add_argument("--case", default="case")
    p.add_argument("--path-nodes", required=True, type=Path)
    p.add_argument("--path-labels", required=True, type=Path)
    p.add_argument("--objects", required=True, type=Path)
    p.add_argument("--pinch", default=None, type=Path)
    p.add_argument("--outdir", required=True, type=Path)

    p.add_argument("--path-id-col", default="path_id")
    p.add_argument("--node-id-col", default="node_id")
    p.add_argument(
        "--object-node-col",
        default=None,
        help="Node ID column in object membership table. If omitted, infer node_id or id.",
    )

    p.add_argument(
        "--label-cols",
        default="path_family,outcome_group,seam_class",
        help="Comma-separated categorical path-label columns.",
    )
    p.add_argument(
        "--numeric-cols",
        default=(
            "n_escalation_windows,n_compression,n_graze,n_dissipation,"
            "near_fraction,mid_fraction,far_fraction,mean_criticality,"
            "max_criticality,mean_unsigned_obstruction,max_unsigned_obstruction,"
            "mean_absolute_holonomy,mean_angle_jump_deg,max_angle_jump_deg,"
            "n_sector_changes,core_windows,near_windows,far_windows"
        ),
        help="Comma-separated numeric path descriptor columns.",
    )

    p.add_argument(
        "--min-count",
        type=int,
        default=25,
        help="Minimum denominator count for enrichment rows emphasized in report.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Top rows to show in report sections.",
    )

    args = p.parse_args()

    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1")
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    return Config(
        case=args.case,
        path_nodes=args.path_nodes,
        path_labels=args.path_labels,
        objects=args.objects,
        pinch=args.pinch,
        outdir=args.outdir,
        path_id_col=args.path_id_col,
        node_id_col=args.node_id_col,
        object_node_col=args.object_node_col,
        label_cols=parse_csv_list(args.label_cols),
        numeric_cols=parse_csv_list(args.numeric_cols),
        min_count=args.min_count,
        top_n=args.top_n,
    )


def normalize_node_id_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()

    # Strip existing PAM prefix if present.
    stripped = raw.str.replace(r"^node_", "", regex=True)

    numeric = pd.to_numeric(stripped, errors="coerce")

    out = raw.copy()
    mask = numeric.notna()

    # Canonical OBS node id form.
    out.loc[mask] = "node_" + numeric.loc[mask].astype(int).astype(str).str.zfill(4)

    return out


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def infer_object_node_col(objects: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in objects.columns:
            raise ValueError(f"requested object node column not found: {requested}")
        return requested

    for candidate in ["node_id", "id"]:
        if candidate in objects.columns:
            return candidate

    raise ValueError("Could not infer object node column; expected node_id or id")


def read_path_nodes(cfg: Config) -> pd.DataFrame:
    if not cfg.path_nodes.exists():
        raise FileNotFoundError(cfg.path_nodes)

    df = pd.read_csv(cfg.path_nodes)
    require_columns(df, [cfg.path_id_col, cfg.node_id_col], "path_nodes")

    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=int)

    out_cols = [cfg.path_id_col, "step", cfg.node_id_col]
    optional = [c for c in ["r", "alpha", "mds1", "mds2"] if c in df.columns]
    out_cols.extend(optional)

    df = df[out_cols].copy()
    df[cfg.path_id_col] = df[cfg.path_id_col].astype(str)
    df[cfg.node_id_col] = df[cfg.node_id_col].astype(str)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")

    return df


def read_path_labels(cfg: Config) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not cfg.path_labels.exists():
        raise FileNotFoundError(cfg.path_labels)

    df = pd.read_csv(cfg.path_labels)
    require_columns(df, [cfg.path_id_col], "path_labels")

    df[cfg.path_id_col] = df[cfg.path_id_col].astype(str)

    label_cols = [c for c in cfg.label_cols if c in df.columns]
    numeric_cols = [c for c in cfg.numeric_cols if c in df.columns]

    for col in label_cols:
        df[col] = df[col].astype("string").fillna("__NA__").astype(str)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [cfg.path_id_col] + label_cols + numeric_cols
    df = df[keep].drop_duplicates(subset=[cfg.path_id_col]).copy()

    return df, label_cols, numeric_cols


def read_objects(cfg: Config) -> tuple[pd.DataFrame, str]:
    if not cfg.objects.exists():
        raise FileNotFoundError(cfg.objects)

    df = pd.read_csv(cfg.objects)
    require_columns(df, ["scale_index", "object"], "objects")

    obj_node_col = infer_object_node_col(df, cfg.object_node_col)

    df = df[[obj_node_col, "scale_index", "object"]].drop_duplicates().copy()
    df[obj_node_col] = df[obj_node_col].astype(str)
    df["scale_index"] = pd.to_numeric(df["scale_index"], errors="raise").astype(int)
    df["object"] = df["object"].astype(str)

    return df, obj_node_col


def read_pinch(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "not_provided"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = ["object", "scale_index_from", "scale_index_to"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return df, f"missing_columns:{missing}"

    df["object"] = df["object"].astype(str)
    df["scale_index_from"] = pd.to_numeric(df["scale_index_from"], errors="coerce").astype("Int64")
    df["scale_index_to"] = pd.to_numeric(df["scale_index_to"], errors="coerce").astype("Int64")

    return df, "ok"


def build_memberships(
    path_nodes: pd.DataFrame,
    path_labels: pd.DataFrame,
    objects: pd.DataFrame,
    cfg: Config,
    obj_node_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      step_weighted: every path-node-step object occupancy row.
      path_weighted: one row per path_id/object/scale.
    """
    
    nodes = path_nodes.rename(columns={cfg.node_id_col: "_node_id"}).copy()
    objs = objects.rename(columns={obj_node_col: "_node_id"}).copy()

    nodes["_node_id_raw"] = nodes["_node_id"].astype(str)
    objs["_node_id_raw"] = objs["_node_id"].astype(str)

    nodes["_node_id"] = normalize_node_id_series(nodes["_node_id"])
    objs["_node_id"] = normalize_node_id_series(objs["_node_id"])

    joined = nodes.merge(
        objs,
        on="_node_id",
        how="inner",
        validate="many_to_many",
    )

    joined = joined.merge(
        path_labels,
        on=cfg.path_id_col,
        how="left",
        validate="many_to_one",
    )

    joined = joined.rename(columns={"_node_id": cfg.node_id_col})

    path_weighted = (
        joined.drop_duplicates(
            subset=[cfg.path_id_col, "scale_index", "object"]
        )
        .copy()
        .reset_index(drop=True)
    )

    return joined.reset_index(drop=True), path_weighted


def global_label_distribution(
    path_labels: pd.DataFrame,
    label_cols: list[str],
    path_id_col: str,
) -> pd.DataFrame:
    rows = []
    n_total = path_labels[path_id_col].nunique()

    for col in label_cols:
        counts = path_labels[col].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "label_col": col,
                    "label_value": str(value),
                    "global_path_count": int(count),
                    "global_path_share": float(count / max(n_total, 1)),
                }
            )

    return pd.DataFrame(rows)


def compute_label_enrichment(
    membership: pd.DataFrame,
    path_labels: pd.DataFrame,
    label_cols: list[str],
    path_id_col: str,
    weighting: str,
) -> pd.DataFrame:
    global_dist = global_label_distribution(path_labels, label_cols, path_id_col)

    rows = []
    group_cols = ["scale_index", "object"]

    for (scale_index, obj), sub in membership.groupby(group_cols, sort=True):
        if weighting == "path_weighted":
            denom = sub[path_id_col].nunique()
        else:
            denom = len(sub)

        for label_col in label_cols:
            if label_col not in sub.columns:
                continue

            if weighting == "path_weighted":
                tmp = sub[[path_id_col, label_col]].drop_duplicates()
                counts = tmp[label_col].value_counts(dropna=False)
            else:
                counts = sub[label_col].value_counts(dropna=False)

            for label_value, count in counts.items():
                object_share = float(count / max(denom, 1))

                g = global_dist[
                    (global_dist["label_col"] == label_col)
                    & (global_dist["label_value"] == str(label_value))
                ]

                if len(g):
                    global_count = int(g["global_path_count"].iloc[0])
                    global_share = float(g["global_path_share"].iloc[0])
                else:
                    global_count = 0
                    global_share = np.nan

                enrichment = (
                    object_share / (global_share + EPS)
                    if np.isfinite(global_share)
                    else np.nan
                )

                rows.append(
                    {
                        "case": "",
                        "weighting": weighting,
                        "scale_index": int(scale_index),
                        "object": obj,
                        "label_col": label_col,
                        "label_value": str(label_value),
                        "count": int(count),
                        "denominator": int(denom),
                        "object_share": object_share,
                        "global_path_count": global_count,
                        "global_path_share": global_share,
                        "enrichment": float(enrichment),
                        "log2_enrichment": float(np.log2(enrichment + EPS))
                        if np.isfinite(enrichment)
                        else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def compute_numeric_summary(
    membership: pd.DataFrame,
    numeric_cols: list[str],
    path_id_col: str,
    weighting: str,
) -> pd.DataFrame:
    rows = []

    for (scale_index, obj), sub in membership.groupby(["scale_index", "object"], sort=True):
        if weighting == "path_weighted":
            sub = sub.drop_duplicates(subset=[path_id_col, "scale_index", "object"])

        for col in numeric_cols:
            if col not in sub.columns:
                continue

            vals = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "case": "",
                    "weighting": weighting,
                    "scale_index": int(scale_index),
                    "object": obj,
                    "numeric_col": col,
                    "n": int(vals.notna().sum()),
                    "mean": float(vals.mean(skipna=True)),
                    "median": float(vals.median(skipna=True)),
                    "std": float(vals.std(skipna=True)),
                    "min": float(vals.min(skipna=True)),
                    "max": float(vals.max(skipna=True)),
                }
            )

    return pd.DataFrame(rows)


def project_pinch_candidates(
    pinch: pd.DataFrame | None,
    path_weighted: pd.DataFrame,
    path_labels: pd.DataFrame,
    label_cols: list[str],
    path_id_col: str,
    min_count: int,
) -> pd.DataFrame:
    if pinch is None or pinch.empty:
        return pd.DataFrame()

    global_dist = global_label_distribution(path_labels, label_cols, path_id_col)

    rows = []

    for rank, cand in pinch.reset_index(drop=True).iterrows():
        obj = str(cand["object"])
        s_from = cand.get("scale_index_from")
        s_to = cand.get("scale_index_to")

        if pd.isna(s_from) or pd.isna(s_to):
            continue

        s_from = int(s_from)
        s_to = int(s_to)

        # Paths touching object at either side of transition.
        before = path_weighted[
            (path_weighted["object"] == obj)
            & (path_weighted["scale_index"] == s_from)
        ]
        after = path_weighted[
            (path_weighted["object"] == obj)
            & (path_weighted["scale_index"] == s_to)
        ]

        before_paths = set(before[path_id_col].astype(str))
        after_paths = set(after[path_id_col].astype(str))

        sets = {
            "before": before_paths,
            "after": after_paths,
            "entered": after_paths - before_paths,
            "exited": before_paths - after_paths,
            "persisted": before_paths & after_paths,
            "union": before_paths | after_paths,
        }

        for cohort, path_set in sets.items():
            if not path_set:
                denom = 0
                cohort_labels = path_labels.iloc[0:0].copy()
            else:
                cohort_labels = path_labels[path_labels[path_id_col].astype(str).isin(path_set)]
                denom = cohort_labels[path_id_col].nunique()

            for label_col in label_cols:
                if label_col not in cohort_labels.columns:
                    continue

                counts = cohort_labels[label_col].value_counts(dropna=False)
                for label_value, count in counts.items():
                    object_share = float(count / max(denom, 1))

                    g = global_dist[
                        (global_dist["label_col"] == label_col)
                        & (global_dist["label_value"] == str(label_value))
                    ]

                    if len(g):
                        global_share = float(g["global_path_share"].iloc[0])
                    else:
                        global_share = np.nan

                    enrichment = (
                        object_share / (global_share + EPS)
                        if np.isfinite(global_share)
                        else np.nan
                    )

                    rows.append(
                        {
                            "candidate_rank": int(rank + 1),
                            "object": obj,
                            "scale_index_from": s_from,
                            "scale_index_to": s_to,
                            "cohort": cohort,
                            "n_paths": int(denom),
                            "label_col": label_col,
                            "label_value": str(label_value),
                            "count": int(count),
                            "share": object_share,
                            "global_share": global_share,
                            "enrichment": float(enrichment),
                            "log2_enrichment": float(np.log2(enrichment + EPS))
                            if np.isfinite(enrichment)
                            else np.nan,
                            "pinch_score_total": cand.get("pinch_score_total", cand.get("pinch_score", np.nan)),
                            "dominant_family": cand.get("dominant_family", ""),
                            "dominant_reason": cand.get("dominant_reason", ""),
                        }
                    )

            # Include empty cohort row so auditing is explicit.
            if denom == 0:
                rows.append(
                    {
                        "candidate_rank": int(rank + 1),
                        "object": obj,
                        "scale_index_from": s_from,
                        "scale_index_to": s_to,
                        "cohort": cohort,
                        "n_paths": 0,
                        "label_col": "",
                        "label_value": "",
                        "count": 0,
                        "share": np.nan,
                        "global_share": np.nan,
                        "enrichment": np.nan,
                        "log2_enrichment": np.nan,
                        "pinch_score_total": cand.get("pinch_score_total", cand.get("pinch_score", np.nan)),
                        "dominant_family": cand.get("dominant_family", ""),
                        "dominant_reason": cand.get("dominant_reason", ""),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["passes_min_count"] = pd.to_numeric(out["n_paths"], errors="coerce").fillna(0) >= min_count
    return out


def write_audit(
    cfg: Config,
    path_nodes: pd.DataFrame,
    path_labels: pd.DataFrame,
    objects: pd.DataFrame,
    step_weighted: pd.DataFrame,
    path_weighted: pd.DataFrame,
    label_cols: list[str],
    numeric_cols: list[str],
    obj_node_col: str,
    pinch_status: str,
) -> pd.DataFrame:
    node_path_set = set(path_nodes[cfg.path_id_col].astype(str))
    label_path_set = set(path_labels[cfg.path_id_col].astype(str))

    node_id_set = set(normalize_node_id_series(path_nodes[cfg.node_id_col]))
    obj_node_set = set(normalize_node_id_series(objects[obj_node_col]))

    rows = [
        {
            "audit": "path_id_overlap",
            "value": len(node_path_set & label_path_set),
            "details": (
                f"path_nodes_unique={len(node_path_set)};"
                f"path_labels_unique={len(label_path_set)};"
                f"nodes_only={len(node_path_set - label_path_set)};"
                f"labels_only={len(label_path_set - node_path_set)}"
            ),
        },
        {
            "audit": "node_id_overlap",
            "value": len(node_id_set & obj_node_set),
            "details": (
                f"path_nodes_unique_nodes={len(node_id_set)};"
                f"objects_unique_nodes={len(obj_node_set)};"
                f"path_nodes_only={len(node_id_set - obj_node_set)};"
                f"objects_only={len(obj_node_set - node_id_set)}"
            ),
        },
        {
            "audit": "step_weighted_rows",
            "value": len(step_weighted),
            "details": "",
        },
        {
            "audit": "path_weighted_rows",
            "value": len(path_weighted),
            "details": "",
        },
        {
            "audit": "label_cols_used",
            "value": len(label_cols),
            "details": ",".join(label_cols),
        },
        {
            "audit": "numeric_cols_used",
            "value": len(numeric_cols),
            "details": ",".join(numeric_cols),
        },
        {
            "audit": "pinch_status",
            "value": 0,
            "details": pinch_status,
        },
    ]

    return pd.DataFrame(rows)


def fmt(x: object, digits: int = 4) -> str:
    try:
        y = float(x)
    except Exception:
        return ""
    if not np.isfinite(y):
        return ""
    return f"{y:.{digits}g}"


def write_report(
    cfg: Config,
    input_manifest: pd.DataFrame,
    audit: pd.DataFrame,
    enrichment: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    pinch_projection: pd.DataFrame,
    label_cols: list[str],
    numeric_cols: list[str],
) -> None:
    lines = [
        "# OBS-077b — Path-label projection",
        "",
        "## Scope",
        "",
        "OBS-077b projects path-level instrument labels onto OBS-076c scale-space structural supports and OBS-077a pinch-point candidates.",
        "",
        "This is not text-level semantic annotation yet.",
        "",
        "## Inputs",
        "",
        "| artifact | status | details | path |",
        "| --- | --- | --- | --- |",
    ]

    for row in input_manifest.itertuples(index=False):
        lines.append(f"| {row.artifact} | {row.status} | {row.details} | `{row.path}` |")

    lines.extend(
        [
            "",
            "## Join audit",
            "",
            "| audit | value | details |",
            "| --- | ---: | --- |",
        ]
    )

    for row in audit.itertuples(index=False):
        lines.append(f"| {row.audit} | {row.value} | `{row.details}` |")

    lines.extend(
        [
            "",
            "## Labels used",
            "",
            f"- categorical labels: `{', '.join(label_cols)}`",
            f"- numeric descriptors: `{', '.join(numeric_cols)}`",
            "",
            "## Strongest path-weighted label enrichments",
            "",
        ]
    )

    if enrichment.empty:
        lines.append("No enrichment rows computed.")
    else:
        top = enrichment[
            (enrichment["weighting"] == "path_weighted")
            & (pd.to_numeric(enrichment["denominator"], errors="coerce") >= cfg.min_count)
        ].copy()

        top = top.sort_values(["log2_enrichment", "denominator"], ascending=[False, False]).head(cfg.top_n)

        lines.extend(
            [
                "| scale | object | label | value | count | denom | share | global | enrichment | log2 |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for row in top.itertuples(index=False):
            lines.append(
                f"| {int(row.scale_index)} | {row.object} | {row.label_col} | {row.label_value} | "
                f"{int(row.count)} | {int(row.denominator)} | {fmt(row.object_share)} | "
                f"{fmt(row.global_path_share)} | {fmt(row.enrichment)} | {fmt(row.log2_enrichment)} |"
            )

    lines.extend(
        [
            "",
            "## Strongest step-weighted label enrichments",
            "",
        ]
    )

    if not enrichment.empty:
        top = enrichment[
            (enrichment["weighting"] == "step_weighted")
            & (pd.to_numeric(enrichment["denominator"], errors="coerce") >= cfg.min_count)
        ].copy()

        top = top.sort_values(["log2_enrichment", "denominator"], ascending=[False, False]).head(cfg.top_n)

        lines.extend(
            [
                "| scale | object | label | value | count | denom | share | global | enrichment | log2 |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for row in top.itertuples(index=False):
            lines.append(
                f"| {int(row.scale_index)} | {row.object} | {row.label_col} | {row.label_value} | "
                f"{int(row.count)} | {int(row.denominator)} | {fmt(row.object_share)} | "
                f"{fmt(row.global_path_share)} | {fmt(row.enrichment)} | {fmt(row.log2_enrichment)} |"
            )

    lines.extend(
        [
            "",
            "## Pinch candidate path-label projection",
            "",
        ]
    )

    if pinch_projection.empty:
        lines.append("No pinch projection rows computed.")
    else:
        top = pinch_projection[
            (pinch_projection.get("passes_min_count", False) == True)
            & (pinch_projection["label_col"] != "")
        ].copy()

        top = top.sort_values(
            ["candidate_rank", "cohort", "log2_enrichment"],
            ascending=[True, True, False],
        ).head(cfg.top_n)

        lines.extend(
            [
                "| rank | object | transition | cohort | n_paths | label | value | share | enrichment | dominant_family |",
                "| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )

        for row in top.itertuples(index=False):
            lines.append(
                f"| {int(row.candidate_rank)} | {row.object} | "
                f"{int(row.scale_index_from)}→{int(row.scale_index_to)} | "
                f"{row.cohort} | {int(row.n_paths)} | {row.label_col} | {row.label_value} | "
                f"{fmt(row.share)} | {fmt(row.enrichment)} | {row.dominant_family} |"
            )

    lines.extend(
        [
            "",
            "## Numeric descriptor summaries",
            "",
            "Numeric summaries are written to `obs077b_numeric_summary_by_object_scale.csv`.",
            "",
            "## Interpretation guide",
            "",
            "Use `path_weighted` enrichment for the primary scientific read.",
            "",
            "```text",
            "path_weighted:",
            "  each path_id contributes at most once per object/scale",
            "",
            "step_weighted:",
            "  every path-node-step occupancy contributes",
            "```",
            "",
            "Step-weighted enrichment is useful for occupancy intensity but can be biased by long paths.",
            "",
            "Pinch projection cohorts:",
            "",
            "```text",
            "before: paths touching the object at scale_from",
            "",
            "after: paths touching the object at scale_to",
            "",
            "entered: paths newly entering the object at scale_to",
            "",
            "exited: paths leaving the object after scale_from",
            "",
            "persisted: paths present at both sides",
            "",
            "union: paths present at either side",
            "```",
            "",
            "## Guardrails",
            "",
            "- This is path-label projection, not generated-text analysis.",
            "- Labels are instrument-derived from structural coupling summaries.",
            "- Enrichment is associative, not causal.",
            "- Text/provenance alignment remains a later layer.",
            "",
            "## Output artifacts",
            "",
            "- `obs077b_input_manifest.csv`",
            "- `obs077b_join_audit.csv`",
            "- `obs077b_path_object_membership_step_weighted.csv`",
            "- `obs077b_path_object_membership_path_weighted.csv`",
            "- `obs077b_label_enrichment_by_object_scale.csv`",
            "- `obs077b_numeric_summary_by_object_scale.csv`",
            "- `obs077b_pinch_label_projection.csv`",
            "- `obs077b_report.md`",
            "",
        ]
    )

    # Fix multiline code block text produced above.
    text = "\n".join(lines)
    text = text.replace(
        '"before:\n              paths touching the object at scale_from",',
        "before:\n  paths touching the object at scale_from"
    )
    text = text.replace(
        '"after:\n              paths touching the object at scale_to",',
        "after:\n  paths touching the object at scale_to"
    )
    text = text.replace(
        '"entered:\n              paths newly entering the object at scale_to",',
        "entered:\n  paths newly entering the object at scale_to"
    )
    text = text.replace(
        '"exited:\n              paths leaving the object after scale_from",',
        "exited:\n  paths leaving the object after scale_from"
    )
    text = text.replace(
        '"persisted:\n              paths present at both sides",',
        "persisted:\n  paths present at both sides"
    )
    text = text.replace(
        '"union:\n              paths present at either side",',
        "union:\n  paths present at either side"
    )

    (cfg.outdir / "obs077b_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    path_nodes = read_path_nodes(cfg)
    path_labels, label_cols, numeric_cols = read_path_labels(cfg)
    objects, obj_node_col = read_objects(cfg)
    pinch, pinch_status = read_pinch(cfg.pinch)

    step_weighted, path_weighted = build_memberships(
        path_nodes=path_nodes,
        path_labels=path_labels,
        objects=objects,
        cfg=cfg,
        obj_node_col=obj_node_col,
    )

    enrichment_parts = []
    enrichment_parts.append(
        compute_label_enrichment(
            membership=path_weighted,
            path_labels=path_labels,
            label_cols=label_cols,
            path_id_col=cfg.path_id_col,
            weighting="path_weighted",
        )
    )
    enrichment_parts.append(
        compute_label_enrichment(
            membership=step_weighted,
            path_labels=path_labels,
            label_cols=label_cols,
            path_id_col=cfg.path_id_col,
            weighting="step_weighted",
        )
    )
    enrichment = pd.concat(enrichment_parts, ignore_index=True)
    if not enrichment.empty:
        enrichment["case"] = cfg.case

    numeric_parts = []
    numeric_parts.append(
        compute_numeric_summary(
            membership=path_weighted,
            numeric_cols=numeric_cols,
            path_id_col=cfg.path_id_col,
            weighting="path_weighted",
        )
    )
    numeric_parts.append(
        compute_numeric_summary(
            membership=step_weighted,
            numeric_cols=numeric_cols,
            path_id_col=cfg.path_id_col,
            weighting="step_weighted",
        )
    )
    numeric_summary = pd.concat(numeric_parts, ignore_index=True)
    if not numeric_summary.empty:
        numeric_summary["case"] = cfg.case

    pinch_projection = project_pinch_candidates(
        pinch=pinch,
        path_weighted=path_weighted,
        path_labels=path_labels,
        label_cols=label_cols,
        path_id_col=cfg.path_id_col,
        min_count=cfg.min_count,
    )
    if not pinch_projection.empty:
        pinch_projection["case"] = cfg.case

    audit = write_audit(
        cfg=cfg,
        path_nodes=path_nodes,
        path_labels=path_labels,
        objects=objects,
        step_weighted=step_weighted,
        path_weighted=path_weighted,
        label_cols=label_cols,
        numeric_cols=numeric_cols,
        obj_node_col=obj_node_col,
        pinch_status=pinch_status,
    )

    input_manifest = pd.DataFrame(
        [
            {
                "artifact": "path_nodes",
                "path": str(cfg.path_nodes),
                "status": "ok",
                "details": f"rows={len(path_nodes)}",
            },
            {
                "artifact": "path_labels",
                "path": str(cfg.path_labels),
                "status": "ok",
                "details": f"rows={len(path_labels)}",
            },
            {
                "artifact": "objects",
                "path": str(cfg.objects),
                "status": "ok",
                "details": f"rows={len(objects)};node_col={obj_node_col}",
            },
            {
                "artifact": "pinch",
                "path": str(cfg.pinch) if cfg.pinch else "",
                "status": pinch_status,
                "details": f"rows={len(pinch) if pinch is not None else 0}",
            },
        ]
    )

    input_manifest.to_csv(cfg.outdir / "obs077b_input_manifest.csv", index=False)
    audit.to_csv(cfg.outdir / "obs077b_join_audit.csv", index=False)

    step_weighted.to_csv(
        cfg.outdir / "obs077b_path_object_membership_step_weighted.csv",
        index=False,
    )
    path_weighted.to_csv(
        cfg.outdir / "obs077b_path_object_membership_path_weighted.csv",
        index=False,
    )
    enrichment.to_csv(
        cfg.outdir / "obs077b_label_enrichment_by_object_scale.csv",
        index=False,
    )
    numeric_summary.to_csv(
        cfg.outdir / "obs077b_numeric_summary_by_object_scale.csv",
        index=False,
    )
    pinch_projection.to_csv(
        cfg.outdir / "obs077b_pinch_label_projection.csv",
        index=False,
    )

    write_report(
        cfg=cfg,
        input_manifest=input_manifest,
        audit=audit,
        enrichment=enrichment,
        numeric_summary=numeric_summary,
        pinch_projection=pinch_projection,
        label_cols=label_cols,
        numeric_cols=numeric_cols,
    )

    print("OBS-077b complete")
    print("wrote:", cfg.outdir / "obs077b_input_manifest.csv")
    print("wrote:", cfg.outdir / "obs077b_join_audit.csv")
    print("wrote:", cfg.outdir / "obs077b_path_object_membership_step_weighted.csv")
    print("wrote:", cfg.outdir / "obs077b_path_object_membership_path_weighted.csv")
    print("wrote:", cfg.outdir / "obs077b_label_enrichment_by_object_scale.csv")
    print("wrote:", cfg.outdir / "obs077b_numeric_summary_by_object_scale.csv")
    print("wrote:", cfg.outdir / "obs077b_pinch_label_projection.csv")
    print("wrote:", cfg.outdir / "obs077b_report.md")


if __name__ == "__main__":
    main()
