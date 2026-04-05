#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_family_transition_overlay.py

Overlay geodesic path families with transition-rate / operator-response fields.

Goal
----
Test whether geodesic route families differ in transition exposure.

Expected inputs
---------------
1. geodesic path-family assignments CSV, containing at least:
   - path_id
   - path_family

2. geodesic path-node CSV, containing at least:
   - path_id
   - step
   - r
   - alpha

3. node-level transition/operator CSV, containing:
   - r
   - alpha
   - and at least one of:
       * transition_rate
       * transition_probability
       * transition_prob
       * operator_response
       * transition_exposure

Optional node-level columns:
   - lazarus_score
   - signed_phase
   - criticality

Outputs
-------
- geodesic_family_transition_path_summary.csv
- geodesic_family_transition_family_summary.csv
- geodesic_family_transition_summary.txt
- geodesic_family_vs_mean_transition.png
- geodesic_family_vs_max_transition.png
- geodesic_family_vs_high_transition_fraction.png
- geodesic_family_transition_vs_lazarus.png

Notes
-----
- This is the transition/operator analogue of the Lazarus overlay.
- It auto-detects the transition-like column from common names.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    family_csv: str
    path_nodes_csv: str
    transition_nodes_csv: str
    outdir: str = "outputs/toy_geodesic_family_transition_overlay"
    high_transition_quantile: float = 0.85


def _corr(df: pd.DataFrame, x: str, y: str) -> float:
    if x not in df.columns or y not in df.columns:
        return float("nan")
    work = df[[x, y]].dropna()
    if len(work) < 3:
        return float("nan")
    return float(work[x].corr(work[y], method="spearman"))


def _find_transition_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "transition_rate",
        "transition_probability",
        "transition_prob",
        "transition_exposure",
        "operator_response",
        "response_strength",
        "response_active",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_inputs(cfg: Config):
    fam = pd.read_csv(cfg.family_csv).copy()
    path_nodes = pd.read_csv(cfg.path_nodes_csv).copy()
    trans = pd.read_csv(cfg.transition_nodes_csv).copy()

    need_fam = {"path_id", "path_family"}
    missing_fam = need_fam - set(fam.columns)
    if missing_fam:
        raise ValueError(f"family_csv missing columns: {sorted(missing_fam)}")

    need_path = {"path_id", "r", "alpha"}
    missing_path = need_path - set(path_nodes.columns)
    if missing_path:
        raise ValueError(f"path_nodes_csv missing columns: {sorted(missing_path)}")

    need_trans = {"r", "alpha"}
    missing_trans = need_trans - set(trans.columns)
    if missing_trans:
        raise ValueError(f"transition_nodes_csv missing columns: {sorted(missing_trans)}")

    for df in (path_nodes, trans):
        for c in ["r", "alpha"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in trans.columns:
        if c not in {"r", "alpha"}:
            trans[c] = pd.to_numeric(trans[c], errors="coerce")

    return fam, path_nodes, trans


def build_path_summary(
    fam: pd.DataFrame,
    path_nodes: pd.DataFrame,
    trans: pd.DataFrame,
    high_transition_quantile: float,
) -> tuple[pd.DataFrame, str]:
    trans_col = _find_transition_col(trans)
    if trans_col is None:
        raise ValueError(
            "No transition-like column found. Expected one of: "
            "transition_rate, transition_probability, transition_prob, "
            "transition_exposure, operator_response"
        )

    keep_cols = ["r", "alpha", trans_col]
    optional_cols = [c for c in ["lazarus_score", "signed_phase", "criticality"] if c in trans.columns]
    keep_cols.extend(optional_cols)

    ann = path_nodes.merge(
        trans[keep_cols].drop_duplicates(subset=["r", "alpha"]),
        on=["r", "alpha"],
        how="left",
    )

    vals = pd.to_numeric(ann[trans_col], errors="coerce")
    thr = float(vals.quantile(high_transition_quantile)) if len(vals.dropna()) else float("nan")
    ann["high_transition"] = vals >= thr if pd.notna(thr) else False

    rows: list[dict[str, object]] = []
    for path_id, sub in ann.groupby("path_id", dropna=False):
        rec: dict[str, object] = {
            "path_id": path_id,
            "n_nodes": int(len(sub)),
            "mean_transition_exposure": float(pd.to_numeric(sub[trans_col], errors="coerce").mean()),
            "max_transition_exposure": float(pd.to_numeric(sub[trans_col], errors="coerce").max()),
            "high_transition_fraction": float(pd.to_numeric(sub["high_transition"], errors="coerce").fillna(0).mean()),
        }

        if "lazarus_score" in sub.columns:
            rec["mean_lazarus"] = float(pd.to_numeric(sub["lazarus_score"], errors="coerce").mean())
            rec["max_lazarus"] = float(pd.to_numeric(sub["lazarus_score"], errors="coerce").max())
        else:
            rec["mean_lazarus"] = np.nan
            rec["max_lazarus"] = np.nan

        if "criticality" in sub.columns:
            rec["mean_criticality"] = float(pd.to_numeric(sub["criticality"], errors="coerce").mean())
            rec["max_criticality"] = float(pd.to_numeric(sub["criticality"], errors="coerce").max())
        else:
            rec["mean_criticality"] = np.nan
            rec["max_criticality"] = np.nan

        rows.append(rec)

    path_summary = pd.DataFrame(rows)
    path_summary = fam.merge(path_summary, on="path_id", how="left")
    path_summary["high_transition_threshold"] = thr
    return path_summary, trans_col


def build_family_summary(path_summary: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "n_paths": ("path_family", "size"),
    }

    optional_aggs = {
        "mean_transition_exposure": ("mean_transition_exposure", "mean"),
        "mean_max_transition_exposure": ("max_transition_exposure", "mean"),
        "mean_high_transition_fraction": ("high_transition_fraction", "mean"),
        "mean_lazarus": ("mean_lazarus", "mean"),
        "mean_criticality": ("mean_criticality", "mean"),
    }

    for out_col, (src_col, fn) in optional_aggs.items():
        if src_col in path_summary.columns:
            agg_map[out_col] = (src_col, fn)

    summary = (
        path_summary.groupby("path_family", dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def summarize_text(
    path_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    trans_col: str,
) -> str:
    lines: list[str] = []
    lines.append("=== Geodesic Family × Transition Overlay Summary ===")
    lines.append("")
    lines.append(f"n_paths = {len(path_summary)}")
    if len(path_summary):
        thr = pd.to_numeric(path_summary["high_transition_threshold"], errors="coerce").iloc[0]
        lines.append(f"high_transition_threshold = {thr:.4f}")
    lines.append(f"transition_column = {trans_col}")
    lines.append("")
    lines.append("Path-level correlations")
    lines.append(
        f"  corr(mean_transition_exposure, high_transition_fraction) = "
        f"{_corr(path_summary, 'mean_transition_exposure', 'high_transition_fraction'):.4f}"
    )
    if "mean_lazarus" in path_summary.columns:
        lines.append(
            f"  corr(mean_transition_exposure, mean_lazarus) = "
            f"{_corr(path_summary, 'mean_transition_exposure', 'mean_lazarus'):.4f}"
        )
    if "mean_criticality" in path_summary.columns:
        lines.append(
            f"  corr(mean_transition_exposure, mean_criticality) = "
            f"{_corr(path_summary, 'mean_transition_exposure', 'mean_criticality'):.4f}"
        )
    lines.append("")
    lines.append("Family summary")
    for _, row in family_summary.iterrows():
        parts = [
            f"  {row['path_family']}: ",
            f"n_paths={int(row['n_paths'])}",
        ]
        for col in [
            "mean_transition_exposure",
            "mean_max_transition_exposure",
            "mean_high_transition_fraction",
            "mean_lazarus",
            "mean_criticality",
        ]:
            if col in family_summary.columns:
                parts.append(f"{col}={row[col]:.4f}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def plot_family_box(
    df: pd.DataFrame,
    value_col: str,
    outpath: Path,
    title: str,
) -> None:
    plot_df = df[["path_family", value_col]].dropna().copy()
    fams = list(plot_df["path_family"].dropna().unique())
    groups = [plot_df.loc[plot_df["path_family"] == fam, value_col].to_numpy() for fam in fams]

    fig, ax = plt.subplots(figsize=(9, 6))
    if groups:
        ax.boxplot(groups, labels=fams, showfliers=True)
    ax.set_title(title)
    ax.set_xlabel("path_family")
    ax.set_ylabel(value_col)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    plot_df = df[[x, y, "path_family"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for fam, sub in plot_df.groupby("path_family", dropna=False):
        ax.scatter(sub[x], sub[y], s=45, label=str(fam))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay geodesic route families with transition/operator diagnostics.")
    parser.add_argument("--family-csv", required=True)
    parser.add_argument("--path-nodes-csv", required=True)
    parser.add_argument("--transition-nodes-csv", required=True)
    parser.add_argument("--outdir", default="outputs/toy_geodesic_family_transition_overlay")
    parser.add_argument("--high-transition-quantile", type=float, default=0.85)
    args = parser.parse_args()

    cfg = Config(
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        transition_nodes_csv=args.transition_nodes_csv,
        outdir=args.outdir,
        high_transition_quantile=args.high_transition_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fam, path_nodes, trans = load_inputs(cfg)
    path_summary, trans_col = build_path_summary(
        fam,
        path_nodes,
        trans,
        cfg.high_transition_quantile,
    )
    family_summary = build_family_summary(path_summary)

    path_summary.to_csv(outdir / "geodesic_family_transition_path_summary.csv", index=False)
    family_summary.to_csv(outdir / "geodesic_family_transition_family_summary.csv", index=False)
    (outdir / "geodesic_family_transition_summary.txt").write_text(
        summarize_text(path_summary, family_summary, trans_col),
        encoding="utf-8",
    )

    plot_family_box(
        path_summary,
        "mean_transition_exposure",
        outdir / "geodesic_family_vs_mean_transition.png",
        "Geodesic family vs mean transition exposure",
    )
    plot_family_box(
        path_summary,
        "max_transition_exposure",
        outdir / "geodesic_family_vs_max_transition.png",
        "Geodesic family vs max transition exposure",
    )
    plot_family_box(
        path_summary,
        "high_transition_fraction",
        outdir / "geodesic_family_vs_high_transition_fraction.png",
        "Geodesic family vs high-transition fraction",
    )
    plot_scatter(
        path_summary,
        "mean_transition_exposure",
        "mean_lazarus",
        outdir / "geodesic_family_transition_vs_lazarus.png",
        "Transition exposure vs Lazarus by path family",
    )

    print(outdir / "geodesic_family_transition_path_summary.csv")
    print(outdir / "geodesic_family_transition_family_summary.csv")
    print(outdir / "geodesic_family_transition_summary.txt")
    print(outdir / "geodesic_family_vs_mean_transition.png")
    print(outdir / "geodesic_family_vs_max_transition.png")
    print(outdir / "geodesic_family_vs_high_transition_fraction.png")
    print(outdir / "geodesic_family_transition_vs_lazarus.png")


if __name__ == "__main__":
    main()
