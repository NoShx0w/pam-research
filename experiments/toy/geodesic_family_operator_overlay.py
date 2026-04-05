#!/usr/bin/env python3
from __future__ import annotations

"""
geodesic_family_operator_overlay.py

Overlay geodesic path families with operator / Lazarus diagnostics.

Goal
----
Test whether geodesic route families differ in dynamical/operator structure.

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

3. node-level operator CSV, containing:
   - r
   - alpha
   - lazarus_score

Optional node-level columns:
   - transition_rate
   - transition_probability
   - transition_prob
   - operator_response
   - any other scalar operator field

Outputs
-------
- geodesic_family_operator_path_summary.csv
- geodesic_family_operator_family_summary.csv
- geodesic_family_operator_summary.txt
- geodesic_family_vs_mean_lazarus.png
- geodesic_family_vs_max_lazarus.png
- geodesic_family_vs_transition_exposure.png

Notes
-----
- This is a file-first overlay study.
- It does not assume a particular operator semantics beyond scalar node fields.
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
    operator_nodes_csv: str
    outdir: str = "outputs/toy_geodesic_family_operator_overlay"
    lazarus_high_quantile: float = 0.85


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
        "operator_response",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_inputs(cfg: Config):
    fam = pd.read_csv(cfg.family_csv).copy()
    path_nodes = pd.read_csv(cfg.path_nodes_csv).copy()
    ops = pd.read_csv(cfg.operator_nodes_csv).copy()

    need_fam = {"path_id", "path_family"}
    missing_fam = need_fam - set(fam.columns)
    if missing_fam:
        raise ValueError(f"family_csv missing columns: {sorted(missing_fam)}")

    need_path = {"path_id", "r", "alpha"}
    missing_path = need_path - set(path_nodes.columns)
    if missing_path:
        raise ValueError(f"path_nodes_csv missing columns: {sorted(missing_path)}")

    need_ops = {"r", "alpha", "lazarus_score"}
    missing_ops = need_ops - set(ops.columns)
    if missing_ops:
        raise ValueError(f"operator_nodes_csv missing columns: {sorted(missing_ops)}")

    for df in (path_nodes, ops):
        for c in ["r", "alpha"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ops.columns:
        if c not in {"r", "alpha"}:
            ops[c] = pd.to_numeric(ops[c], errors="coerce")

    return fam, path_nodes, ops


def build_path_summary(
    fam: pd.DataFrame,
    path_nodes: pd.DataFrame,
    ops: pd.DataFrame,
    lazarus_high_quantile: float,
) -> tuple[pd.DataFrame, str | None]:
    trans_col = _find_transition_col(ops)

    keep_cols = ["r", "alpha", "lazarus_score"]
    if trans_col is not None:
        keep_cols.append(trans_col)

    ann = path_nodes.merge(
        ops[keep_cols].drop_duplicates(subset=["r", "alpha"]),
        on=["r", "alpha"],
        how="left",
    )

    laz = pd.to_numeric(ann["lazarus_score"], errors="coerce")
    laz_thr = float(laz.quantile(lazarus_high_quantile)) if len(laz.dropna()) else float("nan")
    ann["high_lazarus"] = laz >= laz_thr if pd.notna(laz_thr) else False

    rows: list[dict[str, object]] = []
    for path_id, sub in ann.groupby("path_id", dropna=False):
        rec: dict[str, object] = {
            "path_id": path_id,
            "n_nodes": int(len(sub)),
            "mean_lazarus": float(pd.to_numeric(sub["lazarus_score"], errors="coerce").mean()),
            "max_lazarus": float(pd.to_numeric(sub["lazarus_score"], errors="coerce").max()),
            "high_lazarus_fraction": float(pd.to_numeric(sub["high_lazarus"], errors="coerce").fillna(0).mean()),
        }

        if trans_col is not None:
            rec["mean_transition_exposure"] = float(pd.to_numeric(sub[trans_col], errors="coerce").mean())
            rec["max_transition_exposure"] = float(pd.to_numeric(sub[trans_col], errors="coerce").max())
        else:
            rec["mean_transition_exposure"] = np.nan
            rec["max_transition_exposure"] = np.nan

        rows.append(rec)

    path_summary = pd.DataFrame(rows)
    path_summary = fam.merge(path_summary, on="path_id", how="left")
    path_summary["lazarus_high_threshold"] = laz_thr
    return path_summary, trans_col


def build_family_summary(path_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "path_family",
        "mean_lazarus",
        "max_lazarus",
        "high_lazarus_fraction",
        "mean_transition_exposure",
        "max_transition_exposure",
    ]
    work = path_summary[cols].copy()
    return (
        work.groupby("path_family", dropna=False)
        .agg(
            n_paths=("path_family", "size"),
            mean_lazarus=("mean_lazarus", "mean"),
            mean_max_lazarus=("max_lazarus", "mean"),
            mean_high_lazarus_fraction=("high_lazarus_fraction", "mean"),
            mean_transition_exposure=("mean_transition_exposure", "mean"),
            mean_max_transition_exposure=("max_transition_exposure", "mean"),
        )
        .reset_index()
        .sort_values("n_paths", ascending=False)
        .reset_index(drop=True)
    )


def summarize_text(
    path_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    trans_col: str | None,
) -> str:
    lines: list[str] = []
    lines.append("=== Geodesic Family × Operator Overlay Summary ===")
    lines.append("")
    lines.append(f"n_paths = {len(path_summary)}")
    if len(path_summary):
        thr = pd.to_numeric(path_summary["lazarus_high_threshold"], errors="coerce").iloc[0]
        lines.append(f"high_lazarus_threshold = {thr:.4f}")
    lines.append("")
    lines.append("Path-level correlations")
    lines.append(
        f"  corr(mean_lazarus, high_lazarus_fraction) = "
        f"{_corr(path_summary, 'mean_lazarus', 'high_lazarus_fraction'):.4f}"
    )
    if "mean_transition_exposure" in path_summary.columns:
        lines.append(
            f"  corr(mean_lazarus, mean_transition_exposure) = "
            f"{_corr(path_summary, 'mean_lazarus', 'mean_transition_exposure'):.4f}"
        )
    lines.append("")

    lines.append("Family summary")
    for _, row in family_summary.iterrows():
        lines.append(
            f"  {row['path_family']}: "
            f"n_paths={int(row['n_paths'])}, "
            f"mean_lazarus={row['mean_lazarus']:.4f}, "
            f"mean_max_lazarus={row['mean_max_lazarus']:.4f}, "
            f"mean_high_lazarus_fraction={row['mean_high_lazarus_fraction']:.4f}, "
            f"mean_transition_exposure={row['mean_transition_exposure']:.4f}, "
            f"mean_max_transition_exposure={row['mean_max_transition_exposure']:.4f}"
        )

    lines.append("")
    lines.append(f"transition_column = {trans_col if trans_col is not None else 'none found'}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay geodesic route families with operator/Lazarus diagnostics.")
    parser.add_argument("--family-csv", required=True)
    parser.add_argument("--path-nodes-csv", required=True)
    parser.add_argument("--operator-nodes-csv", required=True)
    parser.add_argument("--outdir", default="outputs/toy_geodesic_family_operator_overlay")
    parser.add_argument("--lazarus-high-quantile", type=float, default=0.85)
    args = parser.parse_args()

    cfg = Config(
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        operator_nodes_csv=args.operator_nodes_csv,
        outdir=args.outdir,
        lazarus_high_quantile=args.lazarus_high_quantile,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fam, path_nodes, ops = load_inputs(cfg)
    path_summary, trans_col = build_path_summary(
        fam,
        path_nodes,
        ops,
        cfg.lazarus_high_quantile,
    )
    family_summary = build_family_summary(path_summary)

    path_summary.to_csv(outdir / "geodesic_family_operator_path_summary.csv", index=False)
    family_summary.to_csv(outdir / "geodesic_family_operator_family_summary.csv", index=False)
    (outdir / "geodesic_family_operator_summary.txt").write_text(
        summarize_text(path_summary, family_summary, trans_col),
        encoding="utf-8",
    )

    plot_family_box(
        path_summary,
        "mean_lazarus",
        outdir / "geodesic_family_vs_mean_lazarus.png",
        "Geodesic family vs mean Lazarus",
    )
    plot_family_box(
        path_summary,
        "max_lazarus",
        outdir / "geodesic_family_vs_max_lazarus.png",
        "Geodesic family vs max Lazarus",
    )
    if trans_col is not None:
        plot_family_box(
            path_summary,
            "mean_transition_exposure",
            outdir / "geodesic_family_vs_transition_exposure.png",
            "Geodesic family vs transition exposure",
        )

    print(outdir / "geodesic_family_operator_path_summary.csv")
    print(outdir / "geodesic_family_operator_family_summary.csv")
    print(outdir / "geodesic_family_operator_summary.txt")
    print(outdir / "geodesic_family_vs_mean_lazarus.png")
    print(outdir / "geodesic_family_vs_max_lazarus.png")
    if trans_col is not None:
        print(outdir / "geodesic_family_vs_transition_exposure.png")


if __name__ == "__main__":
    main()
