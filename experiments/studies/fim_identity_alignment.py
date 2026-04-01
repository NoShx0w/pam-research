#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_identity_node_proxy(identity_edges_csv: str | Path) -> pd.DataFrame:
    edges = pd.read_csv(identity_edges_csv)

    required = {
        "src_node_id",
        "src_r",
        "src_alpha",
        "identity_distance",
    }
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"identity edges CSV missing required columns: {sorted(missing)}")

    out = (
        edges.groupby(["src_node_id", "src_r", "src_alpha"], as_index=False)
        .agg(
            identity_distance_mean=("identity_distance", "mean"),
            identity_distance_max=("identity_distance", "max"),
            identity_distance_sum=("identity_distance", "sum"),
            n_identity_edges=("identity_distance", "count"),
        )
        .rename(
            columns={
                "src_node_id": "node_id",
                "src_r": "r",
                "src_alpha": "alpha",
            }
        )
    )
    return out


def load_alignment_base(
    *,
    identity_node_summary_csv: str | Path,
    identity_edges_csv: str | Path,
    criticality_csv: str | Path,
) -> pd.DataFrame:
    ident_nodes = pd.read_csv(identity_node_summary_csv)
    ident_proxy = load_identity_node_proxy(identity_edges_csv)
    crit = pd.read_csv(criticality_csv)

    required_ident = {"node_id", "r", "alpha", "criticality", "distance_to_seam"}
    missing_ident = required_ident - set(ident_nodes.columns)
    if missing_ident:
        raise ValueError(f"identity node summary CSV missing required columns: {sorted(missing_ident)}")

    required_crit = {"r", "alpha", "scalar_curvature", "criticality"}
    missing_crit = required_crit - set(crit.columns)
    if missing_crit:
        raise ValueError(f"criticality surface CSV missing required columns: {sorted(missing_crit)}")

    df = (
        ident_nodes.merge(
            ident_proxy,
            on=["node_id", "r", "alpha"],
            how="left",
        )
        .merge(
            crit[["r", "alpha", "scalar_curvature"]],
            on=["r", "alpha"],
            how="left",
        )
        .copy()
    )

    for col in [
        "criticality",
        "distance_to_seam",
        "scalar_curvature",
        "identity_distance_mean",
        "identity_distance_max",
        "identity_distance_sum",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def corr_summary(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    work = df[[x, y]].dropna()
    if len(work) < 3:
        return {
            "metric_x": x,
            "metric_y": y,
            "n": len(work),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    return {
        "metric_x": x,
        "metric_y": y,
        "n": int(len(work)),
        "pearson": float(work[x].corr(work[y], method="pearson")),
        "spearman": float(work[x].corr(work[y], method="spearman")),
    }


def render_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> None:
    work = df[[x, y]].dropna()
    plt.figure(figsize=(6.6, 5.0))
    plt.scatter(work[x], work[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare first-pass PAM identity field against existing structural fields.")
    parser.add_argument("--identity-node-summary-csv", default="outputs/fim_identity/identity_node_summary.csv")
    parser.add_argument("--identity-edges-csv", default="outputs/fim_identity/identity_field_edges.csv")
    parser.add_argument("--criticality-csv", default="outputs/fim_critical/criticality_surface.csv")
    parser.add_argument("--outdir", default="outputs/fim_identity_alignment")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_alignment_base(
        identity_node_summary_csv=args.identity_node_summary_csv,
        identity_edges_csv=args.identity_edges_csv,
        criticality_csv=args.criticality_csv,
    )

    df.to_csv(outdir / "identity_alignment_nodes.csv", index=False)

    rows = [
        corr_summary(df, "identity_distance_mean", "distance_to_seam"),
        corr_summary(df, "identity_distance_mean", "criticality"),
        corr_summary(df, "identity_distance_mean", "scalar_curvature"),
        corr_summary(df, "identity_distance_max", "distance_to_seam"),
        corr_summary(df, "identity_distance_max", "criticality"),
        corr_summary(df, "identity_distance_max", "scalar_curvature"),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "identity_alignment_summary.csv", index=False)

    render_scatter(
        df,
        "distance_to_seam",
        "identity_distance_mean",
        outdir / "identity_vs_seam_distance.png",
        "Identity vs Seam Distance",
    )
    render_scatter(
        df,
        "criticality",
        "identity_distance_mean",
        outdir / "identity_vs_criticality.png",
        "Identity vs Criticality",
    )
    render_scatter(
        df,
        "scalar_curvature",
        "identity_distance_mean",
        outdir / "identity_vs_curvature.png",
        "Identity vs Curvature",
    )

    print(outdir / "identity_alignment_nodes.csv")
    print(outdir / "identity_alignment_summary.csv")
    print(outdir / "identity_vs_seam_distance.png")
    print(outdir / "identity_vs_criticality.png")
    print(outdir / "identity_vs_curvature.png")


if __name__ == "__main__":
    main()
