#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_nodes(identity_nodes_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(identity_nodes_csv)

    required = {
        "node_id",
        "r",
        "alpha",
        "mds1",
        "mds2",
        "identity_magnitude",
        "identity_spin",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"identity diagnostics nodes CSV missing required columns: {sorted(missing)}"
        )

    for col in ["r", "alpha", "mds1", "mds2", "identity_magnitude", "identity_spin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["abs_identity_spin"] = df["identity_spin"].abs()
    return df


def render_overlay(
    df: pd.DataFrame,
    *,
    outpath: Path,
    top_k: int = 12,
    label_k: int = 6,
) -> None:
    work = df.dropna(subset=["mds1", "mds2", "identity_magnitude", "identity_spin"]).copy()
    top = work.sort_values("abs_identity_spin", ascending=False).head(top_k).copy()
    label_df = top.head(label_k).copy()

    plt.figure(figsize=(8.0, 6.2))

    sc = plt.scatter(
        work["mds1"],
        work["mds2"],
        c=work["identity_magnitude"],
        s=80,
        alpha=0.85,
        cmap="viridis",
    )
    plt.colorbar(sc, label="identity magnitude")

    pos = top[top["identity_spin"] > 0]
    neg = top[top["identity_spin"] < 0]
    zer = top[top["identity_spin"] == 0]

    if len(pos):
        plt.scatter(
            pos["mds1"],
            pos["mds2"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label="top +spin singularities",
        )

    if len(neg):
        plt.scatter(
            neg["mds1"],
            neg["mds2"],
            s=220,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2.0,
            label="top -spin singularities",
        )

    if len(zer):
        plt.scatter(
            zer["mds1"],
            zer["mds2"],
            s=220,
            facecolors="none",
            edgecolors="white",
            linewidths=2.0,
            label="top zero-spin singularities",
        )

    for _, row in label_df.iterrows():
        plt.text(
            row["mds1"],
            row["mds2"],
            f"({row['r']:.2f}, {row['alpha']:.3f})",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.title("Identity Magnitude with Spin Singularity Overlay")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def build_singularity_overlay_table(df: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    cols = [
        "node_id",
        "r",
        "alpha",
        "mds1",
        "mds2",
        "identity_magnitude",
        "identity_spin",
        "abs_identity_spin",
    ]
    return (
        df.sort_values("abs_identity_spin", ascending=False)
        .loc[:, cols]
        .head(top_k)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay top identity singularities on the PAM MDS manifold.")
    parser.add_argument(
        "--identity-diagnostics-nodes-csv",
        default="outputs/fim_identity_diagnostics/identity_diagnostics_nodes.csv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/fim_identity_singularity_overlay",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--label-k",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_nodes(args.identity_diagnostics_nodes_csv)
    overlay_table = build_singularity_overlay_table(df, top_k=args.top_k)
    overlay_table.to_csv(outdir / "identity_singularity_overlay_table.csv", index=False)

    render_overlay(
        df,
        outpath=outdir / "identity_singularity_overlay_on_mds.png",
        top_k=args.top_k,
        label_k=args.label_k,
    )

    print(outdir / "identity_singularity_overlay_table.csv")
    print(outdir / "identity_singularity_overlay_on_mds.png")


if __name__ == "__main__":
    main()
