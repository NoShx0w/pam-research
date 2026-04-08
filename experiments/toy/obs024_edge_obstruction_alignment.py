#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    edge_csv: str = "outputs/toy_identity_transport_alignment/edge_transport_alignment.csv"
    cell_csv: str = "outputs/fim_identity_holonomy/identity_holonomy_cells.csv"
    outdir: str = "outputs/obs024_edge_obstruction_alignment"
    seam_threshold: float = 0.15
    top_k: int = 20


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def canon_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def expand_cells_to_edges(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cells.iterrows():
        A = int(row["A_node_id"])
        B = int(row["B_node_id"])
        C = int(row["C_node_id"])
        D = int(row["D_node_id"])

        boundary = [
            canon_edge(A, B),
            canon_edge(B, C),
            canon_edge(C, D),
            canon_edge(A, D),
        ]

        for u, v in boundary:
            rows.append(
                {
                    "u": u,
                    "v": v,
                    "abs_holonomy_residual": pd.to_numeric(row.get("abs_holonomy_residual"), errors="coerce"),
                    "holonomy_residual": pd.to_numeric(row.get("holonomy_residual"), errors="coerce"),
                    "mean_abs_corner_spin": pd.to_numeric(row.get("mean_abs_corner_spin"), errors="coerce"),
                    "max_abs_corner_spin": pd.to_numeric(row.get("max_abs_corner_spin"), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_edge_obstruction(cell_edges: pd.DataFrame) -> pd.DataFrame:
    return (
        cell_edges.groupby(["u", "v"], as_index=False)
        .agg(
            edge_incident_mean_abs_holonomy=("abs_holonomy_residual", "mean"),
            edge_incident_max_abs_holonomy=("abs_holonomy_residual", "max"),
            edge_incident_mean_signed_holonomy=("holonomy_residual", "mean"),
            edge_incident_mean_abs_corner_spin=("mean_abs_corner_spin", "mean"),
            edge_incident_max_abs_corner_spin=("max_abs_corner_spin", "max"),
            n_incident_cells=("u", "size"),
        )
    )


def prepare_transport_edges(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    out["u"] = np.minimum(out["src_id"], out["dst_id"]).astype(int)
    out["v"] = np.maximum(out["src_id"], out["dst_id"]).astype(int)

    # collapse symmetric duplicate directions
    out = (
        out.groupby(["u", "v"], as_index=False)
        .agg(
            misalignment_deg=("misalignment_deg", "mean"),
            edge_distance_to_seam_mid=("edge_distance_to_seam_mid", "mean"),
            edge_signed_phase_mid=("edge_signed_phase_mid", "mean"),
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-csv", default=Config.edge_csv)
    parser.add_argument("--cell-csv", default=Config.cell_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k", type=int, default=Config.top_k)
    args = parser.parse_args()

    cfg = Config(
        edge_csv=args.edge_csv,
        cell_csv=args.cell_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k=args.top_k,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(cfg.edge_csv)
    cells = pd.read_csv(cfg.cell_csv)

    edge_transport = prepare_transport_edges(edges)
    cell_edges = expand_cells_to_edges(cells)
    edge_obs = aggregate_edge_obstruction(cell_edges)

    merged = edge_transport.merge(edge_obs, on=["u", "v"], how="left")

    seam_mask = pd.to_numeric(merged["edge_distance_to_seam_mid"], errors="coerce") <= cfg.seam_threshold

    metrics = [
        "edge_incident_mean_abs_holonomy",
        "edge_incident_max_abs_holonomy",
        "edge_incident_mean_abs_corner_spin",
        "edge_incident_max_abs_corner_spin",
    ]

    rows = []
    for m in metrics:
        corr = safe_corr(merged["misalignment_deg"], merged[m])
        seam_mean = float(pd.to_numeric(merged.loc[seam_mask, m], errors="coerce").mean())
        off_mean = float(pd.to_numeric(merged.loc[~seam_mask, m], errors="coerce").mean())

        top_mis = set(merged.sort_values("misalignment_deg", ascending=False).head(cfg.top_k).index.tolist())
        top_obs = set(merged.sort_values(m, ascending=False).head(cfg.top_k).index.tolist())
        overlap = len(top_mis & top_obs)

        rows.append(
            {
                "metric": m,
                "corr_with_misalignment": corr,
                "seam_mean": seam_mean,
                "off_mean": off_mean,
                "top_k_overlap": overlap,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["corr_with_misalignment", "top_k_overlap"],
        ascending=[False, False],
    )

    summary.to_csv(outdir / "edge_obstruction_alignment_summary.csv", index=False)
    merged.to_csv(outdir / "edge_obstruction_alignment_edges.csv", index=False)

    lines = ["=== OBS-024 Edge Obstruction Alignment ===", ""]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['metric']}: corr={row['corr_with_misalignment']:.4f}, "
            f"seam_mean={row['seam_mean']:.4f}, off_mean={row['off_mean']:.4f}, "
            f"top_k_overlap={int(row['top_k_overlap'])}"
        )
    (outdir / "obs024_edge_obstruction_alignment_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(summary["metric"], summary["corr_with_misalignment"])
    ax.invert_yaxis()
    ax.set_xlabel("corr with edge misalignment")
    ax.set_title("OBS-024 edge obstruction alignment")
    ax.grid(alpha=0.15, axis="x")
    fig.tight_layout()
    fig.savefig(outdir / "obs024_edge_obstruction_alignment.png", dpi=220)
    plt.close(fig)

    print(outdir / "edge_obstruction_alignment_summary.csv")
    print(outdir / "edge_obstruction_alignment_edges.csv")
    print(outdir / "obs024_edge_obstruction_alignment_summary.txt")
    print(outdir / "obs024_edge_obstruction_alignment.png")


if __name__ == "__main__":
    main()
