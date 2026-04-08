#!/usr/bin/env python3
"""
OBS-025 — Response anisotropy vs relational obstruction.

Unify two seam-side structural fields:

1. response anisotropy
   -> symmetric traceless norm of the local response tensor

2. relational obstruction
   -> neighbor-level directional mismatch / transport-aware relational field

This script compares:
- seam localization of both fields
- global correlation
- hotspot overlap
- family of top nodes shared by both fields

Inputs
------
outputs/fim_response_operator_decomposition/response_operator_decomposition_nodes.csv
outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/obs025_anisotropy_vs_relational_obstruction/
  obs025_anisotropy_vs_relational_obstruction_nodes.csv
  obs025_anisotropy_vs_relational_obstruction_summary.txt
  obs025_anisotropy_vs_relational_obstruction_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    anisotropy_csv: str = "outputs/fim_response_operator_decomposition/response_operator_decomposition_nodes.csv"
    mismatch_csv: str = "outputs/obs023_local_direction_mismatch/local_direction_mismatch_nodes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs025_anisotropy_vs_relational_obstruction"
    seam_threshold: float = 0.15
    hotspot_quantile: float = 0.85
    top_k_overlap: int = 10
    top_k_labels: int = 8


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    aniso = pd.read_csv(cfg.anisotropy_csv)
    mm = pd.read_csv(cfg.mismatch_csv)
    seam = pd.read_csv(cfg.seam_csv)

    for df in (aniso, mm, seam):
        for col in df.columns:
            if col not in {"path_id", "path_family"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    keep_aniso = [
        c
        for c in [
            "node_id",
            "r",
            "alpha",
            "mds1",
            "mds2",
            "signed_phase",
            "distance_to_seam",
            "node_holonomy_proxy",
            "sym_traceless_norm",
            "antisymmetric_norm",
            "scalar_norm",
            "commutator_norm_rsp",
        ]
        if c in aniso.columns
    ]
    keep_mm = [
        c
        for c in [
            "node_id",
            "neighbor_direction_mismatch_mean",
            "local_direction_mismatch_deg",
            "transport_align_mean_deg",
        ]
        if c in mm.columns
    ]

    nodes = aniso[keep_aniso].merge(mm[keep_mm], on="node_id", how="left")
    return nodes, seam


def add_hotspots(nodes: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = nodes.copy()

    aniso_thr = float(pd.to_numeric(out["sym_traceless_norm"], errors="coerce").quantile(quantile))
    rel_thr = float(pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce").quantile(quantile))

    out["anisotropy_hotspot"] = (
        pd.to_numeric(out["sym_traceless_norm"], errors="coerce") >= aniso_thr
    ).astype(int)
    out["relational_hotspot"] = (
        pd.to_numeric(out["neighbor_direction_mismatch_mean"], errors="coerce") >= rel_thr
    ).astype(int)
    out["shared_hotspot"] = ((out["anisotropy_hotspot"] == 1) & (out["relational_hotspot"] == 1)).astype(int)

    out.attrs["anisotropy_threshold"] = aniso_thr
    out.attrs["relational_threshold"] = rel_thr
    return out


def build_summary(nodes: pd.DataFrame, cfg: Config) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    transport_col = "transport_align_mean_deg" if "transport_align_mean_deg" in nodes.columns else "neighbor_direction_mismatch_mean"

    top_aniso = set(
        nodes.sort_values("sym_traceless_norm", ascending=False)
        .head(cfg.top_k_overlap)["node_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    top_rel = set(
        nodes.sort_values("neighbor_direction_mismatch_mean", ascending=False)
        .head(cfg.top_k_overlap)["node_id"]
        .dropna()
        .astype(int)
        .tolist()
    )
    overlap = len(top_aniso & top_rel)

    lines = [
        "=== OBS-025 Anisotropy vs Relational Obstruction Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {cfg.seam_threshold:.4f}",
        f"hotspot_quantile = {cfg.hotspot_quantile:.4f}",
        f"anisotropy_threshold = {float(nodes.attrs['anisotropy_threshold']):.6f}",
        f"relational_threshold = {float(nodes.attrs['relational_threshold']):.6f}",
        "",
        "Field means",
        f"  mean sym_traceless_norm              = {safe_mean(nodes['sym_traceless_norm']):.6f}",
        f"  mean neighbor_direction_mismatch     = {safe_mean(nodes['neighbor_direction_mismatch_mean']):.6f}",
        "",
        "Seam localization",
        f"  seam-band mean sym_traceless_norm    = {safe_mean(nodes.loc[seam_mask, 'sym_traceless_norm']):.6f}",
        f"  off-seam mean sym_traceless_norm     = {safe_mean(nodes.loc[~seam_mask, 'sym_traceless_norm']):.6f}",
        f"  seam-band mean relational mismatch   = {safe_mean(nodes.loc[seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        f"  off-seam mean relational mismatch    = {safe_mean(nodes.loc[~seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        "",
        "Correlations",
        f"  corr(sym_traceless_norm, relational mismatch) = {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.4f}",
        f"  corr(sym_traceless_norm, {transport_col})     = {safe_corr(nodes['sym_traceless_norm'], nodes[transport_col]):.4f}",
        f"  corr(sym_traceless_norm, distance_to_seam)    = {safe_corr(nodes['sym_traceless_norm'], nodes['distance_to_seam']):.4f}",
        f"  corr(relational mismatch, distance_to_seam)   = {safe_corr(nodes['neighbor_direction_mismatch_mean'], nodes['distance_to_seam']):.4f}",
        f"  corr(sym_traceless_norm, node_holonomy_proxy) = {safe_corr(nodes['sym_traceless_norm'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        f"  corr(relational mismatch, node_holonomy_proxy)= {safe_corr(nodes['neighbor_direction_mismatch_mean'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        "",
        "Hotspot overlap",
        f"  n_anisotropy_hotspots = {int(nodes['anisotropy_hotspot'].sum())}",
        f"  n_relational_hotspots = {int(nodes['relational_hotspot'].sum())}",
        f"  n_shared_hotspots     = {int(nodes['shared_hotspot'].sum())}",
        f"  top_{cfg.top_k_overlap}_overlap     = {overlap}",
        "",
        "Top shared hotspots",
    ]

    top_shared = nodes[nodes["shared_hotspot"] == 1].copy()
    top_shared = top_shared.sort_values(
        ["sym_traceless_norm", "neighbor_direction_mismatch_mean"],
        ascending=False,
    ).head(10)

    for _, row in top_shared.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, "
            f"r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"sym_traceless_norm={float(row['sym_traceless_norm']):.6f}, "
            f"neighbor_direction_mismatch_mean={float(row['neighbor_direction_mismatch_mean']):.6f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )

    return "\n".join(lines)


def render_figure(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold

    fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1.15], height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_overlap = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")

    # anisotropy field
    if len(seam_draw):
        ax_a.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.5, alpha=0.65, zorder=1)
        ax_a.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.6, alpha=0.96, zorder=2)

    sc_a = ax_a.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce"),
        cmap="viridis",
        s=92,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )
    ax_a.scatter(
        nodes.loc[nodes["anisotropy_hotspot"] == 1, "mds1"],
        nodes.loc[nodes["anisotropy_hotspot"] == 1, "mds2"],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )
    cbar_a = fig.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.02)
    cbar_a.set_label("sym. traceless norm")
    ax_a.set_title("Response anisotropy", fontsize=15, pad=8)
    ax_a.set_xlabel("MDS 1")
    ax_a.set_ylabel("MDS 2")
    ax_a.grid(alpha=0.08)
    ax_a.set_aspect("equal", adjustable="box")

    # relational field
    if len(seam_draw):
        ax_b.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.5, alpha=0.65, zorder=1)
        ax_b.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.6, alpha=0.96, zorder=2)

    sc_b = ax_b.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["neighbor_direction_mismatch_mean"], errors="coerce"),
        cmap="magma",
        s=92,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )
    ax_b.scatter(
        nodes.loc[nodes["relational_hotspot"] == 1, "mds1"],
        nodes.loc[nodes["relational_hotspot"] == 1, "mds2"],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )
    cbar_b = fig.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.02)
    cbar_b.set_label("relational mismatch (deg)")
    ax_b.set_title("Relational obstruction", fontsize=15, pad=8)
    ax_b.set_xlabel("MDS 1")
    ax_b.set_ylabel("MDS 2")
    ax_b.grid(alpha=0.08)
    ax_b.set_aspect("equal", adjustable="box")

    # scatter relation
    x = pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce")
    y = pd.to_numeric(nodes["neighbor_direction_mismatch_mean"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.86)
    ax_sc.set_xlabel("sym. traceless norm")
    ax_sc.set_ylabel("relational mismatch (deg)")
    ax_sc.set_title("Field correspondence", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98,
        0.05,
        f"corr = {safe_corr(x, y):.3f}",
        transform=ax_sc.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    # seam/off-seam bars
    bar_labels = ["anisotropy", "relational"]
    seam_vals = [
        safe_mean(nodes.loc[seam_mask, "sym_traceless_norm"]),
        safe_mean(nodes.loc[seam_mask, "neighbor_direction_mismatch_mean"]),
    ]
    off_vals = [
        safe_mean(nodes.loc[~seam_mask, "sym_traceless_norm"]),
        safe_mean(nodes.loc[~seam_mask, "neighbor_direction_mismatch_mean"]),
    ]
    xx = np.arange(len(bar_labels))
    width = 0.34
    ax_bar.bar(xx - width / 2, seam_vals, width, label="seam-band", alpha=0.9)
    ax_bar.bar(xx + width / 2, off_vals, width, label="off-seam", alpha=0.9)
    ax_bar.set_xticks(xx)
    ax_bar.set_xticklabels(bar_labels)
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")
    ax_bar.legend()

    # hotspot overlap map
    if len(seam_draw):
        ax_overlap.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.5, alpha=0.65, zorder=1)
        ax_overlap.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.6, alpha=0.96, zorder=2)

    base = nodes.copy()
    ax_overlap.scatter(base["mds1"], base["mds2"], s=48, c="lightgray", alpha=0.55, linewidths=0, zorder=2.5)
    aniso_only = base[(base["anisotropy_hotspot"] == 1) & (base["shared_hotspot"] == 0)]
    rel_only = base[(base["relational_hotspot"] == 1) & (base["shared_hotspot"] == 0)]
    shared = base[base["shared_hotspot"] == 1]

    if len(aniso_only):
        ax_overlap.scatter(aniso_only["mds1"], aniso_only["mds2"], s=120, c="#2A9D8F", alpha=0.95, zorder=3)
    if len(rel_only):
        ax_overlap.scatter(rel_only["mds1"], rel_only["mds2"], s=120, c="#B23A48", alpha=0.95, zorder=3.2)
    if len(shared):
        ax_overlap.scatter(shared["mds1"], shared["mds2"], s=155, c="#FFD166", edgecolors="black", linewidths=1.0, alpha=0.98, zorder=4)

    top_shared = shared.sort_values(
        ["sym_traceless_norm", "neighbor_direction_mismatch_mean"], ascending=False
    ).head(cfg.top_k_labels)
    for _, row in top_shared.iterrows():
        ax_overlap.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=5,
        )

    ax_overlap.set_title("Hotspot overlap", fontsize=14, pad=8)
    ax_overlap.set_xlabel("MDS 1")
    ax_overlap.set_ylabel("MDS 2")
    ax_overlap.grid(alpha=0.08)
    ax_overlap.set_aspect("equal", adjustable="box")

    # diagnostics box
    ax_diag.axis("off")
    transport_col = "transport_align_mean_deg" if "transport_align_mean_deg" in nodes.columns else "neighbor_direction_mismatch_mean"

    text = (
        "OBS-025 diagnostics\n\n"
        f"n anisotropy hotspots: {int(nodes['anisotropy_hotspot'].sum())}\n"
        f"n relational hotspots: {int(nodes['relational_hotspot'].sum())}\n"
        f"n shared hotspots: {int(nodes['shared_hotspot'].sum())}\n\n"
        f"corr(aniso, relational): {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.3f}\n"
        f"corr(aniso, {transport_col}): {safe_corr(nodes['sym_traceless_norm'], nodes[transport_col]):.3f}\n\n"
        "cyan   = anisotropy-only hotspot\n"
        "crimson= relational-only hotspot\n"
        "gold   = shared hotspot"
    )
    ax_diag.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-025 anisotropy vs relational obstruction", fontsize=20)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare response anisotropy against relational obstruction.")
    parser.add_argument("--anisotropy-csv", default=Config.anisotropy_csv)
    parser.add_argument("--mismatch-csv", default=Config.mismatch_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--hotspot-quantile", type=float, default=Config.hotspot_quantile)
    parser.add_argument("--top-k-overlap", type=int, default=Config.top_k_overlap)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        anisotropy_csv=args.anisotropy_csv,
        mismatch_csv=args.mismatch_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        hotspot_quantile=args.hotspot_quantile,
        top_k_overlap=args.top_k_overlap,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, seam = load_inputs(cfg)
    nodes = add_hotspots(nodes, cfg.hotspot_quantile)

    csv_path = outdir / "obs025_anisotropy_vs_relational_obstruction_nodes.csv"
    txt_path = outdir / "obs025_anisotropy_vs_relational_obstruction_summary.txt"
    png_path = outdir / "obs025_anisotropy_vs_relational_obstruction_figure.png"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg), encoding="utf-8")
    render_figure(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
