#!/usr/bin/env python3
"""
OBS-025 — Two-field seam panel.

Present the seam as a multi-field regime by comparing:

1. response anisotropy
   -> symmetric traceless norm of the response tensor

2. relational obstruction
   -> neighbor-level directional mismatch

and their hotspot overlap structure.

Inputs
------
outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/obs025_two_field_seam_panel/
  obs025_two_field_seam_panel_summary.txt
  obs025_two_field_seam_panel.png
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
    nodes_csv: str = "outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs025_two_field_seam_panel"
    seam_threshold: float = 0.15
    top_k_labels: int = 4


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
    nodes = pd.read_csv(cfg.nodes_csv)
    seam = pd.read_csv(cfg.seam_csv)

    for df in (nodes, seam):
        for col in df.columns:
            if col not in {"path_id", "path_family", "dominant_component"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    required = [
        "node_id",
        "mds1",
        "mds2",
        "distance_to_seam",
        "sym_traceless_norm",
        "neighbor_direction_mismatch_mean",
        "anisotropy_hotspot",
        "relational_hotspot",
        "shared_hotspot",
    ]
    missing = [c for c in required if c not in nodes.columns]
    if missing:
        raise ValueError(f"Missing required node columns: {missing}")

    return nodes, seam


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    aniso_only = (nodes["anisotropy_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)
    rel_only = (nodes["relational_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)
    shared = nodes["shared_hotspot"] == 1

    lines = [
        "=== OBS-025 Two-Field Seam Panel Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Field summary",
        f"  mean sym_traceless_norm          = {safe_mean(nodes['sym_traceless_norm']):.6f}",
        f"  mean relational mismatch         = {safe_mean(nodes['neighbor_direction_mismatch_mean']):.6f}",
        f"  corr(anisotropy, relational)     = {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.4f}",
        "",
        "Seam localization",
        f"  seam-band mean anisotropy        = {safe_mean(nodes.loc[seam_mask, 'sym_traceless_norm']):.6f}",
        f"  off-seam mean anisotropy         = {safe_mean(nodes.loc[~seam_mask, 'sym_traceless_norm']):.6f}",
        f"  seam-band mean relational        = {safe_mean(nodes.loc[seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        f"  off-seam mean relational         = {safe_mean(nodes.loc[~seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        "",
        "Hotspot structure",
        f"  anisotropy-only hotspots         = {int(aniso_only.sum())}",
        f"  relational-only hotspots         = {int(rel_only.sum())}",
        f"  shared hotspots                  = {int(shared.sum())}",
        "",
        "Interpretation",
        "- both fields are seam-biased",
        "- anisotropy and relational obstruction only weakly align pointwise",
        "- the seam therefore behaves as a multi-field structural regime",
    ]
    return "\n".join(lines)


def render_panel(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    aniso_only = nodes[(nodes["anisotropy_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)].copy()
    rel_only = nodes[(nodes["relational_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)].copy()
    shared = nodes[nodes["shared_hotspot"] == 1].copy()

    fig = plt.figure(figsize=(16.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1.05], height_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_overlap = fig.add_subplot(gs[0, 2])
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")

    def draw_seam(ax):
        if len(seam_draw):
            ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.5, alpha=0.65, zorder=1)
            ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.6, alpha=0.96, zorder=2)

    # Panel A: anisotropy field
    draw_seam(ax_a)
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
        nodes.loc[seam_mask, "mds1"],
        nodes.loc[seam_mask, "mds2"],
        s=160,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )
    cbar_a = fig.colorbar(sc_a, ax=ax_a, fraction=0.046, pad=0.02)
    cbar_a.set_label("sym. traceless norm")
    ax_a.set_title("Field A — response anisotropy", fontsize=15, pad=8)
    ax_a.set_xlabel("MDS 1")
    ax_a.set_ylabel("MDS 2")
    ax_a.grid(alpha=0.08)
    ax_a.set_aspect("equal", adjustable="box")

    # Panel B: relational field
    draw_seam(ax_b)
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
        nodes.loc[seam_mask, "mds1"],
        nodes.loc[seam_mask, "mds2"],
        s=160,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )
    cbar_b = fig.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.02)
    cbar_b.set_label("relational mismatch (deg)")
    ax_b.set_title("Field B — relational obstruction", fontsize=15, pad=8)
    ax_b.set_xlabel("MDS 1")
    ax_b.set_ylabel("MDS 2")
    ax_b.grid(alpha=0.08)
    ax_b.set_aspect("equal", adjustable="box")

    # Panel C: overlap structure
    draw_seam(ax_overlap)
    ax_overlap.scatter(nodes["mds1"], nodes["mds2"], s=42, c="lightgray", alpha=0.55, linewidths=0, zorder=2.5)

    if len(aniso_only):
        ax_overlap.scatter(aniso_only["mds1"], aniso_only["mds2"], s=120, c="#2A9D8F", alpha=0.95, zorder=3)
    if len(rel_only):
        ax_overlap.scatter(rel_only["mds1"], rel_only["mds2"], s=120, c="#B23A48", alpha=0.95, zorder=3.2)
    if len(shared):
        ax_overlap.scatter(shared["mds1"], shared["mds2"], s=155, c="#FFD166", edgecolors="black", linewidths=1.0, alpha=0.98, zorder=4)

    top_shared = shared.sort_values(
        ["sym_traceless_norm", "neighbor_direction_mismatch_mean"],
        ascending=False,
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

    ax_overlap.set_title("Hotspot overlap structure", fontsize=15, pad=8)
    ax_overlap.set_xlabel("MDS 1")
    ax_overlap.set_ylabel("MDS 2")
    ax_overlap.grid(alpha=0.08)
    ax_overlap.set_aspect("equal", adjustable="box")

    # Panel D: field correspondence scatter
    x = pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce")
    y = pd.to_numeric(nodes["neighbor_direction_mismatch_mean"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.86)
    ax_sc.set_xlabel("sym. traceless norm")
    ax_sc.set_ylabel("relational mismatch (deg)")
    ax_sc.set_title("Anisotropy vs relational obstruction", fontsize=14, pad=8)
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

    # Panel E: normalized seam enrichment
    aniso_seam = safe_mean(nodes.loc[seam_mask, "sym_traceless_norm"])
    aniso_off = safe_mean(nodes.loc[~seam_mask, "sym_traceless_norm"])
    rel_seam = safe_mean(nodes.loc[seam_mask, "neighbor_direction_mismatch_mean"])
    rel_off = safe_mean(nodes.loc[~seam_mask, "neighbor_direction_mismatch_mean"])

    aniso_ratio = aniso_seam / aniso_off if np.isfinite(aniso_off) and abs(aniso_off) > 1e-12 else np.nan
    rel_ratio = rel_seam / rel_off if np.isfinite(rel_off) and abs(rel_off) > 1e-12 else np.nan

    labels = ["anisotropy", "relational"]
    vals = [aniso_ratio, rel_ratio]

    ax_bar.bar(labels, vals, alpha=0.9)
    ax_bar.axhline(1.0, color="black", linewidth=1.2, alpha=0.8)
    ax_bar.set_ylabel("seam / off-seam ratio")
    ax_bar.set_title("Normalized seam enrichment", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax_bar.text(
                i,
                v + 0.02 * max(vals),
                f"{v:.2f}×",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Panel F: interpretation box
    transport_text = (
        "OBS-025 synthesis\n\n"
        f"corr(aniso, relational): {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.3f}\n\n"
        f"anisotropy-only hotspots: {len(aniso_only)}\n"
        f"relational-only hotspots: {len(rel_only)}\n"
        f"shared hotspots: {len(shared)}\n\n"
        "cyan    = anisotropy-only\n"
        "crimson = relational-only\n"
        "gold    = shared\n\n"
        "Conclusion:\n"
        "both fields are seam-enriched,\n"
        "but only weakly aligned\n"
        "node-by-node."
    )
    ax_diag.text(
        0.02,
        0.98,
        transport_text,
        va="top",
        ha="left",
        fontsize=10.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-025 two-field seam panel", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OBS-025 two-field seam panel.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, seam = load_inputs(cfg)

    txt_path = outdir / "obs025_two_field_seam_panel_summary.txt"
    png_path = outdir / "obs025_two_field_seam_panel.png"

    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")
    render_panel(cfg, nodes, seam, png_path)

    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
