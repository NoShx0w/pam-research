#!/usr/bin/env python3
"""
OBS-027 — Seam-regime synthesis.

Unify OBS-023 through OBS-026 into one canonical seam figure.

Panels
------
A. Relational obstruction field
B. Response anisotropy field
C. Two-field hotspot overlap
D. Family two-field occupancy summary
E. Seam-regime synthesis text box

Inputs
------
outputs/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_nodes.csv
outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv
outputs/obs022_scene_bundle/scene_seam.csv

Outputs
-------
outputs/obs027_seam_regime_synthesis/
  obs027_seam_regime_synthesis_summary.txt
  obs027_seam_regime_synthesis.png
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
    family_summary_csv: str = "outputs/obs026_family_two_field_occupancy/family_two_field_class_summary.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/obs027_seam_regime_synthesis"
    seam_threshold: float = 0.15
    top_k_shared_labels: int = 4


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


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


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(cfg.nodes_csv)
    fam = pd.read_csv(cfg.family_summary_csv)
    seam = pd.read_csv(cfg.seam_csv)

    for df in (nodes, fam, seam):
        for col in df.columns:
            if col not in {"path_id", "path_family", "route_class", "hotspot_class", "dominant_component"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    required_nodes = [
        "node_id", "mds1", "mds2", "distance_to_seam",
        "sym_traceless_norm", "neighbor_direction_mismatch_mean",
        "anisotropy_hotspot", "relational_hotspot", "shared_hotspot",
    ]
    missing = [c for c in required_nodes if c not in nodes.columns]
    if missing:
        raise ValueError(f"Missing required node columns: {missing}")

    if "route_class" in fam.columns:
        order = {k: i for i, k in enumerate(CLASS_ORDER)}
        fam["order"] = fam["route_class"].map(lambda x: order.get(x, 999))
        fam = fam.sort_values("order").drop(columns="order").reset_index(drop=True)

    return nodes, fam, seam


def build_summary(nodes: pd.DataFrame, fam: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    aniso_only = ((nodes["anisotropy_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)).sum()
    rel_only = ((nodes["relational_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)).sum()
    shared = (nodes["shared_hotspot"] == 1).sum()

    lines = [
        "=== OBS-027 Seam Regime Synthesis Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Field synthesis",
        f"  mean anisotropy                = {safe_mean(nodes['sym_traceless_norm']):.6f}",
        f"  mean relational obstruction    = {safe_mean(nodes['neighbor_direction_mismatch_mean']):.6f}",
        f"  corr(aniso, relational)        = {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.4f}",
        f"  seam-band mean anisotropy      = {safe_mean(nodes.loc[seam_mask, 'sym_traceless_norm']):.6f}",
        f"  off-seam mean anisotropy       = {safe_mean(nodes.loc[~seam_mask, 'sym_traceless_norm']):.6f}",
        f"  seam-band mean relational      = {safe_mean(nodes.loc[seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        f"  off-seam mean relational       = {safe_mean(nodes.loc[~seam_mask, 'neighbor_direction_mismatch_mean']):.6f}",
        "",
        "Hotspot structure",
        f"  anisotropy-only hotspots       = {int(aniso_only)}",
        f"  relational-only hotspots       = {int(rel_only)}",
        f"  shared hotspots                = {int(shared)}",
        "",
        "Family occupancy synthesis",
    ]

    for _, row in fam.iterrows():
        lines.append(
            f"  {row['route_class']}: "
            f"row(aniso-only)={float(row['row_share_anisotropy_only']):.4f}, "
            f"row(rel-only)={float(row['row_share_relational_only']):.4f}, "
            f"row(shared)={float(row['row_share_shared']):.4f}, "
            f"touch(shared)={float(row['path_touch_shared']):.4f}, "
            f"mean_d2s={float(row['mean_distance_to_seam']):.4f}"
        )

    lines.extend(
        [
            "",
            "Canonical interpretation",
            "- the seam is a multi-field structural regime",
            "- relational obstruction is the stronger seam discriminator",
            "- response anisotropy is a distinct seam-side field",
            "- hotspot overlap is small, so the fields are not reducible to one another",
            "- families differ by residency pattern within this two-field regime",
        ]
    )
    return "\n".join(lines)


def render_figure(cfg: Config, nodes: pd.DataFrame, fam: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold
    aniso_only = nodes[(nodes["anisotropy_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)].copy()
    rel_only = nodes[(nodes["relational_hotspot"] == 1) & (nodes["shared_hotspot"] == 0)].copy()
    shared = nodes[nodes["shared_hotspot"] == 1].copy()

    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1.15], height_ratios=[1.0, 1.0])

    ax_rel = fig.add_subplot(gs[0, 0])
    ax_aniso = fig.add_subplot(gs[0, 1])
    ax_overlap = fig.add_subplot(gs[0, 2])
    ax_occ = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")

    def draw_seam(ax):
        if len(seam_draw):
            ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=5.5, alpha=0.65, zorder=1)
            ax.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.6, alpha=0.96, zorder=2)

    # A relational
    draw_seam(ax_rel)
    sc_rel = ax_rel.scatter(
        nodes["mds1"], nodes["mds2"],
        c=pd.to_numeric(nodes["neighbor_direction_mismatch_mean"], errors="coerce"),
        cmap="magma", s=92, alpha=0.96, linewidths=0.35, edgecolors="white", zorder=3
    )
    ax_rel.scatter(
        nodes.loc[seam_mask, "mds1"], nodes.loc[seam_mask, "mds2"],
        s=160, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4
    )
    cbar_rel = fig.colorbar(sc_rel, ax=ax_rel, fraction=0.046, pad=0.02)
    cbar_rel.set_label("relational mismatch (deg)")
    ax_rel.set_title("A — relational obstruction", fontsize=15, pad=8)
    ax_rel.set_xlabel("MDS 1")
    ax_rel.set_ylabel("MDS 2")
    ax_rel.grid(alpha=0.08)
    ax_rel.set_aspect("equal", adjustable="box")

    # B anisotropy
    draw_seam(ax_aniso)
    sc_aniso = ax_aniso.scatter(
        nodes["mds1"], nodes["mds2"],
        c=pd.to_numeric(nodes["sym_traceless_norm"], errors="coerce"),
        cmap="viridis", s=92, alpha=0.96, linewidths=0.35, edgecolors="white", zorder=3
    )
    ax_aniso.scatter(
        nodes.loc[seam_mask, "mds1"], nodes.loc[seam_mask, "mds2"],
        s=160, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4
    )
    cbar_aniso = fig.colorbar(sc_aniso, ax=ax_aniso, fraction=0.046, pad=0.02)
    cbar_aniso.set_label("sym. traceless norm")
    ax_aniso.set_title("B — response anisotropy", fontsize=15, pad=8)
    ax_aniso.set_xlabel("MDS 1")
    ax_aniso.set_ylabel("MDS 2")
    ax_aniso.grid(alpha=0.08)
    ax_aniso.set_aspect("equal", adjustable="box")

    # C overlap
    draw_seam(ax_overlap)
    ax_overlap.scatter(nodes["mds1"], nodes["mds2"], s=42, c="lightgray", alpha=0.55, linewidths=0, zorder=2.5)
    if len(aniso_only):
        ax_overlap.scatter(aniso_only["mds1"], aniso_only["mds2"], s=120, c="#2A9D8F", alpha=0.95, zorder=3)
    if len(rel_only):
        ax_overlap.scatter(rel_only["mds1"], rel_only["mds2"], s=120, c="#B23A48", alpha=0.95, zorder=3.2)
    if len(shared):
        ax_overlap.scatter(shared["mds1"], shared["mds2"], s=155, c="#FFD166",
                           edgecolors="black", linewidths=1.0, alpha=0.98, zorder=4)

    top_shared = shared.sort_values(
        ["sym_traceless_norm", "neighbor_direction_mismatch_mean"],
        ascending=False,
    ).head(cfg.top_k_shared_labels)
    for _, row in top_shared.iterrows():
        ax_overlap.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=5,
        )

    ax_overlap.set_title("C — two-field hotspot overlap", fontsize=15, pad=8)
    ax_overlap.set_xlabel("MDS 1")
    ax_overlap.set_ylabel("MDS 2")
    ax_overlap.grid(alpha=0.08)
    ax_overlap.set_aspect("equal", adjustable="box")

    # D family occupancy
    classes = fam["route_class"].tolist()
    x = np.arange(len(classes))
    width = 0.24

    ax_occ.bar(x - width, fam["row_share_anisotropy_only"], width, label="anisotropy-only")
    ax_occ.bar(x, fam["row_share_relational_only"], width, label="relational-only")
    ax_occ.bar(x + width, fam["row_share_shared"], width, label="shared")
    ax_occ.set_xticks(x)
    ax_occ.set_xticklabels(classes, rotation=12)
    ax_occ.set_ylabel("row share")
    ax_occ.set_title("D — family occupancy on seam fields", fontsize=15, pad=8)
    ax_occ.grid(alpha=0.15, axis="y")
    ax_occ.legend()

    # E synthesis box
    ax_diag.axis("off")

    aniso_ratio = (
        safe_mean(nodes.loc[seam_mask, "sym_traceless_norm"])
        / safe_mean(nodes.loc[~seam_mask, "sym_traceless_norm"])
    )
    rel_ratio = (
        safe_mean(nodes.loc[seam_mask, "neighbor_direction_mismatch_mean"])
        / safe_mean(nodes.loc[~seam_mask, "neighbor_direction_mismatch_mean"])
    )

    text = (
        "OBS-027 synthesis\n\n"
        f"corr(aniso, relational): {safe_corr(nodes['sym_traceless_norm'], nodes['neighbor_direction_mismatch_mean']):.3f}\n"
        f"aniso seam enrichment: {aniso_ratio:.2f}×\n"
        f"rel seam enrichment: {rel_ratio:.2f}×\n\n"
        f"aniso-only hotspots: {len(aniso_only)}\n"
        f"rel-only hotspots: {len(rel_only)}\n"
        f"shared hotspots: {len(shared)}\n\n"
        "stable corridor:\nstrongest two-field resident\n\n"
        "reorg-heavy:\nbalanced but more diffuse\n\n"
        "branch-exit:\ntouch-and-leave class\n\n"
        "Conclusion:\n"
        "the seam is a multi-field,\n"
        "family-selective residency\n"
        "landscape."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-027 seam-regime synthesis", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OBS-027 seam-regime synthesis.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--family-summary-csv", default=Config.family_summary_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-shared-labels", type=int, default=Config.top_k_shared_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        family_summary_csv=args.family_summary_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_shared_labels=args.top_k_shared_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes, fam, seam = load_inputs(cfg)

    txt_path = outdir / "obs027_seam_regime_synthesis_summary.txt"
    png_path = outdir / "obs027_seam_regime_synthesis.png"

    txt_path.write_text(build_summary(nodes, fam, cfg.seam_threshold), encoding="utf-8")
    render_figure(cfg, nodes, fam, seam, png_path)

    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
