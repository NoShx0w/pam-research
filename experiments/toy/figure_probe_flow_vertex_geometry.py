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
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    outdir: str = "outputs/figure_probe_flow_vertex_geometry"
    reps_per_family: int = 6
    seam_threshold: float = 0.15
    hub_top_k: int = 12


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",   # gold
    "reorganization_heavy": "#B23A48",   # crimson
}


def load_nodes(phase_csv: str, lazarus_csv: str) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    laz = pd.read_csv(lazarus_csv).copy()

    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase"] if c in phase.columns]
    keep_laz = [c for c in ["node_id", "r", "alpha", "distance_to_seam", "lazarus_score"] if c in laz.columns]

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in phase.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    nodes = phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam", "lazarus_score"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
    return nodes


def load_paths(path_nodes_csv: str, family_csv: str, nodes: pd.DataFrame) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    extra = [c for c in ["distance_to_seam", "lazarus_score"] if c in nodes.columns and c not in paths.columns]
    if join_cols and extra:
        paths = paths.merge(nodes[join_cols + extra], on=join_cols, how="left")

    for col in ["mds1", "mds2", "distance_to_seam", "lazarus_score"]:
        if col in paths.columns:
            paths[col] = pd.to_numeric(paths[col], errors="coerce")
    return paths


def pick_hubs(paths: pd.DataFrame, top_k: int) -> pd.DataFrame:
    traffic = (
        paths.groupby(["node_id", "r", "alpha"], as_index=False)
        .agg(n_unique_paths=("path_id", "nunique"))
        .sort_values("n_unique_paths", ascending=False)
        .reset_index(drop=True)
    )
    return traffic.head(top_k).copy()


def pick_representative_paths(paths: pd.DataFrame, reps_per_family: int) -> pd.DataFrame:
    grouped = (
        paths.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_lazarus=("lazarus_score", "mean"),
            mean_distance_to_seam=("distance_to_seam", "mean"),
        )
    )

    keep_ids: list[str] = []
    for fam in ["stable_seam_corridor", "reorganization_heavy"]:
        sub = grouped[grouped["path_family"] == fam].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values(
            ["mean_lazarus", "n_steps", "mean_distance_to_seam"],
            ascending=[False, False, True],
        )
        keep_ids.extend(sub.head(reps_per_family)["path_id"].tolist())

    return paths[paths["path_id"].isin(keep_ids)].copy()


def draw_background(ax, nodes: pd.DataFrame):
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=nodes["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=70,
        alpha=0.28,
        linewidths=0,
        zorder=1,
    )
    return sc


def draw_paths(
    ax,
    reps: pd.DataFrame,
    hubs_full: pd.DataFrame,
    seam_threshold: float,
):
    # hub lookup for marker sizing
    hub_lookup = {
        (int(r.node_id), float(r.r), float(r.alpha)): float(r.n_unique_paths)
        for r in hubs_full.itertuples(index=False)
    }
    max_hub = max(hub_lookup.values()) if hub_lookup else 1.0

    laz_min = pd.to_numeric(reps["lazarus_score"], errors="coerce").min()
    laz_max = pd.to_numeric(reps["lazarus_score"], errors="coerce").max()

    last_vertex_scatter = None

    for fam in ["reorganization_heavy", "stable_seam_corridor"]:
        sub = reps[reps["path_family"] == fam].copy()
        color = FAMILY_COLORS[fam]

        for _, grp in sub.groupby("path_id", sort=False):
            grp = grp.sort_values("step").copy()

            # edges
            ax.plot(
                grp["mds1"],
                grp["mds2"],
                color=color,
                linewidth=2.6 if fam == "stable_seam_corridor" else 1.8,
                alpha=0.95 if fam == "stable_seam_corridor" else 0.75,
                zorder=3 if fam == "stable_seam_corridor" else 2,
            )

            # vertices colored by lazarus
            last_vertex_scatter = ax.scatter(
                grp["mds1"],
                grp["mds2"],
                c=grp["lazarus_score"],
                cmap="magma",
                vmin=laz_min,
                vmax=laz_max,
                s=80 if fam == "stable_seam_corridor" else 60,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=5,
            )

            # seam-contact vertex rings
            seam_mask = pd.to_numeric(grp["distance_to_seam"], errors="coerce") <= seam_threshold
            seam_pts = grp[seam_mask].copy()
            if len(seam_pts):
                ax.scatter(
                    seam_pts["mds1"],
                    seam_pts["mds2"],
                    s=150,
                    facecolors="none",
                    edgecolors="#8E5EA2",
                    linewidths=1.6,
                    alpha=0.85,
                    zorder=6,
                )

            # hub overlays
            hub_sizes = []
            hub_x = []
            hub_y = []
            for row in grp.itertuples(index=False):
                key = (int(row.node_id), float(row.r), float(row.alpha))
                if key in hub_lookup:
                    occ = hub_lookup[key] / max_hub
                    hub_sizes.append(150 + 220 * occ)
                    hub_x.append(float(row.mds1))
                    hub_y.append(float(row.mds2))

            if hub_sizes:
                ax.scatter(
                    hub_x,
                    hub_y,
                    s=hub_sizes,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    alpha=0.9,
                    zorder=7,
                )

            # start/end markers
            start = grp.iloc[0]
            end = grp.iloc[-1]
            ax.scatter(
                [start["mds1"]], [start["mds2"]],
                marker="*",
                s=320,
                c="#2B6CB0",
                edgecolors="black",
                linewidths=0.8,
                zorder=8,
            )
            ax.scatter(
                [end["mds1"]], [end["mds2"]],
                marker="X",
                s=260,
                c="#ED8936",
                edgecolors="black",
                linewidths=0.6,
                zorder=8,
            )

    return last_vertex_scatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Render vertex-emphasized probe flow geometry in the style of a sampled graph plate.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--outdir", default="outputs/figure_probe_flow_vertex_geometry")
    parser.add_argument("--reps-per-family", type=int, default=6)
    parser.add_argument("--seam-threshold", type=float, default=0.15)
    parser.add_argument("--hub-top-k", type=int, default=12)
    args = parser.parse_args()

    cfg = Config(
        phase_csv=args.phase_csv,
        lazarus_csv=args.lazarus_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        outdir=args.outdir,
        reps_per_family=args.reps_per_family,
        seam_threshold=args.seam_threshold,
        hub_top_k=args.hub_top_k,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(cfg.phase_csv, cfg.lazarus_csv)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv, nodes)
    reps = pick_representative_paths(paths, cfg.reps_per_family)
    hubs = pick_hubs(paths, cfg.hub_top_k)

    fig, ax = plt.subplots(figsize=(12, 9))
    bg = draw_background(ax, nodes)
    vertex_scatter = draw_paths(ax, reps, hubs, cfg.seam_threshold)

    ax.set_title("Probe flow and transition geometry", fontsize=20, pad=12)
    ax.set_xlabel("MDS 1", fontsize=16)
    ax.set_ylabel("MDS 2", fontsize=16)
    ax.grid(alpha=0.15)

    # concise in-figure legend text
    ax.text(
        0.02, 0.98,
        "gold lines = stable_seam_corridor\n"
        "crimson lines = reorganization_heavy\n"
        "purple rings = seam-contact vertices\n"
        "black rings = top traffic hubs\n"
        "blue star = path start, orange X = path end",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    if vertex_scatter is not None:
        cbar = fig.colorbar(vertex_scatter, ax=ax, fraction=0.04, pad=0.045)
        cbar.set_label("lazarus_score", fontsize=13)

    fig.tight_layout()

    outpath = outdir / "figure_probe_flow_vertex_geometry.png"
    fig.savefig(outpath, dpi=240)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
