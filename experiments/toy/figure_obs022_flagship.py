#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    phase_csv: str = "outputs/fim_phase/signed_phase_coords.csv"
    seam_csv: str = "outputs/fim_phase/phase_boundary_mds_backprojected.csv"
    lazarus_csv: str = "outputs/fim_lazarus/lazarus_scores.csv"
    response_csv: str = "outputs/fim_response_operator/response_operator_nodes.csv"
    family_csv: str = "outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv"
    path_nodes_csv: str = "outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv"
    convergence_csv: str = "outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv"
    outdir: str = "outputs/figure_obs022_flagship"
    top_hubs: int = 12
    reps_per_family: int = 10
    eigen_stride: int = 3


FAMILY_ORDER = [
    "stable_seam_corridor",
    "reorganization_heavy",
    "settled_distant",
    "off_seam_reorganizing",
]


def principal_response_eig(
    t_xx: np.ndarray,
    t_xy: np.ndarray,
    t_yx: np.ndarray,
    t_yy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t_xx)
    eig1 = np.full(n, np.nan)
    eig2 = np.full(n, np.nan)
    theta = np.full(n, np.nan)

    for i in range(n):
        M = np.array([[t_xx[i], t_xy[i]], [t_yx[i], t_yy[i]]], dtype=float)
        if not np.isfinite(M).all():
            continue
        vals, vecs = np.linalg.eig(M)
        order = np.argsort(np.abs(vals))[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        v = np.real(vecs[:, 0])

        eig1[i] = float(np.real(vals[0]))
        eig2[i] = float(np.real(vals[1]))
        theta[i] = float(math.atan2(v[1], v[0]))

    return eig1, eig2, theta


def load_phase_nodes(phase_csv: str, lazarus_csv: str) -> pd.DataFrame:
    phase = pd.read_csv(phase_csv).copy()
    laz = pd.read_csv(lazarus_csv).copy()

    keep_phase = [c for c in ["node_id", "r", "alpha", "mds1", "mds2", "signed_phase"] if c in phase.columns]
    keep_laz = [c for c in ["node_id", "r", "alpha", "distance_to_seam", "lazarus_score"] if c in laz.columns]

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in phase.columns and c in laz.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]

    df = phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")
    return df


def load_response_field(response_csv: str) -> pd.DataFrame:
    rsp = pd.read_csv(response_csv).copy()
    keep = [c for c in ["r", "alpha", "mds1", "mds2", "T_xx", "T_xy", "T_yx", "T_yy", "response_strength"] if c in rsp.columns]
    rsp = rsp[keep].copy()

    eig1, eig2, theta = principal_response_eig(
        pd.to_numeric(rsp["T_xx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_xy"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_yx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_yy"], errors="coerce").to_numpy(dtype=float),
    )
    rsp["rsp_eig1"] = eig1
    rsp["rsp_eig2"] = eig2
    rsp["rsp_theta"] = theta
    return rsp


def load_paths(path_nodes_csv: str, family_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    paths = paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")
    return paths


def pick_hubs(paths: pd.DataFrame, nodes: pd.DataFrame, top_k: int) -> pd.DataFrame:
    traffic = (
        paths.groupby(["node_id", "r", "alpha"], as_index=False)
        .agg(n_unique_paths=("path_id", "nunique"))
        .sort_values("n_unique_paths", ascending=False)
        .reset_index(drop=True)
    )
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in traffic.columns and c in nodes.columns]
    if not join_cols:
        join_cols = ["r", "alpha"]
    hubs = traffic.merge(
        nodes[[c for c in ["node_id", "r", "alpha", "mds1", "mds2", "distance_to_seam", "lazarus_score"] if c in nodes.columns]],
        on=join_cols,
        how="left",
    )
    return hubs.head(top_k).copy()


def pick_representative_paths(paths: pd.DataFrame, reps_per_family: int) -> pd.DataFrame:
    # representative = longest-ish family members with good seam coverage, but deterministic and simple
    g = (
        paths.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            mean_distance_to_seam=("distance_to_seam", "mean") if "distance_to_seam" in paths.columns else ("step", "count"),
        )
    )
    selected = []
    for fam in ["stable_seam_corridor", "reorganization_heavy"]:
        sub = g[g["path_family"] == fam].copy()
        if len(sub) == 0:
            continue
        sort_cols = ["n_steps"]
        ascending = [False]
        sub = sub.sort_values(sort_cols, ascending=ascending).head(reps_per_family)
        selected.extend(sub["path_id"].tolist())
    return paths[paths["path_id"].isin(selected)].copy()


def draw_background(ax, nodes: pd.DataFrame, seam: pd.DataFrame) -> None:
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=nodes["signed_phase"],
        s=70,
        alpha=0.9,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0,
        zorder=1,
    )
    if len(seam):
        seam = seam.sort_values("mds1")
        ax.plot(
            seam["mds1"],
            seam["mds2"],
            color="black",
            linewidth=2.4,
            alpha=0.9,
            zorder=3,
        )
    return sc


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.12)


def panel_a(ax, nodes: pd.DataFrame, seam: pd.DataFrame, hubs: pd.DataFrame, fig) -> None:
    sc = draw_background(ax, nodes, seam)

    sizes = 120 + 380 * (hubs["n_unique_paths"] / max(hubs["n_unique_paths"].max(), 1))
    ax.scatter(
        hubs["mds1"],
        hubs["mds2"],
        s=sizes,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=5,
    )
    ax.scatter(
        hubs["mds1"],
        hubs["mds2"],
        s=18,
        c="black",
        zorder=6,
    )

    for _, row in hubs.iterrows():
        ax.text(row["mds1"], row["mds2"], str(int(row["node_id"])), fontsize=8, zorder=7)

    style_axis(ax, "A. Phase manifold, seam, and routing hubs")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("signed phase")


def panel_b(ax, nodes: pd.DataFrame, seam: pd.DataFrame, reps: pd.DataFrame) -> None:
    draw_background(ax, nodes, seam)

    fam_styles = {
        "stable_seam_corridor": dict(linewidth=2.6, alpha=0.95),
        "reorganization_heavy": dict(linewidth=1.6, alpha=0.75),
    }

    for fam in ["reorganization_heavy", "stable_seam_corridor"]:
        sub = reps[reps["path_family"] == fam].copy()
        for path_id, grp in sub.groupby("path_id", sort=False):
            grp = grp.sort_values("step")
            ax.plot(
                grp["mds1"],
                grp["mds2"],
                zorder=4 if fam == "stable_seam_corridor" else 3,
                **fam_styles[fam],
            )

    # direct labels
    ax.text(0.03, 0.97, "thick = stable_seam_corridor\nthin = reorganization_heavy",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))

    style_axis(ax, "B. Seam-contact route bundles")


def panel_c(ax, nodes: pd.DataFrame, seam: pd.DataFrame, rsp: pd.DataFrame, stride: int) -> None:
    draw_background(ax, nodes, seam)

    sub = rsp.dropna(subset=["mds1", "mds2", "rsp_theta"]).copy().iloc[::max(stride, 1), :]
    u = np.cos(pd.to_numeric(sub["rsp_theta"], errors="coerce").to_numpy(dtype=float))
    v = np.sin(pd.to_numeric(sub["rsp_theta"], errors="coerce").to_numpy(dtype=float))

    ax.quiver(
        sub["mds1"], sub["mds2"],
        u, v,
        angles="xy", scale_units="xy", scale=10,
        width=0.0025,
        alpha=0.8,
        zorder=4,
    )

    style_axis(ax, "C. Response principal-direction field")


def panel_d(ax, tidy: pd.DataFrame) -> None:
    tidy = tidy.copy()
    tidy["scale_order"] = pd.to_numeric(tidy["scale"], errors="coerce")

    for fam in FAMILY_ORDER:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale_order")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["scale"],
            sub["share_paths"],
            marker="o",
            linewidth=2,
            label=fam,
        )

    ax.set_title("D. Family-share convergence")
    ax.set_xlabel("scale")
    ax.set_ylabel("share")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a flagship observatory figure for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--response-csv", default="outputs/fim_response_operator/response_operator_nodes.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--convergence-csv", default="outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_flagship")
    parser.add_argument("--top-hubs", type=int, default=12)
    parser.add_argument("--reps-per-family", type=int, default=10)
    parser.add_argument("--eigen-stride", type=int, default=3)
    args = parser.parse_args()

    cfg = Config(
        phase_csv=args.phase_csv,
        seam_csv=args.seam_csv,
        lazarus_csv=args.lazarus_csv,
        response_csv=args.response_csv,
        family_csv=args.family_csv,
        path_nodes_csv=args.path_nodes_csv,
        convergence_csv=args.convergence_csv,
        outdir=args.outdir,
        top_hubs=args.top_hubs,
        reps_per_family=args.reps_per_family,
        eigen_stride=args.eigen_stride,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_phase_nodes(cfg.phase_csv, cfg.lazarus_csv)
    seam = pd.read_csv(cfg.seam_csv).copy() if Path(cfg.seam_csv).exists() else pd.DataFrame()
    rsp = load_response_field(cfg.response_csv)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv)
    tidy = pd.read_csv(cfg.convergence_csv).copy()

    # enrich paths with seam distance if present in nodes
    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    if join_cols and "distance_to_seam" in nodes.columns and "distance_to_seam" not in paths.columns:
        paths = paths.merge(nodes[join_cols + ["distance_to_seam"]], on=join_cols, how="left")

    hubs = pick_hubs(paths, nodes, cfg.top_hubs)
    reps = pick_representative_paths(paths, cfg.reps_per_family)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    panel_a(axs[0, 0], nodes, seam, hubs, fig)
    panel_b(axs[0, 1], nodes, seam, reps)
    panel_c(axs[1, 0], nodes, seam, rsp, cfg.eigen_stride)
    panel_d(axs[1, 1], tidy)

    fig.suptitle("Stable seam corridors on the PAM manifold", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outpath = outdir / "figure_obs022_flagship.png"
    fig.savefig(outpath, dpi=240)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
