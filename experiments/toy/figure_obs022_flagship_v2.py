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
    outdir: str = "outputs/figure_obs022_flagship_v2"
    top_hubs: int = 10
    reps_per_family: int = 5
    seam_field_threshold: float = 0.20
    field_stride: int = 1


FAMILY_COLORS = {
    "stable_seam_corridor": "#D4A72C",   # gold
    "reorganization_heavy": "#B23A48",   # crimson
    "settled_distant": "#5C6B73",        # slate
    "off_seam_reorganizing": "#2A9D8F",  # teal
}

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

    return phase[keep_phase].merge(laz[keep_laz], on=join_cols, how="left")


def load_response_field(response_csv: str) -> pd.DataFrame:
    rsp = pd.read_csv(response_csv).copy()
    keep = [c for c in ["r", "alpha", "mds1", "mds2", "distance_to_seam", "response_strength", "T_xx", "T_xy", "T_yx", "T_yy"] if c in rsp.columns]
    rsp = rsp[keep].copy()

    eig1, eig2, theta = principal_response_eig(
        pd.to_numeric(rsp["T_xx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_xy"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_yx"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(rsp["T_yy"], errors="coerce").to_numpy(dtype=float),
    )
    rsp["rsp_theta"] = theta
    rsp["rsp_eig1"] = eig1
    rsp["rsp_eig2"] = eig2
    return rsp


def load_paths(path_nodes_csv: str, family_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(path_nodes_csv).copy()
    fam = pd.read_csv(family_csv).copy()

    if "probe_id" in paths.columns and "path_id" not in paths.columns:
        paths = paths.rename(columns={"probe_id": "path_id"})

    return paths.merge(fam[["path_id", "path_family"]], on="path_id", how="left")


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
    g = (
        paths.groupby(["path_id", "path_family"], as_index=False)
        .agg(
            n_steps=("step", "max"),
            start_x=("mds1", "first"),
            start_y=("mds2", "first"),
        )
    )
    selected = []
    for fam in ["stable_seam_corridor", "reorganization_heavy"]:
        sub = g[g["path_family"] == fam].copy().sort_values(["n_steps", "start_x"], ascending=[False, True])
        selected.extend(sub.head(reps_per_family)["path_id"].tolist())
    return paths[paths["path_id"].isin(selected)].copy()


def draw_background(ax, nodes: pd.DataFrame, seam: pd.DataFrame, fig=None, add_cbar=False):
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=nodes["signed_phase"],
        s=55,
        alpha=0.88,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0,
        zorder=1,
    )

    if len(seam):
        seam = seam.sort_values("mds1")
        ax.plot(seam["mds1"], seam["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=2)
        ax.plot(seam["mds1"], seam["mds2"], color="black", linewidth=2.8, alpha=0.95, zorder=3)

    if add_cbar and fig is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("signed phase")
    return sc


def style_axis(ax, title: str):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.10)


def panel_a(ax, fig, nodes: pd.DataFrame, seam: pd.DataFrame, hubs: pd.DataFrame):
    draw_background(ax, nodes, seam, fig=fig, add_cbar=True)

    sizes = 130 + 280 * (hubs["n_unique_paths"] / max(hubs["n_unique_paths"].max(), 1))
    ax.scatter(
        hubs["mds1"],
        hubs["mds2"],
        s=sizes,
        facecolors="none",
        edgecolors="black",
        linewidths=1.7,
        zorder=5,
    )
    ax.scatter(
        hubs["mds1"],
        hubs["mds2"],
        s=20,
        c="black",
        zorder=6,
    )

    ax.text(
        0.03, 0.97, "routing hubs",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9)
    )
    style_axis(ax, "A. Phase manifold, seam, and routing hubs")


def panel_b(ax, nodes: pd.DataFrame, seam: pd.DataFrame, reps: pd.DataFrame):
    draw_background(ax, nodes, seam)

    for fam in ["reorganization_heavy", "stable_seam_corridor"]:
        sub = reps[reps["path_family"] == fam].copy()
        for _, grp in sub.groupby("path_id", sort=False):
            grp = grp.sort_values("step")
            ax.plot(
                grp["mds1"],
                grp["mds2"],
                color=FAMILY_COLORS[fam],
                linewidth=3.0 if fam == "stable_seam_corridor" else 1.8,
                alpha=0.95 if fam == "stable_seam_corridor" else 0.55,
                zorder=5 if fam == "stable_seam_corridor" else 4,
            )

    ax.text(
        0.03, 0.97,
        "gold = stable_seam_corridor\ncrimson = reorganization_heavy",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9)
    )
    style_axis(ax, "B. Seam-contact route bundles")


def panel_c(ax, nodes: pd.DataFrame, seam: pd.DataFrame, rsp: pd.DataFrame, seam_threshold: float, stride: int):
    draw_background(ax, nodes, seam)

    sub = rsp.copy()
    if "distance_to_seam" in sub.columns:
        sub = sub[pd.to_numeric(sub["distance_to_seam"], errors="coerce") <= seam_threshold].copy()
    sub = sub.dropna(subset=["mds1", "mds2", "rsp_theta"]).iloc[::max(1, stride), :]

    u = np.cos(pd.to_numeric(sub["rsp_theta"], errors="coerce").to_numpy(dtype=float))
    v = np.sin(pd.to_numeric(sub["rsp_theta"], errors="coerce").to_numpy(dtype=float))

    ax.quiver(
        sub["mds1"], sub["mds2"],
        u, v,
        angles="xy", scale_units="xy", scale=11,
        width=0.0028,
        color="black",
        alpha=0.7,
        zorder=5,
    )

    ax.text(
        0.03, 0.97, f"response directions\n(seam neighborhood ≤ {seam_threshold:.2f})",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9)
    )
    style_axis(ax, "C. Response principal-direction field")


def panel_d(ax, tidy: pd.DataFrame):
    tidy = tidy.copy()
    tidy["scale"] = pd.to_numeric(tidy["scale"], errors="coerce")

    for fam in FAMILY_ORDER:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["scale"],
            sub["share_paths"],
            marker="o",
            linewidth=2.2,
            color=FAMILY_COLORS[fam],
            label=fam,
        )

    ax.set_xscale("log")
    ax.set_title("D. Family-share convergence")
    ax.set_xlabel("scale")
    ax.set_ylabel("share")
    ax.grid(alpha=0.20)
    ax.legend(fontsize=8, loc="lower right")


def main():
    parser = argparse.ArgumentParser(description="Build a refined flagship observatory figure for OBS-022.")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--response-csv", default="outputs/fim_response_operator/response_operator_nodes.csv")
    parser.add_argument("--family-csv", default="outputs/scales/100000/toy_scaled_probe_path_families/geodesic_path_family_assignments.csv")
    parser.add_argument("--path-nodes-csv", default="outputs/scales/100000/fim_ops_scaled/scaled_probe_paths_for_family_clean.csv")
    parser.add_argument("--convergence-csv", default="outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv")
    parser.add_argument("--outdir", default="outputs/figure_obs022_flagship_v2")
    parser.add_argument("--top-hubs", type=int, default=10)
    parser.add_argument("--reps-per-family", type=int, default=5)
    parser.add_argument("--seam-field-threshold", type=float, default=0.20)
    parser.add_argument("--field-stride", type=int, default=1)
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
        seam_field_threshold=args.seam_field_threshold,
        field_stride=args.field_stride,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_phase_nodes(cfg.phase_csv, cfg.lazarus_csv)
    seam = pd.read_csv(cfg.seam_csv).copy() if Path(cfg.seam_csv).exists() else pd.DataFrame()
    rsp = load_response_field(cfg.response_csv)
    paths = load_paths(cfg.path_nodes_csv, cfg.family_csv)
    tidy = pd.read_csv(cfg.convergence_csv).copy()

    join_cols = [c for c in ["node_id", "r", "alpha"] if c in paths.columns and c in nodes.columns]
    if join_cols and "distance_to_seam" in nodes.columns and "distance_to_seam" not in paths.columns:
        paths = paths.merge(nodes[join_cols + ["distance_to_seam"]], on=join_cols, how="left")

    hubs = pick_hubs(paths, nodes, cfg.top_hubs)
    reps = pick_representative_paths(paths, cfg.reps_per_family)

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))

    panel_a(axs[0, 0], fig, nodes, seam, hubs)
    panel_b(axs[0, 1], nodes, seam, reps)
    panel_c(axs[1, 0], nodes, seam, rsp, cfg.seam_field_threshold, cfg.field_stride)
    panel_d(axs[1, 1], tidy)

    fig.suptitle("Stable seam corridors on the PAM manifold", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    outpath = outdir / "figure_obs022_flagship_v2.png"
    fig.savefig(outpath, dpi=240)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
