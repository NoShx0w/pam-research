#!/usr/bin/env python3
"""
fim_response_complex_compatibility.py

Test whether the local response operator T is compatible with a candidate
almost-complex structure J on the PAM manifold.

Core idea
---------
In 2D, a local almost-complex structure J is easy to define, so the interesting
question is not whether J exists, but whether the response operator T respects it.

We therefore measure, per node:

    commutator_norm     = || T J - J T ||_F
    anticommutator_norm = || T J + J T ||_F

for two candidate J fields:

1. J_fim : built from fim_theta
2. J_rsp : built from rsp_theta

Outputs
-------
outputs/fim_response_complex_compatibility/
  response_complex_compatibility_nodes.csv
  response_complex_compatibility_summary.txt
  response_complex_compatibility_panel.png
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
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/fim_response_complex_compatibility"
    seam_threshold: float = 0.15
    top_k_labels: int = 10


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def basis_from_theta(theta: float) -> tuple[np.ndarray, np.ndarray]:
    e1 = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    e2 = np.array([-e1[1], e1[0]], dtype=float)
    return e1, e2


def J_from_theta(theta: float) -> np.ndarray:
    """
    Build the canonical 2D almost-complex structure in the local orthonormal frame
    oriented by theta, then express it in global coordinates.

    In 2D Euclidean coordinates this reduces to the standard J, but we keep the
    construction explicit for clarity and future generalization.
    """
    e1, e2 = basis_from_theta(theta)
    R = np.column_stack([e1, e2])
    J0 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    return R @ J0 @ R.T


def fro_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def response_matrix(row: pd.Series) -> np.ndarray | None:
    cols = ["T_xx", "T_xy", "T_yx", "T_yy"]
    if not all(c in row.index for c in cols):
        return None
    vals = [pd.to_numeric(row[c], errors="coerce") for c in cols]
    if any(pd.isna(v) for v in vals):
        return None
    return np.array([[float(vals[0]), float(vals[1])], [float(vals[2]), float(vals[3])]], dtype=float)


def build_node_table(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        fim_theta = pd.to_numeric(row.get("fim_theta"), errors="coerce")
        rsp_theta = pd.to_numeric(row.get("rsp_theta"), errors="coerce")
        T = response_matrix(row)

        out = {"node_id": node_id}

        if pd.notna(fim_theta):
            J_fim = J_from_theta(float(fim_theta))
            out.update(
                {
                    "J_fim_xx": J_fim[0, 0],
                    "J_fim_xy": J_fim[0, 1],
                    "J_fim_yx": J_fim[1, 0],
                    "J_fim_yy": J_fim[1, 1],
                }
            )
        else:
            J_fim = None
            out.update(
                {
                    "J_fim_xx": np.nan,
                    "J_fim_xy": np.nan,
                    "J_fim_yx": np.nan,
                    "J_fim_yy": np.nan,
                }
            )

        if pd.notna(rsp_theta):
            J_rsp = J_from_theta(float(rsp_theta))
            out.update(
                {
                    "J_rsp_xx": J_rsp[0, 0],
                    "J_rsp_xy": J_rsp[0, 1],
                    "J_rsp_yx": J_rsp[1, 0],
                    "J_rsp_yy": J_rsp[1, 1],
                }
            )
        else:
            J_rsp = None
            out.update(
                {
                    "J_rsp_xx": np.nan,
                    "J_rsp_xy": np.nan,
                    "J_rsp_yx": np.nan,
                    "J_rsp_yy": np.nan,
                }
            )

        for prefix, J in [("fim", J_fim), ("rsp", J_rsp)]:
            if T is None or J is None:
                out[f"commutator_norm_{prefix}"] = np.nan
                out[f"anticommutator_norm_{prefix}"] = np.nan
                out[f"commutator_trace_{prefix}"] = np.nan
                out[f"anticommutator_trace_{prefix}"] = np.nan
            else:
                C = commutator(T, J)
                A = anticommutator(T, J)
                out[f"commutator_norm_{prefix}"] = fro_norm(C)
                out[f"anticommutator_norm_{prefix}"] = fro_norm(A)
                out[f"commutator_trace_{prefix}"] = float(np.trace(C))
                out[f"anticommutator_trace_{prefix}"] = float(np.trace(A))

        rows.append(out)

    compat = pd.DataFrame(rows)
    return nodes.merge(compat, on="node_id", how="left")


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    lines = [
        "=== FIM Response Complex Compatibility Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "Mean commutator norms",
        f"  mean commutator_norm_fim = {float(pd.to_numeric(nodes['commutator_norm_fim'], errors='coerce').mean()):.6f}",
        f"  mean commutator_norm_rsp = {float(pd.to_numeric(nodes['commutator_norm_rsp'], errors='coerce').mean()):.6f}",
        "",
        "Mean anticommutator norms",
        f"  mean anticommutator_norm_fim = {float(pd.to_numeric(nodes['anticommutator_norm_fim'], errors='coerce').mean()):.6f}",
        f"  mean anticommutator_norm_rsp = {float(pd.to_numeric(nodes['anticommutator_norm_rsp'], errors='coerce').mean()):.6f}",
        "",
        "Seam localization",
        f"  seam-band mean commutator_norm_fim = {float(pd.to_numeric(nodes.loc[seam_mask, 'commutator_norm_fim'], errors='coerce').mean()):.6f}",
        f"  off-seam mean commutator_norm_fim  = {float(pd.to_numeric(nodes.loc[~seam_mask, 'commutator_norm_fim'], errors='coerce').mean()):.6f}",
        f"  seam-band mean commutator_norm_rsp = {float(pd.to_numeric(nodes.loc[seam_mask, 'commutator_norm_rsp'], errors='coerce').mean()):.6f}",
        f"  off-seam mean commutator_norm_rsp  = {float(pd.to_numeric(nodes.loc[~seam_mask, 'commutator_norm_rsp'], errors='coerce').mean()):.6f}",
        "",
        "Correlations",
        f"  corr(commutator_norm_fim, distance_to_seam) = {safe_corr(nodes['commutator_norm_fim'], nodes['distance_to_seam']):.4f}",
        f"  corr(commutator_norm_rsp, distance_to_seam) = {safe_corr(nodes['commutator_norm_rsp'], nodes['distance_to_seam']):.4f}",
        f"  corr(commutator_norm_fim, node_holonomy_proxy) = {safe_corr(nodes['commutator_norm_fim'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        f"  corr(commutator_norm_rsp, node_holonomy_proxy) = {safe_corr(nodes['commutator_norm_rsp'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        "",
        "Top incompatibility nodes (rsp frame)",
    ]

    top = nodes.sort_values("commutator_norm_rsp", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"  node_id={int(row['node_id'])}, "
            f"r={float(row['r']):.4f}, alpha={float(row['alpha']):.4f}, "
            f"commutator_norm_rsp={float(row['commutator_norm_rsp']):.6f}, "
            f"anticommutator_norm_rsp={float(row['anticommutator_norm_rsp']):.6f}, "
            f"distance_to_seam={float(row['distance_to_seam']):.4f}"
        )

    return "\n".join(lines)


def render_panel(cfg: Config, nodes: pd.DataFrame, seam: pd.DataFrame, outpath: Path) -> None:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= cfg.seam_threshold

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.3], height_ratios=[1.0, 1.0])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 1])

    seam_draw = seam.dropna(subset=["mds1", "mds2"]).sort_values("mds1")
    if len(seam_draw):
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="white", linewidth=6.0, alpha=0.65, zorder=1)
        ax_main.plot(seam_draw["mds1"], seam_draw["mds2"], color="black", linewidth=2.8, alpha=0.96, zorder=2)

    sc = ax_main.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=pd.to_numeric(nodes["commutator_norm_rsp"], errors="coerce"),
        cmap="magma",
        s=95,
        alpha=0.96,
        linewidths=0.35,
        edgecolors="white",
        zorder=3,
    )

    seam_nodes = nodes[seam_mask]
    if len(seam_nodes):
        ax_main.scatter(
            seam_nodes["mds1"],
            seam_nodes["mds2"],
            s=170,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
        )

    top = nodes.sort_values("commutator_norm_rsp", ascending=False).head(cfg.top_k_labels)
    for _, row in top.iterrows():
        ax_main.scatter(
            [row["mds1"]],
            [row["mds2"]],
            s=145,
            facecolors="none",
            edgecolors="#FFD166",
            linewidths=1.8,
            zorder=5,
        )
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("||T J_rsp - J_rsp T||_F")

    ax_main.set_title("Response / complex-structure incompatibility", fontsize=17, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.text(
        0.02,
        0.97,
        "black seam = detected phase boundary\nblack rings = seam neighborhood\nyellow labels = highest incompatibility nodes",
        transform=ax_main.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )

    vals = [
        float(pd.to_numeric(nodes.loc[seam_mask, "commutator_norm_rsp"], errors="coerce").mean()),
        float(pd.to_numeric(nodes.loc[~seam_mask, "commutator_norm_rsp"], errors="coerce").mean()),
    ]
    ax_bar.bar(["seam-band", "off-seam"], vals, alpha=0.9)
    ax_bar.set_ylabel("mean commutator norm")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    y = pd.to_numeric(nodes["commutator_norm_rsp"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("distance to seam")
    ax_sc.set_ylabel("commutator norm")
    ax_sc.set_title("Incompatibility vs seam distance", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98,
        0.05,
        f"corr = {safe_corr(y, x):.3f}",
        transform=ax_sc.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — Response / Complex Compatibility", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test response-operator compatibility with candidate complex structures.")
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

    nodes = pd.read_csv(cfg.nodes_csv)
    for col in nodes.columns:
        if col not in {"path_id", "path_family"}:
            try:
                nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
            except Exception:
                pass

    seam = pd.read_csv(cfg.seam_csv)
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam"]:
        if col in seam.columns:
            seam[col] = pd.to_numeric(seam[col], errors="coerce")

    nodes = build_node_table(nodes)

    csv_path = outdir / "response_complex_compatibility_nodes.csv"
    txt_path = outdir / "response_complex_compatibility_summary.txt"
    png_path = outdir / "response_complex_compatibility_panel.png"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")
    render_panel(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
