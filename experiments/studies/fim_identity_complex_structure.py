#!/usr/bin/env python3
"""
fim_identity_complex_structure.py

Construct and test candidate almost-complex structure fields on the PAM manifold.

Two candidate constructions are tested:

1. response-frame J
   Built from the local response principal direction rsp_theta

2. identity-frame J
   Built from the local identity / Fisher direction fim_theta

For each candidate we compute:
- J_xx, J_xy, J_yx, J_yy
- J_squared_error
- transport_J_error
- alignment with obstruction / spin proxies (if present)

Outputs
-------
outputs/fim_identity_complex_structure/
  complex_structure_nodes.csv
  complex_structure_summary.txt
  complex_structure_panel.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pam.geometry.directional_field import DirectionalField, wrap_angle
from pam.geometry.parallel_transport import edge_parallel_transport_table


@dataclass(frozen=True)
class Config:
    nodes_csv: str = "outputs/obs022_scene_bundle/scene_nodes.csv"
    edges_csv: str = "outputs/obs022_scene_bundle/scene_edges.csv"
    seam_csv: str = "outputs/obs022_scene_bundle/scene_seam.csv"
    outdir: str = "outputs/fim_identity_complex_structure"
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
    Build J in global coordinates from a local orthonormal frame
    whose first axis is angle theta.

    J(e1) = e2
    J(e2) = -e1
    """
    e1, e2 = basis_from_theta(theta)
    # columns are J(e_x), J(e_y) in global basis after basis conversion
    # equivalently R * J0 * R^{-1}; in 2D this reduces to the standard
    # complex structure in any orthonormal frame, but we keep explicit construction.
    R = np.column_stack([e1, e2])
    J0 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    return R @ J0 @ R.T


def J_squared_error(J: np.ndarray) -> float:
    I = np.eye(2, dtype=float)
    err = J @ J + I
    return float(np.linalg.norm(err, ord="fro"))


def matrix_transport_error(J_src: np.ndarray, J_dst: np.ndarray, delta_conn: float) -> float:
    """
    Transport a (1,1)-tensor by conjugation with the edge rotation:
        J' = R J R^{-1}
    and compare with destination J.
    """
    c = float(np.cos(delta_conn))
    s = float(np.sin(delta_conn))
    R = np.array([[c, -s], [s, c]], dtype=float)
    J_transported = R @ J_src @ R.T
    return float(np.linalg.norm(J_transported - J_dst, ord="fro"))


def build_node_table(field: DirectionalField) -> pd.DataFrame:
    nodes = field.nodes.copy()

    rows = []
    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        fim_theta = float(row["fim_theta"])
        rsp_theta = float(row["rsp_theta"])

        J_fim = J_from_theta(fim_theta)
        J_rsp = J_from_theta(rsp_theta)

        rows.append(
            {
                "node_id": node_id,
                "J_fim_xx": J_fim[0, 0],
                "J_fim_xy": J_fim[0, 1],
                "J_fim_yx": J_fim[1, 0],
                "J_fim_yy": J_fim[1, 1],
                "J_rsp_xx": J_rsp[0, 0],
                "J_rsp_xy": J_rsp[0, 1],
                "J_rsp_yx": J_rsp[1, 0],
                "J_rsp_yy": J_rsp[1, 1],
                "J_fim_squared_error": J_squared_error(J_fim),
                "J_rsp_squared_error": J_squared_error(J_rsp),
            }
        )

    out = pd.DataFrame(rows)
    return nodes.merge(out, on="node_id", how="left")


def build_edge_transport_errors(field: DirectionalField) -> pd.DataFrame:
    nodes = field.nodes.set_index("node_id", drop=False)
    edge_pt = edge_parallel_transport_table(field)

    rows = []
    for _, row in edge_pt.iterrows():
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        delta_conn = np.radians(float(row["delta_connection_theta_deg"]))

        src = nodes.loc[src_id]
        dst = nodes.loc[dst_id]

        J_fim_src = J_from_theta(float(src["fim_theta"]))
        J_fim_dst = J_from_theta(float(dst["fim_theta"]))
        J_rsp_src = J_from_theta(float(src["rsp_theta"]))
        J_rsp_dst = J_from_theta(float(dst["rsp_theta"]))

        rows.append(
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "edge_distance_to_seam_mid": 0.5 * (
                    float(src["distance_to_seam"]) + float(dst["distance_to_seam"])
                ),
                "J_fim_transport_error": matrix_transport_error(J_fim_src, J_fim_dst, delta_conn),
                "J_rsp_transport_error": matrix_transport_error(J_rsp_src, J_rsp_dst, delta_conn),
            }
        )

    return pd.DataFrame(rows)


def build_node_transport_summary(edge_df: pd.DataFrame) -> pd.DataFrame:
    return (
        edge_df.groupby("src_id", as_index=False)
        .agg(
            J_fim_transport_error_mean=("J_fim_transport_error", "mean"),
            J_fim_transport_error_max=("J_fim_transport_error", "max"),
            J_rsp_transport_error_mean=("J_rsp_transport_error", "mean"),
            J_rsp_transport_error_max=("J_rsp_transport_error", "max"),
        )
        .rename(columns={"src_id": "node_id"})
    )


def build_summary(nodes: pd.DataFrame, seam_threshold: float) -> str:
    seam_mask = pd.to_numeric(nodes["distance_to_seam"], errors="coerce") <= seam_threshold

    lines = [
        "=== FIM Identity Complex Structure Summary ===",
        "",
        f"n_nodes = {len(nodes)}",
        f"seam_threshold = {seam_threshold:.4f}",
        "",
        "J^2 + I errors",
        f"  mean J_fim_squared_error = {float(pd.to_numeric(nodes['J_fim_squared_error'], errors='coerce').mean()):.8f}",
        f"  mean J_rsp_squared_error = {float(pd.to_numeric(nodes['J_rsp_squared_error'], errors='coerce').mean()):.8f}",
        "",
        "Transport consistency",
        f"  mean J_fim_transport_error_mean = {float(pd.to_numeric(nodes['J_fim_transport_error_mean'], errors='coerce').mean()):.6f}",
        f"  mean J_rsp_transport_error_mean = {float(pd.to_numeric(nodes['J_rsp_transport_error_mean'], errors='coerce').mean()):.6f}",
        f"  seam-band mean J_fim_transport_error = {float(pd.to_numeric(nodes.loc[seam_mask, 'J_fim_transport_error_mean'], errors='coerce').mean()):.6f}",
        f"  off-seam mean J_fim_transport_error  = {float(pd.to_numeric(nodes.loc[~seam_mask, 'J_fim_transport_error_mean'], errors='coerce').mean()):.6f}",
        f"  seam-band mean J_rsp_transport_error = {float(pd.to_numeric(nodes.loc[seam_mask, 'J_rsp_transport_error_mean'], errors='coerce').mean()):.6f}",
        f"  off-seam mean J_rsp_transport_error  = {float(pd.to_numeric(nodes.loc[~seam_mask, 'J_rsp_transport_error_mean'], errors='coerce').mean()):.6f}",
        "",
        "Correlations",
        f"  corr(J_fim_transport_error, distance_to_seam) = {safe_corr(nodes['J_fim_transport_error_mean'], nodes['distance_to_seam']):.4f}",
        f"  corr(J_rsp_transport_error, distance_to_seam) = {safe_corr(nodes['J_rsp_transport_error_mean'], nodes['distance_to_seam']):.4f}",
        f"  corr(J_fim_transport_error, node_holonomy_proxy) = {safe_corr(nodes['J_fim_transport_error_mean'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
        f"  corr(J_rsp_transport_error, node_holonomy_proxy) = {safe_corr(nodes['J_rsp_transport_error_mean'], nodes.get('node_holonomy_proxy', pd.Series(dtype=float))):.4f}",
    ]
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
        c=pd.to_numeric(nodes["J_rsp_transport_error_mean"], errors="coerce"),
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
            seam_nodes["mds1"], seam_nodes["mds2"],
            s=170, facecolors="none", edgecolors="black", linewidths=1.3, zorder=4
        )

    top = nodes.sort_values("J_rsp_transport_error_mean", ascending=False).head(cfg.top_k_labels)
    for _, row in top.iterrows():
        ax_main.scatter([row["mds1"]], [row["mds2"]], s=145, facecolors="none", edgecolors="#FFD166", linewidths=1.8, zorder=5)
        ax_main.text(
            float(row["mds1"]) + 0.05,
            float(row["mds2"]) + 0.05,
            f"{int(row['node_id'])}",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.8", alpha=0.9),
            zorder=6,
        )

    cbar = fig.colorbar(sc, ax=ax_main, fraction=0.038, pad=0.02)
    cbar.set_label("J_rsp transport error")

    ax_main.set_title("Candidate almost-complex field transport error", fontsize=17, pad=10)
    ax_main.set_xlabel("MDS 1")
    ax_main.set_ylabel("MDS 2")
    ax_main.grid(alpha=0.08)
    ax_main.set_aspect("equal", adjustable="box")

    vals = [
        float(pd.to_numeric(nodes.loc[seam_mask, "J_rsp_transport_error_mean"], errors="coerce").mean()),
        float(pd.to_numeric(nodes.loc[~seam_mask, "J_rsp_transport_error_mean"], errors="coerce").mean()),
    ]
    ax_bar.bar(["seam-band", "off-seam"], vals, alpha=0.9)
    ax_bar.set_ylabel("mean J_rsp transport error")
    ax_bar.set_title("Seam localization", fontsize=14, pad=8)
    ax_bar.grid(alpha=0.15, axis="y")

    x = pd.to_numeric(nodes["distance_to_seam"], errors="coerce")
    y = pd.to_numeric(nodes["J_rsp_transport_error_mean"], errors="coerce")
    mask = x.notna() & y.notna()
    ax_sc.scatter(x[mask], y[mask], s=38, alpha=0.88)
    ax_sc.set_xlabel("distance to seam")
    ax_sc.set_ylabel("J_rsp transport error")
    ax_sc.set_title("Transport error vs seam distance", fontsize=14, pad=8)
    ax_sc.grid(alpha=0.15)
    ax_sc.text(
        0.98, 0.05, f"corr = {safe_corr(y, x):.3f}",
        transform=ax_sc.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.82", alpha=0.9),
    )

    fig.suptitle("PAM Observatory — Candidate Complex Structure Test", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test candidate almost-complex structure fields.")
    parser.add_argument("--nodes-csv", default=Config.nodes_csv)
    parser.add_argument("--edges-csv", default=Config.edges_csv)
    parser.add_argument("--seam-csv", default=Config.seam_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--top-k-labels", type=int, default=Config.top_k_labels)
    args = parser.parse_args()

    cfg = Config(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        seam_csv=args.seam_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        top_k_labels=args.top_k_labels,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    field = DirectionalField.from_csv(
        cfg.nodes_csv,
        cfg.edges_csv,
        connection_theta_col="fim_theta",
        response_theta_col="rsp_theta",
    )

    seam = pd.read_csv(cfg.seam_csv)
    for col in ["mds1", "mds2", "signed_phase", "distance_to_seam"]:
        if col in seam.columns:
            seam[col] = pd.to_numeric(seam[col], errors="coerce")

    nodes = build_node_table(field)
    edge_df = build_edge_transport_errors(field)
    node_trn = build_node_transport_summary(edge_df)
    nodes = nodes.merge(node_trn, on="node_id", how="left")

    csv_path = outdir / "complex_structure_nodes.csv"
    txt_path = outdir / "complex_structure_summary.txt"
    png_path = outdir / "complex_structure_panel.png"

    nodes.to_csv(csv_path, index=False)
    txt_path.write_text(build_summary(nodes, cfg.seam_threshold), encoding="utf-8")
    render_panel(cfg, nodes, seam, png_path)

    print(csv_path)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
