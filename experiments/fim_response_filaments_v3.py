#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation


def load_nodes(mds_csv: str | Path, phase_csv: str | Path, lazarus_csv: str | Path) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(phase_csv)
    laz = pd.read_csv(lazarus_csv)

    df = mds.copy()
    if {"mds1", "mds2"}.issubset(phase.columns):
        keep_phase = [c for c in ["r", "alpha", "mds1", "mds2", "signed_phase", "distance_to_seam"] if c in phase.columns]
        df = phase[keep_phase].copy()
    else:
        keep_phase = [c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
        df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")

    keep_laz = [c for c in ["r", "alpha", "lazarus_score"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def load_seam(seam_csv: str | Path, nodes: pd.DataFrame) -> pd.DataFrame:
    p = Path(seam_csv)
    if not p.exists():
        return pd.DataFrame()
    seam = pd.read_csv(p)
    if not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(nodes[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")
    return seam.dropna(subset=["mds1", "mds2"]).copy()


def load_paths(paths_csv: str | Path) -> pd.DataFrame:
    p = Path(paths_csv)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def estimate_local_gradients(
    df: pd.DataFrame,
    value_col: str,
    x_col: str = "mds1",
    y_col: str = "mds2",
    k: int = 8,
) -> pd.DataFrame:
    work = df[[x_col, y_col, value_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    X = work[[x_col, y_col]].to_numpy(dtype=float)
    z = work[value_col].to_numpy(dtype=float)

    gx = np.full(len(work), np.nan, dtype=float)
    gy = np.full(len(work), np.nan, dtype=float)
    gnorm = np.full(len(work), np.nan, dtype=float)

    for i in range(len(work)):
        if not np.isfinite(z[i]):
            continue

        d2 = np.sum((X - X[i]) ** 2, axis=1)
        order = np.argsort(d2)
        neigh = order[1:min(k + 1, len(order))]
        neigh = neigh[np.isfinite(z[neigh])]
        if len(neigh) < 3:
            continue

        dx = X[neigh, 0] - X[i, 0]
        dy = X[neigh, 1] - X[i, 1]
        dz = z[neigh] - z[i]
        A = np.column_stack([dx, dy])

        try:
            beta, *_ = np.linalg.lstsq(A, dz, rcond=None)
        except np.linalg.LinAlgError:
            continue

        gx[i] = float(beta[0])
        gy[i] = float(beta[1])
        gnorm[i] = float(np.sqrt(beta[0] ** 2 + beta[1] ** 2))

    return pd.DataFrame(
        {
            f"grad_{value_col}_x": gx,
            f"grad_{value_col}_y": gy,
            f"grad_{value_col}_norm": gnorm,
        },
        index=df.index,
    )


def build_response_fields(nodes: pd.DataFrame) -> pd.DataFrame:
    df = nodes.copy()
    df["signed_phase"] = pd.to_numeric(df["signed_phase"], errors="coerce")
    df["lazarus_score"] = pd.to_numeric(df["lazarus_score"], errors="coerce")

    phase_grads = estimate_local_gradients(df, "signed_phase")
    laz_grads = estimate_local_gradients(df, "lazarus_score")
    df = pd.concat([df, phase_grads, laz_grads], axis=1)

    gx_phi = df["grad_signed_phase_x"].to_numpy(dtype=float)
    gy_phi = df["grad_signed_phase_y"].to_numpy(dtype=float)
    gx_l = df["grad_lazarus_score_x"].to_numpy(dtype=float)
    gy_l = df["grad_lazarus_score_y"].to_numpy(dtype=float)

    norm_phi = np.sqrt(gx_phi ** 2 + gy_phi ** 2)
    norm_l = np.sqrt(gx_l ** 2 + gy_l ** 2)
    dot = gx_phi * gx_l + gy_phi * gy_l
    strength = norm_phi * norm_l
    cosine = np.where((strength > 0) & np.isfinite(strength), dot / strength, np.nan)

    df["response_strength"] = strength
    df["signed_coupling"] = dot
    df["cosine_alignment"] = np.clip(cosine, -1.0, 1.0)

    ux = np.where(norm_phi > 0, gx_phi / norm_phi, np.nan)
    uy = np.where(norm_phi > 0, gy_phi / norm_phi, np.nan)
    df["phase_dir_x"] = ux
    df["phase_dir_y"] = uy
    return df


def select_active_nodes(df: pd.DataFrame, top_quantile: float = 0.8, max_nodes: int = 24) -> pd.DataFrame:
    work = df.copy()
    rs = pd.to_numeric(work["response_strength"], errors="coerce")
    threshold = float(rs.quantile(top_quantile))
    work = work[rs >= threshold].copy()
    work = work.sort_values("response_strength", ascending=False)

    if len(work) <= max_nodes:
        return work

    selected = []
    span = max(work["mds1"].max() - work["mds1"].min(), work["mds2"].max() - work["mds2"].min())
    min_dist = 0.08 * span

    for idx, row in work.iterrows():
        p = np.array([row["mds1"], row["mds2"]], dtype=float)
        if not selected:
            selected.append(idx)
        else:
            ok = True
            for j in selected:
                q = np.array([work.loc[j, "mds1"], work.loc[j, "mds2"]], dtype=float)
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                selected.append(idx)
        if len(selected) >= max_nodes:
            break
    return work.loc[selected].copy()


def pick_canonical_path(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty or "probe_id" not in paths.columns:
        return pd.DataFrame()

    rows = []
    for pid, grp in paths.groupby("probe_id"):
        grp = grp.sort_values("step")
        laz_max = float(pd.to_numeric(grp.get("lazarus_score"), errors="coerce").max()) if "lazarus_score" in grp.columns else 0.0
        flip = 0
        prev = 0
        if "signed_phase" in grp.columns:
            for v in pd.to_numeric(grp["signed_phase"], errors="coerce").fillna(0.0):
                s = -1 if v < 0 else (1 if v > 0 else 0)
                if s == 0:
                    continue
                if prev != 0 and s != prev:
                    flip = 1
                    break
                prev = s
        rows.append((pid, flip, laz_max, len(grp)))
    rank = pd.DataFrame(rows, columns=["probe_id", "flip", "laz_max", "n"]).sort_values(
        ["flip", "laz_max", "n"], ascending=[False, False, False]
    )
    chosen = rank.iloc[0]["probe_id"]
    return paths[paths["probe_id"] == chosen].sort_values("step").copy()


def nearest_node_z(xv: np.ndarray, yv: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = np.full(len(xv), np.nan, dtype=float)
    pts = np.column_stack([x, y])
    for i, (xx, yy) in enumerate(zip(xv, yv)):
        d2 = np.sum((pts - np.array([xx, yy])) ** 2, axis=1)
        j = int(np.argmin(d2))
        out[i] = z[j]
    return out


def build_triangulation(df: pd.DataFrame, mask_cross_seam: bool = True):
    x = pd.to_numeric(df["mds1"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["mds2"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["signed_phase"], errors="coerce").to_numpy(dtype=float)
    laz = pd.to_numeric(df["lazarus_score"], errors="coerce").to_numpy(dtype=float)

    tri = Triangulation(x, y)

    finite_mask = np.all(np.isfinite(z[tri.triangles]), axis=1)
    if mask_cross_seam:
        tri_signs = np.sign(z[tri.triangles])
        cross_mask = np.ptp(tri_signs, axis=1) > 0
        tri.set_mask(~finite_mask | cross_mask)
    else:
        tri.set_mask(~finite_mask)

    return tri, x, y, z, laz


def add_filaments(ax, active: pd.DataFrame, x: np.ndarray, y: np.ndarray, z: np.ndarray, z_offset: float = 0.03) -> None:
    if active.empty:
        return

    strength = pd.to_numeric(active["response_strength"], errors="coerce")
    smin = float(strength.min())
    smax = float(strength.max())

    xs = active["mds1"].to_numpy(dtype=float)
    ys = active["mds2"].to_numpy(dtype=float)
    zs = nearest_node_z(xs, ys, x, y, z) + z_offset

    for i, (_, row) in enumerate(active.iterrows()):
        length = 0.08
        if smax > smin:
            length = 0.08 + 0.14 * (float(row["response_strength"]) - smin) / (smax - smin)
        ux = float(row["phase_dir_x"])
        uy = float(row["phase_dir_y"])
        x0 = xs[i] - 0.5 * length * ux
        x1 = xs[i] + 0.5 * length * ux
        y0 = ys[i] - 0.5 * length * uy
        y1 = ys[i] + 0.5 * length * uy
        zz = zs[i]
        ax.plot([x0, x1], [y0, y1], [zz, zz], color="white", linewidth=1.7, alpha=0.92)


def add_quiver(ax, active: pd.DataFrame, x: np.ndarray, y: np.ndarray, z: np.ndarray, z_offset: float = 0.03) -> None:
    if active.empty:
        return

    strength = pd.to_numeric(active["response_strength"], errors="coerce")
    smin = float(strength.min())
    smax = float(strength.max())

    xs = active["mds1"].to_numpy(dtype=float)
    ys = active["mds2"].to_numpy(dtype=float)
    zs = nearest_node_z(xs, ys, x, y, z) + z_offset
    ux = active["phase_dir_x"].to_numpy(dtype=float)
    uy = active["phase_dir_y"].to_numpy(dtype=float)

    base = 0.18
    if smax > smin:
        lengths = base * (0.4 + 0.6 * (strength - smin) / (smax - smin))
    else:
        lengths = np.full(len(active), base)

    ax.quiver(
        xs, ys, zs,
        ux * lengths.to_numpy(dtype=float),
        uy * lengths.to_numpy(dtype=float),
        np.zeros(len(active)),
        color="white",
        linewidth=1.0,
        arrow_length_ratio=0.22,
        alpha=0.95,
    )


def render(df: pd.DataFrame, seam: pd.DataFrame, paths: pd.DataFrame, outpath: Path, mode: str = "filaments", mask_cross_seam: bool = True) -> None:
    active = select_active_nodes(df, top_quantile=0.8, max_nodes=24)
    canonical = pick_canonical_path(paths)

    tri, x, y, z, laz = build_triangulation(df, mask_cross_seam=mask_cross_seam)

    laz_norm = laz - np.nanmin(laz)
    maxv = np.nanmax(laz_norm)
    if maxv > 0:
        laz_norm = laz_norm / maxv
    colors = plt.cm.magma(laz_norm)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        tri,
        z,
        linewidth=0.25,
        antialiased=True,
        shade=False,
        alpha=0.97,
    )
    surf.set_facecolors(colors[tri.triangles].mean(axis=1))
    surf.set_edgecolor((0, 0, 0, 0.18))

    mappable = plt.cm.ScalarMappable(cmap="magma")
    mappable.set_array(laz_norm)

    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        sx = seam_ord["mds1"].to_numpy(dtype=float)
        sy = seam_ord["mds2"].to_numpy(dtype=float)
        sz = nearest_node_z(sx, sy, x, y, z) + 0.025
        ax.plot(sx, sy, sz, color="white", linewidth=2.7, alpha=0.96)

    if not canonical.empty:
        px = canonical["mds1"].to_numpy(dtype=float)
        py = canonical["mds2"].to_numpy(dtype=float)
        pz = nearest_node_z(px, py, x, y, z) + 0.02
        ax.plot(px, py, pz, color="white", linewidth=2.4, alpha=0.96)

    if mode == "quiver":
        add_quiver(ax, active, x, y, z)
        title = "Phase Geometry and Boundary-Activated Response (quiver)"
    else:
        add_filaments(ax, active, x, y, z)
        title = "Phase Geometry and Boundary-Activated Response (filaments)"

    ax.set_title(title)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("Signed phase")
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("Lazarus intensity")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a triangulated response surface with sparse filaments or quiver.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--mode", choices=["filaments", "quiver"], default="filaments")
    parser.add_argument("--no-seam-mask", action="store_true")
    parser.add_argument("--outdir", default="outputs/fim_response_filaments_v3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.mds_csv, args.phase_csv, args.lazarus_csv)
    nodes = build_response_fields(nodes)
    seam = load_seam(args.seam_csv, nodes)
    paths = load_paths(args.paths_csv)

    outpath = outdir / f"response_surface_{args.mode}_v3.png"
    render(
        nodes,
        seam,
        paths,
        outpath,
        mode=args.mode,
        mask_cross_seam=not args.no_seam_mask,
    )
    print(outpath)


if __name__ == "__main__":
    main()
