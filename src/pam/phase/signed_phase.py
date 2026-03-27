"""Canonical signed-phase stage for the PAM phase pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def nearest_segment_sign(point: np.ndarray, polyline: np.ndarray) -> float:
    best_d2 = np.inf
    best_sign = 1.0

    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        t = b - a
        tt = float(np.dot(t, t))
        if tt == 0:
            continue

        u = np.clip(float(np.dot(point - a, t) / tt), 0.0, 1.0)
        q = a + u * t
        v = point - q
        d2 = float(np.dot(v, v))

        if d2 < best_d2:
            cross = float(t[0] * v[1] - t[1] * v[0])
            best_sign = -1.0 if cross < 0 else 1.0
            best_d2 = d2

    return best_sign


def render_grid(df: pd.DataFrame, value_col: str, title: str, outpath: Path, cbar_label: str):
    r_vals = np.sort(df["r"].unique())
    a_vals = np.sort(df["alpha"].unique())
    grid = (
        df.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def render_mds(df: pd.DataFrame, seam_df: pd.DataFrame, title: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    sc = ax.scatter(
        df["mds1"],
        df["mds2"],
        c=df["signed_phase"],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=60,
    )
    fig.colorbar(sc, ax=ax, label="signed phase")

    seam_ord = seam_df.sort_values("mds1")
    ax.plot(seam_ord["mds1"], seam_ord["mds2"], linewidth=2.5)

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_signed_phase(
    mds_csv,
    seam_csv,
    phase_distance_csv,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mds = pd.read_csv(mds_csv)
    seam = pd.read_csv(seam_csv)
    dist = pd.read_csv(phase_distance_csv)

    df = mds.merge(
        dist[["r", "alpha", "distance_to_seam"]],
        on=["r", "alpha"],
        how="left",
    )

    seam_small = seam.copy()
    if "mds1" not in seam_small.columns or "mds2" not in seam_small.columns:
        seam_small = seam_small[["r", "alpha"]].merge(
            mds[["r", "alpha", "mds1", "mds2"]],
            on=["r", "alpha"],
            how="left",
        )

    seam_small = seam_small.dropna(subset=["mds1", "mds2"]).drop_duplicates(subset=["r", "alpha"]).copy()
    seam_poly = seam_small.sort_values("mds1")[["mds1", "mds2"]].to_numpy(dtype=float)

    pts = df[["mds1", "mds2"]].to_numpy(dtype=float)
    signs = np.array([nearest_segment_sign(p, seam_poly) for p in pts], dtype=float)

    d = df["distance_to_seam"].to_numpy(dtype=float)
    dmax_raw = np.nanmax(d)
    dmax = float(dmax_raw) if np.isfinite(dmax_raw) and dmax_raw > 0 else 1.0

    df["phase_sign"] = signs
    df["signed_phase"] = signs * (d / dmax)

    out_csv = outdir / "signed_phase_coords.csv"
    df.to_csv(out_csv, index=False)

    render_grid(
        df,
        value_col="signed_phase",
        title="Signed phase coordinate on parameter grid",
        outpath=outdir / "signed_phase_on_grid.png",
        cbar_label="signed phase",
    )

    render_mds(
        df,
        seam_small,
        title="Signed phase coordinate on PAM manifold",
        outpath=outdir / "signed_phase_on_mds.png",
    )

    print(out_csv)
    print(outdir / "signed_phase_on_grid.png")
    print(outdir / "signed_phase_on_mds.png")

    return df
