"""Canonical Lazarus regime stage for the PAM operators layer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_series(s: pd.Series) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return pd.Series(np.zeros(len(s)), index=s.index)
    vmin = np.nanmin(arr[finite])
    vmax = np.nanmax(arr[finite])
    if vmax <= vmin:
        return pd.Series(np.zeros(len(s)), index=s.index)
    out = np.zeros(len(s), dtype=float)
    out[finite] = (arr[finite] - vmin) / (vmax - vmin)
    return pd.Series(out, index=s.index)


def render_grid(df: pd.DataFrame, value_col: str, title: str, outpath: Path, cbar_label: str):
    r_vals = np.sort(df["r"].unique())
    a_vals = np.sort(df["alpha"].unique())
    grid = (
        df.pivot_table(index="r", columns="alpha", values=value_col, aggfunc="mean")
        .reindex(index=r_vals, columns=a_vals)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xticks(range(len(a_vals)))
    ax.set_xticklabels([f"{x:.3f}" for x in a_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(r_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in r_vals])
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def render_mds(df: pd.DataFrame, seam_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(8.0, 6.2))

    sc = ax.scatter(
        df["mds1"],
        df["mds2"],
        c=df["lazarus_score"],
        s=80 + 140 * df["lazarus_score"],
        alpha=0.85,
    )
    fig.colorbar(sc, ax=ax, label="lazarus_score")

    if not seam_df.empty and {"mds1", "mds2"}.issubset(seam_df.columns):
        seam_ord = seam_df.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], linewidth=2.0, alpha=0.8)

    hot = df[df["lazarus_hit"] == 1]
    if not hot.empty:
        ax.scatter(hot["mds1"], hot["mds2"], s=170, marker="*", edgecolors="black", linewidths=0.5)

    ax.set_title("Lazarus regime on PAM manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def run_lazarus(
    signed_phase_csv,
    curvature_csv,
    seam_csv,
    outdir,
    threshold_quantile: float = 0.85,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase = pd.read_csv(signed_phase_csv)
    curv = pd.read_csv(curvature_csv)
    seam = pd.read_csv(seam_csv) if Path(seam_csv).exists() else pd.DataFrame()

    df = phase.merge(
        curv[[c for c in ["r", "alpha", "scalar_curvature"] if c in curv.columns]],
        on=["r", "alpha"],
        how="left",
    )

    df["abs_curvature"] = df["scalar_curvature"].abs()
    df["curvature_norm"] = normalize_series(np.log10(1.0 + df["abs_curvature"]))
    df["seam_proximity"] = 1.0 - normalize_series(df["distance_to_seam"])
    df["phase_centering"] = 1.0 - normalize_series(df["signed_phase"].abs())

    df["lazarus_score"] = (
        0.45 * df["curvature_norm"]
        + 0.35 * df["seam_proximity"]
        + 0.20 * df["phase_centering"]
    )

    thresh = float(df["lazarus_score"].quantile(threshold_quantile))
    df["lazarus_hit"] = (df["lazarus_score"] >= thresh).astype(int)

    df.to_csv(outdir / "lazarus_scores.csv", index=False)

    render_grid(
        df,
        value_col="lazarus_score",
        title="Lazarus score on parameter grid",
        outpath=outdir / "lazarus_region_on_grid.png",
        cbar_label="lazarus_score",
    )

    if not seam.empty and not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")

    render_mds(df, seam, outdir / "lazarus_region_on_mds.png")

    summary = pd.DataFrame(
        {
            "threshold_quantile": [threshold_quantile],
            "threshold_value": [thresh],
            "num_hits": [int(df["lazarus_hit"].sum())],
            "num_total": [int(len(df))],
        }
    )
    summary.to_csv(outdir / "lazarus_summary.csv", index=False)

    print(outdir / "lazarus_scores.csv")
    print(outdir / "lazarus_summary.csv")
    print(outdir / "lazarus_region_on_grid.png")
    print(outdir / "lazarus_region_on_mds.png")

    return {
        "scores": df,
        "summary": summary,
    }
