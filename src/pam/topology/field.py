"""Canonical field-alignment stage for the PAM topology pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_nodes(mds_csv, phase_csv, lazarus_csv) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    phase = pd.read_csv(phase_csv)
    laz = pd.read_csv(lazarus_csv)

    df = mds.copy()
    keep_phase = [c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")
    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def load_paths(paths_csv) -> pd.DataFrame:
    return pd.read_csv(paths_csv)


def build_node_alignment(nodes: pd.DataFrame):
    work = nodes.copy()
    work["distance_to_seam"] = pd.to_numeric(work["distance_to_seam"], errors="coerce")
    work["lazarus_score"] = pd.to_numeric(work["lazarus_score"], errors="coerce")
    if "lazarus_hit" in work.columns:
        work["lazarus_hit"] = pd.to_numeric(work["lazarus_hit"], errors="coerce").fillna(0).astype(int)
    else:
        work["lazarus_hit"] = 0

    peak_idx = work["lazarus_score"].idxmax()
    peak_df = work.loc[[peak_idx]].copy()

    median_laz = float(work["lazarus_score"].median())
    work["lazarus_group"] = work["lazarus_score"].apply(
        lambda x: "high" if pd.notna(x) and x >= median_laz else "low"
    )

    summary = pd.DataFrame(
        {
            "n_nodes": [int(len(work))],
            "mean_distance_all_nodes": [work["distance_to_seam"].mean()],
            "median_distance_all_nodes": [work["distance_to_seam"].median()],
            "mean_distance_high_lazarus": [work.loc[work["lazarus_group"] == "high", "distance_to_seam"].mean()],
            "mean_distance_low_lazarus": [work.loc[work["lazarus_group"] == "low", "distance_to_seam"].mean()],
            "mean_lazarus_hit_distance": [work.loc[work["lazarus_hit"] == 1, "distance_to_seam"].mean()],
            "peak_lazarus_score": [float(peak_df["lazarus_score"].iloc[0])],
            "peak_lazarus_distance_to_seam": [float(peak_df["distance_to_seam"].iloc[0])],
        }
    )
    return work, summary


def build_path_alignment(paths: pd.DataFrame):
    req = ["probe_id", "step", "distance_to_seam", "lazarus_score", "signed_phase"]
    missing = [c for c in req if c not in paths.columns]
    if missing:
        raise ValueError(f"Missing required path columns: {missing}")

    rows = []
    for probe_id, grp in paths.groupby("probe_id"):
        grp = grp.sort_values("step").reset_index(drop=True)

        dist = pd.to_numeric(grp["distance_to_seam"], errors="coerce")
        laz = pd.to_numeric(grp["lazarus_score"], errors="coerce")
        peak_i = int(laz.idxmax())
        ridge_i = int(dist.idxmin())

        prev = 0
        flip_step = pd.NA
        for _, row in grp.iterrows():
            val = float(row["signed_phase"])
            s = -1 if val < 0 else (1 if val > 0 else 0)
            if s == 0:
                continue
            if prev != 0 and s != prev:
                flip_step = int(row["step"])
                break
            prev = s

        peak_step = int(grp.loc[peak_i, "step"])
        ridge_step = int(grp.loc[ridge_i, "step"])

        if pd.isna(flip_step):
            lag_ridge_to_flip = pd.NA
            lag_peak_to_flip = pd.NA
        else:
            lag_ridge_to_flip = int(flip_step - ridge_step)
            lag_peak_to_flip = int(flip_step - peak_step)

        rows.append(
            {
                "probe_id": probe_id,
                "family": grp["family"].iloc[0] if "family" in grp.columns else "",
                "lazarus_peak_step": peak_step,
                "ridge_contact_step": ridge_step,
                "phase_flip_step": flip_step,
                "peak_distance_to_seam": float(grp.loc[peak_i, "distance_to_seam"]),
                "min_distance_to_seam": float(grp.loc[ridge_i, "distance_to_seam"]),
                "lag_peak_to_ridge": int(ridge_step - peak_step),
                "lag_ridge_to_flip": lag_ridge_to_flip,
                "lag_peak_to_flip": lag_peak_to_flip,
            }
        )

    path_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "n_paths": [int(len(path_df))],
            "mean_peak_distance_to_seam": [pd.to_numeric(path_df["peak_distance_to_seam"], errors="coerce").mean()],
            "median_peak_distance_to_seam": [pd.to_numeric(path_df["peak_distance_to_seam"], errors="coerce").median()],
            "mean_lag_peak_to_ridge": [pd.to_numeric(path_df["lag_peak_to_ridge"], errors="coerce").mean()],
            "median_lag_peak_to_ridge": [pd.to_numeric(path_df["lag_peak_to_ridge"], errors="coerce").median()],
            "mean_lag_ridge_to_flip": [pd.to_numeric(path_df["lag_ridge_to_flip"], errors="coerce").mean()],
            "median_lag_ridge_to_flip": [pd.to_numeric(path_df["lag_ridge_to_flip"], errors="coerce").median()],
            "mean_lag_peak_to_flip": [pd.to_numeric(path_df["lag_peak_to_flip"], errors="coerce").mean()],
            "median_lag_peak_to_flip": [pd.to_numeric(path_df["lag_peak_to_flip"], errors="coerce").median()],
        }
    )
    return path_df, summary


def render_plots(node_df: pd.DataFrame, path_df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(node_df["distance_to_seam"], node_df["lazarus_score"], alpha=0.8)
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("lazarus_score")
    ax.set_title("Lazarus score vs distance to seam")
    fig.tight_layout()
    fig.savefig(outdir / "lazarus_seam_alignment_scatter.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vals = pd.to_numeric(path_df["peak_distance_to_seam"], errors="coerce").dropna()
    ax.hist(vals, bins=min(20, max(5, len(vals) // 5)))
    ax.set_xlabel("peak distance to seam")
    ax.set_ylabel("count")
    ax.set_title("Distribution of Lazarus-peak distances to seam")
    fig.tight_layout()
    fig.savefig(outdir / "lazarus_peak_distance_hist.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    plot_df = path_df[["lag_peak_to_ridge", "lag_ridge_to_flip"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(plot_df):
        ax.scatter(plot_df["lag_peak_to_ridge"], plot_df["lag_ridge_to_flip"], alpha=0.65)
    ax.set_xlabel("lag_peak_to_ridge")
    ax.set_ylabel("lag_ridge_to_flip")
    ax.set_title("Field ordering: peak → ridge → flip")
    fig.tight_layout()
    fig.savefig(outdir / "field_ordering_scatter.png", dpi=220)
    plt.close(fig)


def run_field_alignment(
    mds_csv,
    phase_csv,
    lazarus_csv,
    paths_csv,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(mds_csv, phase_csv, lazarus_csv)
    node_df, node_summary = build_node_alignment(nodes)

    paths = load_paths(paths_csv)
    path_df, path_summary = build_path_alignment(paths)

    node_df.to_csv(outdir / "lazarus_to_seam_distances.csv", index=False)
    node_summary.to_csv(outdir / "field_alignment_node_summary.csv", index=False)
    path_df.to_csv(outdir / "field_alignment_path_metrics.csv", index=False)
    path_summary.to_csv(outdir / "field_alignment_path_summary.csv", index=False)

    render_plots(node_df, path_df, outdir)

    print(outdir / "lazarus_to_seam_distances.csv")
    print(outdir / "field_alignment_node_summary.csv")
    print(outdir / "field_alignment_path_metrics.csv")
    print(outdir / "field_alignment_path_summary.csv")
    print(outdir / "lazarus_seam_alignment_scatter.png")
    print(outdir / "lazarus_peak_distance_hist.png")
    print(outdir / "field_ordering_scatter.png")

    return {
        "node_df": node_df,
        "node_summary": node_summary,
        "path_df": path_df,
        "path_summary": path_summary,
    }
