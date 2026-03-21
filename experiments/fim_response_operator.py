#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def load_paths(paths_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(paths_csv)


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


def build_response_operator(df: pd.DataFrame):
    work = df.copy()
    work["signed_phase"] = pd.to_numeric(work["signed_phase"], errors="coerce")
    work["lazarus_score"] = pd.to_numeric(work["lazarus_score"], errors="coerce")
    work["distance_to_seam"] = pd.to_numeric(work["distance_to_seam"], errors="coerce")

    phase_grads = estimate_local_gradients(work, "signed_phase")
    laz_grads = estimate_local_gradients(work, "lazarus_score")
    work = pd.concat([work, phase_grads, laz_grads], axis=1)

    gx_phi = work["grad_signed_phase_x"].to_numpy(dtype=float)
    gy_phi = work["grad_signed_phase_y"].to_numpy(dtype=float)
    gx_l = work["grad_lazarus_score_x"].to_numpy(dtype=float)
    gy_l = work["grad_lazarus_score_y"].to_numpy(dtype=float)

    dot = gx_phi * gx_l + gy_phi * gy_l
    norm_phi = np.sqrt(gx_phi ** 2 + gy_phi ** 2)
    norm_l = np.sqrt(gx_l ** 2 + gy_l ** 2)
    strength = norm_phi * norm_l
    cosine = np.where((strength > 0) & np.isfinite(strength), dot / strength, np.nan)

    work["response_strength"] = strength
    work["signed_coupling"] = dot
    work["cosine_alignment"] = np.clip(cosine, -1.0, 1.0)

    work["T_xx"] = gx_l * gx_phi
    work["T_xy"] = gx_l * gy_phi
    work["T_yx"] = gy_l * gx_phi
    work["T_yy"] = gy_l * gy_phi
    work["trace_T"] = work["T_xx"] + work["T_yy"]
    work["frobenius_T"] = np.sqrt(
        work["T_xx"] ** 2 + work["T_xy"] ** 2 + work["T_yx"] ** 2 + work["T_yy"] ** 2
    )

    valid_strength = pd.to_numeric(work["response_strength"], errors="coerce")
    tau = float(valid_strength.quantile(0.9))
    work["response_active"] = (valid_strength >= tau).astype(int)

    summary = pd.DataFrame(
        {
            "n_nodes": [int(len(work))],
            "mean_response_strength": [pd.to_numeric(work["response_strength"], errors="coerce").mean()],
            "median_response_strength": [pd.to_numeric(work["response_strength"], errors="coerce").median()],
            "mean_signed_coupling": [pd.to_numeric(work["signed_coupling"], errors="coerce").mean()],
            "median_signed_coupling": [pd.to_numeric(work["signed_coupling"], errors="coerce").median()],
            "mean_cosine_alignment": [pd.to_numeric(work["cosine_alignment"], errors="coerce").mean()],
            "median_cosine_alignment": [pd.to_numeric(work["cosine_alignment"], errors="coerce").median()],
            "mean_strength_near_seam": [
                pd.to_numeric(
                    work.loc[work["distance_to_seam"] <= work["distance_to_seam"].median(), "response_strength"],
                    errors="coerce",
                ).mean()
            ],
            "mean_strength_far_from_seam": [
                pd.to_numeric(
                    work.loc[work["distance_to_seam"] > work["distance_to_seam"].median(), "response_strength"],
                    errors="coerce",
                ).mean()
            ],
            "response_activation_threshold_q90": [tau],
            "active_fraction": [work["response_active"].mean()],
        }
    )
    return work, summary


def attach_operator_to_paths(paths: pd.DataFrame, op_nodes: pd.DataFrame):
    keep = [
        "r", "alpha", "response_strength", "signed_coupling", "cosine_alignment",
        "response_active", "trace_T", "frobenius_T"
    ]
    merged = paths.merge(op_nodes[keep], on=["r", "alpha"], how="left")

    rows = []
    for probe_id, grp in merged.groupby("probe_id"):
        grp = grp.sort_values("step").reset_index(drop=True)

        flip = 0
        prev = 0
        flip_step = pd.NA
        if "signed_phase" in grp.columns:
            for _, row in grp.iterrows():
                val = float(row["signed_phase"])
                s = -1 if val < 0 else (1 if val > 0 else 0)
                if s == 0:
                    continue
                if prev != 0 and s != prev:
                    flip = 1
                    flip_step = int(row["step"])
                    break
                prev = s

        strength = pd.to_numeric(grp["response_strength"], errors="coerce")
        coupling = pd.to_numeric(grp["signed_coupling"], errors="coerce")
        align = pd.to_numeric(grp["cosine_alignment"], errors="coerce")
        active = pd.to_numeric(grp["response_active"], errors="coerce")

        peak_idx = strength.idxmax()
        peak_step = int(grp.loc[peak_idx, "step"]) if pd.notna(peak_idx) else pd.NA

        lag_peak_to_flip = pd.NA
        if flip and pd.notna(peak_step):
            lag_peak_to_flip = int(flip_step - peak_step)

        rows.append(
            {
                "probe_id": probe_id,
                "family": grp["family"].iloc[0] if "family" in grp.columns else "",
                "phase_flip": flip,
                "phase_flip_step": flip_step,
                "response_peak_step": peak_step,
                "lag_response_peak_to_flip": lag_peak_to_flip,
                "mean_response_strength": strength.mean(),
                "max_response_strength": strength.max(),
                "mean_signed_coupling": coupling.mean(),
                "max_signed_coupling": coupling.max(),
                "mean_cosine_alignment": align.mean(),
                "share_response_active": active.mean(),
            }
        )

    path_summary = pd.DataFrame(rows)
    agg = pd.DataFrame(
        {
            "n_paths": [int(len(path_summary))],
            "mean_response_strength_flip": [
                pd.to_numeric(path_summary.loc[path_summary["phase_flip"] == 1, "mean_response_strength"], errors="coerce").mean()
            ],
            "mean_response_strength_no_flip": [
                pd.to_numeric(path_summary.loc[path_summary["phase_flip"] == 0, "mean_response_strength"], errors="coerce").mean()
            ],
            "mean_share_response_active_flip": [
                pd.to_numeric(path_summary.loc[path_summary["phase_flip"] == 1, "share_response_active"], errors="coerce").mean()
            ],
            "mean_share_response_active_no_flip": [
                pd.to_numeric(path_summary.loc[path_summary["phase_flip"] == 0, "share_response_active"], errors="coerce").mean()
            ],
            "mean_lag_response_peak_to_flip": [
                pd.to_numeric(path_summary["lag_response_peak_to_flip"], errors="coerce").mean()
            ],
            "median_lag_response_peak_to_flip": [
                pd.to_numeric(path_summary["lag_response_peak_to_flip"], errors="coerce").median()
            ],
        }
    )
    return merged, path_summary, agg


def render_plots(node_df: pd.DataFrame, path_summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sc = ax.scatter(node_df["mds1"], node_df["mds2"], c=node_df["response_strength"], s=70, alpha=0.9)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Response-operator strength on manifold")
    fig.colorbar(sc, ax=ax, label="response_strength")
    fig.tight_layout()
    fig.savefig(outdir / "response_strength_map.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sc = ax.scatter(node_df["mds1"], node_df["mds2"], c=node_df["signed_coupling"], s=70, alpha=0.9)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Signed coupling on manifold")
    fig.colorbar(sc, ax=ax, label="signed_coupling")
    fig.tight_layout()
    fig.savefig(outdir / "signed_coupling_map.png", dpi=220)
    plt.close(fig)

    plot_df = node_df[["distance_to_seam", "cosine_alignment"]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(plot_df["distance_to_seam"], plot_df["cosine_alignment"], alpha=0.75)
    ax.set_xlabel("distance_to_seam")
    ax.set_ylabel("cosine_alignment")
    ax.set_title("Response-operator alignment vs seam distance")
    fig.tight_layout()
    fig.savefig(outdir / "response_alignment_vs_seam_distance.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    flip_df = path_summary.copy()
    flip_df["phase_flip_label"] = flip_df["phase_flip"].map({0: "no_flip", 1: "flip"})
    grouped = flip_df.groupby("phase_flip_label", observed=False)["mean_response_strength"].mean().reindex(["no_flip", "flip"])
    ax.bar(grouped.index, grouped.values)
    ax.set_ylabel("mean response strength")
    ax.set_title("Path-level response strength by outcome")
    fig.tight_layout()
    fig.savefig(outdir / "response_strength_by_flip.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct and analyze a local response operator T = ∇L ⊗ ∇φ.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_response_operator")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.mds_csv, args.phase_csv, args.lazarus_csv)
    op_nodes, node_summary = build_response_operator(nodes)
    path_nodes, path_summary, path_agg = attach_operator_to_paths(load_paths(args.paths_csv), op_nodes)

    op_nodes.to_csv(outdir / "response_operator_nodes.csv", index=False)
    node_summary.to_csv(outdir / "response_operator_node_summary.csv", index=False)
    path_nodes.to_csv(outdir / "response_operator_paths.csv", index=False)
    path_summary.to_csv(outdir / "response_operator_path_summary.csv", index=False)
    path_agg.to_csv(outdir / "response_operator_path_aggregate.csv", index=False)

    render_plots(op_nodes, path_summary, outdir)

    print(outdir / "response_operator_nodes.csv")
    print(outdir / "response_operator_node_summary.csv")
    print(outdir / "response_operator_paths.csv")
    print(outdir / "response_operator_path_summary.csv")
    print(outdir / "response_operator_path_aggregate.csv")
    print(outdir / "response_strength_map.png")
    print(outdir / "signed_coupling_map.png")
    print(outdir / "response_alignment_vs_seam_distance.png")
    print(outdir / "response_strength_by_flip.png")


if __name__ == "__main__":
    main()
