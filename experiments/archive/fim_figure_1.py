import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_node_table(mds_csv: str | Path, curvature_csv: str | Path, phase_csv: str | Path, lazarus_csv: str | Path) -> pd.DataFrame:
    mds = pd.read_csv(mds_csv)
    curv = pd.read_csv(curvature_csv)
    phase = pd.read_csv(phase_csv)
    laz = pd.read_csv(lazarus_csv)

    df = mds.copy()
    if "scalar_curvature" in curv.columns:
        df = df.merge(
            curv[[c for c in ["r", "alpha", "scalar_curvature"] if c in curv.columns]],
            on=["r", "alpha"],
            how="left",
        )
    keep_phase = [c for c in ["r", "alpha", "signed_phase", "distance_to_seam"] if c in phase.columns]
    df = df.merge(phase[keep_phase], on=["r", "alpha"], how="left")
    keep_laz = [c for c in ["r", "alpha", "lazarus_score", "lazarus_hit"] if c in laz.columns]
    df = df.merge(laz[keep_laz], on=["r", "alpha"], how="left")
    return df


def load_seam(seam_csv: str | Path, node_df: pd.DataFrame) -> pd.DataFrame:
    p = Path(seam_csv)
    if not p.exists():
        return pd.DataFrame()
    seam = pd.read_csv(p)
    if not {"mds1", "mds2"}.issubset(seam.columns):
        seam = seam.merge(node_df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")
    return seam.dropna(subset=["mds1", "mds2"]).copy()


def load_critical(critical_csv: str | Path, node_df: pd.DataFrame) -> pd.DataFrame:
    p = Path(critical_csv)
    if not p.exists():
        return pd.DataFrame()
    crit = pd.read_csv(p)
    if not {"mds1", "mds2"}.issubset(crit.columns):
        crit = crit.merge(node_df[["r", "alpha", "mds1", "mds2"]], on=["r", "alpha"], how="left")
    return crit.dropna(subset=["mds1", "mds2"]).copy()


def load_paths(paths_csv: str | Path) -> pd.DataFrame:
    p = Path(paths_csv)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def select_representative_paths(paths: pd.DataFrame, max_paths: int = 6) -> list[str]:
    if paths.empty or "probe_id" not in paths.columns:
        return []

    summaries = []
    for probe_id, grp in paths.groupby("probe_id"):
        grp = grp.sort_values("step")
        laz_max = float(grp["lazarus_score"].max()) if "lazarus_score" in grp.columns else 0.0
        path_len = len(grp)
        sign_change = 0
        if "signed_phase" in grp.columns:
            prev = 0
            for v in grp["signed_phase"].astype(float):
                s = -1 if v < 0 else (1 if v > 0 else 0)
                if s == 0:
                    continue
                if prev != 0 and s != prev:
                    sign_change = 1
                    break
                prev = s
        summaries.append({
            "probe_id": probe_id,
            "laz_max": laz_max,
            "path_len": path_len,
            "sign_change": sign_change,
        })

    sdf = pd.DataFrame(summaries)
    sdf = sdf.sort_values(["sign_change", "laz_max", "path_len"], ascending=[False, False, False])
    return list(sdf["probe_id"].head(max_paths))


def select_transition_path(paths: pd.DataFrame) -> str | None:
    if paths.empty or "probe_id" not in paths.columns:
        return None

    summaries = []
    for probe_id, grp in paths.groupby("probe_id"):
        grp = grp.sort_values("step")
        laz_max = float(grp["lazarus_score"].max()) if "lazarus_score" in grp.columns else 0.0
        sign_change = 0
        if "signed_phase" in grp.columns:
            prev = 0
            for v in grp["signed_phase"].astype(float):
                s = -1 if v < 0 else (1 if v > 0 else 0)
                if s == 0:
                    continue
                if prev != 0 and s != prev:
                    sign_change = 1
                    break
                prev = s
        summaries.append({"probe_id": probe_id, "laz_max": laz_max, "sign_change": sign_change})

    sdf = pd.DataFrame(summaries)
    sdf = sdf.sort_values(["sign_change", "laz_max"], ascending=[False, False])
    if len(sdf) == 0:
        return None
    return str(sdf.iloc[0]["probe_id"])


def mark_transition_point(grp: pd.DataFrame):
    prev = 0
    for _, row in grp.iterrows():
        val = float(row["signed_phase"])
        s = -1 if val < 0 else (1 if val > 0 else 0)
        if s == 0:
            continue
        if prev != 0 and s != prev:
            return row
        prev = s
    return None


def mark_lazarus_peak(grp: pd.DataFrame):
    if "lazarus_score" not in grp.columns:
        return None
    idx = grp["lazarus_score"].astype(float).idxmax()
    return grp.loc[idx]


def render_figure_1(nodes: pd.DataFrame, seam: pd.DataFrame, critical: pd.DataFrame, paths: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))

    # A. Phase manifold
    ax = axes[0, 0]
    sc = ax.scatter(
        nodes["mds1"], nodes["mds2"],
        c=nodes["signed_phase"], cmap="coolwarm", vmin=-1, vmax=1,
        s=46, alpha=0.82
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.8, alpha=0.95)
    ax.set_title("A. Phase manifold")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.colorbar(sc, ax=ax, shrink=0.84, label="signed phase")

    # B. Compression field
    ax = axes[0, 1]
    sc = ax.scatter(
        nodes["mds1"], nodes["mds2"],
        c=nodes["lazarus_score"], cmap="plasma",
        s=46, alpha=0.86
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.8, alpha=0.95)
    ax.set_title("B. Compression field")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.colorbar(sc, ax=ax, shrink=0.84, label="lazarus_score")

    # C. Operator trajectories
    ax = axes[1, 0]
    bg = ax.scatter(
        nodes["mds1"], nodes["mds2"],
        c=nodes["lazarus_score"], cmap="plasma",
        s=24, alpha=0.22
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.4, alpha=0.95)
    keep_ids = select_representative_paths(paths, max_paths=6)
    for probe_id in keep_ids:
        grp = paths[paths["probe_id"] == probe_id].sort_values("step")
        ax.plot(grp["mds1"], grp["mds2"], linewidth=2.0, alpha=0.92, label=probe_id)
        ax.scatter(grp.iloc[[0]]["mds1"], grp.iloc[[0]]["mds2"], s=70, marker="o")
        ax.scatter(grp.iloc[[-1]]["mds1"], grp.iloc[[-1]]["mds2"], s=90, marker="X")
    ax.set_title("C. Operator trajectories")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    if keep_ids:
        ax.legend(loc="best", fontsize=8)
    fig.colorbar(bg, ax=ax, shrink=0.84, label="lazarus_score")

    # D. Compression-driven transition
    ax = axes[1, 1]
    bg = ax.scatter(
        nodes["mds1"], nodes["mds2"],
        c=nodes["lazarus_score"], cmap="plasma",
        s=26, alpha=0.18
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.8, alpha=0.95, label="seam")
    if not critical.empty:
        ax.scatter(critical["mds1"], critical["mds2"], marker="*", s=130, linewidths=0.6, edgecolors="black", label="critical points")

    chosen = select_transition_path(paths)
    if chosen is not None:
        grp = paths[paths["probe_id"] == chosen].sort_values("step").reset_index(drop=True)
        ax.plot(grp["mds1"], grp["mds2"], linewidth=2.6, alpha=0.96, label=f"{chosen} trajectory")
        ax.scatter(grp.iloc[[0]]["mds1"], grp.iloc[[0]]["mds2"], s=80, marker="o", label="start")

        peak = mark_lazarus_peak(grp)
        if peak is not None:
            ax.scatter([peak["mds1"]], [peak["mds2"]], s=130, marker="o", edgecolors="black", linewidths=0.8, label="Lazarus peak")

        flip = mark_transition_point(grp)
        if flip is not None:
            ax.scatter([flip["mds1"]], [flip["mds2"]], s=120, marker="X", label="transition")

    ax.set_title("D. Compression-driven transition")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(bg, ax=ax, shrink=0.84, label="lazarus_score")

    fig.suptitle("Figure 1 — PAM Observatory Overview", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render Figure 1 — PAM Observatory Overview.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_figure_1")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_node_table(args.mds_csv, args.curvature_csv, args.phase_csv, args.lazarus_csv)
    seam = load_seam(args.seam_csv, nodes)
    critical = load_critical(args.critical_csv, nodes)
    paths = load_paths(args.paths_csv)

    outpath = outdir / "figure_1_observatory_overview.png"
    render_figure_1(nodes, seam, critical, paths, outpath)
    print(outpath)


if __name__ == "__main__":
    main()
