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
        df = df.merge(curv[[c for c in ["r", "alpha", "scalar_curvature"] if c in curv.columns]], on=["r", "alpha"], how="left")
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


def load_paths(paths_csv: str | Path, max_paths: int = 80) -> pd.DataFrame:
    p = Path(paths_csv)
    if not p.exists():
        return pd.DataFrame()
    paths = pd.read_csv(p)
    if "probe_id" in paths.columns:
        keep = list(paths["probe_id"].drop_duplicates().head(max_paths))
        paths = paths[paths["probe_id"].isin(keep)].copy()
    return paths


def _base_panel(ax, nodes: pd.DataFrame, seam: pd.DataFrame, critical: pd.DataFrame, color_col: str, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    sc = ax.scatter(
        nodes["mds1"],
        nodes["mds2"],
        c=nodes[color_col],
        cmap=cmap,
        s=42,
        alpha=0.82,
        vmin=vmin,
        vmax=vmax,
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        ax.plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.0, alpha=0.9)
    if not critical.empty:
        ax.scatter(critical["mds1"], critical["mds2"], marker="*", s=120, linewidths=0.5, edgecolors="black")
    ax.set_title(title)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    return sc


def render_pretty_geometry(nodes: pd.DataFrame, seam: pd.DataFrame, critical: pd.DataFrame, paths: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # Panel A: signed phase
    sc1 = _base_panel(
        axes[0, 0], nodes, seam, critical,
        color_col="signed_phase", title="A. Phase manifold", cmap="coolwarm", vmin=-1, vmax=1
    )
    fig.colorbar(sc1, ax=axes[0, 0], shrink=0.84, label="signed phase")

    # Panel B: curvature
    curv = nodes["scalar_curvature"].abs() if "scalar_curvature" in nodes.columns else pd.Series([0]*len(nodes))
    nodes = nodes.copy()
    nodes["abs_curvature"] = curv
    sc2 = _base_panel(
        axes[0, 1], nodes, seam, critical,
        color_col="abs_curvature", title="B. Curvature field", cmap="magma"
    )
    fig.colorbar(sc2, ax=axes[0, 1], shrink=0.84, label="|curvature|")

    # Panel C: Lazarus field
    sc3 = _base_panel(
        axes[1, 0], nodes, seam, critical,
        color_col="lazarus_score", title="C. Lazarus compression field", cmap="plasma"
    )
    fig.colorbar(sc3, ax=axes[1, 0], shrink=0.84, label="lazarus_score")

    # Panel D: probe flow over Lazarus
    sc4 = axes[1, 1].scatter(
        nodes["mds1"], nodes["mds2"], c=nodes["lazarus_score"], cmap="plasma", s=28, alpha=0.35
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        axes[1, 1].plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.0, alpha=0.9)
    if not critical.empty:
        axes[1, 1].scatter(critical["mds1"], critical["mds2"], marker="*", s=120, linewidths=0.5, edgecolors="black")

    # Panel D: probe flow over Lazarus
    sc4 = axes[1, 1].scatter(
        nodes["mds1"], nodes["mds2"], c=nodes["lazarus_score"], cmap="plasma", s=28, alpha=0.20
    )
    if not seam.empty:
        seam_ord = seam.sort_values("mds1")
        axes[1, 1].plot(seam_ord["mds1"], seam_ord["mds2"], color="black", linewidth=2.2, alpha=0.9)
    if not critical.empty:
        axes[1, 1].scatter(
            critical["mds1"], critical["mds2"],
            marker="*", s=120, linewidths=0.5, edgecolors="black", zorder=6
        )

    if not paths.empty and {"probe_id", "step", "mds1", "mds2"}.issubset(paths.columns):
        # Normalize path-vertex coloring to path Lazarus if available
        path_has_laz = "lazarus_score" in paths.columns

        marks = []
        seam_marks_x = []
        seam_marks_y = []

        for probe_id, grp in paths.groupby("probe_id"):
            grp = grp.sort_values("step").reset_index(drop=True)

            # 1) connecting edges
            axes[1, 1].plot(
                grp["mds1"], grp["mds2"],
                linewidth=1.6,
                alpha=0.75,
                zorder=3,
            )

            # 2) explicit vertices
            if path_has_laz:
                axes[1, 1].scatter(
                    grp["mds1"], grp["mds2"],
                    c=grp["lazarus_score"],
                    cmap="plasma",
                    s=54,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=4,
                )
            else:
                axes[1, 1].scatter(
                    grp["mds1"], grp["mds2"],
                    s=54,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=4,
                )

            # 3) seam-contact vertex rings, if seam distance is available
            if "distance_to_seam" in grp.columns:
                seam_hit = pd.to_numeric(grp["distance_to_seam"], errors="coerce") <= 0.15
                if seam_hit.any():
                    seam_marks_x.extend(grp.loc[seam_hit, "mds1"].tolist())
                    seam_marks_y.extend(grp.loc[seam_hit, "mds2"].tolist())

            # 4) first sign-flip / transition marker
            if "signed_phase" in grp.columns:
                prev = 0
                for _, row in grp.iterrows():
                    val = float(row["signed_phase"])
                    sign = -1 if val < 0 else (1 if val > 0 else 0)
                    if sign == 0:
                        continue
                    if prev != 0 and sign != prev:
                        marks.append((row["mds1"], row["mds2"]))
                        break
                    prev = sign

        # seam-contact rings
        if seam_marks_x:
            axes[1, 1].scatter(
                seam_marks_x, seam_marks_y,
                s=95,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                alpha=0.9,
                zorder=5,
            )

        # transition X marks
        if marks:
            mx, my = zip(*marks)
            axes[1, 1].scatter(
                mx, my,
                marker="X",
                s=90,
                linewidths=0.8,
                edgecolors="black",
                zorder=6,
            )

    axes[1, 1].set_title("D. Probe flow and transition geometry")
    axes[1, 1].set_xlabel("MDS 1")
    axes[1, 1].set_ylabel("MDS 2")
    fig.colorbar(sc4, ax=axes[1, 1], shrink=0.84, label="lazarus_score")

    fig.suptitle("PAM Observatory — Pretty Geometry", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a canonical pretty geometry panel for the PAM Observatory.")
    parser.add_argument("--mds-csv", default="outputs/fim_mds/mds_coords.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--phase-csv", default="outputs/fim_phase/signed_phase_coords.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--paths-csv", default="outputs/fim_ops_scaled/scaled_probe_paths.csv")
    parser.add_argument("--outdir", default="outputs/fim_pretty")
    parser.add_argument("--max-paths", type=int, default=80)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    nodes = load_node_table(args.mds_csv, args.curvature_csv, args.phase_csv, args.lazarus_csv)
    seam = load_seam(args.seam_csv, nodes)
    critical = load_critical(args.critical_csv, nodes)
    paths = load_paths(args.paths_csv, max_paths=args.max_paths)

    outpath = outdir / "pretty_geometry_panel.png"
    render_pretty_geometry(nodes, seam, critical, paths, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
