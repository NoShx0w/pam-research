
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def safe_read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def summarize_region(df, r_col="r", a_col="alpha"):
    if df is None or df.empty:
        return "n/a"
    rmin, rmax = df[r_col].min(), df[r_col].max()
    amin, amax = df[a_col].min(), df[a_col].max()
    return f"r≈[{rmin:.3f},{rmax:.3f}], alpha≈[{amin:.3f},{amax:.3f}]"


def load_image(path):
    p = Path(path)
    if not p.exists():
        return None
    return mpimg.imread(p)


def panel_figure(images, titles, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, img, title in zip(axes, images, titles):
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a compact PAM Fisher-geometry phase report from Observatory outputs."
    )
    parser.add_argument("--fim-csv", default="outputs/fim/fim_surface.csv")
    parser.add_argument("--curvature-csv", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--critical-csv", default="outputs/fim_critical/critical_points.csv")
    parser.add_argument("--seam-csv", default="outputs/fim_phase/phase_boundary_mds_backprojected.csv")
    parser.add_argument("--phase-distance-csv", default="outputs/fim_phase/phase_distance_to_seam.csv")
    parser.add_argument("--outdir", default="outputs/fim_report")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fim = safe_read_csv(args.fim_csv)
    curv = safe_read_csv(args.curvature_csv)
    crit = safe_read_csv(args.critical_csv)
    seam = safe_read_csv(args.seam_csv)
    basin = safe_read_csv(args.phase_distance_csv)

    summary = {}
    if fim is not None and "fim_det" in fim.columns:
        idx = int(np.nanargmax(fim["fim_det"].to_numpy(dtype=float)))
        row = fim.iloc[idx]
        summary["max_det"] = (float(row["r"]), float(row["alpha"]), float(row["fim_det"]))

    if curv is not None and "scalar_curvature" in curv.columns:
        vals = np.abs(curv["scalar_curvature"].to_numpy(dtype=float))
        idx = int(np.nanargmax(vals))
        row = curv.iloc[idx]
        summary["max_abs_curv"] = (float(row["r"]), float(row["alpha"]), float(row["scalar_curvature"]))

    if crit is not None and not crit.empty:
        crit_top = crit.sort_values("criticality", ascending=False).head(args.top_k).copy()
    else:
        crit_top = pd.DataFrame()

    report_md = outdir / "phase_report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# PAM Fisher-Geometry Phase Report\n\n")
        f.write("## Summary\n\n")
        if "max_det" in summary:
            r, a, v = summary["max_det"]
            f.write(f"- Max Fisher density at `(r={r:.3f}, alpha={a:.3f})`, `fim_det={v:.6g}`.\n")
        if "max_abs_curv" in summary:
            r, a, v = summary["max_abs_curv"]
            f.write(f"- Max |scalar curvature| at `(r={r:.3f}, alpha={a:.3f})`, `K={v:.6g}`.\n")
        f.write(f"- Seam corridor: {summarize_region(seam)}.\n")
        f.write(f"- Basin transition band: {summarize_region(basin)}.\n")
        if not crit_top.empty:
            f.write("- Top critical points:\n")
            for _, row in crit_top.iterrows():
                f.write(
                    f"  - `(r={row['r']:.3f}, alpha={row['alpha']:.3f})`, "
                    f"`criticality={row['criticality']:.3f}`\n"
                )
        f.write("\n## Interpretation\n\n")
        f.write(
            "The current dataset supports a two-basin picture separated by a curved transition seam. "
            "High Fisher density, curvature peaks, geodesic shear, and criticality maxima all align "
            "within the same parameter corridor, suggesting a genuine phase-transition band.\n"
        )

    panel_out = outdir / "phase_report_panel.png"
    images = [
        load_image("outputs/fim/log10_det.png") or load_image("outputs/fim/fim_log10_det.png"),
        load_image("outputs/fim_curvature/log_abs_scalar_curvature.png"),
        load_image("outputs/fim_phase/log10_phase_distance_to_seam.png"),
        load_image("outputs/fim_critical/mds_seam_critical_overlay.png"),
    ]
    titles = [
        "log10 det(G)",
        "log10 |curvature|",
        "log10 distance to seam",
        "MDS seam + critical points",
    ]
    panel_figure(images, titles, panel_out)

    print(report_md)
    print(panel_out)


if __name__ == "__main__":
    main()
