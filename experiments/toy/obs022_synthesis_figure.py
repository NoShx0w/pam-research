#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    convergence_tidy_csv: str = "outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv"
    eigen_summary_csv: str = "outputs/toy_eigenvector_alignment/eigenvector_alignment_family_summary.csv"
    seam_residency_csv: str = "outputs/toy_seam_residency/seam_residency_family_summary.csv"
    outdir: str = "outputs/obs022_synthesis_figure"
    focal_scale: str = "100000"


FAMILY_ORDER = [
    "settled_distant",
    "off_seam_reorganizing",
    "reorganization_heavy",
    "stable_seam_corridor",
]

FAMILY_LABELS = {
    "settled_distant": "settled_distant",
    "off_seam_reorganizing": "off_seam_reorganizing",
    "reorganization_heavy": "reorganization_heavy",
    "stable_seam_corridor": "stable_seam_corridor",
}


def load_convergence_tidy(path: str, focal_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tidy = pd.read_csv(path).copy()
    tidy["scale"] = tidy["scale"].astype(str)
    tidy["scale_order"] = pd.to_numeric(tidy["scale"], errors="coerce")
    focal = tidy[tidy["scale"] == str(focal_scale)].copy()
    return tidy, focal


def load_eigen_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(path).copy()


def load_seam_residency(path: str) -> pd.DataFrame:
    return pd.read_csv(path).copy()


def reindex_family(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Index(FAMILY_ORDER, name="path_family")
    return df.set_index("path_family").reindex(idx).reset_index()


def panel_a_family_share(ax, tidy: pd.DataFrame) -> None:
    tidy = tidy.copy()
    tidy["scale_order"] = pd.to_numeric(tidy["scale"], errors="coerce")

    for fam in FAMILY_ORDER:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale_order")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["scale"],
            sub["share_paths"],
            marker="o",
            linewidth=2,
            label=FAMILY_LABELS[fam],
        )

    ax.set_title("A. Family share across probe scale")
    ax.set_xlabel("scale")
    ax.set_ylabel("share of sampled paths")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_b_dynamic_privilege(ax, focal: pd.DataFrame) -> None:
    focal = reindex_family(focal)
    x = np.arange(len(focal))
    width = 0.38

    laz = pd.to_numeric(focal["mean_lazarus"], errors="coerce").to_numpy(dtype=float)
    rsp = pd.to_numeric(focal["mean_transition_exposure"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - width / 2, laz, width=width, label="mean Lazarus")
    ax.bar(x + width / 2, rsp, width=width, label="mean response")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in focal["path_family"]], rotation=20, ha="right")
    ax.set_title("B. Dynamic privilege at scale 100000")
    ax.set_ylabel("family mean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_c_eigen_alignment(ax, eigen: pd.DataFrame) -> None:
    eigen = reindex_family(eigen)
    x = np.arange(len(eigen))
    width = 0.38

    fim = pd.to_numeric(eigen["mean_align_fim"], errors="coerce").to_numpy(dtype=float)
    rsp = pd.to_numeric(eigen["mean_align_rsp"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - width / 2, fim, width=width, label="Fisher alignment")
    ax.bar(x + width / 2, rsp, width=width, label="Response alignment")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in eigen["path_family"]], rotation=20, ha="right")
    ax.set_title("C. Principal-direction alignment")
    ax.set_ylabel("mean alignment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_d_seam_mode(ax, seam: pd.DataFrame) -> None:
    seam = reindex_family(seam)
    x = np.arange(len(seam))
    width = 0.25

    seam_frac = pd.to_numeric(seam["mean_seam_fraction"], errors="coerce").to_numpy(dtype=float)
    seam_eps = pd.to_numeric(seam["mean_n_seam_episodes"], errors="coerce").to_numpy(dtype=float)
    seam_run = pd.to_numeric(seam["mean_seam_run_length"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - width, seam_frac, width=width, label="seam fraction")
    ax.bar(x, seam_eps, width=width, label="seam episodes")
    ax.bar(x + width, seam_run, width=width, label="run length")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABELS[f] for f in seam["path_family"]], rotation=20, ha="right")
    ax.set_title("D. Seam traversal mode")
    ax.set_ylabel("family mean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def write_summary_text(
    focal: pd.DataFrame,
    eigen: pd.DataFrame,
    seam: pd.DataFrame,
    outpath: Path,
) -> None:
    focal = reindex_family(focal)
    eigen = reindex_family(eigen)
    seam = reindex_family(seam)

    lines = ["=== OBS-022 Synthesis Figure Summary ===", ""]
    lines.append("Large-scale family ranking (scale 100000)")
    for fam in FAMILY_ORDER:
        f = focal[focal["path_family"] == fam].iloc[0]
        e = eigen[eigen["path_family"] == fam].iloc[0]
        s = seam[seam["path_family"] == fam].iloc[0]
        lines.append(
            f"{fam}: "
            f"share={float(f['share_paths']):.4f}, "
            f"mean_lazarus={float(f['mean_lazarus']):.4f}, "
            f"mean_response={float(f['mean_transition_exposure']):.4f}, "
            f"align_fim={float(e['mean_align_fim']):.4f}, "
            f"align_rsp={float(e['mean_align_rsp']):.4f}, "
            f"seam_fraction={float(s['mean_seam_fraction']):.4f}, "
            f"n_seam_episodes={float(s['mean_n_seam_episodes']):.4f}, "
            f"seam_run_length={float(s['mean_seam_run_length']):.4f}"
        )

    lines.append("")
    lines.append("Interpretive summary")
    lines.append("- stable_seam_corridor dominates dynamic privilege at large scale")
    lines.append("- stable_seam_corridor is most aligned with Fisher/response principal directions")
    lines.append("- reorganization_heavy is more seam-immersed and more fragmented")
    lines.append("- stable_seam_corridor is the coherent seam traversal mode")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 4-panel synthesis figure for OBS-019 to OBS-022.")
    parser.add_argument("--convergence-tidy-csv", default="outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv")
    parser.add_argument("--eigen-summary-csv", default="outputs/toy_eigenvector_alignment/eigenvector_alignment_family_summary.csv")
    parser.add_argument("--seam-residency-csv", default="outputs/toy_seam_residency/seam_residency_family_summary.csv")
    parser.add_argument("--outdir", default="outputs/obs022_synthesis_figure")
    parser.add_argument("--focal-scale", default="100000")
    args = parser.parse_args()

    cfg = Config(
        convergence_tidy_csv=args.convergence_tidy_csv,
        eigen_summary_csv=args.eigen_summary_csv,
        seam_residency_csv=args.seam_residency_csv,
        outdir=args.outdir,
        focal_scale=args.focal_scale,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tidy, focal = load_convergence_tidy(cfg.convergence_tidy_csv, cfg.focal_scale)
    eigen = load_eigen_summary(cfg.eigen_summary_csv)
    seam = load_seam_residency(cfg.seam_residency_csv)

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    panel_a_family_share(axs[0, 0], tidy)
    panel_b_dynamic_privilege(axs[0, 1], focal)
    panel_c_eigen_alignment(axs[1, 0], eigen)
    panel_d_seam_mode(axs[1, 1], seam)

    fig.suptitle("OBS-022 synthesis: convergence, privilege, alignment, and seam traversal", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = outdir / "obs022_synthesis_figure.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    focal.to_csv(outdir / "obs022_focal_scale_family_table.csv", index=False)
    write_summary_text(focal, eigen, seam, outdir / "obs022_synthesis_summary.txt")

    print(fig_path)
    print(outdir / "obs022_focal_scale_family_table.csv")
    print(outdir / "obs022_synthesis_summary.txt")


if __name__ == "__main__":
    main()