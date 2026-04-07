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
    outdir: str = "outputs/obs022_synthesis_figure_v2"
    focal_scale: str = "100000"


FAMILY_ORDER = [
    "settled_distant",
    "off_seam_reorganizing",
    "reorganization_heavy",
    "stable_seam_corridor",
]


def reindex_family(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Index(FAMILY_ORDER, name="path_family")
    return df.set_index("path_family").reindex(idx).reset_index()


def load_convergence_tidy(path: str, focal_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tidy = pd.read_csv(path).copy()
    tidy["scale"] = tidy["scale"].astype(str)
    tidy["scale_order"] = pd.to_numeric(tidy["scale"], errors="coerce")
    focal = tidy[tidy["scale"] == str(focal_scale)].copy()
    return tidy, focal


def panel_a_family_share(ax, tidy: pd.DataFrame) -> None:
    tidy = tidy.copy()
    tidy["scale_order"] = pd.to_numeric(tidy["scale"], errors="coerce")

    for fam in FAMILY_ORDER:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale_order")
        if len(sub) == 0:
            continue
        ax.plot(sub["scale"], sub["share_paths"], marker="o", linewidth=2, label=fam)

    ax.set_title("A. Family share across probe scale")
    ax.set_xlabel("scale")
    ax.set_ylabel("share of sampled paths")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def panel_b_lazarus(ax, focal: pd.DataFrame) -> None:
    focal = reindex_family(focal)
    x = np.arange(len(focal))
    vals = pd.to_numeric(focal["mean_lazarus"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(focal["path_family"], rotation=20, ha="right")
    ax.set_title("B. Mean Lazarus at scale 100000")
    ax.set_ylabel("mean Lazarus")
    ax.grid(alpha=0.25, axis="y")


def panel_c_response(ax, focal: pd.DataFrame) -> None:
    focal = reindex_family(focal)
    x = np.arange(len(focal))
    vals = pd.to_numeric(focal["mean_transition_exposure"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(focal["path_family"], rotation=20, ha="right")
    ax.set_title("C. Mean response at scale 100000")
    ax.set_ylabel("mean response")
    ax.grid(alpha=0.25, axis="y")


def panel_d_eigen_alignment(ax, eigen: pd.DataFrame) -> None:
    eigen = reindex_family(eigen)
    x = np.arange(len(eigen))
    width = 0.38

    fim = pd.to_numeric(eigen["mean_align_fim"], errors="coerce").to_numpy(dtype=float)
    rsp = pd.to_numeric(eigen["mean_align_rsp"], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - width / 2, fim, width=width, label="Fisher alignment")
    ax.bar(x + width / 2, rsp, width=width, label="Response alignment")

    ax.set_xticks(x)
    ax.set_xticklabels(eigen["path_family"], rotation=20, ha="right")
    ax.set_title("D. Principal-direction alignment")
    ax.set_ylabel("mean alignment")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_e_seam_mode(ax, seam: pd.DataFrame) -> None:
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
    ax.set_xticklabels(seam["path_family"], rotation=20, ha="right")
    ax.set_title("E. Seam traversal mode")
    ax.set_ylabel("family mean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def panel_f_text(ax) -> None:
    ax.axis("off")
    text = (
        "OBS-022 summary\n\n"
        "• stable_seam_corridor emerges and stabilizes at large scale\n"
        "• dominates Lazarus and response at scale 100000\n"
        "• most aligned with Fisher and response eigenvectors\n"
        "• reorganization_heavy is more seam-immersed and more fragmented\n"
        "• corridor is the coherent seam traversal mode"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2x3 synthesis figure for OBS-022.")
    parser.add_argument("--convergence-tidy-csv", default="outputs/obs019_scale_convergence/obs019_scale_convergence_tidy.csv")
    parser.add_argument("--eigen-summary-csv", default="outputs/toy_eigenvector_alignment/eigenvector_alignment_family_summary.csv")
    parser.add_argument("--seam-residency-csv", default="outputs/toy_seam_residency/seam_residency_family_summary.csv")
    parser.add_argument("--outdir", default="outputs/obs022_synthesis_figure_v2")
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
    eigen = pd.read_csv(cfg.eigen_summary_csv).copy()
    seam = pd.read_csv(cfg.seam_residency_csv).copy()

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    panel_a_family_share(axs[0, 0], tidy)
    panel_b_lazarus(axs[0, 1], focal)
    panel_c_response(axs[0, 2], focal)
    panel_d_eigen_alignment(axs[1, 0], eigen)
    panel_e_seam_mode(axs[1, 1], seam)
    panel_f_text(axs[1, 2])

    fig.suptitle("OBS-022 synthesis: convergence, privilege, alignment, and seam traversal", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    outpath = outdir / "obs022_synthesis_figure_v2.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

    print(outpath)


if __name__ == "__main__":
    main()
