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
    root: str = "outputs/scales"
    scales: tuple[str, ...] = ("10", "100", "1000", "10000", "100000")
    outdir: str = "outputs/obs019_scale_convergence"


def read_family_counts(scale_root: Path) -> pd.DataFrame:
    path = scale_root / "toy_scaled_probe_path_families" / "geodesic_path_family_summary.csv"
    df = pd.read_csv(path).copy()
    return df


def read_lazarus(scale_root: Path) -> pd.DataFrame:
    path = scale_root / "toy_scaled_probe_family_operator_overlay" / "geodesic_family_operator_family_summary.csv"
    df = pd.read_csv(path).copy()
    return df


def read_response(scale_root: Path) -> pd.DataFrame:
    path = scale_root / "toy_scaled_probe_family_transition_overlay" / "geodesic_family_transition_family_summary.csv"
    df = pd.read_csv(path).copy()
    return df


def build_tidy_summary(cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scale in cfg.scales:
        sroot = Path(cfg.root) / scale

        fam = read_family_counts(sroot)
        laz = read_lazarus(sroot)
        rsp = read_response(sroot)

        merged = fam.merge(laz, on=["path_family", "n_paths"], how="left", suffixes=("", "_laz"))
        merged = merged.merge(rsp, on=["path_family", "n_paths"], how="left", suffixes=("", "_rsp"))

        total_paths = float(merged["n_paths"].sum()) if len(merged) else float("nan")

        for _, row in merged.iterrows():
            rows.append(
                {
                    "scale": scale,
                    "path_family": row["path_family"],
                    "n_paths": int(row["n_paths"]),
                    "share_paths": float(row["n_paths"] / total_paths) if total_paths > 0 else float("nan"),
                    "mean_length": float(row["mean_length"]) if "mean_length" in row else float("nan"),
                    "mean_near_fraction": float(row["mean_near_fraction"]) if "mean_near_fraction" in row else float("nan"),
                    "mean_angle_jump": float(row["mean_angle_jump"]) if "mean_angle_jump" in row else float("nan"),
                    "mean_sector_changes": float(row["mean_sector_changes"]) if "mean_sector_changes" in row else float("nan"),
                    "mean_lazarus": float(row["mean_lazarus"]) if "mean_lazarus" in row else float("nan"),
                    "mean_high_lazarus_fraction": float(row["mean_high_lazarus_fraction"]) if "mean_high_lazarus_fraction" in row else float("nan"),
                    "mean_transition_exposure": float(row["mean_transition_exposure"]) if "mean_transition_exposure" in row else float("nan"),
                    "mean_high_transition_fraction": float(row["mean_high_transition_fraction"]) if "mean_high_transition_fraction" in row else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    scale_order = {s: i for i, s in enumerate(cfg.scales)}
    df["scale_order"] = df["scale"].map(scale_order)
    df = df.sort_values(["scale_order", "path_family"]).reset_index(drop=True)
    return df


def build_scale_winners(tidy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scale, sub in tidy.groupby("scale", sort=False):
        sub = sub.copy()

        laz_sub = sub.dropna(subset=["mean_lazarus"])
        rsp_sub = sub.dropna(subset=["mean_transition_exposure"])

        top_lazarus_family = None
        top_lazarus_value = float("nan")
        if len(laz_sub):
            idx = laz_sub["mean_lazarus"].astype(float).idxmax()
            top_lazarus_family = str(laz_sub.loc[idx, "path_family"])
            top_lazarus_value = float(laz_sub.loc[idx, "mean_lazarus"])

        top_response_family = None
        top_response_value = float("nan")
        if len(rsp_sub):
            idx = rsp_sub["mean_transition_exposure"].astype(float).idxmax()
            top_response_family = str(rsp_sub.loc[idx, "path_family"])
            top_response_value = float(rsp_sub.loc[idx, "mean_transition_exposure"])

        stable = sub[sub["path_family"] == "stable_seam_corridor"]
        stable_share = float(stable["share_paths"].iloc[0]) if len(stable) else float("nan")

        rows.append(
            {
                "scale": scale,
                "top_lazarus_family": top_lazarus_family,
                "top_lazarus_value": top_lazarus_value,
                "top_response_family": top_response_family,
                "top_response_value": top_response_value,
                "stable_seam_corridor_share": stable_share,
            }
        )

    return pd.DataFrame(rows)


def write_summary_text(tidy: pd.DataFrame, winners: pd.DataFrame, outpath: Path) -> None:
    lines: list[str] = []
    lines.append("=== OBS-019 Scale Convergence Summary ===")
    lines.append("")
    for _, row in winners.iterrows():
        lines.append(
            f"scale={row['scale']}: "
            f"top_lazarus_family={row['top_lazarus_family']} ({row['top_lazarus_value']:.4f}), "
            f"top_response_family={row['top_response_family']} ({row['top_response_value']:.4f}), "
            f"stable_seam_corridor_share={row['stable_seam_corridor_share']:.4f}"
        )
    lines.append("")
    lines.append("Per-scale family detail")
    for scale, sub in tidy.groupby("scale", sort=False):
        lines.append(f"  scale {scale}")
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['path_family']}: "
                f"n={int(row['n_paths'])}, "
                f"share={row['share_paths']:.4f}, "
                f"mean_lazarus={row['mean_lazarus']:.4f}, "
                f"mean_transition_exposure={row['mean_transition_exposure']:.4f}"
            )

    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_lines(
    tidy: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    family_order = [
        "settled_distant",
        "off_seam_reorganizing",
        "reorganization_heavy",
        "stable_seam_corridor",
    ]

    for fam in family_order:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale_order")
        if len(sub) == 0:
            continue
        ax.plot(sub["scale"], sub[value_col], marker="o", linewidth=2, label=fam)

    ax.set_title(title)
    ax.set_xlabel("scale")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_family_shares(tidy: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    family_order = [
        "settled_distant",
        "off_seam_reorganizing",
        "reorganization_heavy",
        "stable_seam_corridor",
    ]

    for fam in family_order:
        sub = tidy[tidy["path_family"] == fam].sort_values("scale_order")
        if len(sub) == 0:
            continue
        ax.plot(sub["scale"], sub["share_paths"], marker="o", linewidth=2, label=fam)

    ax.set_title("OBS-019: family share across probe scale")
    ax.set_xlabel("scale")
    ax.set_ylabel("share of sampled probe paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OBS-019 cross-scale convergence summary artifacts.")
    parser.add_argument("--root", default="outputs/scales")
    parser.add_argument("--scales", default="10,100,1000,10000,100000")
    parser.add_argument("--outdir", default="outputs/obs019_scale_convergence")
    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        scales=tuple(x.strip() for x in args.scales.split(",") if x.strip()),
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tidy = build_tidy_summary(cfg)
    winners = build_scale_winners(tidy)

    tidy.to_csv(outdir / "obs019_scale_convergence_tidy.csv", index=False)
    winners.to_csv(outdir / "obs019_scale_convergence_winners.csv", index=False)
    write_summary_text(tidy, winners, outdir / "obs019_scale_convergence_summary.txt")

    plot_family_shares(
        tidy,
        outdir / "obs019_family_share_by_scale.png",
    )
    plot_lines(
        tidy,
        "mean_lazarus",
        "mean Lazarus",
        "OBS-019: mean Lazarus by family across probe scale",
        outdir / "obs019_mean_lazarus_by_scale.png",
    )
    plot_lines(
        tidy,
        "mean_transition_exposure",
        "mean response strength",
        "OBS-019: mean response by family across probe scale",
        outdir / "obs019_mean_response_by_scale.png",
    )

    print(outdir / "obs019_scale_convergence_tidy.csv")
    print(outdir / "obs019_scale_convergence_winners.csv")
    print(outdir / "obs019_scale_convergence_summary.txt")
    print(outdir / "obs019_family_share_by_scale.png")
    print(outdir / "obs019_mean_lazarus_by_scale.png")
    print(outdir / "obs019_mean_response_by_scale.png")


if __name__ == "__main__":
    main()
