#!/usr/bin/env python3
"""
OBS-033 — Reversible core vs directed escape in the seam proto-groupoid.

Purpose
-------
Partition the seam proto-groupoid into two sectors:

1. reversible core
   - quasi-inverse shuttle-like generator pairs
   - seam-internal local symmetry remnants

2. directed escape sector
   - one-way release / persistence / transition arrows
   - irreversible seam departure and off-seam continuation

This study turns the asymmetry result of OBS-032 into an explicit structural
decomposition.

Inputs
------
outputs/obs032_quasi_inverses_and_asymmetry/quasi_inverse_pairs.csv
outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_generators.csv
outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_compositions.csv

Outputs
-------
outputs/obs033_reversible_core_vs_directed_escape/
  reversible_core_pairs.csv
  directed_escape_pairs.csv
  reversible_vs_directed_family_summary.csv
  obs033_reversible_core_vs_directed_escape_summary.txt
  obs033_reversible_core_vs_directed_escape_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    quasi_pairs_csv: str = (
        "outputs/obs032_quasi_inverses_and_asymmetry/quasi_inverse_pairs.csv"
    )
    generators_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_generators.csv"
    )
    compositions_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_compositions.csv"
    )
    outdir: str = "outputs/obs033_reversible_core_vs_directed_escape"
    quasi_inverse_threshold: float = 0.40
    min_pair_count: float = 1.0
    top_k_pairs: int = 10


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_cols = {
        "route_class",
        "pair_key",
        "forward_label",
        "reverse_label",
        "generator_completed",
        "source_object",
        "target_object",
        "generator_1",
        "generator_2",
        "composition_typed",
        "composition",
        "object_a",
        "object_b",
        "sector",
    }
    for col in df.columns:
        if col not in text_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df

def classify_pair_sector(row: pd.Series, cfg: Config) -> str:
    sym = pd.to_numeric(row.get("symmetry_score"), errors="coerce")
    rev = pd.to_numeric(row.get("reverse_exists"), errors="coerce")
    total = pd.to_numeric(row.get("total_count"), errors="coerce")

    if pd.isna(total) or total < cfg.min_pair_count:
        return "other"

    if rev == 1 and pd.notna(sym) and sym >= cfg.quasi_inverse_threshold:
        return "reversible_core"

    return "directed_escape"


def build_pair_sectors(pair_table: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = pair_table.copy()
    work["sector"] = work.apply(lambda row: classify_pair_sector(row, cfg), axis=1)

    rev = work[work["sector"] == "reversible_core"].copy().reset_index(drop=True)
    direct = work[work["sector"] == "directed_escape"].copy().reset_index(drop=True)
    return rev, direct


def build_family_summary(rev: pd.DataFrame, direct: pd.DataFrame, generators: pd.DataFrame, compositions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    gsub = generators[generators["route_class"] != "overall"].copy()
    csub = compositions[compositions["route_class"] != "overall"].copy()

    for cls in CLASS_ORDER:
        r = rev[rev["route_class"] == cls].copy()
        d = direct[direct["route_class"] == cls].copy()
        g = gsub[gsub["route_class"] == cls].copy()
        c = csub[csub["route_class"] == cls].copy()

        total_pair_weight = float(r["total_count"].sum() + d["total_count"].sum())
        reversible_pair_share = (float(r["total_count"].sum()) / total_pair_weight) if total_pair_weight > 0 else 0.0
        directed_pair_share = (float(d["total_count"].sum()) / total_pair_weight) if total_pair_weight > 0 else 0.0

        # Generator-sector proxy by generator names that appeared in reversible pairs
        rev_generators = set(r["generator_completed"].dropna().astype(str))
        direct_generators = set(d["generator_completed"].dropna().astype(str))

        g_total = float(g["n_instances"].sum()) if len(g) else 0.0
        rev_gen_share = float(g[g["generator_completed"].astype(str).isin(rev_generators)]["n_instances"].sum()) / g_total if g_total > 0 else 0.0
        direct_gen_share = float(g[g["generator_completed"].astype(str).isin(direct_generators)]["n_instances"].sum()) / g_total if g_total > 0 else 0.0

        # Composition-sector proxy
        rev_comp_share = 0.0
        direct_comp_share = 0.0
        if len(c):
            rev_mask = c["generator_1"].astype(str).isin(rev_generators) & c["generator_2"].astype(str).isin(rev_generators)
            direct_mask = c["generator_1"].astype(str).isin(direct_generators) | c["generator_2"].astype(str).isin(direct_generators)
            comp_total = float(c["n_compositions"].sum())
            if comp_total > 0:
                rev_comp_share = float(c.loc[rev_mask, "n_compositions"].sum()) / comp_total
                direct_comp_share = float(c.loc[direct_mask, "n_compositions"].sum()) / comp_total

        top_rev = r.sort_values("total_count", ascending=False).head(3).copy()
        top_dir = d.sort_values("total_count", ascending=False).head(3).copy()

        for frame in (top_rev, top_dir):
            if "object_a" in frame.columns:
                frame["object_a"] = frame["object_a"].fillna("?").astype(str)
            if "object_b" in frame.columns:
                frame["object_b"] = frame["object_b"].fillna("?").astype(str)

        rows.append(
            {
                "route_class": cls,
                "n_reversible_pairs": int(len(r)),
                "n_directed_pairs": int(len(d)),
                "reversible_pair_share": reversible_pair_share,
                "directed_pair_share": directed_pair_share,
                "reversible_generator_share": rev_gen_share,
                "directed_generator_share": direct_gen_share,
                "reversible_composition_share": rev_comp_share,
                "directed_composition_share": direct_comp_share,
                "top_reversible_pair_1": (
                    f"{top_rev.iloc[0]['generator_completed']}:{top_rev.iloc[0]['object_a']}↔{top_rev.iloc[0]['object_b']}"
                    if len(top_rev) > 0 else np.nan
                ),
                "top_reversible_pair_2": (
                    f"{top_rev.iloc[1]['generator_completed']}:{top_rev.iloc[1]['object_a']}↔{top_rev.iloc[1]['object_b']}"
                    if len(top_rev) > 1 else np.nan
                ),
                "top_directed_pair_1": (
                    f"{top_dir.iloc[0]['generator_completed']}:{top_dir.iloc[0]['object_a']}↔{top_dir.iloc[0]['object_b']}"
                    if len(top_dir) > 0 else np.nan
                ),
                "top_directed_pair_2": (
                    f"{top_dir.iloc[1]['generator_completed']}:{top_dir.iloc[1]['object_a']}↔{top_dir.iloc[1]['object_b']}"
                    if len(top_dir) > 1 else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(rev: pd.DataFrame, direct: pd.DataFrame, fam: pd.DataFrame) -> str:
    total_rev = int(len(rev))
    total_dir = int(len(direct))
    total_weight = float(rev["total_count"].sum() + direct["total_count"].sum())
    rev_weight_share = float(rev["total_count"].sum() / total_weight) if total_weight > 0 else 0.0
    dir_weight_share = float(direct["total_count"].sum() / total_weight) if total_weight > 0 else 0.0

    lines = [
        "=== OBS-033 Reversible Core vs Directed Escape Summary ===",
        "",
        f"n_reversible_pairs = {total_rev}",
        f"n_directed_pairs = {total_dir}",
        f"reversible_pair_weight_share = {rev_weight_share:.4f}",
        f"directed_pair_weight_share = {dir_weight_share:.4f}",
        "",
        "Interpretive guide",
        "- reversible core consists of quasi-inverse pairs with reverse partners and moderate symmetry",
        "- directed escape sector consists of one-way or strongly biased pairs",
        "- generator/composition shares estimate how much of each family lives in each sector",
        "",
        "Family sector summaries",
    ]

    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_reversible_pairs          = {int(row['n_reversible_pairs'])}",
                f"  n_directed_pairs            = {int(row['n_directed_pairs'])}",
                f"  reversible_pair_share       = {float(row['reversible_pair_share']):.4f}",
                f"  directed_pair_share         = {float(row['directed_pair_share']):.4f}",
                f"  reversible_generator_share  = {float(row['reversible_generator_share']):.4f}",
                f"  directed_generator_share    = {float(row['directed_generator_share']):.4f}",
                f"  reversible_composition_share= {float(row['reversible_composition_share']):.4f}",
                f"  directed_composition_share  = {float(row['directed_composition_share']):.4f}",
                f"  top_reversible_pair_1       = {row['top_reversible_pair_1']}",
                f"  top_reversible_pair_2       = {row['top_reversible_pair_2']}",
                f"  top_directed_pair_1         = {row['top_directed_pair_1']}",
                f"  top_directed_pair_2         = {row['top_directed_pair_2']}",
                "",
            ]
        )

    if total_rev:
        top_rev = rev.sort_values("total_count", ascending=False).head(8)
        lines.append("Top reversible-core pairs")
        for _, row in top_rev.iterrows():
            lines.append(
                f"  {row['route_class']} | {row['generator_completed']}:{row['object_a']}↔{row['object_b']} "
                f"| count={row['total_count']:.0f}, symmetry={row['symmetry_score']:.4f}"
            )
    if total_dir:
        top_dir = direct.sort_values("total_count", ascending=False).head(8)
        lines.append("")
        lines.append("Top directed-escape pairs")
        for _, row in top_dir.iterrows():
            lines.append(
                f"  {row['route_class']} | {row['generator_completed']}:{row['object_a']}↔{row['object_b']} "
                f"| count={row['total_count']:.0f}, forward={row['forward_count']:.0f}, reverse={row['reverse_count']:.0f}"
            )

    return "\n".join(lines)


def render_figure(rev: pd.DataFrame, direct: pd.DataFrame, fam: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_pairs = fig.add_subplot(gs[0, 0])
    ax_gens = fig.add_subplot(gs[0, 1])
    ax_comps = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(fam))
    width = 0.34

    ax_pairs.bar(x - width / 2, fam["reversible_pair_share"], width, label="reversible core")
    ax_pairs.bar(x + width / 2, fam["directed_pair_share"], width, label="directed escape")
    ax_pairs.set_xticks(x)
    ax_pairs.set_xticklabels(fam["route_class"], rotation=12)
    ax_pairs.set_title("Pair-sector weight share", fontsize=14, pad=8)
    ax_pairs.grid(alpha=0.15, axis="y")
    ax_pairs.legend()

    ax_gens.bar(x - width / 2, fam["reversible_generator_share"], width, label="reversible gen share")
    ax_gens.bar(x + width / 2, fam["directed_generator_share"], width, label="directed gen share")
    ax_gens.set_xticks(x)
    ax_gens.set_xticklabels(fam["route_class"], rotation=12)
    ax_gens.set_title("Generator-sector share", fontsize=14, pad=8)
    ax_gens.grid(alpha=0.15, axis="y")
    ax_gens.legend()

    ax_comps.bar(x - width / 2, fam["reversible_composition_share"], width, label="reversible comp share")
    ax_comps.bar(x + width / 2, fam["directed_composition_share"], width, label="directed comp share")
    ax_comps.set_xticks(x)
    ax_comps.set_xticklabels(fam["route_class"], rotation=12)
    ax_comps.set_title("Composition-sector share", fontsize=14, pad=8)
    ax_comps.grid(alpha=0.15, axis="y")
    ax_comps.legend()

    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"rev1: {row['top_reversible_pair_1']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"rev2: {row['top_reversible_pair_2']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"dir1: {row['top_directed_pair_1']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"dir2: {row['top_directed_pair_2']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Representative sector pairs", fontsize=14, pad=8)

    total_weight = float(rev["total_count"].sum() + direct["total_count"].sum())
    rev_share = float(rev["total_count"].sum() / total_weight) if total_weight > 0 else 0.0
    dir_share = float(direct["total_count"].sum() / total_weight) if total_weight > 0 else 0.0

    ax_diag.axis("off")
    text = (
        "OBS-033 diagnostics\n\n"
        f"reversible weight share:\n{rev_share:.3f}\n\n"
        f"directed weight share:\n{dir_share:.3f}\n\n"
        f"n reversible pairs:\n{len(rev)}\n\n"
        f"n directed pairs:\n{len(direct)}\n\n"
        "Interpretation:\n"
        "the seam algebra splits\n"
        "into a small reversible\n"
        "core and a dominant\n"
        "directed escape sector."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-033 reversible core vs directed escape", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition the seam proto-groupoid into reversible core vs directed escape.")
    parser.add_argument("--quasi-pairs-csv", default=Config.quasi_pairs_csv)
    parser.add_argument("--generators-csv", default=Config.generators_csv)
    parser.add_argument("--compositions-csv", default=Config.compositions_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--quasi-inverse-threshold", type=float, default=Config.quasi_inverse_threshold)
    parser.add_argument("--min-pair-count", type=float, default=Config.min_pair_count)
    parser.add_argument("--top-k-pairs", type=int, default=Config.top_k_pairs)
    args = parser.parse_args()

    cfg = Config(
        quasi_pairs_csv=args.quasi_pairs_csv,
        generators_csv=args.generators_csv,
        compositions_csv=args.compositions_csv,
        outdir=args.outdir,
        quasi_inverse_threshold=args.quasi_inverse_threshold,
        min_pair_count=args.min_pair_count,
        top_k_pairs=args.top_k_pairs,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pair_table = load_csv(cfg.quasi_pairs_csv)
    generators = load_csv(cfg.generators_csv)
    compositions = load_csv(cfg.compositions_csv)

    rev, direct = build_pair_sectors(pair_table, cfg)
    fam = build_family_summary(rev, direct, generators, compositions)

    rev_csv = outdir / "reversible_core_pairs.csv"
    dir_csv = outdir / "directed_escape_pairs.csv"
    fam_csv = outdir / "reversible_vs_directed_family_summary.csv"
    txt_path = outdir / "obs033_reversible_core_vs_directed_escape_summary.txt"
    png_path = outdir / "obs033_reversible_core_vs_directed_escape_figure.png"

    rev.to_csv(rev_csv, index=False)
    direct.to_csv(dir_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(rev, direct, fam), encoding="utf-8")
    render_figure(rev, direct, fam, png_path, cfg)

    print(rev_csv)
    print(dir_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
