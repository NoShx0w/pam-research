#!/usr/bin/env python3
"""
OBS-032 — Quasi-inverses and asymmetry in the seam proto-groupoid.

Purpose
-------
Measure which empirical generators behave approximately reversibly, and which
encode genuinely one-way seam dynamics.

This study sits directly on top of OBS-031.

Core questions
--------------
1. Which typed arrows have an observed reverse partner?
2. How asymmetric are forward vs reverse frequencies?
3. Do forward-then-reverse compositions exist as local identity surrogates?
4. Are corridor generators more reversible than branch-exit generators?
5. Are release generators structurally one-way?

Inputs
------
outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_generators.csv
outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_compositions.csv

Outputs
-------
outputs/obs032_quasi_inverses_and_asymmetry/
  quasi_inverse_pairs.csv
  quasi_inverse_family_summary.csv
  obs032_quasi_inverses_and_asymmetry_summary.txt
  obs032_quasi_inverses_and_asymmetry_figure.png
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
    generators_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_generators.csv"
    )
    compositions_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_compositions.csv"
    )
    outdir: str = "outputs/obs032_quasi_inverses_and_asymmetry"
    min_share: float = 0.0
    top_k_pairs: int = 12


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {
            "route_class",
            "generator_completed",
            "source_object",
            "target_object",
            "generator_1",
            "generator_2",
            "composition_typed",
            "composition",
        }:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def canonical_pair_key(gen: str, src: str, tgt: str) -> str:
    a = f"{gen}:{src}->{tgt}"
    b = f"{gen}:{tgt}->{src}"
    return " || ".join(sorted([a, b]))


def pair_orientation_label(gen: str, src: str, tgt: str) -> str:
    return f"{gen}:{src}->{tgt}"


def build_pair_table(generators: pd.DataFrame, compositions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    gen_sub = generators[generators["route_class"] != "overall"].copy()
    comp_sub = compositions[compositions["route_class"] != "overall"].copy()

    for cls in CLASS_ORDER:
        g = gen_sub[gen_sub["route_class"] == cls].copy()
        c = comp_sub[comp_sub["route_class"] == cls].copy()

        # Candidate reversible pairs are same generator name with reversed source/target
        grouped = g.groupby("generator_completed", sort=False)

        for gen_name, gg in grouped:
            seen = set()
            for _, row in gg.iterrows():
                src = str(row["source_object"])
                tgt = str(row["target_object"])
                key = (gen_name, frozenset([src, tgt]))
                if key in seen:
                    continue
                seen.add(key)

                fwd = gg[(gg["source_object"] == src) & (gg["target_object"] == tgt)]
                rev = gg[(gg["source_object"] == tgt) & (gg["target_object"] == src)]

                fwd_n = float(fwd["n_instances"].sum()) if len(fwd) else 0.0
                rev_n = float(rev["n_instances"].sum()) if len(rev) else 0.0
                total = fwd_n + rev_n

                # symmetric balance: 1 is perfect symmetry, 0 is one-way
                symmetry_score = 1.0 - abs(fwd_n - rev_n) / total if total > 0 else np.nan

                # existence of reverse partner
                reverse_exists = int(rev_n > 0)

                # identity-surrogate compositions:
                # gen(src->tgt) ; gen(tgt->src)
                comp_match_1 = c[
                    (c["generator_1"] == gen_name)
                    & (c["generator_2"] == gen_name)
                    & (c["source_1"] == src)
                    & (c["target_1"] == tgt)
                    & (c["source_2"] == tgt)
                    & (c["target_2"] == src)
                ]
                comp_match_2 = c[
                    (c["generator_1"] == gen_name)
                    & (c["generator_2"] == gen_name)
                    & (c["source_1"] == tgt)
                    & (c["target_1"] == src)
                    & (c["source_2"] == src)
                    & (c["target_2"] == tgt)
                ]
                id_surrogate_count = float(comp_match_1["n_compositions"].sum() + comp_match_2["n_compositions"].sum())
                id_surrogate_share = float(comp_match_1["composition_share"].sum() + comp_match_2["composition_share"].sum())

                rows.append(
                    {
                        "route_class": cls,
                        "generator_completed": gen_name,
                        "object_a": src,
                        "object_b": tgt,
                        "pair_key": canonical_pair_key(gen_name, src, tgt),
                        "forward_label": pair_orientation_label(gen_name, src, tgt),
                        "reverse_label": pair_orientation_label(gen_name, tgt, src),
                        "forward_count": fwd_n,
                        "reverse_count": rev_n,
                        "total_count": total,
                        "reverse_exists": reverse_exists,
                        "symmetry_score": symmetry_score,
                        "directional_bias": safe_div(fwd_n - rev_n, total),
                        "identity_surrogate_count": id_surrogate_count,
                        "identity_surrogate_share": id_surrogate_share,
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # remove degenerate self-pairs like O->O vs O->O
    out = out[out["object_a"] != out["object_b"]].copy()

    # deduplicate canonical pairs within each class
    out["pair_order"] = out["pair_key"]
    out = out.sort_values(["route_class", "total_count"], ascending=[True, False])
    out = out.drop_duplicates(subset=["route_class", "pair_key"]).drop(columns=["pair_order"]).reset_index(drop=True)
    return out


def build_family_summary(pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = pair_table[pair_table["route_class"] == cls].copy()

        if len(sub) == 0:
            rows.append(
                {
                    "route_class": cls,
                    "n_pairs": 0,
                    "mean_symmetry_score": np.nan,
                    "mean_reverse_exists": np.nan,
                    "mean_identity_surrogate_share": np.nan,
                    "n_quasi_inverse_pairs": 0,
                    "n_one_way_pairs": 0,
                    "top_pair_1": np.nan,
                    "top_pair_2": np.nan,
                    "top_pair_3": np.nan,
                }
            )
            continue

        quasi = sub[
            (sub["reverse_exists"] == 1)
            & (sub["symmetry_score"] >= 0.40)
        ]
        one_way = sub[sub["reverse_exists"] == 0]

        top = sub.sort_values("total_count", ascending=False).head(5)

        rows.append(
            {
                "route_class": cls,
                "n_pairs": int(len(sub)),
                "mean_symmetry_score": float(sub["symmetry_score"].mean()),
                "mean_reverse_exists": float(sub["reverse_exists"].mean()),
                "mean_identity_surrogate_share": float(sub["identity_surrogate_share"].mean()),
                "n_quasi_inverse_pairs": int(len(quasi)),
                "n_one_way_pairs": int(len(one_way)),
                "top_pair_1": (
                    f"{top.iloc[0]['generator_completed']}:{top.iloc[0]['object_a']}↔{top.iloc[0]['object_b']}"
                    if len(top) > 0 else np.nan
                ),
                "top_pair_2": (
                    f"{top.iloc[1]['generator_completed']}:{top.iloc[1]['object_a']}↔{top.iloc[1]['object_b']}"
                    if len(top) > 1 else np.nan
                ),
                "top_pair_3": (
                    f"{top.iloc[2]['generator_completed']}:{top.iloc[2]['object_a']}↔{top.iloc[2]['object_b']}"
                    if len(top) > 2 else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(pair_table: pd.DataFrame, family_summary: pd.DataFrame) -> str:
    overall_pairs = len(pair_table)
    quasi_total = int(
        ((pair_table["reverse_exists"] == 1) & (pair_table["symmetry_score"] >= 0.40)).sum()
    ) if len(pair_table) else 0
    one_way_total = int((pair_table["reverse_exists"] == 0).sum()) if len(pair_table) else 0

    lines = [
        "=== OBS-032 Quasi-Inverses and Asymmetry Summary ===",
        "",
        f"n_candidate_pairs = {overall_pairs}",
        f"n_quasi_inverse_pairs = {quasi_total}",
        f"n_one_way_pairs = {one_way_total}",
        "",
        "Interpretive guide",
        "- symmetry_score near 1 means forward/reverse are balanced",
        "- reverse_exists=0 means no observed reverse partner",
        "- identity_surrogate_share measures forward-then-reverse compositions as local identity proxies",
        "",
        "Family summaries",
    ]

    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_pairs                      = {int(row['n_pairs'])}",
                f"  mean_symmetry_score          = {float(row['mean_symmetry_score']):.4f}",
                f"  mean_reverse_exists          = {float(row['mean_reverse_exists']):.4f}",
                f"  mean_identity_surrogate_share= {float(row['mean_identity_surrogate_share']):.4f}",
                f"  n_quasi_inverse_pairs        = {int(row['n_quasi_inverse_pairs'])}",
                f"  n_one_way_pairs              = {int(row['n_one_way_pairs'])}",
                f"  top_pair_1                   = {row['top_pair_1']}",
                f"  top_pair_2                   = {row['top_pair_2']}",
                f"  top_pair_3                   = {row['top_pair_3']}",
                "",
            ]
        )

    # add top globally asymmetric pairs
    if len(pair_table):
        asym = pair_table.sort_values(["reverse_exists", "symmetry_score", "total_count"], ascending=[True, True, False]).head(8)
        lines.append("Top asymmetric / one-way candidates")
        for _, row in asym.iterrows():
            lines.append(
                f"  {row['route_class']} | {row['generator_completed']}:{row['object_a']}↔{row['object_b']} "
                f"| forward={row['forward_count']:.0f}, reverse={row['reverse_count']:.0f}, "
                f"symmetry={row['symmetry_score']:.4f}, id_share={row['identity_surrogate_share']:.4f}"
            )

    return "\n".join(lines)


def render_figure(pair_table: pd.DataFrame, family_summary: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_sym = fig.add_subplot(gs[0, 0])
    ax_rev = fig.add_subplot(gs[0, 1])
    ax_id = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(family_summary))
    width = 0.34

    ax_sym.bar(x, family_summary["mean_symmetry_score"])
    ax_sym.set_xticks(x)
    ax_sym.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_sym.set_title("Mean symmetry score", fontsize=14, pad=8)
    ax_sym.grid(alpha=0.15, axis="y")

    ax_rev.bar(x - width / 2, family_summary["mean_reverse_exists"], width, label="reverse exists")
    ax_rev.bar(x + width / 2, family_summary["n_quasi_inverse_pairs"] / family_summary["n_pairs"].replace(0, np.nan), width, label="quasi-inverse fraction")
    ax_rev.set_xticks(x)
    ax_rev.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_rev.set_title("Reverse partners / quasi-inverses", fontsize=14, pad=8)
    ax_rev.grid(alpha=0.15, axis="y")
    ax_rev.legend()

    ax_id.bar(x, family_summary["mean_identity_surrogate_share"])
    ax_id.set_xticks(x)
    ax_id.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_id.set_title("Identity-surrogate composition share", fontsize=14, pad=8)
    ax_id.grid(alpha=0.15, axis="y")

    # top pairs
    ax_top.axis("off")
    y = 0.95
    for cls in CLASS_ORDER:
        sub = pair_table[pair_table["route_class"] == cls].sort_values("total_count", ascending=False).head(4)
        ax_top.text(0.02, y, cls, fontsize=12, fontweight="bold")
        y -= 0.06
        for _, row in sub.iterrows():
            txt = (
                f"{row['generator_completed']}:{row['object_a']}↔{row['object_b']} | "
                f"f={row['forward_count']:.0f}, r={row['reverse_count']:.0f}, "
                f"sym={row['symmetry_score']:.2f}, id={row['identity_surrogate_share']:.3f}"
            )
            ax_top.text(0.04, y, txt, fontsize=9.5, family="monospace")
            y -= 0.045
        y -= 0.04
    ax_top.set_title("Top quasi-inverse / asymmetric pairs", fontsize=14, pad=8)

    overall_mean_sym = float(pair_table["symmetry_score"].mean()) if len(pair_table) else float("nan")
    overall_rev = float(pair_table["reverse_exists"].mean()) if len(pair_table) else float("nan")
    overall_id = float(pair_table["identity_surrogate_share"].mean()) if len(pair_table) else float("nan")

    ax_diag.axis("off")
    text = (
        "OBS-032 diagnostics\n\n"
        f"overall mean symmetry:\n{overall_mean_sym:.3f}\n\n"
        f"overall reverse-exists rate:\n{overall_rev:.3f}\n\n"
        f"overall identity-surrogate share:\n{overall_id:.3f}\n\n"
        "Interpretation:\n"
        "high symmetry suggests\n"
        "quasi-reversible local\n"
        "structure; low symmetry\n"
        "suggests one-way dynamics."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-032 quasi-inverses and asymmetry", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure quasi-inverses and asymmetry in the seam proto-groupoid.")
    parser.add_argument("--generators-csv", default=Config.generators_csv)
    parser.add_argument("--compositions-csv", default=Config.compositions_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-share", type=float, default=Config.min_share)
    parser.add_argument("--top-k-pairs", type=int, default=Config.top_k_pairs)
    args = parser.parse_args()

    cfg = Config(
        generators_csv=args.generators_csv,
        compositions_csv=args.compositions_csv,
        outdir=args.outdir,
        min_share=args.min_share,
        top_k_pairs=args.top_k_pairs,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    generators = load_csv(cfg.generators_csv)
    compositions = load_csv(cfg.compositions_csv)

    pair_table = build_pair_table(generators, compositions)
    family_summary = build_family_summary(pair_table)

    pair_csv = outdir / "quasi_inverse_pairs.csv"
    fam_csv = outdir / "quasi_inverse_family_summary.csv"
    txt_path = outdir / "obs032_quasi_inverses_and_asymmetry_summary.txt"
    png_path = outdir / "obs032_quasi_inverses_and_asymmetry_figure.png"

    pair_table.to_csv(pair_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(pair_table, family_summary), encoding="utf-8")
    render_figure(pair_table, family_summary, png_path, cfg)

    print(pair_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
