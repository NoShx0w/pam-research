#!/usr/bin/env python3
"""
OBS-030e — Complete generator basis.

Final cleanup pass on the motif generator algebra before any proto-groupoid step.

Goal
----
Resolve the remaining dominant `g_other` mass in OBS-030d by promoting a small
set of seam-edge / bridge generators:

- g_low_flank_shuttle
- g_off_to_edge
- g_edge_to_post

This should reduce the residual bucket enough that:
- family signatures are mostly expressible in named generators
- top compositions are mostly between named generators
- the algebra is stable enough for a proto-groupoid framing

Inputs
------
outputs/obs030d_resolve_other_generators/resolved_generator_assignments.csv

Outputs
-------
outputs/obs030e_complete_generator_basis/
  completed_generator_assignments.csv
  completed_generator_counts.csv
  completed_generator_compositions.csv
  completed_generator_family_summary.csv
  obs030e_complete_generator_basis_summary.txt
  obs030e_complete_generator_basis_figure.png
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
    assignments_csv: str = "outputs/obs030d_resolve_other_generators/resolved_generator_assignments.csv"
    outdir: str = "outputs/obs030e_complete_generator_basis"
    top_k_words: int = 12


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

GENERATOR_ORDER = [
    "g_rel_release",
    "g_aniso_release",
    "g_flank_shuttle",
    "g_low_flank_shuttle",
    "g_low_residency",
    "g_off_persist",
    "g_post_persist",
    "g_reentry",
    "g_core_behavior",
    "g_off_to_post",
    "g_flank_to_off",
    "g_low_to_off",
    "g_off_to_edge",
    "g_edge_to_post",
    "g_post_to_off",
    "g_other",
]


def load_assignments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {
            "path_id", "path_family", "route_class", "motif", "motif_class",
            "state_a", "state_b", "state_c", "state_a_red", "state_b_red",
            "state_c_red", "generator", "generator_word", "generator_resolved"
        }:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def finalize_generator(word: str, current: str) -> str:
    if current != "g_other":
        return current

    # New seam-edge / bridge generators
    if word in {"L~L~R", "L~R~L", "R~L~L", "L~L~A", "L~A~L", "A~L~L"}:
        return "g_low_flank_shuttle"

    if word in {"O~O~R", "O~O~A", "O~O~L"}:
        return "g_off_to_edge"

    if word in {"O~P~P", "L~L~P", "L~P~P", "R~P~P", "A~P~P"}:
        return "g_edge_to_post"

    return "g_other"


def build_word_table(assignments: pd.DataFrame) -> pd.DataFrame:
    gen_col = "generator_resolved" if "generator_resolved" in assignments.columns else "generator"
    other = assignments[assignments[gen_col] == "g_other"].copy()
    if len(other) == 0:
        return pd.DataFrame(columns=["route_class", "generator_word", "n_words", "n_paths", "word_share"])

    out = (
        other.groupby(["route_class", "generator_word"], as_index=False)
        .agg(
            n_words=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = out.groupby("route_class")["n_words"].transform("sum")
    out["word_share"] = out["n_words"] / total.clip(lower=1)
    return out.sort_values(["route_class", "n_words"], ascending=[True, False]).reset_index(drop=True)


def build_completed_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    source_col = "generator_resolved" if "generator_resolved" in assignments.columns else "generator"
    out = assignments.copy()
    out["generator_completed"] = [
        finalize_generator(word, gen)
        for word, gen in zip(out["generator_word"], out[source_col])
    ]
    return out


def build_generator_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(["route_class", "generator_completed"], as_index=False)
        .agg(
            n_motifs=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = counts.groupby("route_class")["n_motifs"].transform("sum")
    counts["generator_share"] = counts["n_motifs"] / total.clip(lower=1)

    order = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    counts["g_order"] = counts["generator_completed"].map(lambda x: order.get(x, 999))
    return counts.sort_values(["route_class", "g_order"]).drop(columns=["g_order"]).reset_index(drop=True)


def build_generator_compositions(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path_id, grp in assignments.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]
            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "generator_1": a["generator_completed"],
                    "generator_2": b["generator_completed"],
                    "composition": f"{a['generator_completed']} ; {b['generator_completed']}",
                }
            )

    comp = pd.DataFrame(rows)
    if len(comp) == 0:
        return pd.DataFrame(columns=["route_class", "generator_1", "generator_2", "composition", "n_compositions", "n_paths", "composition_share"])

    out = (
        comp.groupby(["route_class", "generator_1", "generator_2", "composition"], as_index=False)
        .agg(
            n_compositions=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = out.groupby("route_class")["n_compositions"].transform("sum")
    out["composition_share"] = out["n_compositions"] / total.clip(lower=1)
    return out.sort_values(["route_class", "n_compositions"], ascending=[True, False]).reset_index(drop=True)


def build_family_summary(counts: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = counts[counts["route_class"] == cls].copy()
        comp = comps[comps["route_class"] == cls].copy()

        row = {"route_class": cls, "n_generators": int(sub["n_motifs"].sum()) if len(sub) else 0}
        for g in GENERATOR_ORDER:
            hit = sub[sub["generator_completed"] == g]
            row[f"{g}_share"] = float(hit["generator_share"].sum()) if len(hit) else 0.0

        topg = sub.sort_values("n_motifs", ascending=False).head(5) if len(sub) else pd.DataFrame()
        for i in range(5):
            row[f"top_generator_{i+1}"] = topg.iloc[i]["generator_completed"] if len(topg) > i else np.nan

        topc = comp.sort_values("n_compositions", ascending=False).head(5) if len(comp) else pd.DataFrame()
        for i in range(5):
            row[f"top_composition_{i+1}"] = topc.iloc[i]["composition"] if len(topc) > i else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(word_table: pd.DataFrame, family_summary: pd.DataFrame) -> str:
    lines = [
        "=== OBS-030e Complete Generator Basis Summary ===",
        "",
        "Final completion policy",
        "- L~L~R / L~R~L / R~L~L (+ A variants) -> g_low_flank_shuttle",
        "- O~O~R / O~O~A / O~O~L -> g_off_to_edge",
        "- O~P~P / L~L~P / *~P~P edge patterns -> g_edge_to_post",
        "",
        "Top residual words before final completion",
    ]

    for cls in CLASS_ORDER:
        sub = word_table[word_table["route_class"] == cls].head(5)
        lines.append(f"  {cls}")
        if len(sub) == 0:
            lines.append("    none")
        else:
            for _, row in sub.iterrows():
                lines.append(
                    f"    {row['generator_word']}: n={int(row['n_words'])}, share={float(row['word_share']):.4f}"
                )

    lines.extend(["", "Completed family signatures"])
    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  g_rel_release_share        = {float(row['g_rel_release_share']):.4f}",
                f"  g_flank_shuttle_share      = {float(row['g_flank_shuttle_share']):.4f}",
                f"  g_low_flank_shuttle_share  = {float(row['g_low_flank_shuttle_share']):.4f}",
                f"  g_off_persist_share        = {float(row['g_off_persist_share']):.4f}",
                f"  g_post_persist_share       = {float(row['g_post_persist_share']):.4f}",
                f"  g_reentry_share            = {float(row['g_reentry_share']):.4f}",
                f"  g_off_to_post_share        = {float(row['g_off_to_post_share']):.4f}",
                f"  g_flank_to_off_share       = {float(row['g_flank_to_off_share']):.4f}",
                f"  g_low_to_off_share         = {float(row['g_low_to_off_share']):.4f}",
                f"  g_off_to_edge_share        = {float(row['g_off_to_edge_share']):.4f}",
                f"  g_edge_to_post_share       = {float(row['g_edge_to_post_share']):.4f}",
                f"  g_post_to_off_share        = {float(row['g_post_to_off_share']):.4f}",
                f"  g_other_share              = {float(row['g_other_share']):.4f}",
                f"  top_generator_1            = {row['top_generator_1']}",
                f"  top_generator_2            = {row['top_generator_2']}",
                f"  top_generator_3            = {row['top_generator_3']}",
                f"  top_composition_1          = {row['top_composition_1']}",
                f"  top_composition_2          = {row['top_composition_2']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- this step succeeds if g_other is no longer dominant",
            "- family signatures should now be carried by named generators",
            "- once residual mass is small, the generator algebra is stable enough for proto-groupoid framing",
        ]
    )
    return "\n".join(lines)


def render_figure(family_summary: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_complete = fig.add_subplot(gs[0, 0])
    ax_persist = fig.add_subplot(gs[0, 1])
    ax_edge = fig.add_subplot(gs[0, 2])
    ax_other = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(family_summary))
    width = 0.24

    ax_complete.bar(x - width, family_summary["g_low_flank_shuttle_share"], width, label="low↔flank")
    ax_complete.bar(x, family_summary["g_off_to_edge_share"], width, label="off→edge")
    ax_complete.bar(x + width, family_summary["g_edge_to_post_share"], width, label="edge→post")
    ax_complete.set_xticks(x)
    ax_complete.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_complete.set_title("Final completed generators", fontsize=14, pad=8)
    ax_complete.grid(alpha=0.15, axis="y")
    ax_complete.legend()

    ax_persist.bar(x - width / 2, family_summary["g_off_persist_share"], width, label="off persist")
    ax_persist.bar(x + width / 2, family_summary["g_post_persist_share"], width, label="post persist")
    ax_persist.set_xticks(x)
    ax_persist.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_persist.set_title("Persistence generators", fontsize=14, pad=8)
    ax_persist.grid(alpha=0.15, axis="y")
    ax_persist.legend()

    ax_edge.bar(x - width / 2, family_summary["g_flank_shuttle_share"], width, label="flank shuttle")
    ax_edge.bar(x + width / 2, family_summary["g_reentry_share"], width, label="reentry")
    ax_edge.set_xticks(x)
    ax_edge.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_edge.set_title("Seam-edge generators", fontsize=14, pad=8)
    ax_edge.grid(alpha=0.15, axis="y")
    ax_edge.legend()

    ax_other.bar(x, family_summary["g_other_share"])
    ax_other.set_xticks(x)
    ax_other.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_other.set_title("Residual g_other share", fontsize=14, pad=8)
    ax_other.grid(alpha=0.15, axis="y")

    ax_top.axis("off")
    y = 0.95
    for _, row in family_summary.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        for i in range(1, 4):
            ax_top.text(0.04, y, f"g{i}: {row[f'top_generator_{i}']}", fontsize=10, family="monospace")
            y -= 0.045
        for i in range(1, 3):
            ax_top.text(0.04, y, f"c{i}: {row[f'top_composition_{i}']}", fontsize=10, family="monospace")
            y -= 0.045
        y -= 0.04
    ax_top.set_title("Top completed generators / compositions", fontsize=14, pad=8)

    ax_diag.axis("off")
    worst = family_summary.sort_values("g_other_share", ascending=False).iloc[0]
    best_post = family_summary.sort_values("g_post_persist_share", ascending=False).iloc[0]
    best_shuttle = family_summary.sort_values("g_flank_shuttle_share", ascending=False).iloc[0]
    text = (
        "OBS-030e diagnostics\n\n"
        f"largest residual g_other:\n{worst['route_class']} ({worst['g_other_share']:.3f})\n\n"
        f"strongest post persistence:\n{best_post['route_class']} ({best_post['g_post_persist_share']:.3f})\n\n"
        f"strongest flank shuttle:\n{best_shuttle['route_class']} ({best_shuttle['g_flank_shuttle_share']:.3f})\n\n"
        "Success criterion:\n"
        "g_other is no longer\n"
        "dominant, so the named\n"
        "generator set can support\n"
        "a proto-groupoid step."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-030e complete generator basis", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the seam motif generator basis.")
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--top-k-words", type=int, default=Config.top_k_words)
    args = parser.parse_args()

    cfg = Config(
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        top_k_words=args.top_k_words,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assignments = load_assignments(cfg.assignments_csv)
    word_table = build_word_table(assignments)
    completed = build_completed_assignments(assignments)
    counts = build_generator_counts(completed)
    comps = build_generator_compositions(completed)
    family_summary = build_family_summary(counts, comps)

    assign_csv = outdir / "completed_generator_assignments.csv"
    counts_csv = outdir / "completed_generator_counts.csv"
    comps_csv = outdir / "completed_generator_compositions.csv"
    fam_csv = outdir / "completed_generator_family_summary.csv"
    txt_path = outdir / "obs030e_complete_generator_basis_summary.txt"
    png_path = outdir / "obs030e_complete_generator_basis_figure.png"

    completed.to_csv(assign_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    comps.to_csv(comps_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(word_table, family_summary), encoding="utf-8")
    render_figure(family_summary, png_path)

    print(assign_csv)
    print(counts_csv)
    print(comps_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
