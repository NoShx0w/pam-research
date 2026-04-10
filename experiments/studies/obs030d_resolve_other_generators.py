#!/usr/bin/env python3
"""
OBS-030d — Resolve residual motif generators.

Purpose
-------
Inspect the motifs currently assigned to g_other in OBS-030c and promote the
dominant recurring reduced words into explicit generators.

This is the final cleanup step before any proto-groupoid framing.

Inputs
------
outputs/obs030c_motif_generator_algebra/motif_generator_assignments.csv

Outputs
-------
outputs/obs030d_resolve_other_generators/
  resolved_generator_assignments.csv
  resolved_generator_counts.csv
  resolved_generator_compositions.csv
  resolved_generator_family_summary.csv
  obs030d_resolve_other_generators_summary.txt
  obs030d_resolve_other_generators_figure.png
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
    assignments_csv: str = "outputs/obs030c_motif_generator_algebra/motif_generator_assignments.csv"
    outdir: str = "outputs/obs030d_resolve_other_generators"
    min_word_count: int = 3
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
    "g_low_residency",
    "g_off_persist",
    "g_post_persist",
    "g_reentry",
    "g_core_behavior",
    "g_off_to_post",
    "g_flank_to_off",
    "g_low_to_off",
    "g_off_to_flank_edge",
    "g_post_to_off",
    "g_other",
]


def load_assignments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {
            "path_id", "path_family", "route_class", "motif", "motif_class",
            "state_a", "state_b", "state_c", "state_a_red", "state_b_red",
            "state_c_red", "generator", "generator_word"
        }:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def resolve_other(word: str, current: str) -> str:
    if current != "g_other":
        return current

    # explicit residual patterns
    if word == "O~O~P":
        return "g_off_to_post"
    if word == "R~A~O" or word == "A~R~O" or word == "A~A~O" or word == "R~R~O":
        return "g_flank_to_off"
    if word == "L~L~O" or word == "L~O~O":
        return "g_low_to_off"
    if word == "O~R~R" or word == "O~A~A" or word == "O~R~A" or word == "O~A~R":
        return "g_off_to_flank_edge"
    if word == "P~O~O" or word == "P~O~P":
        return "g_post_to_off"

    return "g_other"


def build_word_table(assignments: pd.DataFrame) -> pd.DataFrame:
    other = assignments[assignments["generator"] == "g_other"].copy()
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


def build_resolved_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    out = assignments.copy()
    out["generator_resolved"] = [
        resolve_other(word, gen)
        for word, gen in zip(out["generator_word"], out["generator"])
    ]
    return out


def build_generator_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(["route_class", "generator_resolved"], as_index=False)
        .agg(
            n_motifs=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = counts.groupby("route_class")["n_motifs"].transform("sum")
    counts["generator_share"] = counts["n_motifs"] / total.clip(lower=1)

    order = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    counts["g_order"] = counts["generator_resolved"].map(lambda x: order.get(x, 999))
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
                    "generator_1": a["generator_resolved"],
                    "generator_2": b["generator_resolved"],
                    "composition": f"{a['generator_resolved']} ; {b['generator_resolved']}",
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
            hit = sub[sub["generator_resolved"] == g]
            row[f"{g}_share"] = float(hit["generator_share"].sum()) if len(hit) else 0.0

        topg = sub.sort_values("n_motifs", ascending=False).head(5) if len(sub) else pd.DataFrame()
        for i in range(5):
            row[f"top_generator_{i+1}"] = topg.iloc[i]["generator_resolved"] if len(topg) > i else np.nan

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
        "=== OBS-030d Resolve Other Generators Summary ===",
        "",
        "Residual resolution policy",
        "- O~O~P -> g_off_to_post",
        "- flank~flank/off patterns -> g_flank_to_off",
        "- L~L~O or L~O~O -> g_low_to_off",
        "- O~flank patterns -> g_off_to_flank_edge",
        "- P~O~* patterns -> g_post_to_off",
        "",
        "Top residual words before resolution",
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

    lines.extend(["", "Resolved family signatures"])
    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  g_rel_release_share     = {float(row['g_rel_release_share']):.4f}",
                f"  g_flank_shuttle_share   = {float(row['g_flank_shuttle_share']):.4f}",
                f"  g_off_persist_share     = {float(row['g_off_persist_share']):.4f}",
                f"  g_post_persist_share    = {float(row['g_post_persist_share']):.4f}",
                f"  g_reentry_share         = {float(row['g_reentry_share']):.4f}",
                f"  g_off_to_post_share     = {float(row['g_off_to_post_share']):.4f}",
                f"  g_flank_to_off_share    = {float(row['g_flank_to_off_share']):.4f}",
                f"  g_low_to_off_share      = {float(row['g_low_to_off_share']):.4f}",
                f"  g_off_to_flank_edge_share = {float(row['g_off_to_flank_edge_share']):.4f}",
                f"  g_post_to_off_share     = {float(row['g_post_to_off_share']):.4f}",
                f"  g_other_share           = {float(row['g_other_share']):.4f}",
                f"  top_generator_1         = {row['top_generator_1']}",
                f"  top_generator_2         = {row['top_generator_2']}",
                f"  top_generator_3         = {row['top_generator_3']}",
                f"  top_composition_1       = {row['top_composition_1']}",
                f"  top_composition_2       = {row['top_composition_2']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- this step is successful if g_other is no longer dominant",
            "- named generators should capture the leading family signatures and compositions",
            "- once residual mass is small, the generator algebra is stable enough for proto-groupoid framing",
        ]
    )
    return "\n".join(lines)


def render_figure(family_summary: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_new = fig.add_subplot(gs[0, 0])
    ax_persist = fig.add_subplot(gs[0, 1])
    ax_internal = fig.add_subplot(gs[0, 2])
    ax_other = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(family_summary))
    width = 0.24

    ax_new.bar(x - width, family_summary["g_off_to_post_share"], width, label="off→post")
    ax_new.bar(x, family_summary["g_flank_to_off_share"], width, label="flank→off")
    ax_new.bar(x + width, family_summary["g_low_to_off_share"], width, label="low→off")
    ax_new.set_xticks(x)
    ax_new.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_new.set_title("Resolved residual generators", fontsize=14, pad=8)
    ax_new.grid(alpha=0.15, axis="y")
    ax_new.legend()

    ax_persist.bar(x - width / 2, family_summary["g_off_persist_share"], width, label="off persist")
    ax_persist.bar(x + width / 2, family_summary["g_post_persist_share"], width, label="post persist")
    ax_persist.set_xticks(x)
    ax_persist.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_persist.set_title("Persistence generators", fontsize=14, pad=8)
    ax_persist.grid(alpha=0.15, axis="y")
    ax_persist.legend()

    ax_internal.bar(x - width / 2, family_summary["g_flank_shuttle_share"], width, label="flank shuttle")
    ax_internal.bar(x + width / 2, family_summary["g_reentry_share"], width, label="reentry")
    ax_internal.set_xticks(x)
    ax_internal.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_internal.set_title("Seam-edge generators", fontsize=14, pad=8)
    ax_internal.grid(alpha=0.15, axis="y")
    ax_internal.legend()

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
    ax_top.set_title("Top resolved generators / compositions", fontsize=14, pad=8)

    ax_diag.axis("off")
    worst = family_summary.sort_values("g_other_share", ascending=False).iloc[0]
    best_post = family_summary.sort_values("g_post_persist_share", ascending=False).iloc[0]
    best_shuttle = family_summary.sort_values("g_flank_shuttle_share", ascending=False).iloc[0]
    text = (
        "OBS-030d diagnostics\n\n"
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

    fig.suptitle("PAM Observatory — OBS-030d resolve other generators", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve residual motif generators into named generators.")
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-word-count", type=int, default=Config.min_word_count)
    parser.add_argument("--top-k-words", type=int, default=Config.top_k_words)
    args = parser.parse_args()

    cfg = Config(
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        min_word_count=args.min_word_count,
        top_k_words=args.top_k_words,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assignments = load_assignments(cfg.assignments_csv)
    word_table = build_word_table(assignments)
    resolved = build_resolved_assignments(assignments)
    counts = build_generator_counts(resolved)
    comps = build_generator_compositions(resolved)
    family_summary = build_family_summary(counts, comps)

    resolved_csv = outdir / "resolved_generator_assignments.csv"
    counts_csv = outdir / "resolved_generator_counts.csv"
    comps_csv = outdir / "resolved_generator_compositions.csv"
    fam_csv = outdir / "resolved_generator_family_summary.csv"
    txt_path = outdir / "obs030d_resolve_other_generators_summary.txt"
    png_path = outdir / "obs030d_resolve_other_generators_figure.png"

    resolved.to_csv(resolved_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    comps.to_csv(comps_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(word_table, family_summary), encoding="utf-8")
    render_figure(family_summary, png_path)

    print(resolved_csv)
    print(counts_csv)
    print(comps_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
