#!/usr/bin/env python3
"""
OBS-030c — Motif generator algebra.

Bridge the seam observatory from empirical motif structure toward a compact
algebraic object.

This study compresses OBS-030b motif behavior into:

1. a reduced state alphabet
2. a reduced generator alphabet
3. family-specific generator usage
4. generator composition patterns

Reduced state alphabet
----------------------
R = relational_flank
A = anisotropy_flank
L = seam_resident_low
O = off_seam
P = post_exit
C = shared_core

Generator alphabet
------------------
g_rel_release      : relational release toward post-exit
g_aniso_release    : anisotropy-side release toward post-exit
g_flank_shuttle    : seam-internal R/A oscillation
g_low_residency    : low seam residency / L persistence
g_off_persist      : off-seam persistence
g_post_persist     : post-exit persistence
g_reentry          : off/post re-entry into seam structure
g_core_behavior    : shared-core retention/release/decay
g_other            : uncategorized motif

Inputs
------
outputs/obs030b_seam_transition_motifs/seam_transition_motifs.csv

Outputs
-------
outputs/obs030c_motif_generator_algebra/
  motif_generator_assignments.csv
  motif_generator_counts.csv
  motif_generator_compositions.csv
  motif_generator_family_summary.csv
  obs030c_motif_generator_algebra_summary.txt
  obs030c_motif_generator_algebra_figure.png
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
    motifs_csv: str = "outputs/obs030b_seam_transition_motifs/seam_transition_motifs.csv"
    outdir: str = "outputs/obs030c_motif_generator_algebra"
    top_k_generators: int = 8
    top_k_compositions: int = 10


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
    "g_other",
]


def reduce_state(s: str) -> str:
    mapping = {
        "relational_flank": "R",
        "anisotropy_flank": "A",
        "seam_resident_low": "L",
        "off_seam": "O",
        "post_exit": "P",
        "shared_core": "C",
        "mixed_seam": "L",
    }
    return mapping.get(str(s), "L")


def assign_generator(motif_class: str, a: str, b: str, c: str) -> str:
    ra, rb, rc = reduce_state(a), reduce_state(b), reduce_state(c)

    if motif_class == "relational_release_motif":
        return "g_rel_release"
    if motif_class == "anisotropy_release_motif":
        return "g_aniso_release"

    if motif_class == "flank_shuttle_motif":
        return "g_flank_shuttle"

    if motif_class == "low_residency_motif":
        return "g_low_residency"

    if motif_class == "off_seam_persistence_motif":
        return "g_off_persist"
    if motif_class == "post_exit_persistence_motif":
        return "g_post_persist"

    if motif_class == "reentry_motif":
        return "g_reentry"

    if motif_class in {"core_retention_motif", "core_release_motif", "core_to_low_motif"}:
        return "g_core_behavior"

    # fallback structural rules
    if ra == "R" and rc == "P":
        return "g_rel_release"
    if ra == "A" and rc == "P":
        return "g_aniso_release"
    if {ra, rb, rc}.issubset({"R", "A"}):
        return "g_flank_shuttle"
    if ra == rb == rc == "L":
        return "g_low_residency"
    if ra == rb == rc == "O":
        return "g_off_persist"
    if ra == rb == rc == "P":
        return "g_post_persist"
    if ra in {"O", "P"} and rb in {"R", "A", "L", "C"}:
        return "g_reentry"
    if "C" in {ra, rb, rc}:
        return "g_core_behavior"

    return "g_other"


def load_motifs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {"path_id", "path_family", "route_class", "motif", "motif_class", "state_a", "state_b", "state_c"}:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def build_generator_assignments(motifs: pd.DataFrame) -> pd.DataFrame:
    out = motifs.copy()
    out["state_a_red"] = out["state_a"].map(reduce_state)
    out["state_b_red"] = out["state_b"].map(reduce_state)
    out["state_c_red"] = out["state_c"].map(reduce_state)
    out["generator"] = [
        assign_generator(mc, a, b, c)
        for mc, a, b, c in zip(out["motif_class"], out["state_a"], out["state_b"], out["state_c"])
    ]
    out["generator_word"] = out["state_a_red"] + "~" + out["state_b_red"] + "~" + out["state_c_red"]
    return out


def build_generator_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = (
        assignments.groupby(["route_class", "generator"], as_index=False)
        .agg(
            n_motifs=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = counts.groupby("route_class")["n_motifs"].transform("sum")
    counts["generator_share"] = counts["n_motifs"] / total.clip(lower=1)

    order = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    counts["g_order"] = counts["generator"].map(lambda x: order.get(x, 999))
    counts = counts.sort_values(["route_class", "g_order"]).drop(columns=["g_order"]).reset_index(drop=True)
    return counts


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
                    "generator_1": a["generator"],
                    "generator_2": b["generator"],
                    "composition": f"{a['generator']} ; {b['generator']}",
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


def build_family_summary(counts: pd.DataFrame, compositions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        sub = counts[counts["route_class"] == cls].copy()
        comp = compositions[compositions["route_class"] == cls].copy()

        row = {
            "route_class": cls,
            "n_generators": int(sub["n_motifs"].sum()) if len(sub) else 0,
        }

        for g in GENERATOR_ORDER:
            hit = sub[sub["generator"] == g]
            row[f"{g}_share"] = float(hit["generator_share"].sum()) if len(hit) else 0.0
            row[f"{g}_count"] = int(hit["n_motifs"].sum()) if len(hit) else 0

        topg = sub.sort_values("n_motifs", ascending=False).head(5) if len(sub) else pd.DataFrame()
        for i in range(5):
            row[f"top_generator_{i+1}"] = topg.iloc[i]["generator"] if len(topg) > i else np.nan

        topc = comp.sort_values("n_compositions", ascending=False).head(5) if len(comp) else pd.DataFrame()
        for i in range(5):
            row[f"top_composition_{i+1}"] = topc.iloc[i]["composition"] if len(topc) > i else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(family_summary: pd.DataFrame) -> str:
    lines = [
        "=== OBS-030c Motif Generator Algebra Summary ===",
        "",
        "Reduced state alphabet",
        "  R = relational_flank",
        "  A = anisotropy_flank",
        "  L = seam_resident_low",
        "  O = off_seam",
        "  P = post_exit",
        "  C = shared_core",
        "",
        "Generator alphabet",
        "  g_rel_release",
        "  g_aniso_release",
        "  g_flank_shuttle",
        "  g_low_residency",
        "  g_off_persist",
        "  g_post_persist",
        "  g_reentry",
        "  g_core_behavior",
        "  g_other",
        "",
    ]

    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_generators          = {int(row['n_generators'])}",
                f"  g_rel_release_share   = {float(row['g_rel_release_share']):.4f}",
                f"  g_aniso_release_share = {float(row['g_aniso_release_share']):.4f}",
                f"  g_flank_shuttle_share = {float(row['g_flank_shuttle_share']):.4f}",
                f"  g_low_residency_share = {float(row['g_low_residency_share']):.4f}",
                f"  g_off_persist_share   = {float(row['g_off_persist_share']):.4f}",
                f"  g_post_persist_share  = {float(row['g_post_persist_share']):.4f}",
                f"  g_reentry_share       = {float(row['g_reentry_share']):.4f}",
                f"  g_core_behavior_share = {float(row['g_core_behavior_share']):.4f}",
                f"  top_generator_1       = {row['top_generator_1']}",
                f"  top_generator_2       = {row['top_generator_2']}",
                f"  top_generator_3       = {row['top_generator_3']}",
                f"  top_composition_1     = {row['top_composition_1']}",
                f"  top_composition_2     = {row['top_composition_2']}",
                f"  top_composition_3     = {row['top_composition_3']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- generators are compressed motif-types, not single steps",
            "- compositions reveal family-specific concatenation laws",
            "- this is the immediate precursor to a partial algebra / proto-groupoid framing",
        ]
    )
    return "\n".join(lines)


def render_figure(family_summary: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_release = fig.add_subplot(gs[0, 0])
    ax_persist = fig.add_subplot(gs[0, 1])
    ax_internal = fig.add_subplot(gs[0, 2])
    ax_reentry = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    colors = {
        "branch_exit": "#1f77b4",
        "stable_seam_corridor": "#2ca02c",
        "reorganization_heavy": "#d62728",
    }
    x = np.arange(len(family_summary))
    c = [colors[k] for k in family_summary["route_class"]]
    width = 0.36

    ax_release.bar(x - width / 2, family_summary["g_rel_release_share"], width, label="rel release")
    ax_release.bar(x + width / 2, family_summary["g_aniso_release_share"], width, label="aniso release")
    ax_release.set_xticks(x)
    ax_release.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_release.set_title("Release generators", fontsize=14, pad=8)
    ax_release.grid(alpha=0.15, axis="y")
    ax_release.legend()

    ax_persist.bar(x - width / 2, family_summary["g_off_persist_share"], width, label="off persist")
    ax_persist.bar(x + width / 2, family_summary["g_post_persist_share"], width, label="post persist")
    ax_persist.set_xticks(x)
    ax_persist.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_persist.set_title("Persistence generators", fontsize=14, pad=8)
    ax_persist.grid(alpha=0.15, axis="y")
    ax_persist.legend()

    ax_internal.bar(x - width / 2, family_summary["g_flank_shuttle_share"], width, label="flank shuttle")
    ax_internal.bar(x + width / 2, family_summary["g_low_residency_share"], width, label="low residency")
    ax_internal.set_xticks(x)
    ax_internal.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_internal.set_title("Seam-internal generators", fontsize=14, pad=8)
    ax_internal.grid(alpha=0.15, axis="y")
    ax_internal.legend()

    ax_reentry.bar(x - width / 2, family_summary["g_reentry_share"], width, label="reentry")
    ax_reentry.bar(x + width / 2, family_summary["g_core_behavior_share"], width, label="core behavior")
    ax_reentry.set_xticks(x)
    ax_reentry.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_reentry.set_title("Reentry / core generators", fontsize=14, pad=8)
    ax_reentry.grid(alpha=0.15, axis="y")
    ax_reentry.legend()

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
    ax_top.set_title("Top generators / compositions", fontsize=14, pad=8)

    ax_diag.axis("off")
    best_post = family_summary.sort_values("g_post_persist_share", ascending=False).iloc[0]
    best_shuttle = family_summary.sort_values("g_flank_shuttle_share", ascending=False).iloc[0]
    best_re = family_summary.sort_values("g_reentry_share", ascending=False).iloc[0]
    text = (
        "OBS-030c diagnostics\n\n"
        f"strongest post persistence:\n{best_post['route_class']} ({best_post['g_post_persist_share']:.3f})\n\n"
        f"strongest flank shuttle:\n{best_shuttle['route_class']} ({best_shuttle['g_flank_shuttle_share']:.3f})\n\n"
        f"strongest reentry:\n{best_re['route_class']} ({best_re['g_reentry_share']:.3f})\n\n"
        "Meaning:\n"
        "motif dynamics can now be\n"
        "compressed into a small\n"
        "generator set with family-\n"
        "specific composition laws."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-030c motif generator algebra", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress seam motifs into a generator algebra.")
    parser.add_argument("--motifs-csv", default=Config.motifs_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--top-k-generators", type=int, default=Config.top_k_generators)
    parser.add_argument("--top-k-compositions", type=int, default=Config.top_k_compositions)
    args = parser.parse_args()

    cfg = Config(
        motifs_csv=args.motifs_csv,
        outdir=args.outdir,
        top_k_generators=args.top_k_generators,
        top_k_compositions=args.top_k_compositions,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    motifs = load_motifs(cfg.motifs_csv)
    assignments = build_generator_assignments(motifs)
    counts = build_generator_counts(assignments)
    compositions = build_generator_compositions(assignments)
    family_summary = build_family_summary(counts, compositions)

    assign_csv = outdir / "motif_generator_assignments.csv"
    counts_csv = outdir / "motif_generator_counts.csv"
    comp_csv = outdir / "motif_generator_compositions.csv"
    fam_csv = outdir / "motif_generator_family_summary.csv"
    txt_path = outdir / "obs030c_motif_generator_algebra_summary.txt"
    png_path = outdir / "obs030c_motif_generator_algebra_figure.png"

    assignments.to_csv(assign_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    compositions.to_csv(comp_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(family_summary), encoding="utf-8")
    render_figure(family_summary, png_path)

    print(assign_csv)
    print(counts_csv)
    print(comp_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
