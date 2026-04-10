#!/usr/bin/env python3
"""
OBS-030b — Seam transition motifs.

Refine OBS-030 by promoting short-horizon transition motifs to first-class
objects, instead of relying only on one-step typed transitions.

Why
---
OBS-030 showed that the transition-algebra direction is correct, but that
single-step release arrows are too strict. In practice, seam release often
appears as a short motif such as:

    relational_flank -> off_seam -> post_exit
    anisotropy_flank -> off_seam -> post_exit

rather than a single immediate jump into post_exit.

This study therefore classifies 3-state motifs and groups them into
interpretable motif families.

Inputs
------
outputs/obs030_seam_transition_algebra/seam_transition_steps.csv

Outputs
-------
outputs/obs030b_seam_transition_motifs/
  seam_transition_motifs.csv
  seam_motif_counts.csv
  seam_motif_family_summary.csv
  obs030b_seam_transition_motifs_summary.txt
  obs030b_seam_transition_motifs_figure.png
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
    transition_steps_csv: str = "outputs/obs030_seam_transition_algebra/seam_transition_steps.csv"
    outdir: str = "outputs/obs030b_seam_transition_motifs"
    top_k_motifs: int = 12


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_mean(s: pd.Series | np.ndarray) -> float:
    ss = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(ss.mean()) if ss.notna().any() else float("nan")


def classify_motif(a: str, b: str, c: str) -> str:
    # release motifs
    if a == "relational_flank" and b in {"off_seam", "post_exit"} and c == "post_exit":
        return "relational_release_motif"
    if a == "anisotropy_flank" and b in {"off_seam", "post_exit"} and c == "post_exit":
        return "anisotropy_release_motif"
    if a == "shared_core" and b in {"off_seam", "post_exit", "seam_resident_low"} and c in {"off_seam", "post_exit"}:
        return "core_release_motif"

    # seam-core residency / low-stress residency
    if a == "shared_core" and b == "shared_core" and c == "shared_core":
        return "core_retention_motif"
    if a == "seam_resident_low" and b == "seam_resident_low" and c == "seam_resident_low":
        return "low_residency_motif"

    # flank oscillation / shuttling
    if {a, b, c}.issubset({"relational_flank", "anisotropy_flank"}):
        return "flank_shuttle_motif"

    # off-seam persistence
    if a == "off_seam" and b == "off_seam" and c == "off_seam":
        return "off_seam_persistence_motif"
    if a == "post_exit" and b == "post_exit" and c == "post_exit":
        return "post_exit_persistence_motif"

    # re-entry motifs
    if a == "off_seam" and b in {"relational_flank", "anisotropy_flank", "shared_core"}:
        return "reentry_motif"
    if a == "post_exit" and b in {"relational_flank", "anisotropy_flank", "shared_core"}:
        return "reentry_motif"

    # seam to low / low to flank
    if a == "shared_core" and b == "seam_resident_low":
        return "core_to_low_motif"
    if a == "seam_resident_low" and c in {"relational_flank", "anisotropy_flank"}:
        return "low_to_flank_motif"

    return "other_motif"


def build_motifs(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path_id, grp in steps.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            state_a = str(a["state_from"])
            state_b = str(a["state_to"])
            state_c = str(b["state_to"]) if a["state_to"] == b["state_from"] else None
            if state_c is None:
                continue

            motif = f"{state_a} -> {state_b} -> {state_c}"
            motif_class = classify_motif(state_a, state_b, state_c)

            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "step": pd.to_numeric(a.get("step"), errors="coerce"),
                    "motif": motif,
                    "motif_class": motif_class,
                    "state_a": state_a,
                    "state_b": state_b,
                    "state_c": state_c,
                    "distance_a": pd.to_numeric(a.get("distance_from"), errors="coerce"),
                    "distance_b": pd.to_numeric(a.get("distance_to"), errors="coerce"),
                    "relational_a": pd.to_numeric(a.get("relational_from"), errors="coerce"),
                    "anisotropy_a": pd.to_numeric(a.get("anisotropy_from"), errors="coerce"),
                }
            )

    return pd.DataFrame(rows)


def build_motif_counts(motifs: pd.DataFrame) -> pd.DataFrame:
    if len(motifs) == 0:
        return pd.DataFrame(
            columns=[
                "route_class", "motif_class", "motif", "n_motifs",
                "n_paths", "motif_share", "mean_distance_a",
                "mean_relational_a", "mean_anisotropy_a"
            ]
        )

    out = (
        motifs.groupby(["route_class", "motif_class", "motif"], as_index=False)
        .agg(
            n_motifs=("path_id", "size"),
            n_paths=("path_id", "nunique"),
            mean_distance_a=("distance_a", "mean"),
            mean_relational_a=("relational_a", "mean"),
            mean_anisotropy_a=("anisotropy_a", "mean"),
        )
    )
    total = out.groupby("route_class")["n_motifs"].transform("sum")
    out["motif_share"] = out["n_motifs"] / total.clip(lower=1)
    return out.sort_values(["route_class", "n_motifs"], ascending=[True, False]).reset_index(drop=True)


def build_family_summary(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    target_classes = [
        "relational_release_motif",
        "anisotropy_release_motif",
        "core_release_motif",
        "core_retention_motif",
        "low_residency_motif",
        "flank_shuttle_motif",
        "off_seam_persistence_motif",
        "post_exit_persistence_motif",
        "reentry_motif",
        "core_to_low_motif",
    ]

    for cls in CLASS_ORDER:
        sub = counts[counts["route_class"] == cls].copy()

        row = {
            "route_class": cls,
            "n_motifs": int(sub["n_motifs"].sum()) if len(sub) else 0,
        }

        for mc in target_classes:
            hit = sub[sub["motif_class"] == mc]
            row[f"{mc}_count"] = int(hit["n_motifs"].sum()) if len(hit) else 0
            row[f"{mc}_share"] = float(hit["motif_share"].sum()) if len(hit) else 0.0

        top = sub.sort_values("n_motifs", ascending=False).head(5) if len(sub) else pd.DataFrame()
        for i in range(5):
            row[f"top_motif_{i+1}"] = top.iloc[i]["motif"] if len(top) > i else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(family_summary: pd.DataFrame) -> str:
    lines = [
        "=== OBS-030b Seam Transition Motifs Summary ===",
        "",
        "Motif classes",
        "  relational_release_motif",
        "  anisotropy_release_motif",
        "  core_release_motif",
        "  core_retention_motif",
        "  low_residency_motif",
        "  flank_shuttle_motif",
        "  off_seam_persistence_motif",
        "  post_exit_persistence_motif",
        "  reentry_motif",
        "  core_to_low_motif",
        "",
    ]

    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_motifs                         = {int(row['n_motifs'])}",
                f"  relational_release_motif_share   = {float(row['relational_release_motif_share']):.4f}",
                f"  anisotropy_release_motif_share   = {float(row['anisotropy_release_motif_share']):.4f}",
                f"  core_release_motif_share         = {float(row['core_release_motif_share']):.4f}",
                f"  core_retention_motif_share       = {float(row['core_retention_motif_share']):.4f}",
                f"  low_residency_motif_share        = {float(row['low_residency_motif_share']):.4f}",
                f"  flank_shuttle_motif_share        = {float(row['flank_shuttle_motif_share']):.4f}",
                f"  off_seam_persistence_motif_share = {float(row['off_seam_persistence_motif_share']):.4f}",
                f"  post_exit_persistence_motif_share= {float(row['post_exit_persistence_motif_share']):.4f}",
                f"  reentry_motif_share              = {float(row['reentry_motif_share']):.4f}",
                f"  core_to_low_motif_share          = {float(row['core_to_low_motif_share']):.4f}",
                f"  top_motif_1                      = {row['top_motif_1']}",
                f"  top_motif_2                      = {row['top_motif_2']}",
                f"  top_motif_3                      = {row['top_motif_3']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- release motifs are short-horizon typed departures, not single-step jumps only",
            "- flank_shuttle motifs capture seam-internal oscillation between relational and anisotropy sides",
            "- off_seam/post_exit persistence motifs capture staying gone once released",
            "- reentry motifs capture return from off-seam states back into seam structure",
        ]
    )
    return "\n".join(lines)


def render_figure(family_summary: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_rel = fig.add_subplot(gs[0, 0])
    ax_aniso = fig.add_subplot(gs[0, 1])
    ax_core = fig.add_subplot(gs[0, 2])
    ax_res = fig.add_subplot(gs[1, 0])
    ax_re = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    colors = {
        "branch_exit": "#1f77b4",
        "stable_seam_corridor": "#2ca02c",
        "reorganization_heavy": "#d62728",
    }

    x = np.arange(len(family_summary))
    c = [colors[k] for k in family_summary["route_class"]]

    ax_rel.bar(x, family_summary["relational_release_motif_share"], color=c)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_rel.set_title("Relational release motifs", fontsize=14, pad=8)
    ax_rel.grid(alpha=0.15, axis="y")

    ax_aniso.bar(x, family_summary["anisotropy_release_motif_share"], color=c)
    ax_aniso.set_xticks(x)
    ax_aniso.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_aniso.set_title("Anisotropy release motifs", fontsize=14, pad=8)
    ax_aniso.grid(alpha=0.15, axis="y")

    width = 0.36
    ax_core.bar(x - width / 2, family_summary["core_retention_motif_share"], width, label="core retention")
    ax_core.bar(x + width / 2, family_summary["core_release_motif_share"], width, label="core release")
    ax_core.set_xticks(x)
    ax_core.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_core.set_title("Core motifs", fontsize=14, pad=8)
    ax_core.grid(alpha=0.15, axis="y")
    ax_core.legend()

    ax_res.bar(x - width / 2, family_summary["off_seam_persistence_motif_share"], width, label="off-seam persist")
    ax_res.bar(x + width / 2, family_summary["post_exit_persistence_motif_share"], width, label="post-exit persist")
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_res.set_title("Release persistence motifs", fontsize=14, pad=8)
    ax_res.grid(alpha=0.15, axis="y")
    ax_res.legend()

    ax_re.bar(x - width / 2, family_summary["reentry_motif_share"], width, label="re-entry")
    ax_re.bar(x + width / 2, family_summary["flank_shuttle_motif_share"], width, label="flank shuttle")
    ax_re.set_xticks(x)
    ax_re.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_re.set_title("Re-entry / seam-internal motifs", fontsize=14, pad=8)
    ax_re.grid(alpha=0.15, axis="y")
    ax_re.legend()

    ax_diag.axis("off")
    best_rel = family_summary.sort_values("relational_release_motif_share", ascending=False).iloc[0]
    best_aniso = family_summary.sort_values("anisotropy_release_motif_share", ascending=False).iloc[0]
    best_re = family_summary.sort_values("reentry_motif_share", ascending=False).iloc[0]
    text = (
        "OBS-030b diagnostics\n\n"
        f"strongest relational release:\n{best_rel['route_class']} ({best_rel['relational_release_motif_share']:.3f})\n\n"
        f"strongest anisotropy release:\n{best_aniso['route_class']} ({best_aniso['anisotropy_release_motif_share']:.3f})\n\n"
        f"strongest re-entry:\n{best_re['route_class']} ({best_re['reentry_motif_share']:.3f})\n\n"
        "Goal:\n"
        "upgrade one-step arrows\n"
        "into short-horizon typed\n"
        "motifs that match the\n"
        "observed seam dynamics."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-030b seam transition motifs", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote seam transitions to short-horizon typed motifs.")
    parser.add_argument("--transition-steps-csv", default=Config.transition_steps_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--top-k-motifs", type=int, default=Config.top_k_motifs)
    args = parser.parse_args()

    cfg = Config(
        transition_steps_csv=args.transition_steps_csv,
        outdir=args.outdir,
        top_k_motifs=args.top_k_motifs,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(cfg.transition_steps_csv)
    motifs = build_motifs(steps)
    counts = build_motif_counts(motifs)
    family_summary = build_family_summary(counts)

    motifs_csv = outdir / "seam_transition_motifs.csv"
    counts_csv = outdir / "seam_motif_counts.csv"
    fam_csv = outdir / "seam_motif_family_summary.csv"
    txt_path = outdir / "obs030b_seam_transition_motifs_summary.txt"
    png_path = outdir / "obs030b_seam_transition_motifs_figure.png"

    motifs.to_csv(motifs_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(family_summary), encoding="utf-8")
    render_figure(family_summary, png_path, cfg)

    print(motifs_csv)
    print(counts_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
