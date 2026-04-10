#!/usr/bin/env python3
"""
OBS-030 — Seam transition algebra.

Formalize the seam program as a typed local transition system.

Objects
-------
Local seam-state types assigned to route steps from the canonical seam bundle.

State types
-----------
- off_seam
- relational_flank
- anisotropy_flank
- shared_core
- mixed_seam
- seam_resident_low
- post_exit

Arrows
------
Observed step-to-step transitions between state types along route families.

This study asks:
1. Which typed transitions are most common?
2. Which transitions are family-specific?
3. Which two-step compositions form stereotyped motifs?
4. Is branch_exit associated with relational release motifs?
5. Is reorganization_heavy associated with anisotropy-side release motifs?

Inputs
------
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs030_seam_transition_algebra/
  seam_transition_steps.csv
  seam_transition_counts.csv
  seam_transition_compositions.csv
  seam_transition_family_summary.csv
  obs030_seam_transition_algebra_summary.txt
  obs030_seam_transition_algebra_figure.png
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
    seam_nodes_csv: str = "outputs/obs028c_canonical_seam_bundle/seam_nodes.csv"
    routes_csv: str = "outputs/obs022_scene_bundle/scene_routes.csv"
    outdir: str = "outputs/obs030_seam_transition_algebra"
    seam_threshold: float = 0.15
    post_exit_threshold: float = 0.50
    top_k_transitions: int = 12
    top_k_compositions: int = 10


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

STATE_ORDER = [
    "off_seam",
    "post_exit",
    "seam_resident_low",
    "mixed_seam",
    "relational_flank",
    "anisotropy_flank",
    "shared_core",
]


def safe_mean(s: pd.Series | np.ndarray) -> float:
    ss = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(ss.mean()) if ss.notna().any() else float("nan")


def classify_routes(routes: pd.DataFrame) -> pd.DataFrame:
    out = routes.copy()
    fam = out.get("path_family", pd.Series(index=out.index, dtype=object))
    is_branch = pd.to_numeric(out.get("is_branch_away", 0), errors="coerce").fillna(0).eq(1)
    is_rep = pd.to_numeric(out.get("is_representative", 0), errors="coerce").fillna(0).eq(1)

    out["route_class"] = np.select(
        [
            is_branch,
            is_rep & fam.eq("stable_seam_corridor"),
            is_rep & fam.eq("reorganization_heavy"),
        ],
        [
            "branch_exit",
            "stable_seam_corridor",
            "reorganization_heavy",
        ],
        default="other",
    )
    return out


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    seam_nodes = pd.read_csv(cfg.seam_nodes_csv)
    routes = pd.read_csv(cfg.routes_csv)

    for df in (seam_nodes, routes):
        for col in df.columns:
            if col not in {"path_id", "path_family", "route_class", "hotspot_class"}:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass

    routes = classify_routes(routes)

    keep_cols = [
        c for c in [
            "node_id",
            "r",
            "alpha",
            "mds1",
            "mds2",
            "distance_to_seam",
            "neighbor_direction_mismatch_mean",
            "sym_traceless_norm",
            "anisotropy_hotspot",
            "relational_hotspot",
            "shared_hotspot",
            "hotspot_class",
            "seam_band",
        ] if c in seam_nodes.columns
    ]

    seam_use = seam_nodes[keep_cols].drop_duplicates(subset=["node_id"]).copy()
    seam_use = seam_use.rename(columns={c: f"{c}_bundle" for c in seam_use.columns if c != "node_id"})

    routes = routes.merge(seam_use, on="node_id", how="left")

    for base_col in [
        "r",
        "alpha",
        "mds1",
        "mds2",
        "distance_to_seam",
        "neighbor_direction_mismatch_mean",
        "sym_traceless_norm",
        "anisotropy_hotspot",
        "relational_hotspot",
        "shared_hotspot",
        "hotspot_class",
        "seam_band",
    ]:
        bundle_col = f"{base_col}_bundle"
        if base_col not in routes.columns and bundle_col in routes.columns:
            routes[base_col] = routes[bundle_col]
        elif base_col in routes.columns and bundle_col in routes.columns:
            routes[base_col] = routes[base_col].where(routes[base_col].notna(), routes[bundle_col])

    drop_cols = [c for c in routes.columns if c.endswith("_bundle")]
    if drop_cols:
        routes = routes.drop(columns=drop_cols)

    return seam_nodes, routes


def assign_state_type(row: pd.Series, cfg: Config) -> str:
    d2s = pd.to_numeric(row.get("distance_to_seam"), errors="coerce")
    rel = pd.to_numeric(row.get("neighbor_direction_mismatch_mean"), errors="coerce")
    aniso = pd.to_numeric(row.get("sym_traceless_norm"), errors="coerce")
    shared = int(pd.to_numeric(row.get("shared_hotspot"), errors="coerce") == 1)
    rel_hot = int(pd.to_numeric(row.get("relational_hotspot"), errors="coerce") == 1)
    aniso_hot = int(pd.to_numeric(row.get("anisotropy_hotspot"), errors="coerce") == 1)

    if pd.isna(d2s):
        return "off_seam"

    if d2s > cfg.post_exit_threshold:
        return "post_exit"

    if d2s > cfg.seam_threshold:
        return "off_seam"

    # seam-band states
    if shared == 1:
        return "shared_core"
    if rel_hot == 1 and aniso_hot == 0:
        return "relational_flank"
    if aniso_hot == 1 and rel_hot == 0:
        return "anisotropy_flank"
    if rel_hot == 1 and aniso_hot == 1:
        return "mixed_seam"

    # seam-resident but not hotspot
    if np.isfinite(rel) or np.isfinite(aniso):
        return "seam_resident_low"

    return "mixed_seam"


def build_step_table(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = routes[routes["route_class"].isin(CLASS_ORDER)].copy()
    work = work.sort_values(["path_id", "step"]).reset_index(drop=True)

    work["state_type"] = work.apply(lambda row: assign_state_type(row, cfg), axis=1)

    rows = []
    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "path_family": a.get("path_family", np.nan),
                    "step": pd.to_numeric(a.get("step"), errors="coerce"),
                    "node_id": pd.to_numeric(a.get("node_id"), errors="coerce"),
                    "next_node_id": pd.to_numeric(b.get("node_id"), errors="coerce"),
                    "state_from": a["state_type"],
                    "state_to": b["state_type"],
                    "distance_from": pd.to_numeric(a.get("distance_to_seam"), errors="coerce"),
                    "distance_to": pd.to_numeric(b.get("distance_to_seam"), errors="coerce"),
                    "relational_from": pd.to_numeric(a.get("neighbor_direction_mismatch_mean"), errors="coerce"),
                    "anisotropy_from": pd.to_numeric(a.get("sym_traceless_norm"), errors="coerce"),
                    "dx": pd.to_numeric(b.get("mds1"), errors="coerce") - pd.to_numeric(a.get("mds1"), errors="coerce"),
                    "dy": pd.to_numeric(b.get("mds2"), errors="coerce") - pd.to_numeric(a.get("mds2"), errors="coerce"),
                }
            )

    return work, pd.DataFrame(rows)


def build_transition_counts(transition_steps: pd.DataFrame) -> pd.DataFrame:
    counts = (
        transition_steps.groupby(["route_class", "state_from", "state_to"], as_index=False)
        .agg(
            n_transitions=("path_id", "size"),
            n_paths=("path_id", "nunique"),
            mean_distance_from=("distance_from", "mean"),
            mean_distance_to=("distance_to", "mean"),
            mean_relational_from=("relational_from", "mean"),
            mean_anisotropy_from=("anisotropy_from", "mean"),
        )
    )

    total = counts.groupby("route_class")["n_transitions"].transform("sum")
    counts["transition_share"] = counts["n_transitions"] / total.clip(lower=1)

    state_rank = {s: i for i, s in enumerate(STATE_ORDER)}
    counts["from_order"] = counts["state_from"].map(lambda x: state_rank.get(x, 999))
    counts["to_order"] = counts["state_to"].map(lambda x: state_rank.get(x, 999))
    counts = counts.sort_values(
        ["route_class", "n_transitions", "from_order", "to_order"],
        ascending=[True, False, True, True],
    ).drop(columns=["from_order", "to_order"]).reset_index(drop=True)
    return counts


def build_transition_compositions(transition_steps: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path_id, grp in transition_steps.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]
            if a["state_to"] != b["state_from"]:
                continue

            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "composition": f"{a['state_from']} -> {a['state_to']} -> {b['state_to']}",
                    "state_a": a["state_from"],
                    "state_b": a["state_to"],
                    "state_c": b["state_to"],
                }
            )

    comp = pd.DataFrame(rows)
    if len(comp) == 0:
        return pd.DataFrame(columns=["route_class", "composition", "n_compositions", "n_paths", "composition_share"])

    out = (
        comp.groupby(["route_class", "composition"], as_index=False)
        .agg(
            n_compositions=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    total = out.groupby("route_class")["n_compositions"].transform("sum")
    out["composition_share"] = out["n_compositions"] / total.clip(lower=1)
    return out.sort_values(["route_class", "n_compositions"], ascending=[True, False]).reset_index(drop=True)


def build_family_summary(transition_counts: pd.DataFrame, compositions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    target_transitions = {
        "relational_release": ("relational_flank", "post_exit"),
        "anisotropy_release": ("anisotropy_flank", "post_exit"),
        "core_retention": ("shared_core", "shared_core"),
        "core_to_low": ("shared_core", "seam_resident_low"),
        "off_reentry": ("off_seam", "relational_flank"),
    }

    for cls in CLASS_ORDER:
        sub = transition_counts[transition_counts["route_class"] == cls].copy()
        comp = compositions[compositions["route_class"] == cls].copy()

        total_trans = int(sub["n_transitions"].sum()) if len(sub) else 0
        row = {
            "route_class": cls,
            "n_transitions": total_trans,
        }

        for label, (a, b) in target_transitions.items():
            hit = sub[(sub["state_from"] == a) & (sub["state_to"] == b)]
            row[f"{label}_count"] = int(hit["n_transitions"].sum()) if len(hit) else 0
            row[f"{label}_share"] = float(hit["transition_share"].sum()) if len(hit) else 0.0

        if len(sub):
            top = sub.sort_values("n_transitions", ascending=False).head(3)
            row["top_transition_1"] = f"{top.iloc[0]['state_from']} -> {top.iloc[0]['state_to']}" if len(top) >= 1 else np.nan
            row["top_transition_2"] = f"{top.iloc[1]['state_from']} -> {top.iloc[1]['state_to']}" if len(top) >= 2 else np.nan
            row["top_transition_3"] = f"{top.iloc[2]['state_from']} -> {top.iloc[2]['state_to']}" if len(top) >= 3 else np.nan
        else:
            row["top_transition_1"] = np.nan
            row["top_transition_2"] = np.nan
            row["top_transition_3"] = np.nan

        if len(comp):
            topc = comp.sort_values("n_compositions", ascending=False).head(3)
            row["top_composition_1"] = topc.iloc[0]["composition"] if len(topc) >= 1 else np.nan
            row["top_composition_2"] = topc.iloc[1]["composition"] if len(topc) >= 2 else np.nan
            row["top_composition_3"] = topc.iloc[2]["composition"] if len(topc) >= 3 else np.nan
        else:
            row["top_composition_1"] = np.nan
            row["top_composition_2"] = np.nan
            row["top_composition_3"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(family_summary: pd.DataFrame, transition_counts: pd.DataFrame, compositions: pd.DataFrame) -> str:
    lines = [
        "=== OBS-030 Seam Transition Algebra Summary ===",
        "",
        "State types",
        "  off_seam",
        "  post_exit",
        "  seam_resident_low",
        "  mixed_seam",
        "  relational_flank",
        "  anisotropy_flank",
        "  shared_core",
        "",
    ]

    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_transitions            = {int(row['n_transitions'])}",
                f"  relational_release_share = {float(row['relational_release_share']):.4f}",
                f"  anisotropy_release_share = {float(row['anisotropy_release_share']):.4f}",
                f"  core_retention_share     = {float(row['core_retention_share']):.4f}",
                f"  core_to_low_share        = {float(row['core_to_low_share']):.4f}",
                f"  off_reentry_share        = {float(row['off_reentry_share']):.4f}",
                f"  top_transition_1         = {row['top_transition_1']}",
                f"  top_transition_2         = {row['top_transition_2']}",
                f"  top_transition_3         = {row['top_transition_3']}",
                f"  top_composition_1        = {row['top_composition_1']}",
                f"  top_composition_2        = {row['top_composition_2']}",
                f"  top_composition_3        = {row['top_composition_3']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- relational_release tracks typed exit from the relational flank into post-exit state",
            "- anisotropy_release tracks typed exit from the anisotropy flank into post-exit state",
            "- core_retention tracks seam-core staying behavior",
            "- compositions expose stereotyped multistep motifs rather than single-step transitions only",
        ]
    )
    return "\n".join(lines)


def render_figure(transition_counts: pd.DataFrame, compositions: pd.DataFrame, family_summary: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1.0], height_ratios=[1.0, 1.0])

    ax_rel = fig.add_subplot(gs[0, 0])
    ax_aniso = fig.add_subplot(gs[0, 1])
    ax_core = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    colors = {
        "branch_exit": "#1f77b4",
        "stable_seam_corridor": "#2ca02c",
        "reorganization_heavy": "#d62728",
    }

    x = np.arange(len(family_summary))

    ax_rel.bar(x, family_summary["relational_release_share"], color=[colors[c] for c in family_summary["route_class"]])
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_rel.set_title("Relational release share", fontsize=14, pad=8)
    ax_rel.grid(alpha=0.15, axis="y")

    ax_aniso.bar(x, family_summary["anisotropy_release_share"], color=[colors[c] for c in family_summary["route_class"]])
    ax_aniso.set_xticks(x)
    ax_aniso.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_aniso.set_title("Anisotropy release share", fontsize=14, pad=8)
    ax_aniso.grid(alpha=0.15, axis="y")

    ax_core.bar(x - 0.18, family_summary["core_retention_share"], 0.36, label="core retention")
    ax_core.bar(x + 0.18, family_summary["core_to_low_share"], 0.36, label="core→low")
    ax_core.set_xticks(x)
    ax_core.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_core.set_title("Core behavior", fontsize=14, pad=8)
    ax_core.grid(alpha=0.15, axis="y")
    ax_core.legend()

    # top transitions table-like text
    ax_top.axis("off")
    y = 0.95
    for cls in CLASS_ORDER:
        sub = transition_counts[transition_counts["route_class"] == cls].head(cfg.top_k_transitions // 3 + 1)
        ax_top.text(0.02, y, cls, fontsize=12, fontweight="bold")
        y -= 0.06
        for _, row in sub.head(3).iterrows():
            txt = f"{row['state_from']} -> {row['state_to']} : {int(row['n_transitions'])} ({float(row['transition_share']):.3f})"
            ax_top.text(0.04, y, txt, fontsize=10, family="monospace")
            y -= 0.045
        y -= 0.035

    ax_top.set_title("Top typed transitions by family", fontsize=14, pad=8)

    # diagnostics
    ax_diag.axis("off")
    if len(family_summary):
        best_rel = family_summary.sort_values("relational_release_share", ascending=False).iloc[0]
        best_aniso = family_summary.sort_values("anisotropy_release_share", ascending=False).iloc[0]
        best_core = family_summary.sort_values("core_retention_share", ascending=False).iloc[0]
        text = (
            "OBS-030 diagnostics\n\n"
            f"strongest relational release:\n{best_rel['route_class']} ({best_rel['relational_release_share']:.3f})\n\n"
            f"strongest anisotropy release:\n{best_aniso['route_class']} ({best_aniso['anisotropy_release_share']:.3f})\n\n"
            f"strongest core retention:\n{best_core['route_class']} ({best_core['core_retention_share']:.3f})\n\n"
            "Goal:\n"
            "turn seam dynamics into\n"
            "typed local moves and\n"
            "composable motifs."
        )
    else:
        text = "No family summary rows available."

    ax_diag.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=10.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-030 seam transition algebra", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formalize seam dynamics as a typed transition algebra.")
    parser.add_argument("--seam-nodes-csv", default=Config.seam_nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--post-exit-threshold", type=float, default=Config.post_exit_threshold)
    parser.add_argument("--top-k-transitions", type=int, default=Config.top_k_transitions)
    parser.add_argument("--top-k-compositions", type=int, default=Config.top_k_compositions)
    args = parser.parse_args()

    cfg = Config(
        seam_nodes_csv=args.seam_nodes_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        post_exit_threshold=args.post_exit_threshold,
        top_k_transitions=args.top_k_transitions,
        top_k_compositions=args.top_k_compositions,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, routes = load_inputs(cfg)
    step_nodes, transition_steps = build_step_table(routes, cfg)
    transition_counts = build_transition_counts(transition_steps)
    compositions = build_transition_compositions(transition_steps)
    family_summary = build_family_summary(transition_counts, compositions)

    steps_csv = outdir / "seam_transition_steps.csv"
    counts_csv = outdir / "seam_transition_counts.csv"
    comp_csv = outdir / "seam_transition_compositions.csv"
    fam_csv = outdir / "seam_transition_family_summary.csv"
    txt_path = outdir / "obs030_seam_transition_algebra_summary.txt"
    png_path = outdir / "obs030_seam_transition_algebra_figure.png"

    transition_steps.to_csv(steps_csv, index=False)
    transition_counts.to_csv(counts_csv, index=False)
    compositions.to_csv(comp_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(family_summary, transition_counts, compositions), encoding="utf-8")
    render_figure(transition_counts, compositions, family_summary, png_path, cfg)

    print(steps_csv)
    print(counts_csv)
    print(comp_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
