#!/usr/bin/env python3
"""
OBS-031 — Proto-groupoid of seam dynamics.

Purpose
-------
Lift the cleaned generator algebra from OBS-030e into an explicit empirical
proto-groupoid / partial composition system.

This is intentionally disciplined:
- not a formal theorem
- not a claim of full invertibility
- not a maximal abstract construction

Instead, this study builds a compact algebraic object from the observed seam
dynamics:

1. objects = reduced seam states
2. arrows = named empirical generators
3. source/target typing for each generator
4. observed partial compositions between generators
5. family-specific proto-groupoid signatures / subalgebras

Inputs
------
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv
outputs/obs030e_complete_generator_basis/completed_generator_compositions.csv

Outputs
-------
outputs/obs031_proto_groupoid_of_seam_dynamics/
  proto_groupoid_generators.csv
  proto_groupoid_compositions.csv
  proto_groupoid_family_summary.csv
  proto_groupoid_objects.csv
  obs031_proto_groupoid_of_seam_dynamics_summary.txt
  obs031_proto_groupoid_of_seam_dynamics_figure.png
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
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    compositions_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_compositions.csv"
    )
    outdir: str = "outputs/obs031_proto_groupoid_of_seam_dynamics"
    min_generator_share: float = 0.02
    min_composition_share: float = 0.02
    top_k_compositions: int = 12


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

OBJECT_ORDER = ["R", "A", "L", "O", "P", "C"]

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


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in {
            "path_id",
            "path_family",
            "route_class",
            "motif",
            "motif_class",
            "state_a",
            "state_b",
            "state_c",
            "state_a_red",
            "state_b_red",
            "state_c_red",
            "generator",
            "generator_word",
            "generator_resolved",
            "generator_completed",
            "generator_1",
            "generator_2",
            "composition",
        }:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def source_target_from_row(row: pd.Series) -> tuple[str, str]:
    src = str(row["state_a_red"])
    tgt = str(row["state_c_red"])
    return src, tgt


def build_objects(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obj in OBJECT_ORDER:
        n_as_source = int((assignments["state_a_red"] == obj).sum())
        n_as_middle = int((assignments["state_b_red"] == obj).sum())
        n_as_target = int((assignments["state_c_red"] == obj).sum())
        rows.append(
            {
                "object": obj,
                "n_as_source": n_as_source,
                "n_as_middle": n_as_middle,
                "n_as_target": n_as_target,
            }
        )
    return pd.DataFrame(rows)


def build_generators(assignments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = assignments.copy()

    st = work.apply(source_target_from_row, axis=1)
    work["source_object"] = [x[0] for x in st]
    work["target_object"] = [x[1] for x in st]

    out = (
        work.groupby(["route_class", "generator_completed", "source_object", "target_object"], as_index=False)
        .agg(
            n_instances=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )

    total = out.groupby("route_class")["n_instances"].transform("sum")
    out["generator_share"] = out["n_instances"] / total.clip(lower=1)

    # overall aggregation too
    overall = (
        work.groupby(["generator_completed", "source_object", "target_object"], as_index=False)
        .agg(
            n_instances=("path_id", "size"),
            n_paths=("path_id", "nunique"),
        )
    )
    overall["route_class"] = "overall"
    total_overall = float(overall["n_instances"].sum())
    overall["generator_share"] = overall["n_instances"] / max(total_overall, 1.0)

    out = pd.concat([out, overall], ignore_index=True)

    order_map = {g: i for i, g in enumerate(GENERATOR_ORDER)}
    obj_map = {o: i for i, o in enumerate(OBJECT_ORDER)}
    out["g_order"] = out["generator_completed"].map(lambda x: order_map.get(x, 999))
    out["s_order"] = out["source_object"].map(lambda x: obj_map.get(x, 999))
    out["t_order"] = out["target_object"].map(lambda x: obj_map.get(x, 999))
    out = out.sort_values(
        ["route_class", "g_order", "s_order", "t_order"]
    ).drop(columns=["g_order", "s_order", "t_order"]).reset_index(drop=True)
    return out


def build_compositions(compositions: pd.DataFrame, assignments: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Need typing for generators
    gen_type = (
        assignments.groupby("generator_completed", as_index=False)
        .agg(
            source_object=("state_a_red", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
            target_object=("state_c_red", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
        )
    )
    gen_type_map = gen_type.set_index("generator_completed")[["source_object", "target_object"]].to_dict("index")

    work = compositions.copy()
    work["source_1"] = work["generator_1"].map(lambda g: gen_type_map.get(g, {}).get("source_object", np.nan))
    work["target_1"] = work["generator_1"].map(lambda g: gen_type_map.get(g, {}).get("target_object", np.nan))
    work["source_2"] = work["generator_2"].map(lambda g: gen_type_map.get(g, {}).get("source_object", np.nan))
    work["target_2"] = work["generator_2"].map(lambda g: gen_type_map.get(g, {}).get("target_object", np.nan))

    work["composition_typed"] = (
        work["generator_1"]
        + " : "
        + work["source_1"].astype(str)
        + "→"
        + work["target_1"].astype(str)
        + " ; "
        + work["generator_2"]
        + " : "
        + work["source_2"].astype(str)
        + "→"
        + work["target_2"].astype(str)
    )
    work["composable_match"] = (work["target_1"] == work["source_2"]).astype(int)

    # Case 1: already aggregated input from OBS-030e
    if "n_compositions" in work.columns:
        if "n_paths" not in work.columns:
            work["n_paths"] = np.nan

        out = (
            work.groupby(
                [
                    "route_class",
                    "generator_1",
                    "generator_2",
                    "source_1",
                    "target_1",
                    "source_2",
                    "target_2",
                    "composable_match",
                    "composition_typed",
                ],
                as_index=False,
            )
            .agg(
                n_compositions=("n_compositions", "sum"),
                n_paths=("n_paths", "sum"),
            )
        )
    else:
        # Case 2: raw composition rows with path_id
        if "path_id" not in work.columns:
            raise ValueError("Compositions input has neither aggregated counts nor raw path_id rows.")

        out = (
            work.groupby(
                [
                    "route_class",
                    "generator_1",
                    "generator_2",
                    "source_1",
                    "target_1",
                    "source_2",
                    "target_2",
                    "composable_match",
                    "composition_typed",
                ],
                as_index=False,
            )
            .agg(
                n_compositions=("path_id", "size"),
                n_paths=("path_id", "nunique"),
            )
        )

    total = out.groupby("route_class")["n_compositions"].transform("sum")
    out["composition_share"] = out["n_compositions"] / total.clip(lower=1)

    overall = (
        out.groupby(
            [
                "generator_1",
                "generator_2",
                "source_1",
                "target_1",
                "source_2",
                "target_2",
                "composable_match",
                "composition_typed",
            ],
            as_index=False,
        )
        .agg(
            n_compositions=("n_compositions", "sum"),
            n_paths=("n_paths", "sum"),
        )
    )
    overall["route_class"] = "overall"
    total_overall = float(overall["n_compositions"].sum())
    overall["composition_share"] = overall["n_compositions"] / max(total_overall, 1.0)

    out = pd.concat([out, overall], ignore_index=True)
    return out.sort_values(["route_class", "n_compositions"], ascending=[True, False]).reset_index(drop=True)


def build_family_summary(generators: pd.DataFrame, compositions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []

    gen_sub = generators[generators["route_class"] != "overall"].copy()
    comp_sub = compositions[compositions["route_class"] != "overall"].copy()

    for cls in CLASS_ORDER:
        g = gen_sub[gen_sub["route_class"] == cls].copy()
        c = comp_sub[comp_sub["route_class"] == cls].copy()

        row = {
            "route_class": cls,
            "n_generator_instances": int(g["n_instances"].sum()) if len(g) else 0,
            "n_compositions": int(c["n_compositions"].sum()) if len(c) else 0,
            "share_named_nonother": float(g.loc[g["generator_completed"] != "g_other", "generator_share"].sum()) if len(g) else 0.0,
            "share_other": float(g.loc[g["generator_completed"] == "g_other", "generator_share"].sum()) if len(g) else 0.0,
            "share_composable_match": float(
                safe_div(
                    c.loc[c["composable_match"] == 1, "n_compositions"].sum(),
                    c["n_compositions"].sum(),
                )
            ) if len(c) else 0.0,
        }

        for gen in GENERATOR_ORDER:
            hit = g[g["generator_completed"] == gen]
            row[f"{gen}_share"] = float(hit["generator_share"].sum()) if len(hit) else 0.0

        topg = g.sort_values("n_instances", ascending=False).head(5)
        for i in range(5):
            row[f"top_generator_{i+1}"] = (
                f"{topg.iloc[i]['generator_completed']}:{topg.iloc[i]['source_object']}→{topg.iloc[i]['target_object']}"
                if len(topg) > i else np.nan
            )

        topc = c.sort_values("n_compositions", ascending=False).head(5)
        for i in range(5):
            row[f"top_composition_{i+1}"] = topc.iloc[i]["composition_typed"] if len(topc) > i else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def build_summary(objects: pd.DataFrame, generators: pd.DataFrame, compositions: pd.DataFrame, family_summary: pd.DataFrame, cfg: Config) -> str:
    overall_g = generators[generators["route_class"] == "overall"].copy()
    overall_c = compositions[compositions["route_class"] == "overall"].copy()

    named_share = float(overall_g.loc[overall_g["generator_completed"] != "g_other", "generator_share"].sum()) if len(overall_g) else 0.0
    other_share = float(overall_g.loc[overall_g["generator_completed"] == "g_other", "generator_share"].sum()) if len(overall_g) else 0.0
    composable_share = float(
        safe_div(
            overall_c.loc[overall_c["composable_match"] == 1, "n_compositions"].sum(),
            overall_c["n_compositions"].sum(),
        )
    ) if len(overall_c) else 0.0

    lines = [
        "=== OBS-031 Proto-Groupoid of Seam Dynamics Summary ===",
        "",
        "Discipline",
        "- empirical proto-groupoid / partial composition system",
        "- not a theorem of invertibility",
        "- not a claim of full categorical closure",
        "",
        "Objects",
    ]
    for _, row in objects.iterrows():
        lines.append(
            f"  {row['object']}: source={int(row['n_as_source'])}, middle={int(row['n_as_middle'])}, target={int(row['n_as_target'])}"
        )

    lines.extend(
        [
            "",
            "Overall algebra quality",
            f"  named_generator_share = {named_share:.4f}",
            f"  residual_g_other_share = {other_share:.4f}",
            f"  composable_match_share = {composable_share:.4f}",
            "",
            "Overall top generators",
        ]
    )

    topg = overall_g.sort_values("n_instances", ascending=False).head(8)
    for _, row in topg.iterrows():
        lines.append(
            f"  {row['generator_completed']} : {row['source_object']}→{row['target_object']} "
            f"(n={int(row['n_instances'])}, share={float(row['generator_share']):.4f})"
        )

    lines.extend(["", "Family proto-groupoid signatures"])
    for _, row in family_summary.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  share_named_nonother      = {float(row['share_named_nonother']):.4f}",
                f"  share_other               = {float(row['share_other']):.4f}",
                f"  share_composable_match    = {float(row['share_composable_match']):.4f}",
                f"  g_rel_release_share       = {float(row['g_rel_release_share']):.4f}",
                f"  g_flank_shuttle_share     = {float(row['g_flank_shuttle_share']):.4f}",
                f"  g_low_flank_shuttle_share = {float(row['g_low_flank_shuttle_share']):.4f}",
                f"  g_off_persist_share       = {float(row['g_off_persist_share']):.4f}",
                f"  g_post_persist_share      = {float(row['g_post_persist_share']):.4f}",
                f"  g_reentry_share           = {float(row['g_reentry_share']):.4f}",
                f"  g_edge_to_post_share      = {float(row['g_edge_to_post_share']):.4f}",
                f"  top_generator_1           = {row['top_generator_1']}",
                f"  top_generator_2           = {row['top_generator_2']}",
                f"  top_composition_1         = {row['top_composition_1']}",
                f"  top_composition_2         = {row['top_composition_2']}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive conclusion",
            "- objects are reduced seam states",
            "- arrows are named empirical generators",
            "- partial composition is observed generator concatenation",
            "- family-specific subalgebras are present",
            "- this supports a proto-groupoid framing of seam dynamics",
        ]
    )
    return "\n".join(lines)


def render_figure(objects: pd.DataFrame, generators: pd.DataFrame, compositions: pd.DataFrame, family_summary: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_named = fig.add_subplot(gs[0, 0])
    ax_family = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    overall_g = generators[generators["route_class"] == "overall"].copy()
    overall_c = compositions[compositions["route_class"] == "overall"].copy()

    topg = overall_g.sort_values("n_instances", ascending=False).head(8)
    ax_named.barh(
        np.arange(len(topg)),
        topg["generator_share"].to_numpy(dtype=float),
    )
    ax_named.set_yticks(np.arange(len(topg)))
    ax_named.set_yticklabels(
        [f"{g}:{s}→{t}" for g, s, t in zip(topg["generator_completed"], topg["source_object"], topg["target_object"])]
    )
    ax_named.invert_yaxis()
    ax_named.set_title("Overall top typed generators", fontsize=14, pad=8)
    ax_named.grid(alpha=0.15, axis="x")

    x = np.arange(len(family_summary))
    width = 0.22
    ax_family.bar(x - width, family_summary["g_post_persist_share"], width, label="post persist")
    ax_family.bar(x, family_summary["g_flank_shuttle_share"], width, label="flank shuttle")
    ax_family.bar(x + width, family_summary["g_low_flank_shuttle_share"], width, label="low↔flank")
    ax_family.set_xticks(x)
    ax_family.set_xticklabels(family_summary["route_class"], rotation=12)
    ax_family.set_title("Family generator signatures", fontsize=14, pad=8)
    ax_family.grid(alpha=0.15, axis="y")
    ax_family.legend()

    topc = overall_c.sort_values("n_compositions", ascending=False).head(cfg.top_k_compositions)
    ax_comp.barh(
        np.arange(len(topc)),
        topc["composition_share"].to_numpy(dtype=float),
    )
    ax_comp.set_yticks(np.arange(len(topc)))
    ax_comp.set_yticklabels(topc["composition_typed"].tolist(), fontsize=8)
    ax_comp.invert_yaxis()
    ax_comp.set_title("Top typed partial compositions", fontsize=14, pad=8)
    ax_comp.grid(alpha=0.15, axis="x")

    ax_top.axis("off")
    y = 0.95
    for _, row in family_summary.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"g1: {row['top_generator_1']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"g2: {row['top_generator_2']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"c1: {row['top_composition_1']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"c2: {row['top_composition_2']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Family subalgebras", fontsize=14, pad=8)

    named_share = float(overall_g.loc[overall_g["generator_completed"] != "g_other", "generator_share"].sum()) if len(overall_g) else 0.0
    other_share = float(overall_g.loc[overall_g["generator_completed"] == "g_other", "generator_share"].sum()) if len(overall_g) else 0.0
    composable_share = float(
        safe_div(
            overall_c.loc[overall_c["composable_match"] == 1, "n_compositions"].sum(),
            overall_c["n_compositions"].sum(),
        )
    ) if len(overall_c) else 0.0

    ax_diag.axis("off")
    text = (
        "OBS-031 diagnostics\n\n"
        f"named generator share:\n{named_share:.3f}\n\n"
        f"residual g_other share:\n{other_share:.3f}\n\n"
        f"composable match share:\n{composable_share:.3f}\n\n"
        "Interpretation:\n"
        "the seam dynamics now admit\n"
        "a partial typed-arrow system\n"
        "with family-specific\n"
        "subalgebras."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-031 proto-groupoid of seam dynamics", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an empirical proto-groupoid of seam dynamics.")
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--compositions-csv", default=Config.compositions_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-generator-share", type=float, default=Config.min_generator_share)
    parser.add_argument("--min-composition-share", type=float, default=Config.min_composition_share)
    parser.add_argument("--top-k-compositions", type=int, default=Config.top_k_compositions)
    args = parser.parse_args()

    cfg = Config(
        assignments_csv=args.assignments_csv,
        compositions_csv=args.compositions_csv,
        outdir=args.outdir,
        min_generator_share=args.min_generator_share,
        min_composition_share=args.min_composition_share,
        top_k_compositions=args.top_k_compositions,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assignments = load_csv(cfg.assignments_csv)
    compositions = load_csv(cfg.compositions_csv)

    objects = build_objects(assignments)
    generators = build_generators(assignments, cfg)
    comp = build_compositions(compositions, assignments, cfg)
    family_summary = build_family_summary(generators, comp, cfg)

    objects_csv = outdir / "proto_groupoid_objects.csv"
    gen_csv = outdir / "proto_groupoid_generators.csv"
    comp_csv = outdir / "proto_groupoid_compositions.csv"
    fam_csv = outdir / "proto_groupoid_family_summary.csv"
    txt_path = outdir / "obs031_proto_groupoid_of_seam_dynamics_summary.txt"
    png_path = outdir / "obs031_proto_groupoid_of_seam_dynamics_figure.png"

    objects.to_csv(objects_csv, index=False)
    generators.to_csv(gen_csv, index=False)
    comp.to_csv(comp_csv, index=False)
    family_summary.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(objects, generators, comp, family_summary, cfg), encoding="utf-8")
    render_figure(objects, generators, comp, family_summary, png_path, cfg)

    print(objects_csv)
    print(gen_csv)
    print(comp_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
