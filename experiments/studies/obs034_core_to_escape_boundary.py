#!/usr/bin/env python3
"""
OBS-034 — Core-to-escape boundary.

Purpose
-------
Map the gateway between:
1. the reversible shuttle core
2. the directed escape sector

This study asks:
- which generators carry crossings from the reversible core into the directed sector?
- which object states are the main launch points?
- which local fields are elevated at the crossing?
- do families cross that boundary in different ways?

Inputs
------
outputs/obs033_reversible_core_vs_directed_escape/reversible_core_pairs.csv
outputs/obs033_reversible_core_vs_directed_escape/directed_escape_pairs.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv
outputs/obs030e_complete_generator_basis/completed_generator_compositions.csv

Outputs
-------
outputs/obs034_core_to_escape_boundary/
  generator_sector_map.csv
  core_to_escape_crossings.csv
  core_to_escape_family_summary.csv
  obs034_core_to_escape_boundary_summary.txt
  obs034_core_to_escape_boundary_figure.png
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
    reversible_pairs_csv: str = (
        "outputs/obs033_reversible_core_vs_directed_escape/reversible_core_pairs.csv"
    )
    directed_pairs_csv: str = (
        "outputs/obs033_reversible_core_vs_directed_escape/directed_escape_pairs.csv"
    )
    proto_generators_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_generators.csv"
    )
    proto_compositions_csv: str = (
        "outputs/obs031_proto_groupoid_of_seam_dynamics/proto_groupoid_compositions.csv"
    )
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    outdir: str = "outputs/obs034_core_to_escape_boundary"
    min_crossing_count: int = 1
    top_k_generators: int = 10


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_cols = {
        "route_class",
        "generator_completed",
        "generator_1",
        "generator_2",
        "composition",
        "composition_typed",
        "motif",
        "motif_class",
        "state_a",
        "state_b",
        "state_c",
        "state_a_red",
        "state_b_red",
        "state_c_red",
        "generator_word",
        "object_a",
        "object_b",
        "pair_key",
        "forward_label",
        "reverse_label",
        "sector",
        "src1",
        "tgt1",
        "src2",
        "tgt2",
        "sector_1",
        "sector_2",
        "crossing_type",
    }
    for col in df.columns:
        if col not in text_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def safe_mean(s: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def build_generator_sector_map(rev: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    rev_use = rev[["route_class", "generator_completed", "object_a", "object_b", "total_count"]].copy()
    rev_use["sector"] = "reversible_core"

    dir_use = direct[["route_class", "generator_completed", "object_a", "object_b", "total_count"]].copy()
    dir_use["sector"] = "directed_escape"

    both = pd.concat([rev_use, dir_use], ignore_index=True)
    both["object_a"] = both["object_a"].astype(str)
    both["object_b"] = both["object_b"].astype(str)

    out = (
        both.groupby(["route_class", "generator_completed", "object_a", "object_b", "sector"], as_index=False)
        .agg(pair_weight=("total_count", "sum"))
    )
    return out.sort_values(["route_class", "pair_weight"], ascending=[True, False]).reset_index(drop=True)


def classify_generator_sector(
    route_class: str,
    generator_name: str,
    src: str,
    tgt: str,
    sector_map: pd.DataFrame,
) -> str:
    hit = sector_map[
        (sector_map["route_class"] == route_class)
        & (sector_map["generator_completed"] == generator_name)
        & (sector_map["object_a"] == src)
        & (sector_map["object_b"] == tgt)
    ]
    if len(hit):
        return str(hit.iloc[0]["sector"])

    # fallback: generator-level family fallback
    hit2 = sector_map[
        (sector_map["route_class"] == route_class)
        & (sector_map["generator_completed"] == generator_name)
    ]
    if len(hit2):
        return str(hit2.sort_values("pair_weight", ascending=False).iloc[0]["sector"])

    return "unknown"


def build_core_to_escape_crossings(
    proto_generators: pd.DataFrame,
    proto_compositions: pd.DataFrame,
    sector_map: pd.DataFrame,
) -> pd.DataFrame:
    # typed generator lookup from OBS-031
    gtab = proto_generators[proto_generators["route_class"] != "overall"].copy()

    # choose most-supported typing per family+generator
    gtab = gtab.sort_values(
        ["route_class", "generator_completed", "n_instances"],
        ascending=[True, True, False],
    )
    gtab = gtab.drop_duplicates(subset=["route_class", "generator_completed"])

    type_map = {
        (str(r.route_class), str(r.generator_completed)): (str(r.source_object), str(r.target_object))
        for r in gtab.itertuples(index=False)
    }

    comp = proto_compositions[proto_compositions["route_class"] != "overall"].copy()

    rows = []
    for _, row in comp.iterrows():
        cls = str(row["route_class"])
        g1 = str(row["generator_1"])
        g2 = str(row["generator_2"])

        src1, tgt1 = type_map.get((cls, g1), ("?", "?"))
        src2, tgt2 = type_map.get((cls, g2), ("?", "?"))

        sec1 = classify_generator_sector(cls, g1, src1, tgt1, sector_map)
        sec2 = classify_generator_sector(cls, g2, src2, tgt2, sector_map)

        crossing_type = "other"
        if sec1 == "reversible_core" and sec2 == "directed_escape":
            crossing_type = "core_to_escape"
        elif sec1 == "directed_escape" and sec2 == "reversible_core":
            crossing_type = "escape_to_core"
        elif sec1 == "reversible_core" and sec2 == "reversible_core":
            crossing_type = "core_internal"
        elif sec1 == "directed_escape" and sec2 == "directed_escape":
            crossing_type = "escape_internal"

        rows.append(
            {
                "route_class": cls,
                "generator_1": g1,
                "generator_2": g2,
                "src1": src1,
                "tgt1": tgt1,
                "src2": src2,
                "tgt2": tgt2,
                "sector_1": sec1,
                "sector_2": sec2,
                "crossing_type": crossing_type,
                "n_compositions": float(row["n_compositions"]),
                "n_paths": float(row["n_paths"]) if "n_paths" in row else np.nan,
                "composition_share": float(row["composition_share"]) if "composition_share" in row else np.nan,
                "composition_typed": row.get("composition_typed", np.nan),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["route_class", "n_compositions"], ascending=[True, False]).reset_index(drop=True)


def build_family_summary(
    crossings: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    # local field summary for first generator in crossing
    assign = assignments.copy()
    assign["route_class"] = assign["route_class"].astype(str)
    assign["generator_completed"] = assign["generator_completed"].astype(str)

    rows = []
    for cls in CLASS_ORDER:
        sub = crossings[crossings["route_class"] == cls].copy()
        total = float(sub["n_compositions"].sum()) if len(sub) else 0.0

        c2e = sub[sub["crossing_type"] == "core_to_escape"].copy()
        e2c = sub[sub["crossing_type"] == "escape_to_core"].copy()
        cin = sub[sub["crossing_type"] == "core_internal"].copy()
        ein = sub[sub["crossing_type"] == "escape_internal"].copy()

        # estimate launch fields by generator_1 usage weighted by crossing counts
        launch_rows = []
        for _, row in c2e.iterrows():
            a = assign[
                (assign["route_class"] == cls)
                & (assign["generator_completed"] == str(row["generator_1"]))
            ]
            if len(a) == 0:
                continue
            launch_rows.append(
                {
                    "weight": float(row["n_compositions"]),
                    "relational": safe_mean(a.get("relational_a", pd.Series(dtype=float))),
                    "anisotropy": safe_mean(a.get("anisotropy_a", pd.Series(dtype=float))),
                    "distance": safe_mean(a.get("distance_a", pd.Series(dtype=float))),
                }
            )

        launch_df = pd.DataFrame(launch_rows)
        if len(launch_df):
            w = launch_df["weight"].to_numpy(dtype=float)
            rel = float(np.average(launch_df["relational"], weights=w))
            aniso = float(np.average(launch_df["anisotropy"], weights=w))
            dist = float(np.average(launch_df["distance"], weights=w))
        else:
            rel = aniso = dist = float("nan")

        top = c2e.sort_values("n_compositions", ascending=False).head(3)

        rows.append(
            {
                "route_class": cls,
                "n_total_compositions": int(total),
                "core_to_escape_share": float(c2e["n_compositions"].sum() / total) if total > 0 else 0.0,
                "escape_to_core_share": float(e2c["n_compositions"].sum() / total) if total > 0 else 0.0,
                "core_internal_share": float(cin["n_compositions"].sum() / total) if total > 0 else 0.0,
                "escape_internal_share": float(ein["n_compositions"].sum() / total) if total > 0 else 0.0,
                "mean_launch_relational": rel,
                "mean_launch_anisotropy": aniso,
                "mean_launch_distance": dist,
                "top_gateway_1": top.iloc[0]["composition_typed"] if len(top) > 0 else np.nan,
                "top_gateway_2": top.iloc[1]["composition_typed"] if len(top) > 1 else np.nan,
                "top_gateway_3": top.iloc[2]["composition_typed"] if len(top) > 2 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(sector_map: pd.DataFrame, crossings: pd.DataFrame, fam: pd.DataFrame) -> str:
    total = float(crossings["n_compositions"].sum()) if len(crossings) else 0.0
    c2e = float(crossings.loc[crossings["crossing_type"] == "core_to_escape", "n_compositions"].sum())
    e2c = float(crossings.loc[crossings["crossing_type"] == "escape_to_core", "n_compositions"].sum())

    lines = [
        "=== OBS-034 Core-to-Escape Boundary Summary ===",
        "",
        f"core_to_escape_share_overall = {c2e / total if total else 0.0:.4f}",
        f"escape_to_core_share_overall = {e2c / total if total else 0.0:.4f}",
        "",
        "Interpretive guide",
        "- core_to_escape marks the gateway from reversible shuttle structure into directed escape",
        "- escape_to_core marks return from directed sector back into reversible core",
        "- launch-field means estimate the local conditions associated with leaving the core",
        "",
        "Family gateway summaries",
    ]

    for _, row in fam.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  core_to_escape_share   = {float(row['core_to_escape_share']):.4f}",
                f"  escape_to_core_share   = {float(row['escape_to_core_share']):.4f}",
                f"  core_internal_share    = {float(row['core_internal_share']):.4f}",
                f"  escape_internal_share  = {float(row['escape_internal_share']):.4f}",
                f"  mean_launch_relational = {float(row['mean_launch_relational']):.4f}",
                f"  mean_launch_anisotropy = {float(row['mean_launch_anisotropy']):.4f}",
                f"  mean_launch_distance   = {float(row['mean_launch_distance']):.4f}",
                f"  top_gateway_1          = {row['top_gateway_1']}",
                f"  top_gateway_2          = {row['top_gateway_2']}",
                f"  top_gateway_3          = {row['top_gateway_3']}",
                "",
            ]
        )

    if len(crossings):
        top = crossings[crossings["crossing_type"] == "core_to_escape"].sort_values("n_compositions", ascending=False).head(10)
        lines.append("Top core-to-escape gateways")
        for _, row in top.iterrows():
            lines.append(
                f"  {row['route_class']} | {row['composition_typed']} | n={row['n_compositions']:.0f}, share={row['composition_share']:.4f}"
            )

    return "\n".join(lines)


def render_figure(crossings: pd.DataFrame, fam: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])

    ax_share = fig.add_subplot(gs[0, 0])
    ax_fields = fig.add_subplot(gs[0, 1])
    ax_internal = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(fam))
    width = 0.34

    ax_share.bar(x - width / 2, fam["core_to_escape_share"], width, label="core→escape")
    ax_share.bar(x + width / 2, fam["escape_to_core_share"], width, label="escape→core")
    ax_share.set_xticks(x)
    ax_share.set_xticklabels(fam["route_class"], rotation=12)
    ax_share.set_title("Boundary crossing shares", fontsize=14, pad=8)
    ax_share.grid(alpha=0.15, axis="y")
    ax_share.legend()

    ax_fields.bar(x - width, fam["mean_launch_relational"], width, label="relational")
    ax_fields.bar(x, fam["mean_launch_anisotropy"], width, label="anisotropy")
    ax_fields.bar(x + width, fam["mean_launch_distance"], width, label="distance")
    ax_fields.set_xticks(x)
    ax_fields.set_xticklabels(fam["route_class"], rotation=12)
    ax_fields.set_title("Gateway launch fields", fontsize=14, pad=8)
    ax_fields.grid(alpha=0.15, axis="y")
    ax_fields.legend()

    ax_internal.bar(x - width / 2, fam["core_internal_share"], width, label="core internal")
    ax_internal.bar(x + width / 2, fam["escape_internal_share"], width, label="escape internal")
    ax_internal.set_xticks(x)
    ax_internal.set_xticklabels(fam["route_class"], rotation=12)
    ax_internal.set_title("Within-sector compositions", fontsize=14, pad=8)
    ax_internal.grid(alpha=0.15, axis="y")
    ax_internal.legend()

    ax_top.axis("off")
    y = 0.95
    for _, row in fam.iterrows():
        ax_top.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_top.text(0.04, y, f"g1: {row['top_gateway_1']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"g2: {row['top_gateway_2']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_top.text(0.04, y, f"g3: {row['top_gateway_3']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_top.set_title("Top core→escape gateways", fontsize=14, pad=8)

    total = float(crossings["n_compositions"].sum()) if len(crossings) else 0.0
    c2e = float(crossings.loc[crossings["crossing_type"] == "core_to_escape", "n_compositions"].sum()) if len(crossings) else 0.0
    e2c = float(crossings.loc[crossings["crossing_type"] == "escape_to_core", "n_compositions"].sum()) if len(crossings) else 0.0

    ax_diag.axis("off")
    text = (
        "OBS-034 diagnostics\n\n"
        f"overall core→escape:\n{(c2e / total) if total else 0.0:.3f}\n\n"
        f"overall escape→core:\n{(e2c / total) if total else 0.0:.3f}\n\n"
        "Interpretation:\n"
        "this isolates the gateway\n"
        "that converts reversible\n"
        "shuttle structure into\n"
        "directed escape dynamics."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-034 core-to-escape boundary", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map the boundary between reversible core and directed escape.")
    parser.add_argument("--reversible-pairs-csv", default=Config.reversible_pairs_csv)
    parser.add_argument("--directed-pairs-csv", default=Config.directed_pairs_csv)
    parser.add_argument("--proto-generators-csv", default=Config.proto_generators_csv)
    parser.add_argument("--proto-compositions-csv", default=Config.proto_compositions_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--min-crossing-count", type=int, default=Config.min_crossing_count)
    parser.add_argument("--top-k-generators", type=int, default=Config.top_k_generators)
    args = parser.parse_args()

    cfg = Config(
        reversible_pairs_csv=args.reversible_pairs_csv,
        directed_pairs_csv=args.directed_pairs_csv,
        proto_generators_csv=args.proto_generators_csv,
        proto_compositions_csv=args.proto_compositions_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        min_crossing_count=args.min_crossing_count,
        top_k_generators=args.top_k_generators,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rev = load_csv(cfg.reversible_pairs_csv)
    direct = load_csv(cfg.directed_pairs_csv)
    proto_generators = load_csv(cfg.proto_generators_csv)
    proto_compositions = load_csv(cfg.proto_compositions_csv)
    assignments = load_csv(cfg.assignments_csv)

    sector_map = build_generator_sector_map(rev, direct)
    crossings = build_core_to_escape_crossings(proto_generators, proto_compositions, sector_map)
    fam = build_family_summary(crossings, assignments)

    sector_csv = outdir / "generator_sector_map.csv"
    crossing_csv = outdir / "core_to_escape_crossings.csv"
    fam_csv = outdir / "core_to_escape_family_summary.csv"
    txt_path = outdir / "obs034_core_to_escape_boundary_summary.txt"
    png_path = outdir / "obs034_core_to_escape_boundary_figure.png"

    sector_map.to_csv(sector_csv, index=False)
    crossings.to_csv(crossing_csv, index=False)
    fam.to_csv(fam_csv, index=False)
    txt_path.write_text(build_summary(sector_map, crossings, fam), encoding="utf-8")
    render_figure(crossings, fam, png_path, cfg)

    print(sector_csv)
    print(crossing_csv)
    print(fam_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
