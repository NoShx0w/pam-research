#!/usr/bin/env python3
"""
OBS-029 — Seam escape channels.

Study the directional geometry by which branch-exit paths leave the seam-core,
and compare that against seam-resident families.

Core idea
---------
Using the canonical seam bundle plus route data, identify local seam-to-offseam
transitions and ask:

1. Do branch-exit paths leave the seam through consistent directions in MDS space?
2. Are there preferred escape channels from shared / relational hotspot nodes?
3. Do stable corridor and reorganization-heavy paths fail to use those channels?

Inputs
------
outputs/obs028c_canonical_seam_bundle/seam_nodes.csv
outputs/obs022_scene_bundle/scene_routes.csv

Outputs
-------
outputs/obs029_seam_escape_channels/
  seam_escape_steps.csv
  seam_escape_class_summary.csv
  obs029_seam_escape_channels_summary.txt
  obs029_seam_escape_channels_figure.png
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
    outdir: str = "outputs/obs029_seam_escape_channels"
    seam_threshold: float = 0.15
    min_exit_gain: float = 0.20
    min_committed_offsteps: int = 2
    direction_bin_deg: float = 30.0
    top_k_vectors: int = 12


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]


def safe_mean(s: pd.Series | np.ndarray) -> float:
    ss = pd.to_numeric(pd.Series(s), errors="coerce")
    return float(ss.mean()) if ss.notna().any() else float("nan")


def circ_mean_deg(theta_deg: pd.Series | np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(theta_deg), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return float("nan")
    rad = np.radians(a)
    return float((np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) + 360.0) % 360.0)


def circ_resultant_length(theta_deg: pd.Series | np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(theta_deg), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return float("nan")
    rad = np.radians(a)
    return float(np.hypot(np.mean(np.cos(rad)), np.mean(np.sin(rad))))


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
            "signed_phase",
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
    seam_use = seam_use.rename(
        columns={
            c: f"{c}_bundle" for c in seam_use.columns if c != "node_id"
        }
    )

    routes = routes.merge(
        seam_use,
        on="node_id",
        how="left",
    )

    # Normalize canonical columns from bundle copy
    for base_col in [
        "r",
        "alpha",
        "mds1",
        "mds2",
        "signed_phase",
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

    required = ["distance_to_seam", "mds1", "mds2", "hotspot_class"]
    missing = [c for c in required if c not in routes.columns]
    if missing:
        raise ValueError(f"Missing required normalized route columns: {missing}")

    return seam_nodes, routes


def angle_deg(dx: float, dy: float) -> float:
    return float((np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0)


def direction_bin(theta_deg: float, bin_width: float) -> str:
    center = int((np.floor((theta_deg + bin_width / 2.0) / bin_width) * bin_width) % 360)
    return f"{center:03d}"


def compute_escape_steps(routes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []

    work = routes[routes["route_class"].isin(CLASS_ORDER)].copy()
    if "step" in work.columns:
        work = work.sort_values(["path_id", "step"]).reset_index(drop=True)

    for path_id, grp in work.groupby("path_id", sort=False):
        grp = grp.sort_values("step").copy().reset_index(drop=True)
        if len(grp) < 2:
            continue

        d2s = pd.to_numeric(grp["distance_to_seam"], errors="coerce")
        seam_mask = (d2s <= cfg.seam_threshold).fillna(False).to_numpy(dtype=bool)

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            da = pd.to_numeric(a.get("distance_to_seam"), errors="coerce")
            db = pd.to_numeric(b.get("distance_to_seam"), errors="coerce")
            if pd.isna(da) or pd.isna(db):
                continue

            from_seam = bool(da <= cfg.seam_threshold)
            to_off = bool(db > cfg.seam_threshold)
            exit_gain = float(db - da)

            x0 = pd.to_numeric(a.get("mds1"), errors="coerce")
            y0 = pd.to_numeric(a.get("mds2"), errors="coerce")
            x1 = pd.to_numeric(b.get("mds1"), errors="coerce")
            y1 = pd.to_numeric(b.get("mds2"), errors="coerce")
            if any(pd.isna(v) for v in [x0, y0, x1, y1]):
                continue

            dx = float(x1 - x0)
            dy = float(y1 - y0)
            step_len = float(np.hypot(dx, dy))
            theta = angle_deg(dx, dy) if step_len > 1e-12 else np.nan

            hotspot_class = a.get("hotspot_class", np.nan)
            if pd.isna(hotspot_class):
                hotspot_class = "non_hotspot"

            is_escape_step = bool(from_seam and to_off and exit_gain >= cfg.min_exit_gain)

            # committed escape = remain off-seam for k subsequent rows after the crossing
            committed = False
            off_run_length = 0
            if is_escape_step:
                j = i + 1
                while j < len(grp):
                    dj = pd.to_numeric(grp.iloc[j].get("distance_to_seam"), errors="coerce")
                    if pd.isna(dj) or dj <= cfg.seam_threshold:
                        break
                    off_run_length += 1
                    j += 1
                committed = off_run_length >= cfg.min_committed_offsteps

            rows.append(
                {
                    "path_id": path_id,
                    "route_class": a["route_class"],
                    "path_family": a.get("path_family", np.nan),
                    "step": pd.to_numeric(a.get("step"), errors="coerce"),
                    "node_id": pd.to_numeric(a.get("node_id"), errors="coerce"),
                    "from_hotspot_class": hotspot_class,
                    "from_shared_hotspot": int(pd.to_numeric(a.get("shared_hotspot"), errors="coerce") == 1),
                    "from_relational_hotspot": int(pd.to_numeric(a.get("relational_hotspot"), errors="coerce") == 1),
                    "from_anisotropy_hotspot": int(pd.to_numeric(a.get("anisotropy_hotspot"), errors="coerce") == 1),
                    "from_distance_to_seam": float(da),
                    "to_distance_to_seam": float(db),
                    "exit_gain": exit_gain,
                    "from_seam": int(from_seam),
                    "to_offseam": int(to_off),
                    "is_escape_step": int(is_escape_step),
                    "is_committed_escape": int(committed),
                    "off_run_length": int(off_run_length),
                    "dx": dx,
                    "dy": dy,
                    "step_len": step_len,
                    "theta_deg": theta,
                    "theta_bin": direction_bin(theta, cfg.direction_bin_deg) if np.isfinite(theta) else np.nan,
                    "from_mds1": float(x0),
                    "from_mds2": float(y0),
                    "to_mds1": float(x1),
                    "to_mds2": float(y1),
                    "from_relational": pd.to_numeric(a.get("neighbor_direction_mismatch_mean"), errors="coerce"),
                    "from_anisotropy": pd.to_numeric(a.get("sym_traceless_norm"), errors="coerce"),
                }
            )

    return pd.DataFrame(rows)


def summarize_classes(escape_steps: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cls in CLASS_ORDER:
        grp = escape_steps[escape_steps["route_class"] == cls].copy()
        esc = grp[grp["is_escape_step"] == 1].copy()
        com = grp[grp["is_committed_escape"] == 1].copy()

        if len(grp) == 0:
            continue

        path_touch = (
            grp.groupby("path_id", as_index=False)
            .agg(
                has_escape=("is_escape_step", "max"),
                has_committed_escape=("is_committed_escape", "max"),
            )
        )

        counts = com["theta_bin"].value_counts() if len(com) else pd.Series(dtype=int)
        top_bin = counts.index[0] if len(counts) else np.nan
        top_bin_share = float(counts.iloc[0] / max(len(com), 1)) if len(counts) else np.nan

        rows.append(
            {
                "route_class": cls,
                "n_steps": int(len(grp)),
                "n_escape_steps": int(len(esc)),
                "n_committed_escape_steps": int(len(com)),
                "escape_step_share": float(len(esc) / max(len(grp), 1)),
                "committed_escape_share": float(len(com) / max(len(grp), 1)),
                "path_escape_touch_share": safe_mean(path_touch["has_escape"]),
                "path_committed_escape_touch_share": safe_mean(path_touch["has_committed_escape"]),
                "mean_exit_gain": safe_mean(com["exit_gain"]),
                "mean_escape_len": safe_mean(com["step_len"]),
                "mean_off_run_length": safe_mean(com["off_run_length"]),
                "mean_escape_theta_deg": circ_mean_deg(com["theta_deg"]),
                "escape_direction_concentration": circ_resultant_length(com["theta_deg"]),
                "top_direction_bin": top_bin,
                "top_direction_bin_share": top_bin_share,
                "shared_escape_share": safe_mean(com["from_shared_hotspot"]),
                "relational_escape_share": safe_mean(com["from_relational_hotspot"]),
                "anisotropy_escape_share": safe_mean(com["from_anisotropy_hotspot"]),
                "mean_from_relational": safe_mean(com["from_relational"]),
                "mean_from_anisotropy": safe_mean(com["from_anisotropy"]),
                "mean_from_distance_to_seam": safe_mean(com["from_distance_to_seam"]),
            }
        )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def build_summary(summary_df: pd.DataFrame) -> str:
    lines = [
        "=== OBS-029 Seam Escape Channels Summary ===",
        "",
    ]

    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['route_class']}",
                f"  n_steps                              = {int(row['n_steps'])}",
                f"  n_escape_steps                       = {int(row['n_escape_steps'])}",
                f"  n_committed_escape_steps             = {int(row['n_committed_escape_steps'])}",
                f"  escape_step_share                    = {float(row['escape_step_share']):.4f}",
                f"  committed_escape_share               = {float(row['committed_escape_share']):.4f}",
                f"  path_escape_touch_share              = {float(row['path_escape_touch_share']):.4f}",
                f"  path_committed_escape_touch_share    = {float(row['path_committed_escape_touch_share']):.4f}",
                f"  mean_exit_gain                       = {float(row['mean_exit_gain']):.4f}",
                f"  mean_escape_len                      = {float(row['mean_escape_len']):.4f}",
                f"  mean_off_run_length                  = {float(row['mean_off_run_length']):.4f}",
                f"  mean_escape_theta_deg                = {float(row['mean_escape_theta_deg']):.4f}",
                f"  escape_direction_concentration       = {float(row['escape_direction_concentration']):.4f}",
                f"  top_direction_bin                    = {row['top_direction_bin']}",
                f"  top_direction_bin_share              = {float(row['top_direction_bin_share']):.4f}",
                f"  shared_escape_share                  = {float(row['shared_escape_share']):.4f}",
                f"  relational_escape_share              = {float(row['relational_escape_share']):.4f}",
                f"  anisotropy_escape_share              = {float(row['anisotropy_escape_share']):.4f}",
                f"  mean_from_relational                 = {float(row['mean_from_relational']):.4f}",
                f"  mean_from_anisotropy                 = {float(row['mean_from_anisotropy']):.4f}",
                f"  mean_from_distance_to_seam           = {float(row['mean_from_distance_to_seam']):.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "Interpretive guide",
            "- escape_step captures local seam-to-offseam departures",
            "- committed_escape captures departures that remain off-seam for multiple subsequent steps",
            "- high escape_direction_concentration suggests a coherent preferred escape channel",
            "- committed escape is the better measure of true release than local oscillation",
        ]
    )
    return "\n".join(lines)


def render_figure(seam_nodes: pd.DataFrame, escape_steps: pd.DataFrame, summary_df: pd.DataFrame, outpath: Path, cfg: Config) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.05, 1.05], height_ratios=[1.0, 1.0])

    ax_map = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_conc = fig.add_subplot(gs[0, 2])
    ax_touch = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[1, 2])

    # base seam map
    ax_map.scatter(
        seam_nodes["mds1"],
        seam_nodes["mds2"],
        c=pd.to_numeric(seam_nodes["distance_to_seam"], errors="coerce"),
        cmap="magma_r",
        s=70,
        alpha=0.85,
        linewidths=0.25,
        edgecolors="white",
        zorder=1,
    )

    shared = seam_nodes[pd.to_numeric(seam_nodes.get("shared_hotspot", 0), errors="coerce").fillna(0) == 1]
    if len(shared):
        ax_map.scatter(
            shared["mds1"],
            shared["mds2"],
            s=170,
            c="#FFD166",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.98,
            zorder=2,
        )

    colors = {
        "branch_exit": "#1f77b4",
        "stable_seam_corridor": "#2ca02c",
        "reorganization_heavy": "#d62728",
    }

    for cls in CLASS_ORDER:
        esc = escape_steps[(escape_steps["route_class"] == cls) & (escape_steps["is_escape_step"] == 1)].copy()
        if len(esc) == 0:
            continue

        # top-k by exit gain for readability
        esc = escape_steps[(escape_steps["route_class"] == cls) & (escape_steps["is_committed_escape"] == 1)].copy()
        for _, row in esc.iterrows():
            ax_map.arrow(
                float(row["from_mds1"]),
                float(row["from_mds2"]),
                float(row["dx"]),
                float(row["dy"]),
                width=0.004,
                head_width=0.06,
                head_length=0.08,
                length_includes_head=True,
                color=colors[cls],
                alpha=0.85,
                zorder=3,
            )

    ax_map.set_title("OBS-029 seam escape vectors", fontsize=16, pad=8)
    ax_map.set_xlabel("MDS 1")
    ax_map.set_ylabel("MDS 2")
    ax_map.grid(alpha=0.10)
    ax_map.set_aspect("equal", adjustable="box")

    # escape step share
    x = np.arange(len(summary_df))
    ax_bar.bar(x, summary_df["committed_escape_share"], color=[colors[c] for c in summary_df["route_class"]])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_bar.set_title("Committed escape share", fontsize=14, pad=8)
    ax_bar.set_ylabel("share of steps")
    ax_bar.grid(alpha=0.15, axis="y")

    # direction concentration
    ax_conc.bar(x, summary_df["escape_direction_concentration"], color=[colors[c] for c in summary_df["route_class"]])
    ax_conc.set_xticks(x)
    ax_conc.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_conc.set_title("Directional concentration", fontsize=14, pad=8)
    ax_conc.set_ylabel("resultant length")
    ax_conc.grid(alpha=0.15, axis="y")

    # escape origin hotspot shares
    width = 0.24
    ax_touch.bar(x - width, summary_df["shared_escape_share"], width, label="shared", color="#FFD166", edgecolor="black")
    ax_touch.bar(x, summary_df["relational_escape_share"], width, label="relational", color="#B23A48")
    ax_touch.bar(x + width, summary_df["anisotropy_escape_share"], width, label="anisotropy", color="#2A9D8F")
    ax_touch.set_xticks(x)
    ax_touch.set_xticklabels(summary_df["route_class"], rotation=12)
    ax_touch.set_title("Escape-origin hotspot type", fontsize=14, pad=8)
    ax_touch.set_ylabel("share of escape steps")
    ax_touch.grid(alpha=0.15, axis="y")
    ax_touch.legend()

    # diagnostics
    ax_diag.axis("off")
    if len(summary_df):
        best_escape = summary_df.sort_values("committed_escape_share", ascending=False).iloc[0]
        best_conc = summary_df.sort_values("escape_direction_concentration", ascending=False).iloc[0]
        text = (
            "OBS-029 diagnostics\n\n"
            f"highest committed escape:\n{best_escape['route_class']} ({best_escape['committed_escape_share']:.3f})\n\n"
            f"strongest directionality:\n{best_conc['route_class']} ({best_conc['escape_direction_concentration']:.3f})\n\n"
            "Interpretation:\n"
            "if branch_exit dominates both,\n"
            "the seam has coherent escape\n"
            "channels rather than random\n"
            "release directions."
        )
    else:
        text = "No summary rows available."

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

    fig.suptitle("PAM Observatory — OBS-029 seam escape channels", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze seam escape channels from the canonical seam bundle.")
    parser.add_argument("--seam-nodes-csv", default=Config.seam_nodes_csv)
    parser.add_argument("--routes-csv", default=Config.routes_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--min-exit-gain", type=float, default=Config.min_exit_gain)
    parser.add_argument("--direction-bin-deg", type=float, default=Config.direction_bin_deg)
    parser.add_argument("--top-k-vectors", type=int, default=Config.top_k_vectors)
    parser.add_argument("--min-committed-offsteps", type=int, default=Config.min_committed_offsteps)
    args = parser.parse_args()

    cfg = Config(
        seam_nodes_csv=args.seam_nodes_csv,
        routes_csv=args.routes_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        min_exit_gain=args.min_exit_gain,
        direction_bin_deg=args.direction_bin_deg,
        top_k_vectors=args.top_k_vectors,
        min_committed_offsteps=args.min_committed_offsteps,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seam_nodes, routes = load_inputs(cfg)
    escape_steps = compute_escape_steps(routes, cfg)
    summary_df = summarize_classes(escape_steps)

    steps_csv = outdir / "seam_escape_steps.csv"
    class_csv = outdir / "seam_escape_class_summary.csv"
    txt_path = outdir / "obs029_seam_escape_channels_summary.txt"
    png_path = outdir / "obs029_seam_escape_channels_figure.png"

    escape_steps.to_csv(steps_csv, index=False)
    summary_df.to_csv(class_csv, index=False)
    txt_path.write_text(build_summary(summary_df), encoding="utf-8")
    render_figure(seam_nodes, escape_steps, summary_df, png_path, cfg)

    print(steps_csv)
    print(class_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
