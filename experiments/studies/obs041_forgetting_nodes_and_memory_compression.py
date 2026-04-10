#!/usr/bin/env python3
"""
OBS-041 — Forgetting nodes and memory compression.

Purpose
-------
Turn the forgetting-node signal from OBS-040b into a first-class observation.

Core question
-------------
Which seam motifs are suffix-sufficient, i.e. where adding an older step
does not materially improve prediction over the trailing suffix?

Interpretation
--------------
If A->B->C adds little over B->C, then B can act as a:
- forgetting node
- Markov bottleneck
- memory-compression state

This study measures that effect family by family.

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs029_seam_escape_channels/seam_escape_steps.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs041_forgetting_nodes_and_memory_compression/
  forgetting_node_candidates.csv
  memory_compression_summary.csv
  motif_suffix_comparison.csv
  obs041_forgetting_nodes_and_memory_compression_summary.txt
  obs041_forgetting_nodes_and_memory_compression_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    crossings_csv: str = (
        "outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv"
    )
    steps_csv: str = (
        "outputs/obs029_seam_escape_channels/seam_escape_steps.csv"
    )
    assignments_csv: str = (
        "outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv"
    )
    outdir: str = "outputs/obs041_forgetting_nodes_and_memory_compression"
    seam_threshold: float = 0.15
    random_state: int = 42
    max_k: int = 5
    min_count: int = 2
    forgetting_gain_threshold: float = 0.02


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

TEXT_COLS = {
    "route_class",
    "crossing_type",
    "generator_1",
    "generator_2",
    "src1",
    "tgt1",
    "src2",
    "tgt2",
    "sector_1",
    "sector_2",
    "composition_typed",
    "path_family",
    "path_id",
    "route_id",
    "family",
    "from_hotspot_class",
    "theta_bin",
    "generator_completed",
    "state_a_red",
    "state_c_red",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if col not in TEXT_COLS:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass
    return df


def safe_mean(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    return float(x.mean()) if x.notna().any() else float("nan")


def detect_id_col(df: pd.DataFrame) -> str:
    for col in ["path_id", "route_id", "trajectory_id", "id"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect path identifier column.")


def detect_step_col(df: pd.DataFrame) -> str:
    for col in ["step", "t", "time", "idx"]:
        if col in df.columns:
            return col
    raise ValueError("Could not detect step/time column.")


def infer_sector(row: pd.Series, seam_threshold: float) -> str:
    hotspot = str(row.get("from_hotspot_class", ""))
    if hotspot in {"anisotropy_only", "relational_only", "shared"}:
        return "core"
    if hotspot == "non_hotspot":
        return "escape"

    d2s = pd.to_numeric(row.get("from_distance_to_seam", np.nan), errors="coerce")
    if pd.notna(d2s):
        return "core" if d2s <= seam_threshold else "escape"

    committed = pd.to_numeric(row.get("is_committed_escape", np.nan), errors="coerce")
    escaped = pd.to_numeric(row.get("is_escape_step", np.nan), errors="coerce")
    if pd.notna(committed) and committed == 1:
        return "escape"
    if pd.notna(escaped) and escaped == 1:
        return "escape"

    return "core"


def build_launch_pool(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments.copy()
    return pd.DataFrame(
        {
            "route_class": a["route_class"].astype(str),
            "generator_completed": a["generator_completed"].astype(str),
            "state_a_red": a["state_a_red"].astype(str),
            "state_c_red": a["state_c_red"].astype(str),
            "relational_a": pd.to_numeric(a.get("relational_a", np.nan), errors="coerce"),
            "anisotropy_a": pd.to_numeric(a.get("anisotropy_a", np.nan), errors="coerce"),
            "distance_a": pd.to_numeric(a.get("distance_a", np.nan), errors="coerce"),
        }
    )


def build_step_context(steps: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pid = detect_id_col(steps)
    step_col = detect_step_col(steps)

    s = steps.copy()
    s["route_class"] = s.get("route_class", s.get("path_family", s.get("family", "unknown"))).astype(str)
    s["sector_est"] = s.apply(lambda r: infer_sector(r, cfg.seam_threshold), axis=1)
    s["hotspot_est"] = s.get("from_hotspot_class", pd.Series([""] * len(s))).astype(str)
    s["theta_bin"] = s.get("theta_bin", pd.Series([""] * len(s))).astype(str)

    keep = [
        pid,
        step_col,
        "route_class",
        "sector_est",
        "hotspot_est",
        "theta_bin",
        "from_relational",
        "from_anisotropy",
        "from_distance_to_seam",
    ]
    return s[keep].copy()


def sample_launch_candidates(crossing_row: pd.Series, launch_pool: pd.DataFrame, n_rep: int, random_state: int) -> pd.DataFrame:
    fam = str(crossing_row["route_class"])
    gen = str(crossing_row["generator_1"])
    src = str(crossing_row["src1"])
    tgt = str(crossing_row["tgt1"])

    sub = launch_pool[
        (launch_pool["route_class"] == fam)
        & (launch_pool["generator_completed"] == gen)
        & (launch_pool["state_a_red"] == src)
        & (launch_pool["state_c_red"] == tgt)
    ].copy()

    if len(sub) == 0:
        sub = launch_pool[
            (launch_pool["route_class"] == fam)
            & (launch_pool["generator_completed"] == gen)
        ].copy()

    if len(sub) == 0:
        return sub

    return sub.sample(n=n_rep, replace=(len(sub) < n_rep), random_state=random_state).reset_index(drop=True)


def make_history_words(hist: pd.DataFrame, k: int) -> dict[str, str]:
    tail = hist.tail(k).reset_index(drop=True) if k > 0 else hist.iloc[0:0]
    missing = max(0, k - len(tail))

    sectors = ["NONE"] * missing + tail["sector_est"].astype(str).tolist()
    hotspots = ["NONE"] * missing + tail["hotspot_est"].astype(str).tolist()
    thetas = ["NONE"] * missing + tail["theta_bin"].astype(str).tolist()

    out: dict[str, str] = {}
    out[f"sector_word_{k}"] = "|".join(sectors) if k > 0 else "K0"
    out[f"hotspot_word_{k}"] = "|".join(hotspots) if k > 0 else "K0"
    out[f"theta_word_{k}"] = "|".join(thetas) if k > 0 else "K0"
    return out


def build_event_dataset_for_family(
    family: str,
    crossings: pd.DataFrame,
    step_context: pd.DataFrame,
    launch_pool: pd.DataFrame,
    raw_steps: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    pid = detect_id_col(raw_steps)
    step_col = detect_step_col(raw_steps)

    use = crossings[
        (crossings["route_class"].astype(str) == family)
        & (crossings["crossing_type"].isin(["core_internal", "core_to_escape"]))
    ].copy()
    use["y_cross"] = (use["crossing_type"] == "core_to_escape").astype(int)

    step_context = step_context[step_context["route_class"] == family].copy()
    raw_steps = raw_steps[raw_steps["route_class"].astype(str) == family].copy()

    rows = []
    for i, row in use.reset_index(drop=True).iterrows():
        n_rep = int(round(float(row["n_compositions"]))) if pd.notna(row["n_compositions"]) else 1
        n_rep = max(n_rep, 1)

        sampled_launch = sample_launch_candidates(row, launch_pool, n_rep, cfg.random_state + i)
        if len(sampled_launch) == 0:
            continue

        context_candidates = step_context.copy()
        if pid in row.index and pd.notna(row[pid]):
            context_candidates = context_candidates[context_candidates[pid].astype(str) == str(row[pid])]

        if len(context_candidates) == 0:
            continue

        sampled_context = context_candidates.sample(
            n=n_rep,
            replace=(len(context_candidates) < n_rep),
            random_state=cfg.random_state + 1000 + i,
        ).reset_index(drop=True)

        for j in range(n_rep):
            l = sampled_launch.iloc[j]
            c = sampled_context.iloc[j]

            path_value = c[pid]
            current_step = c[step_col]
            hist = step_context[(step_context[pid] == path_value) & (step_context[step_col] < current_step)].sort_values(step_col)

            row_out = {
                "route_class": family,
                "crossing_type": str(row["crossing_type"]),
                "y_cross": int(row["y_cross"]),
                "prev_generator": str(row["generator_1"]),
                "prev_state": str(row["src1"]) if pd.notna(row["src1"]) else str(l["state_a_red"]),
                "prev_target": str(row["tgt1"]) if pd.notna(row["tgt1"]) else str(l["state_c_red"]),
                "mean_relational": pd.to_numeric(c.get("from_relational", l["relational_a"]), errors="coerce"),
                "mean_anisotropy": pd.to_numeric(c.get("from_anisotropy", l["anisotropy_a"]), errors="coerce"),
                "mean_distance": pd.to_numeric(c.get("from_distance_to_seam", l["distance_a"]), errors="coerce"),
            }
            for k in range(2, cfg.max_k + 1):
                row_out.update(make_history_words(hist, k))
            rows.append(row_out)

    return pd.DataFrame(rows)


def compare_word_vs_suffix(df: pd.DataFrame, family: str, word_col: str) -> pd.DataFrame:
    if len(df) == 0 or word_col not in df.columns:
        return pd.DataFrame(columns=[
            "route_class", "word_col", "history_word", "suffix_word", "count",
            "cross_rate_word", "cross_rate_suffix", "gain_over_suffix",
            "middle_state", "family_count_rank"
        ])

    k = int(word_col.split("_")[-1])
    if k < 2:
        raise ValueError("Need k>=2 for suffix comparison.")

    work = df[[word_col, "y_cross"]].copy()
    work["history_word"] = work[word_col].astype(str)
    work["parts"] = work["history_word"].str.split("|")
    work["suffix_word"] = work["parts"].apply(lambda x: "|".join(x[-(k-1):]) if isinstance(x, list) and len(x) >= (k - 1) else "NONE")
    work["middle_state"] = work["parts"].apply(lambda x: x[-2] if isinstance(x, list) and len(x) >= 2 else "NONE")

    gk = work.groupby(["history_word", "suffix_word", "middle_state"], as_index=False).agg(
        count=("y_cross", "size"),
        cross_rate_word=("y_cross", "mean"),
    )
    gs = work.groupby("suffix_word", as_index=False).agg(
        cross_rate_suffix=("y_cross", "mean"),
    )

    out = gk.merge(gs, on="suffix_word", how="left")
    out["gain_over_suffix"] = (out["cross_rate_word"] - out["cross_rate_suffix"]).abs()
    out["route_class"] = family
    out["word_col"] = word_col
    out["family_count_rank"] = out["count"].rank(method="dense", ascending=False)
    return out.sort_values(["gain_over_suffix", "count"], ascending=[True, False]).reset_index(drop=True)


def build_forgetting_candidates(comparisons: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if len(comparisons) == 0:
        return comparisons

    out = comparisons[
        (comparisons["count"] >= cfg.min_count)
        & (comparisons["gain_over_suffix"] <= cfg.forgetting_gain_threshold)
    ].copy()

    grouped = (
        out.groupby(["route_class", "middle_state"], as_index=False)
        .agg(
            n_candidate_words=("history_word", "nunique"),
            total_count=("count", "sum"),
            mean_gain=("gain_over_suffix", "mean"),
            top_example=("history_word", "first"),
            suffix_example=("suffix_word", "first"),
        )
        .sort_values(["route_class", "n_candidate_words", "total_count"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return grouped


def build_memory_compression_summary(comparisons: pd.DataFrame, family_rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for fam in CLASS_ORDER:
        fam_comp = comparisons[comparisons["route_class"] == fam].copy()
        if len(fam_comp) == 0:
            rows.append({
                "route_class": fam,
                "n_comparisons": 0,
                "forgetting_share": np.nan,
                "mean_gain_over_suffix": np.nan,
                "top_middle_state": np.nan,
                "top_middle_state_count": np.nan,
            })
            continue

        forgetting = fam_comp[
            (fam_comp["count"] >= cfg.min_count)
            & (fam_comp["gain_over_suffix"] <= cfg.forgetting_gain_threshold)
        ].copy()

        top_state = np.nan
        top_state_count = np.nan
        if len(forgetting):
            mid = forgetting.groupby("middle_state", as_index=False).agg(n=("history_word", "nunique"))
            mid = mid.sort_values("n", ascending=False).reset_index(drop=True)
            top_state = mid.iloc[0]["middle_state"]
            top_state_count = int(mid.iloc[0]["n"])

        rows.append({
            "route_class": fam,
            "n_comparisons": int(len(fam_comp)),
            "forgetting_share": float(len(forgetting) / len(fam_comp)),
            "mean_gain_over_suffix": float(fam_comp["gain_over_suffix"].mean()),
            "top_middle_state": top_state,
            "top_middle_state_count": top_state_count,
        })

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    out["order"] = out["route_class"].map(lambda x: order.get(x, 999))
    return out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def build_summary(memory_summary: pd.DataFrame, candidates: pd.DataFrame, motif_comp: pd.DataFrame) -> str:
    lines = [
        "=== OBS-041 Forgetting Nodes and Memory Compression Summary ===",
        "",
        "Interpretive guide",
        "- forgetting nodes are middle states where longer motif words add little over suffixes",
        "- forgetting_share measures how often family motifs are suffix-sufficient",
        "- top_middle_state is the dominant candidate compression state in each family",
        "",
        "Family memory-compression summaries",
    ]

    for _, row in memory_summary.iterrows():
        lines.extend([
            f"{row['route_class']}",
            f"  n_comparisons          = {int(row['n_comparisons'])}",
            f"  forgetting_share       = {float(row['forgetting_share']):.4f}",
            f"  mean_gain_over_suffix  = {float(row['mean_gain_over_suffix']):.4f}",
            f"  top_middle_state       = {row['top_middle_state']}",
            f"  top_middle_state_count = {row['top_middle_state_count']}",
            "",
        ])

    lines.append("Top forgetting-node candidates")
    for fam in CLASS_ORDER:
        sub = candidates[candidates["route_class"] == fam].head(5)
        lines.append(f"  {fam}")
        if len(sub) == 0:
            lines.append("    none")
        else:
            for _, r in sub.iterrows():
                lines.append(
                    f"    middle={r['middle_state']} | n_words={int(r['n_candidate_words'])} | "
                    f"count={int(r['total_count'])} | mean_gain={float(r['mean_gain']):.4f} | "
                    f"example={r['top_example']} -> {r['suffix_example']}"
                )

    lines.append("")
    lines.append("Selected motif-vs-suffix comparisons")
    for fam in CLASS_ORDER:
        sub = motif_comp[motif_comp["route_class"] == fam].head(5)
        lines.append(f"  {fam}")
        if len(sub) == 0:
            lines.append("    none")
        else:
            for _, r in sub.iterrows():
                lines.append(
                    f"    {r['history_word']} -> {r['suffix_word']} | "
                    f"count={int(r['count'])} | gain={float(r['gain_over_suffix']):.4f}"
                )

    return "\n".join(lines)


def render_figure(memory_summary: pd.DataFrame, candidates: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_share = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    x = np.arange(len(memory_summary))
    ax_share.bar(x, memory_summary["forgetting_share"])
    ax_share.set_xticks(x)
    ax_share.set_xticklabels(memory_summary["route_class"], rotation=12)
    ax_share.set_title("Forgetting-share by family", fontsize=14, pad=8)
    ax_share.grid(alpha=0.15, axis="y")

    ax_gain.bar(x, memory_summary["mean_gain_over_suffix"])
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels(memory_summary["route_class"], rotation=12)
    ax_gain.set_title("Mean gain over suffix", fontsize=14, pad=8)
    ax_gain.grid(alpha=0.15, axis="y")

    ax_state.axis("off")
    y = 0.95
    for _, row in memory_summary.iterrows():
        ax_state.text(0.02, y, str(row["route_class"]), fontsize=12, fontweight="bold")
        y -= 0.06
        ax_state.text(0.04, y, f"top state: {row['top_middle_state']}", fontsize=10, family="monospace")
        y -= 0.045
        ax_state.text(0.04, y, f"count: {row['top_middle_state_count']}", fontsize=10, family="monospace")
        y -= 0.07
    ax_state.set_title("Top compression states", fontsize=14, pad=8)

    ax_text.axis("off")
    y = 0.95
    for fam in CLASS_ORDER:
        sub = candidates[candidates["route_class"] == fam].head(4)
        ax_text.text(0.02, y, fam, fontsize=12, fontweight="bold")
        y -= 0.06
        if len(sub) == 0:
            ax_text.text(0.04, y, "none", fontsize=10, family="monospace")
            y -= 0.05
        else:
            for _, r in sub.iterrows():
                ax_text.text(
                    0.04, y,
                    f"{r['middle_state']} | n={int(r['n_candidate_words'])} | gain={float(r['mean_gain']):.4f}",
                    fontsize=9.5, family="monospace"
                )
                y -= 0.045
        y -= 0.04
    ax_text.set_title("Forgetting-node candidates", fontsize=14, pad=8)

    best = memory_summary.sort_values("forgetting_share", ascending=False).iloc[0]
    ax_diag.axis("off")
    text = (
        "OBS-041 diagnostics\n\n"
        f"highest forgetting-share:\n{best['route_class']}\n{best['forgetting_share']:.3f}\n\n"
        f"top compression state:\n{best['top_middle_state']}\n\n"
        "Interpretation:\n"
        "tests whether some seam\n"
        "states compress history so\n"
        "that longer motifs add little\n"
        "over trailing suffixes."
    )
    ax_diag.text(
        0.02, 0.98, text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-041 forgetting nodes and memory compression", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify forgetting nodes and memory-compression motifs.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--steps-csv", default=Config.steps_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--max-k", type=int, default=Config.max_k)
    parser.add_argument("--min-count", type=int, default=Config.min_count)
    parser.add_argument("--forgetting-gain-threshold", type=float, default=Config.forgetting_gain_threshold)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        steps_csv=args.steps_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        seam_threshold=args.seam_threshold,
        random_state=args.random_state,
        max_k=args.max_k,
        min_count=args.min_count,
        forgetting_gain_threshold=args.forgetting_gain_threshold,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    raw_steps = load_csv(cfg.steps_csv)
    assignments = load_csv(cfg.assignments_csv)

    raw_steps["route_class"] = raw_steps.get("route_class", raw_steps.get("path_family", "unknown")) \
        .astype(str)
    step_context = build_step_context(raw_steps, cfg)
    launch_pool = build_launch_pool(assignments)

    comp_rows = []
    for fam in CLASS_ORDER:
        ds = build_event_dataset_for_family(fam, crossings, step_context, launch_pool, raw_steps, cfg)
        for k in range(3, cfg.max_k + 1):
            word_col = f"sector_word_{k}"
            comp = compare_word_vs_suffix(ds, fam, word_col)
            comp_rows.append(comp)

    motif_comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame()
    candidates = build_forgetting_candidates(motif_comp, cfg)
    memory_summary = build_memory_compression_summary(motif_comp, candidates, cfg)

    candidates_csv = outdir / "forgetting_node_candidates.csv"
    summary_csv = outdir / "memory_compression_summary.csv"
    comp_csv = outdir / "motif_suffix_comparison.csv"
    txt_path = outdir / "obs041_forgetting_nodes_and_memory_compression_summary.txt"
    png_path = outdir / "obs041_forgetting_nodes_and_memory_compression_figure.png"

    candidates.to_csv(candidates_csv, index=False)
    memory_summary.to_csv(summary_csv, index=False)
    motif_comp.to_csv(comp_csv, index=False)
    txt_path.write_text(build_summary(memory_summary, candidates, motif_comp), encoding="utf-8")
    render_figure(memory_summary, candidates, png_path)

    print(candidates_csv)
    print(summary_csv)
    print(comp_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
