from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LINK_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_corpus_link_map(corpora_py: Path | None) -> dict[str, bool]:
    if corpora_py is None or not corpora_py.exists():
        return {}

    text = corpora_py.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    link_map: dict[str, bool] = {}

    def extract_strings(node: ast.AST) -> list[str]:
        vals: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                vals.append(child.value)
        return vals

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    strings = extract_strings(node.value)
                    joined = "\n".join(strings)
                    if strings:
                        link_map[name] = bool(LINK_RE.search(joined))

    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            d = node.value
            for k, v in zip(d.keys, d.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    strings = extract_strings(v)
                    joined = "\n".join(strings)
                    link_map[k.value] = bool(LINK_RE.search(joined))

    return link_map


def infer_group_col(paths: pd.DataFrame) -> str:
    for candidate in ["probe_id", "trajectory_id", "path_id"]:
        if candidate in paths.columns:
            return candidate
    raise ValueError("Could not find a trajectory grouping column in paths CSV.")


def classify_outcome(
    crossed: bool,
    entered_lazarus: bool,
    phase_flip_count: int,
    min_distance: float,
    graze_threshold: float,
) -> str:
    if crossed and phase_flip_count >= 1:
        return "collapse"
    if entered_lazarus:
        return "lazarus"
    if crossed:
        return "seam_cross"
    if pd.notna(min_distance) and float(min_distance) <= graze_threshold:
        return "seam_graze"
    return "in_basin"


def compute_approach_direction(group: pd.DataFrame, dist_col: str) -> float:
    g = group[[dist_col]].dropna().reset_index(drop=True)
    if len(g) < 2:
        return np.nan
    n = min(3, len(g))
    y = g[dist_col].iloc[:n].to_numpy(dtype=float)
    x = np.arange(n, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def summarize_paths(
    paths: pd.DataFrame,
    metrics: pd.DataFrame | None,
    graze_threshold: float,
    corpus_link_map: dict[str, bool],
) -> pd.DataFrame:
    group_col = infer_group_col(paths)
    work = paths.copy()
    work = safe_numeric(
        work,
        ["r", "alpha", "distance_to_seam", "scalar_curvature", "lazarus_score", "signed_phase", "step"],
    )

    if metrics is not None:
        metrics = metrics.copy()
        metrics = safe_numeric(
            metrics,
            [
                "start_r",
                "start_alpha",
                "end_r",
                "end_alpha",
                "crosses_seam",
                "phase_flip_count",
                "min_distance_to_seam",
                "max_curvature_along_path",
                "lazarus_max",
            ],
        )

    rows = []
    for gid, g in work.groupby(group_col, sort=False):
        g = g.sort_values("step") if "step" in g.columns else g.copy()
        first = g.iloc[0]
        last = g.iloc[-1]

        mrow = None
        if metrics is not None and group_col in metrics.columns:
            sel = metrics[metrics[group_col] == gid]
            if len(sel):
                mrow = sel.iloc[0]

        corpus = first.get("corpus", first.get("family", "unknown"))
        start_r = first.get("r", np.nan)
        start_alpha = first.get("alpha", np.nan)
        seed = first.get("seed", np.nan)
        start_node = first.get("node_id", first.get("start_node", np.nan))

        phase_series = g["signed_phase"].dropna() if "signed_phase" in g.columns else pd.Series(dtype=float)
        crossed_seam = bool(
            (bool(mrow.get("crosses_seam", False)) if mrow is not None and "crosses_seam" in mrow.index else False)
            or ("distance_to_seam" in g.columns and (g["distance_to_seam"].fillna(np.inf) <= 0).any())
            or (len(phase_series) > 1 and (phase_series.diff().fillna(0) != 0).any())
        )

        phase_flip_count = int(
            mrow.get("phase_flip_count", 0)
            if mrow is not None and "phase_flip_count" in mrow.index
            else max(0, int((phase_series.diff().fillna(0) != 0).sum())) if len(phase_series) else 0
        )

        min_distance = float(
            mrow.get("min_distance_to_seam", np.nan)
            if mrow is not None and "min_distance_to_seam" in mrow.index
            else g["distance_to_seam"].min(skipna=True) if "distance_to_seam" in g.columns else np.nan
        )

        max_lazarus = float(
            mrow.get("lazarus_max", np.nan)
            if mrow is not None and "lazarus_max" in mrow.index
            else g["lazarus_score"].max(skipna=True) if "lazarus_score" in g.columns else np.nan
        )

        max_curvature = float(
            mrow.get("max_curvature_along_path", np.nan)
            if mrow is not None and "max_curvature_along_path" in mrow.index
            else g["scalar_curvature"].max(skipna=True) if "scalar_curvature" in g.columns else np.nan
        )

        entered_lazarus = bool(
            (bool(mrow.get("lazarus_hit_any", False)) if mrow is not None and "lazarus_hit_any" in mrow.index else False)
            or ("lazarus_hit" in g.columns and g["lazarus_hit"].fillna(0).astype(int).any())
            or (pd.notna(max_lazarus) and max_lazarus >= 0.5)
        )

        phase_start = mrow.get("phase_start", np.nan) if mrow is not None and "phase_start" in mrow.index else first.get("signed_phase", np.nan)
        phase_end = mrow.get("phase_end", np.nan) if mrow is not None and "phase_end" in mrow.index else last.get("signed_phase", np.nan)
        recovered = bool(crossed_seam and pd.notna(phase_start) and pd.notna(phase_end) and float(phase_start) == float(phase_end))

        operator = first.get("operator", "none")
        operator_time = first.get("operator_time", np.nan)
        operator_strength = first.get("operator_strength", np.nan)

        approach_direction = compute_approach_direction(g, "distance_to_seam") if "distance_to_seam" in g.columns else np.nan
        outcome_class = classify_outcome(crossed_seam, entered_lazarus, phase_flip_count, min_distance, graze_threshold)

        rows.append(
            {
                group_col: gid,
                "corpus": corpus,
                "r": start_r,
                "alpha": start_alpha,
                "seed": seed,
                "start_node": start_node,
                "operator": operator,
                "operator_time": operator_time,
                "operator_strength": operator_strength,
                "outcome_class": outcome_class,
                "entered_lazarus": entered_lazarus,
                "crossed_seam": crossed_seam,
                "phase_flip_count": phase_flip_count,
                "min_distance_to_seam": min_distance,
                "max_lazarus": max_lazarus,
                "max_curvature": max_curvature,
                "recovered": recovered,
                "approach_direction_to_seam": approach_direction,
                "has_link": corpus_link_map.get(str(corpus), np.nan),
            }
        )

    return pd.DataFrame(rows)


def build_outcome_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + ["outcome_class"], dropna=False).size().rename("n").reset_index()
    totals = df.groupby(group_cols, dropna=False).size().rename("n_total").reset_index()
    out = counts.merge(totals, on=group_cols, how="left")
    out["rate"] = out["n"] / out["n_total"]
    return out.sort_values(group_cols + ["outcome_class"]).reset_index(drop=True)


def build_link_geometry_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "has_link" not in work.columns:
        return pd.DataFrame()
    work = work.dropna(subset=["has_link"])
    if len(work) == 0:
        return pd.DataFrame()

    work["has_link"] = work["has_link"].astype(bool)
    summary = (
        work.groupby("has_link", dropna=False)
        .agg(
            n=("outcome_class", "count"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
            mean_max_curvature=("max_curvature", "mean"),
            mean_max_lazarus=("max_lazarus", "mean"),
            crossed_seam_rate=("crossed_seam", "mean"),
            entered_lazarus_rate=("entered_lazarus", "mean"),
        )
        .reset_index()
    )
    return summary


def run_initial_conditions_summary(
    *,
    paths_csv: str | Path,
    metrics_csv: str | Path,
    corpora_py: str | Path | None = "src/corpora.py",
    graze_threshold: float = 0.15,
    outdir: str | Path = "outputs/fim_initial_conditions",
) -> dict[str, str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(paths_csv)
    metrics = pd.read_csv(metrics_csv) if Path(metrics_csv).exists() else None
    corpus_link_map = load_corpus_link_map(Path(corpora_py)) if corpora_py else {}

    ic_df = summarize_paths(paths, metrics, graze_threshold, corpus_link_map)
    initial_conditions_outcomes_csv = outdir / "initial_conditions_outcomes.csv"
    ic_df.to_csv(initial_conditions_outcomes_csv, index=False)

    outcome_summary = build_outcome_summary(ic_df, ["corpus", "r", "alpha"])
    initial_conditions_outcome_summary_csv = outdir / "initial_conditions_outcome_summary.csv"
    outcome_summary.to_csv(initial_conditions_outcome_summary_csv, index=False)

    link_summary = build_link_geometry_summary(ic_df)
    link_geometry_summary_csv = outdir / "link_geometry_summary.csv"
    link_summary.to_csv(link_geometry_summary_csv, index=False)

    return {
        "initial_conditions_outcomes_csv": str(initial_conditions_outcomes_csv),
        "initial_conditions_outcome_summary_csv": str(initial_conditions_outcome_summary_csv),
        "link_geometry_summary_csv": str(link_geometry_summary_csv),
    }
