#!/usr/bin/env python3
"""
OBS-040b — Memory-scale stress test.

Purpose
-------
Stress-test the temporal-depth result from OBS-040.

This study asks whether the observed finite memory horizons are:
- robust dynamical properties
or
- artifacts of sparse long-history encoding

It does this in three ways:
1. pushes k beyond the original ceiling
2. compares multiple history encodings
3. tests for "forgetting nodes" / Markov bottlenecks

Encodings
---------
1. exact_word
   exact sector-history word over last k steps

2. motif_backoff
   exact word for frequent histories, OTHER for tail, plus suffix/backoff features

3. aggregate
   no exact words; only dense summaries over last k steps
   (switch counts, entropy, run lengths, escape contact, etc.)

Families
--------
- branch_exit
- stable_seam_corridor
- reorganization_heavy

Inputs
------
outputs/obs034_core_to_escape_boundary/core_to_escape_crossings.csv
outputs/obs029_seam_escape_channels/seam_escape_steps.csv
outputs/obs030e_complete_generator_basis/completed_generator_assignments.csv

Outputs
-------
outputs/obs040b_memory_scale_stress_test/
  stress_test_metrics.csv
  stress_test_coefficients.csv
  stress_test_forgetting_nodes.csv
  stress_test_support_diagnostics.csv
  obs040b_memory_scale_stress_test_summary.txt
  obs040b_memory_scale_stress_test_figure.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
    outdir: str = "outputs/obs040b_memory_scale_stress_test"
    max_k: int = 10
    recent_window: int = 5
    random_state: int = 42
    seam_threshold: float = 0.15
    min_word_count: int = 3
    saturation_tol: float = 0.02


CLASS_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

ENCODINGS = [
    "exact_word",
    "motif_backoff",
    "aggregate",
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


def shannon_entropy(labels: pd.Series) -> float:
    if len(labels) == 0:
        return 0.0
    p = labels.value_counts(normalize=True, dropna=True).to_numpy(dtype=float)
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p + 1e-12)).sum())


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

    out_rows = []
    for _, grp in s.groupby(pid, sort=False):
        grp = grp.sort_values(step_col).reset_index(drop=True)

        grp["core_flag"] = (grp["sector_est"] == "core").astype(int)
        grp["escape_flag"] = (grp["sector_est"] == "escape").astype(int)

        grp["cum_core_before"] = grp["core_flag"].cumsum().shift(fill_value=0)
        grp["cum_escape_before"] = grp["escape_flag"].cumsum().shift(fill_value=0)
        grp["core_share_before"] = grp["cum_core_before"] / np.maximum(np.arange(len(grp)), 1)
        grp["escape_share_before"] = grp["cum_escape_before"] / np.maximum(np.arange(len(grp)), 1)
        grp["escape_touched_before"] = (grp["cum_escape_before"] > 0).astype(int)

        grp["recent_sector_entropy"] = 0.0
        grp["recent_hotspot_entropy"] = 0.0
        grp["recent_theta_entropy"] = 0.0
        grp["recent_switch_count"] = 0
        grp["recent_escape_count"] = 0

        for i in range(len(grp)):
            lo = max(0, i - cfg.recent_window)
            hist = grp.iloc[lo:i]
            grp.loc[i, "recent_sector_entropy"] = shannon_entropy(hist["sector_est"])
            grp.loc[i, "recent_hotspot_entropy"] = shannon_entropy(hist["hotspot_est"])
            grp.loc[i, "recent_theta_entropy"] = shannon_entropy(hist["theta_bin"])
            if len(hist) >= 2:
                grp.loc[i, "recent_switch_count"] = int((hist["sector_est"].shift() != hist["sector_est"]).sum() - 1)
            grp.loc[i, "recent_escape_count"] = int((hist["sector_est"] == "escape").sum())

        grp["runlen_core_before"] = 0
        grp["runlen_escape_before"] = 0
        core_run = 0
        esc_run = 0
        prev_sector = None
        for i in range(len(grp)):
            grp.loc[i, "runlen_core_before"] = core_run
            grp.loc[i, "runlen_escape_before"] = esc_run
            cur = grp.loc[i, "sector_est"]
            if cur == prev_sector == "core":
                core_run += 1
            elif cur == "core":
                core_run = 1
                esc_run = 0
            elif cur == prev_sector == "escape":
                esc_run += 1
            else:
                esc_run = 1
                core_run = 0
            prev_sector = cur

        keep_cols = [
            pid,
            step_col,
            "route_class",
            "sector_est",
            "hotspot_est",
            "theta_bin",
            "from_relational",
            "from_anisotropy",
            "from_distance_to_seam",
            "cum_core_before",
            "cum_escape_before",
            "core_share_before",
            "escape_share_before",
            "escape_touched_before",
            "recent_sector_entropy",
            "recent_hotspot_entropy",
            "recent_theta_entropy",
            "recent_switch_count",
            "recent_escape_count",
            "runlen_core_before",
            "runlen_escape_before",
        ]
        out_rows.append(grp[keep_cols])

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


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


def history_features(hist: pd.DataFrame, k: int) -> dict[str, object]:
    tail = hist.tail(k).reset_index(drop=True) if k > 0 else hist.iloc[0:0]
    missing = max(0, k - len(tail))

    sectors = ["NONE"] * missing + tail["sector_est"].astype(str).tolist()
    hotspots = ["NONE"] * missing + tail["hotspot_est"].astype(str).tolist()
    thetas = ["NONE"] * missing + tail["theta_bin"].astype(str).tolist()

    out: dict[str, object] = {}
    out["history_sector_word"] = "|".join(sectors) if k > 0 else "K0"
    out["history_hotspot_word"] = "|".join(hotspots) if k > 0 else "K0"
    out["history_theta_word"] = "|".join(thetas) if k > 0 else "K0"

    out["history_sector_suffix2"] = "|".join(sectors[-2:]) if k >= 2 else out["history_sector_word"]
    out["history_sector_suffix3"] = "|".join(sectors[-3:]) if k >= 3 else out["history_sector_word"]

    out["k_switch_count"] = 0 if len(tail) < 2 else int((tail["sector_est"].shift() != tail["sector_est"]).sum() - 1)
    out["k_escape_count"] = int((tail["sector_est"] == "escape").sum())
    out["k_core_count"] = int((tail["sector_est"] == "core").sum())
    out["k_sector_entropy"] = shannon_entropy(tail["sector_est"])
    out["k_hotspot_entropy"] = shannon_entropy(tail["hotspot_est"])
    out["k_theta_entropy"] = shannon_entropy(tail["theta_bin"])
    out["k_recent_last_sector"] = sectors[-1] if len(sectors) else "NONE"
    out["k_recent_last_hotspot"] = hotspots[-1] if len(hotspots) else "NONE"
    out["k_recent_last_theta"] = thetas[-1] if len(thetas) else "NONE"
    return out


def build_dataset_for_family(
    family: str,
    crossings: pd.DataFrame,
    step_context: pd.DataFrame,
    launch_pool: pd.DataFrame,
    raw_steps: pd.DataFrame,
    k: int,
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
            h = history_features(hist, k)

            rows.append(
                {
                    "route_class": family,
                    "k": k,
                    "crossing_type": str(row["crossing_type"]),
                    "y_cross": int(row["y_cross"]),
                    "prev_generator": str(row["generator_1"]),
                    "prev_state": str(row["src1"]) if pd.notna(row["src1"]) else str(l["state_a_red"]),
                    "prev_target": str(row["tgt1"]) if pd.notna(row["tgt1"]) else str(l["state_c_red"]),
                    "mean_relational": pd.to_numeric(c.get("from_relational", l["relational_a"]), errors="coerce"),
                    "mean_anisotropy": pd.to_numeric(c.get("from_anisotropy", l["anisotropy_a"]), errors="coerce"),
                    "mean_distance": pd.to_numeric(c.get("from_distance_to_seam", l["distance_a"]), errors="coerce"),
                    "cum_core_before": c["cum_core_before"],
                    "cum_escape_before": c["cum_escape_before"],
                    "core_share_before": c["core_share_before"],
                    "escape_share_before": c["escape_share_before"],
                    "escape_touched_before": c["escape_touched_before"],
                    "recent_sector_entropy": c["recent_sector_entropy"],
                    "recent_hotspot_entropy": c["recent_hotspot_entropy"],
                    "recent_theta_entropy": c["recent_theta_entropy"],
                    "recent_switch_count": c["recent_switch_count"],
                    "recent_escape_count": c["recent_escape_count"],
                    "runlen_core_before": c["runlen_core_before"],
                    "runlen_escape_before": c["runlen_escape_before"],
                    **h,
                }
            )

    return pd.DataFrame(rows)


def apply_backoff(words: pd.Series, min_word_count: int) -> tuple[pd.Series, pd.DataFrame]:
    counts = words.value_counts(dropna=False)
    keep = set(counts[counts >= min_word_count].index.tolist())
    mapped = words.apply(lambda x: x if x in keep else "OTHER")
    diag = pd.DataFrame(
        {
            "word": counts.index.astype(str),
            "count": counts.values,
            "kept": [w in keep for w in counts.index],
        }
    )
    return mapped, diag


def build_design(df: pd.DataFrame, encoding: str, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    numeric_cols = [
        "mean_relational",
        "mean_anisotropy",
        "mean_distance",
        "cum_core_before",
        "cum_escape_before",
        "core_share_before",
        "escape_share_before",
        "escape_touched_before",
        "recent_sector_entropy",
        "recent_hotspot_entropy",
        "recent_theta_entropy",
        "recent_switch_count",
        "recent_escape_count",
        "runlen_core_before",
        "runlen_escape_before",
        "k_switch_count",
        "k_escape_count",
        "k_core_count",
        "k_sector_entropy",
        "k_hotspot_entropy",
        "k_theta_entropy",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median() if work[col].notna().any() else 0.0)

    diag_rows = []

    if encoding == "exact_word":
        cat_cols = [
            "prev_generator",
            "prev_state",
            "prev_target",
            "history_sector_word",
            "history_theta_word",
            "history_hotspot_word",
        ]
        X = pd.get_dummies(work[numeric_cols + cat_cols], columns=cat_cols, drop_first=False, dtype=float)

    elif encoding == "motif_backoff":
        mapped_word, diag = apply_backoff(work["history_sector_word"].astype(str), cfg.min_word_count)
        work["history_sector_word_backoff"] = mapped_word
        diag["encoding"] = encoding
        diag_rows.append(diag)

        cat_cols = [
            "prev_generator",
            "prev_state",
            "prev_target",
            "history_sector_word_backoff",
            "history_sector_suffix2",
            "history_sector_suffix3",
            "k_recent_last_sector",
        ]
        X = pd.get_dummies(work[numeric_cols + cat_cols], columns=cat_cols, drop_first=False, dtype=float)

    elif encoding == "aggregate":
        cat_cols = [
            "prev_generator",
            "prev_state",
            "prev_target",
            "k_recent_last_sector",
            "k_recent_last_hotspot",
            "k_recent_last_theta",
        ]
        X = pd.get_dummies(work[numeric_cols + cat_cols], columns=cat_cols, drop_first=False, dtype=float)

    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    diag_df = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame(columns=["word", "count", "kept", "encoding"])
    return X, diag_df


def fit_one(df: pd.DataFrame, encoding: str, cfg: Config) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    X, diag_df = build_design(df, encoding, cfg)
    y = df["y_cross"].to_numpy(dtype=int)

    support = {
        "n_rows": float(len(df)),
        "n_features": float(X.shape[1]),
        "n_unique_histories": float(df["history_sector_word"].nunique(dropna=False)),
        "top_history_mass": float(df["history_sector_word"].value_counts(normalize=True, dropna=False).iloc[0]) if len(df) else float("nan"),
        "other_mass": float((diag_df["kept"] == False).sum() / max(len(diag_df), 1)) if len(diag_df) else float("nan"),
    }

    if len(df) == 0 or len(np.unique(y)) < 2:
        coef = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
        metrics = {
            "auc": float("nan"),
            "intercept": float("nan"),
            "n_rows": float(len(df)),
            "positive_rate": float(df["y_cross"].mean()) if len(df) else float("nan"),
            **support,
        }
        return coef, metrics, diag_df

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        random_state=42,
        max_iter=7000,
    )
    model.fit(X.to_numpy(dtype=float), y)
    probs = model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    auc = float(roc_auc_score(y, probs))

    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "intercept": float(model.intercept_[0]),
        "n_rows": float(len(df)),
        "positive_rate": float(df["y_cross"].mean()),
        **support,
    }
    return coef, metrics, diag_df


def forgetting_node_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["history3", "suffix2", "count", "cross_rate_3", "cross_rate_2", "gain_over_suffix"])

    work = df.copy()
    work["history3"] = work["history_sector_word"].astype(str)
    work["suffix2"] = work["history_sector_word"].astype(str).apply(lambda s: "|".join(s.split("|")[-2:]) if "|" in s else s)

    g3 = work.groupby("history3", as_index=False).agg(
        count=("y_cross", "size"),
        cross_rate_3=("y_cross", "mean"),
        suffix2=("suffix2", "first"),
    )

    g2 = work.groupby("suffix2", as_index=False).agg(
        cross_rate_2=("y_cross", "mean"),
    )

    out = g3.merge(g2, on="suffix2", how="left")
    out["gain_over_suffix"] = (out["cross_rate_3"] - out["cross_rate_2"]).abs()
    return out.sort_values(["gain_over_suffix", "count"], ascending=[True, False]).reset_index(drop=True)


def summarize_family(metrics_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for fam in CLASS_ORDER:
        for enc in ENCODINGS:
            sub = metrics_df[(metrics_df["route_class"] == fam) & (metrics_df["encoding"] == enc)].sort_values("k")
            if len(sub) == 0 or sub["auc"].notna().sum() == 0:
                rows.append(
                    {
                        "route_class": fam,
                        "encoding": enc,
                        "best_k": np.nan,
                        "best_auc": np.nan,
                        "k0_auc": np.nan,
                        "delta_best_vs_k0": np.nan,
                        "saturation_k": np.nan,
                    }
                )
                continue

            best_idx = sub["auc"].idxmax()
            best_row = sub.loc[best_idx]
            k0 = sub[sub["k"] == 0]["auc"]
            k0_auc = float(k0.iloc[0]) if len(k0) else float("nan")
            best_auc = float(best_row["auc"])
            best_k = int(best_row["k"])

            saturation_k = np.nan
            for _, r in sub.iterrows():
                if pd.notna(r["auc"]) and (best_auc - float(r["auc"]) <= cfg.saturation_tol):
                    saturation_k = int(r["k"])
                    break

            rows.append(
                {
                    "route_class": fam,
                    "encoding": enc,
                    "best_k": best_k,
                    "best_auc": best_auc,
                    "k0_auc": k0_auc,
                    "delta_best_vs_k0": best_auc - k0_auc if pd.notna(k0_auc) else np.nan,
                    "saturation_k": saturation_k,
                }
            )

    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CLASS_ORDER)}
    enc_order = {k: i for i, k in enumerate(ENCODINGS)}
    out["order1"] = out["route_class"].map(lambda x: order.get(x, 999))
    out["order2"] = out["encoding"].map(lambda x: enc_order.get(x, 999))
    return out.sort_values(["order1", "order2"]).drop(columns=["order1", "order2"]).reset_index(drop=True)


def build_summary(metrics_df: pd.DataFrame, family_summary: pd.DataFrame, forgetting_df: pd.DataFrame) -> str:
    lines = [
        "=== OBS-040b Memory-Scale Stress Test Summary ===",
        "",
        "Interpretive guide",
        "- compares exact, backoff, and aggregate history encodings",
        "- tests whether horizon saturation survives richer long-history representations",
        "- forgetting-node candidates are 3-step histories with little gain over their 2-step suffix",
        "",
        "Family / encoding summaries",
    ]

    for fam in CLASS_ORDER:
        lines.append(f"{fam}")
        sub = family_summary[family_summary["route_class"] == fam]
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['encoding']}: "
                f"k0={float(r['k0_auc']):.4f}, "
                f"best_k={int(r['best_k']) if pd.notna(r['best_k']) else 'nan'}, "
                f"best_auc={float(r['best_auc']):.4f}, "
                f"delta={float(r['delta_best_vs_k0']):.4f}, "
                f"sat_k={int(r['saturation_k']) if pd.notna(r['saturation_k']) else 'nan'}"
            )
        lines.append("")

    lines.append("Selected support diagnostics")
    for fam in CLASS_ORDER:
        sub = metrics_df[(metrics_df["route_class"] == fam) & (metrics_df["encoding"] == "exact_word")].sort_values("k")
        vals = ", ".join(
            f"k={int(r.k)}:rows={int(r.n_rows)},uniq={int(r.n_unique_histories)},top_mass={float(r.top_history_mass):.3f}"
            for r in sub.itertuples(index=False)
            if pd.notna(r.n_rows)
        )
        lines.append(f"  {fam}: {vals}")

    lines.append("")
    lines.append("Top forgetting-node candidates")
    for fam in CLASS_ORDER:
        sub = forgetting_df[forgetting_df["route_class"] == fam].head(5)
        lines.append(f"  {fam}")
        if len(sub) == 0:
            lines.append("    none")
        else:
            for _, r in sub.iterrows():
                lines.append(
                    f"    {r['history3']} -> suffix {r['suffix2']} | "
                    f"count={int(r['count'])}, gain={float(r['gain_over_suffix']):.4f}"
                )

    return "\n".join(lines)


def render_figure(metrics_df: pd.DataFrame, family_summary: pd.DataFrame, outpath: Path) -> None:
    fig = plt.figure(figsize=(16.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_curve = fig.add_subplot(gs[0, 0])
    ax_best = fig.add_subplot(gs[0, 1])
    ax_sat = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[1, 0:2])
    ax_diag = fig.add_subplot(gs[1, 2])

    for fam in CLASS_ORDER:
        sub = metrics_df[(metrics_df["route_class"] == fam) & (metrics_df["encoding"] == "aggregate")].sort_values("k")
        ax_curve.plot(sub["k"], sub["auc"], marker="o", label=f"{fam} (agg)")
    ax_curve.set_xlabel("History horizon k")
    ax_curve.set_ylabel("AUC")
    ax_curve.set_title("Aggregate-encoding AUC vs horizon", fontsize=14, pad=8)
    ax_curve.grid(alpha=0.2)
    ax_curve.legend()

    best_exact = family_summary[family_summary["encoding"] == "exact_word"].copy()
    x = np.arange(len(best_exact))
    ax_best.bar(x - 0.2, family_summary[family_summary["encoding"] == "exact_word"]["best_auc"], 0.2, label="exact")
    ax_best.bar(x, family_summary[family_summary["encoding"] == "motif_backoff"]["best_auc"], 0.2, label="backoff")
    ax_best.bar(x + 0.2, family_summary[family_summary["encoding"] == "aggregate"]["best_auc"], 0.2, label="aggregate")
    ax_best.set_xticks(x)
    ax_best.set_xticklabels(best_exact["route_class"], rotation=12)
    ax_best.set_title("Best AUC by encoding", fontsize=14, pad=8)
    ax_best.grid(alpha=0.15, axis="y")
    ax_best.legend()

    ax_sat.bar(x - 0.2, family_summary[family_summary["encoding"] == "exact_word"]["saturation_k"], 0.2, label="exact")
    ax_sat.bar(x, family_summary[family_summary["encoding"] == "motif_backoff"]["saturation_k"], 0.2, label="backoff")
    ax_sat.bar(x + 0.2, family_summary[family_summary["encoding"] == "aggregate"]["saturation_k"], 0.2, label="aggregate")
    ax_sat.set_xticks(x)
    ax_sat.set_xticklabels(best_exact["route_class"], rotation=12)
    ax_sat.set_title("Saturation horizon by encoding", fontsize=14, pad=8)
    ax_sat.grid(alpha=0.15, axis="y")
    ax_sat.legend()

    ax_text.axis("off")
    y = 0.95
    for fam in CLASS_ORDER:
        ax_text.text(0.02, y, fam, fontsize=12, fontweight="bold")
        y -= 0.06
        sub = family_summary[family_summary["route_class"] == fam]
        for _, r in sub.iterrows():
            ax_text.text(
                0.04,
                y,
                f"{r['encoding']}: best_k={int(r['best_k']) if pd.notna(r['best_k']) else 'nan'}, "
                f"best_auc={float(r['best_auc']):.3f}, sat_k={int(r['saturation_k']) if pd.notna(r['saturation_k']) else 'nan'}",
                fontsize=9.5,
                family="monospace",
            )
            y -= 0.045
        y -= 0.04
    ax_text.set_title("Family / encoding comparison", fontsize=14, pad=8)

    best = family_summary.sort_values("best_auc", ascending=False).iloc[0]
    ax_diag.axis("off")
    text = (
        "OBS-040b diagnostics\n\n"
        f"best family/enc:\n{best['route_class']}\n{best['encoding']}\nAUC={best['best_auc']:.3f}\n\n"
        f"best horizon:\nk={int(best['best_k']) if pd.notna(best['best_k']) else 'nan'}\n\n"
        "Interpretation:\n"
        "tests whether finite\n"
        "memory horizons survive\n"
        "richer encodings and\n"
        "sparsity stress."
    )
    ax_diag.text(
        0.02, 0.98,
        text,
        va="top", ha="left", fontsize=10.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.85", alpha=0.95),
    )

    fig.suptitle("PAM Observatory — OBS-040b memory-scale stress test", fontsize=21)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test family memory scales with richer history encodings.")
    parser.add_argument("--crossings-csv", default=Config.crossings_csv)
    parser.add_argument("--steps-csv", default=Config.steps_csv)
    parser.add_argument("--assignments-csv", default=Config.assignments_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--max-k", type=int, default=Config.max_k)
    parser.add_argument("--recent-window", type=int, default=Config.recent_window)
    parser.add_argument("--random-state", type=int, default=Config.random_state)
    parser.add_argument("--seam-threshold", type=float, default=Config.seam_threshold)
    parser.add_argument("--min-word-count", type=int, default=Config.min_word_count)
    parser.add_argument("--saturation-tol", type=float, default=Config.saturation_tol)
    args = parser.parse_args()

    cfg = Config(
        crossings_csv=args.crossings_csv,
        steps_csv=args.steps_csv,
        assignments_csv=args.assignments_csv,
        outdir=args.outdir,
        max_k=args.max_k,
        recent_window=args.recent_window,
        random_state=args.random_state,
        seam_threshold=args.seam_threshold,
        min_word_count=args.min_word_count,
        saturation_tol=args.saturation_tol,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    crossings = load_csv(cfg.crossings_csv)
    raw_steps = load_csv(cfg.steps_csv)
    assignments = load_csv(cfg.assignments_csv)

    step_context = build_step_context(raw_steps, cfg)
    launch_pool = build_launch_pool(assignments)
    raw_steps["route_class"] = raw_steps.get("route_class", raw_steps.get("path_family", "unknown")).astype(str)

    metric_rows = []
    coef_rows = []
    diag_rows = []
    forgetting_rows = []

    for fam in CLASS_ORDER:
        for k in range(cfg.max_k + 1):
            ds = build_dataset_for_family(fam, crossings, step_context, launch_pool, raw_steps, k, cfg)
            if len(ds) == 0:
                for enc in ENCODINGS:
                    metric_rows.append(
                        {
                            "route_class": fam,
                            "k": k,
                            "encoding": enc,
                            "auc": np.nan,
                            "intercept": np.nan,
                            "n_rows": 0,
                            "positive_rate": np.nan,
                            "n_features": np.nan,
                            "n_unique_histories": np.nan,
                            "top_history_mass": np.nan,
                            "other_mass": np.nan,
                        }
                    )
                continue

            if k == 3:
                fn = forgetting_node_analysis(ds)
                fn["route_class"] = fam
                forgetting_rows.append(fn)

            for enc in ENCODINGS:
                coef, metrics, diag_df = fit_one(ds, enc, cfg)
                metrics["route_class"] = fam
                metrics["k"] = k
                metrics["encoding"] = enc
                metric_rows.append(metrics)

                if len(coef):
                    coef = coef.copy()
                    coef["route_class"] = fam
                    coef["k"] = k
                    coef["encoding"] = enc
                    coef_rows.append(coef)

                if len(diag_df):
                    diag_df = diag_df.copy()
                    diag_df["route_class"] = fam
                    diag_df["k"] = k
                    diag_rows.append(diag_df)

    metrics_df = pd.DataFrame(metric_rows)
    coef_df = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    diag_df = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    forgetting_df = pd.concat(forgetting_rows, ignore_index=True) if forgetting_rows else pd.DataFrame()
    family_summary = summarize_family(metrics_df, cfg)

    metrics_csv = outdir / "stress_test_metrics.csv"
    coef_csv = outdir / "stress_test_coefficients.csv"
    forgetting_csv = outdir / "stress_test_forgetting_nodes.csv"
    diag_csv = outdir / "stress_test_support_diagnostics.csv"
    txt_path = outdir / "obs040b_memory_scale_stress_test_summary.txt"
    png_path = outdir / "obs040b_memory_scale_stress_test_figure.png"

    metrics_df.to_csv(metrics_csv, index=False)
    coef_df.to_csv(coef_csv, index=False)
    forgetting_df.to_csv(forgetting_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)
    txt_path.write_text(build_summary(metrics_df, family_summary, forgetting_df), encoding="utf-8")
    render_figure(metrics_df, family_summary, png_path)

    print(metrics_csv)
    print(coef_csv)
    print(forgetting_csv)
    print(diag_csv)
    print(txt_path)
    print(png_path)


if __name__ == "__main__":
    main()
