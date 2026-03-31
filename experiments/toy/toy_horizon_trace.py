#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-8


@dataclass(frozen=True)
class ColumnSpec:
    id_col: str = "id"
    trajectory_col: str = "trajectory_id"
    step_col: str = "step"
    entropy_col: str = "entropy_joint"
    diversity_col: str = "outcome_diversity"
    mode_count_col: str = "mode_count"
    curvature_col: str = "curvature_proxy"
    dominant_col: str = "dominant_fraction"
    seam_distance_col: str = "seam_distance"
    lazarus_score_col: str = "lazarus_score"
    lazarus_hit_col: str = "lazarus_hit"


def clamp01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def safe_normalize(series: pd.Series, *, max_value: Optional[float] = None) -> pd.Series:
    if max_value is None:
        max_value = float(series.max(skipna=True))
    if not np.isfinite(max_value) or max_value <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return clamp01(series.astype(float) / max_value)


def transform_curvature(curvature: pd.Series, *, mode: str = "raw") -> pd.Series:
    c = curvature.astype(float).clip(lower=0.0)
    if mode == "raw":
        return c
    if mode == "log":
        return np.log1p(c)
    if mode == "sqrt":
        return np.sqrt(c)
    raise ValueError(f"Unknown curvature transform mode: {mode}")


def inverse_curvature(curvature: pd.Series, *, mode: str = "raw") -> pd.Series:
    return 1.0 / (1.0 + transform_curvature(curvature, mode=mode))


def log_product_terms(*terms: pd.Series) -> pd.Series:
    acc = pd.Series(np.zeros(len(terms[0])), index=terms[0].index, dtype=float)
    for term in terms:
        acc = acc + np.log(term.astype(float) + EPS)
    return np.exp(acc)


def horizon_components(
    entropy_norm: pd.Series,
    diversity: pd.Series,
    mode_norm: pd.Series,
    inv_curvature: pd.Series,
    seam_proximity: Optional[pd.Series] = None,
) -> pd.DataFrame:
    data = {
        "log_entropy_term": np.log(entropy_norm + EPS),
        "log_diversity_term": np.log(diversity + EPS),
        "log_mode_term": np.log(mode_norm + EPS),
        "log_inverse_curvature_term": np.log(inv_curvature + EPS),
    }
    if seam_proximity is not None:
        data["log_seam_proximity_term"] = np.log(seam_proximity + EPS)
    return pd.DataFrame(data)


def distance_to_horizon(
    entropy_norm: pd.Series,
    diversity: pd.Series,
    mode_norm: pd.Series,
    inv_curvature: pd.Series,
    seam_proximity: Optional[pd.Series] = None,
    *,
    seam_weight: float = 0.0,
) -> pd.Series:
    base = log_product_terms(entropy_norm, diversity, mode_norm, inv_curvature)
    if seam_proximity is None or seam_weight <= 0:
        return base
    seam_factor = (1.0 - seam_weight) + seam_weight * seam_proximity
    return log_product_terms(base, seam_factor)


def horizon_pressure(
    diversity: pd.Series,
    curvature_raw: pd.Series,
    dominant_fraction: pd.Series,
    lazarus_score: Optional[pd.Series] = None,
    *,
    lazarus_weight: float = 0.0,
) -> pd.Series:
    u = clamp01(diversity.astype(float))
    d = clamp01(dominant_fraction.astype(float))
    c = curvature_raw.astype(float).clip(lower=0.0)
    out = c * (1.0 - u) * (1.0 - d)
    if lazarus_score is not None and lazarus_weight > 0:
        out = out * ((1.0 - lazarus_weight) + lazarus_weight * clamp01(lazarus_score.astype(float)))
    return out


def classify_horizon_type(
    diversity: pd.Series,
    dominant_fraction: pd.Series,
    *,
    diversity_threshold: float = 0.2,
    dominance_threshold: float = 0.85,
) -> pd.Series:
    u = clamp01(diversity.astype(float))
    d = clamp01(dominant_fraction.astype(float))
    labels = np.full(len(u), "open", dtype=object)
    labels[(u < diversity_threshold) & (d < dominance_threshold)] = "false_horizon"
    labels[(u < diversity_threshold) & (d >= dominance_threshold)] = "true_horizon"
    return pd.Series(labels, index=diversity.index, dtype="string")


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_dataframe(df: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    sort_cols = []
    if spec.trajectory_col in df.columns:
        sort_cols.append(spec.trajectory_col)
    if spec.step_col in df.columns:
        sort_cols.append(spec.step_col)
    elif spec.id_col in df.columns:
        sort_cols.append(spec.id_col)
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def compute_horizon_metrics(
    df: pd.DataFrame,
    spec: ColumnSpec,
    *,
    entropy_max: Optional[float] = None,
    mode_count_max: Optional[float] = None,
    diversity_threshold: float = 0.2,
    dominance_threshold: float = 0.85,
    curvature_mode: str = "raw",
    seam_weight: float = 0.0,
    lazarus_weight: float = 0.0,
) -> pd.DataFrame:
    required = [
        spec.entropy_col,
        spec.diversity_col,
        spec.mode_count_col,
        spec.curvature_col,
        spec.dominant_col,
    ]
    require_columns(df, required)

    out = df.copy()
    out["entropy_norm"] = safe_normalize(out[spec.entropy_col], max_value=entropy_max)
    out["outcome_diversity_norm"] = clamp01(out[spec.diversity_col].astype(float))
    out["mode_count_norm"] = safe_normalize(out[spec.mode_count_col], max_value=mode_count_max)
    out["curvature_raw"] = out[spec.curvature_col].astype(float).clip(lower=0.0)
    out["curvature_transformed"] = transform_curvature(out["curvature_raw"], mode=curvature_mode)
    out["inverse_curvature"] = inverse_curvature(out["curvature_raw"], mode=curvature_mode)
    out["dominant_fraction"] = clamp01(out[spec.dominant_col].astype(float))

    seam_proximity = None
    if spec.seam_distance_col in out.columns:
        out["seam_distance"] = out[spec.seam_distance_col].astype(float).clip(lower=0.0)
        out["seam_proximity"] = 1.0 - safe_normalize(out["seam_distance"])
        seam_proximity = out["seam_proximity"]
    else:
        out["seam_distance"] = np.nan
        out["seam_proximity"] = np.nan

    lazarus_score = None
    if spec.lazarus_score_col in out.columns:
        out["lazarus_score"] = safe_normalize(out[spec.lazarus_score_col].astype(float))
        lazarus_score = out["lazarus_score"]
    else:
        out["lazarus_score"] = np.nan

    if spec.lazarus_hit_col in out.columns:
        out["lazarus_hit"] = out[spec.lazarus_hit_col].fillna(0).astype(int)
    else:
        out["lazarus_hit"] = 0

    out["distance_to_horizon"] = distance_to_horizon(
        out["entropy_norm"],
        out["outcome_diversity_norm"],
        out["mode_count_norm"],
        out["inverse_curvature"],
        seam_proximity=seam_proximity,
        seam_weight=seam_weight,
    )
    out["horizon_pressure"] = horizon_pressure(
        out["outcome_diversity_norm"],
        out["curvature_raw"],
        out["dominant_fraction"],
        lazarus_score=lazarus_score,
        lazarus_weight=lazarus_weight,
    )
    out["horizon_type"] = classify_horizon_type(
        out["outcome_diversity_norm"],
        out["dominant_fraction"],
        diversity_threshold=diversity_threshold,
        dominance_threshold=dominance_threshold,
    )

    out["horizon_activation"] = out["horizon_pressure"] * (1.0 - out["distance_to_horizon"])
    collapse_base = (
        0.40 * (1.0 - out["distance_to_horizon"])
        + 0.25 * safe_normalize(out["horizon_pressure"])
        + 0.20 * out["dominant_fraction"]
        + 0.15 * out["seam_proximity"].fillna(0.0)
    )
    if spec.lazarus_score_col in out.columns:
        collapse_base = collapse_base + 0.10 * out["lazarus_score"].fillna(0.0)
    out["collapse_risk"] = clamp01(collapse_base)
    out["precollapse_regime"] = (
        (out["collapse_risk"] >= 0.6)
        & (out["dominant_fraction"] < dominance_threshold)
        & (out["outcome_diversity_norm"] > diversity_threshold)
    ).astype(int)

    comps = horizon_components(
        out["entropy_norm"],
        out["outcome_diversity_norm"],
        out["mode_count_norm"],
        out["inverse_curvature"],
        seam_proximity=seam_proximity,
    )
    return pd.concat([out, comps], axis=1)


def plot_trace(
    df: pd.DataFrame,
    spec: ColumnSpec,
    outdir: Path,
    *,
    max_trajectories: int = 5,
    plot_components: bool = True,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if spec.trajectory_col in df.columns:
        trajectory_ids = list(df[spec.trajectory_col].dropna().unique())[:max_trajectories]
        groups = [(tid, df[df[spec.trajectory_col] == tid].copy()) for tid in trajectory_ids]
    else:
        groups = [("global", df.copy())]

    x_col = spec.step_col if spec.step_col in df.columns else None

    for name, g in groups:
        g = g.reset_index(drop=True)
        x = g[x_col] if x_col is not None else np.arange(len(g))

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(x, g["distance_to_horizon"], label="distance_to_horizon")
        axes[0].plot(x, g["horizon_pressure"], label="horizon_pressure")
        axes[0].plot(x, g["collapse_risk"], label="collapse_risk")
        axes[0].plot(x, g["dominant_fraction"], label="dominant_fraction")
        axes[0].set_title(f"Horizon trace — {name}")
        axes[0].set_ylabel("value")
        axes[0].legend()

        if "seam_proximity" in g.columns and g["seam_proximity"].notna().any():
            axes[1].plot(x, g["seam_proximity"], label="seam_proximity")
        if "lazarus_score" in g.columns and g["lazarus_score"].notna().any():
            axes[1].plot(x, g["lazarus_score"], label="lazarus_score")
        axes[1].plot(x, g["horizon_activation"], label="horizon_activation")
        axes[1].plot(x, g["precollapse_regime"], label="precollapse_regime")
        axes[1].set_title(f"Boundary-aware diagnostics — {name}")
        axes[1].set_xlabel(x_col if x_col is not None else "index")
        axes[1].set_ylabel("diagnostic value")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(outdir / f"horizon_trace_{name}.png", dpi=160)
        plt.close(fig)

        if plot_components:
            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            axes[0].plot(x, g["entropy_norm"], label="entropy_norm")
            axes[0].plot(x, g["outcome_diversity_norm"], label="outcome_diversity_norm")
            axes[0].plot(x, g["mode_count_norm"], label="mode_count_norm")
            axes[0].plot(x, g["inverse_curvature"], label="inverse_curvature")
            axes[0].legend()
            axes[0].set_title(f"Normalized factors — {name}")

            axes[1].plot(x, g["log_entropy_term"], label="log_entropy_term")
            axes[1].plot(x, g["log_diversity_term"], label="log_diversity_term")
            axes[1].plot(x, g["log_mode_term"], label="log_mode_term")
            axes[1].plot(x, g["log_inverse_curvature_term"], label="log_inverse_curvature_term")
            if "log_seam_proximity_term" in g.columns:
                axes[1].plot(x, g["log_seam_proximity_term"], label="log_seam_proximity_term")
            axes[1].legend()
            axes[1].set_title(f"Log contributions — {name}")

            axes[2].plot(x, g["curvature_raw"], label="curvature_raw")
            axes[2].plot(x, g["curvature_transformed"], label="curvature_transformed")
            axes[2].legend()
            axes[2].set_title(f"Curvature transform — {name}")
            axes[2].set_xlabel(x_col if x_col is not None else "index")

            fig.tight_layout()
            fig.savefig(outdir / f"horizon_components_{name}.png", dpi=160)
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Canonical horizon-trace experiment (v3).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plot-dir", type=Path, default=None)

    p.add_argument("--id-col", default="id")
    p.add_argument("--trajectory-col", default="trajectory_id")
    p.add_argument("--step-col", default="step")
    p.add_argument("--entropy-col", default="entropy_joint")
    p.add_argument("--diversity-col", default="outcome_diversity")
    p.add_argument("--mode-count-col", default="mode_count")
    p.add_argument("--curvature-col", default="curvature_proxy")
    p.add_argument("--dominant-col", default="dominant_fraction")
    p.add_argument("--seam-distance-col", default="seam_distance")
    p.add_argument("--lazarus-score-col", default="lazarus_score")
    p.add_argument("--lazarus-hit-col", default="lazarus_hit")

    p.add_argument("--entropy-max", type=float, default=None)
    p.add_argument("--mode-count-max", type=float, default=None)
    p.add_argument("--diversity-threshold", type=float, default=0.2)
    p.add_argument("--dominance-threshold", type=float, default=0.85)

    p.add_argument("--curvature-mode", choices=["raw", "log", "sqrt"], default="log")
    p.add_argument("--seam-weight", type=float, default=0.0)
    p.add_argument("--lazarus-weight", type=float, default=0.25)
    p.add_argument("--no-component-plots", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    spec = ColumnSpec(
        id_col=args.id_col,
        trajectory_col=args.trajectory_col,
        step_col=args.step_col,
        entropy_col=args.entropy_col,
        diversity_col=args.diversity_col,
        mode_count_col=args.mode_count_col,
        curvature_col=args.curvature_col,
        dominant_col=args.dominant_col,
        seam_distance_col=args.seam_distance_col,
        lazarus_score_col=args.lazarus_score_col,
        lazarus_hit_col=args.lazarus_hit_col,
    )

    df = pd.read_csv(args.input)
    df = sort_dataframe(df, spec)
    out = compute_horizon_metrics(
        df,
        spec,
        entropy_max=args.entropy_max,
        mode_count_max=args.mode_count_max,
        diversity_threshold=args.diversity_threshold,
        dominance_threshold=args.dominance_threshold,
        curvature_mode=args.curvature_mode,
        seam_weight=args.seam_weight,
        lazarus_weight=args.lazarus_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    if args.plot_dir is not None:
        plot_trace(
            out,
            spec,
            args.plot_dir,
            plot_components=not args.no_component_plots,
        )

    print(f"Wrote horizon metrics to: {args.output}")
    print(f"Curvature transform: {args.curvature_mode}")
    print(f"Seam weight: {args.seam_weight}")
    print(f"Lazarus weight: {args.lazarus_weight}")
    if args.plot_dir is not None:
        print(f"Wrote plots to: {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
