#!/usr/bin/env python3
"""
toy_horizon_trace.py (v2)

Canonical toy experiment for the PAM Observatory.

v2 patch
--------
Changes from v1:
1. Optional log-scaling for curvature before inverse transform
2. Component plotting for diagnostic visibility
3. Optional raw + transformed curvature columns in output
4. Slightly richer plotting layout
5. More explicit CLI flags for curvature handling

Purpose
-------
Compute a first-pass "distance to horizon" trace and "horizon pressure"
trace along an existing manifold / trajectory dataset.

This script is intentionally simple and explicit:
- no hidden assumptions
- no overloaded symbols
- safe normalization
- inspectable component terms
- optional plotting

Expected input
--------------
A CSV with one row per state / sample / trajectory point. The script looks for
the following columns:

Required:
    - id                      unique row identifier
    - entropy_joint           joint entropy proxy H(x)
    - outcome_diversity       outcome diversity U(x)
    - mode_count              active mode count M_c(x)
    - curvature_proxy         criticality / curvature proxy C(x)
    - dominant_fraction       dominant-mode fraction D(x)

Optional:
    - trajectory_id           group identifier for multiple trajectories
    - step                    within-trajectory order
    - seam_distance           optional diagnostic
    - criticality             optional diagnostic alias

If your column names differ, use the CLI flags to map them.

Outputs
-------
1. A CSV with horizon metrics added
2. Optional plots:
   - distance_to_horizon vs step
   - horizon_pressure vs step
   - dominant_fraction vs step
   - diagnostic component plots

Interpretation
--------------
Low distance_to_horizon:
    the system is near outcome-collapse

High horizon_pressure:
    the system is under strong transition pressure,
    i.e. diversity is shrinking under high curvature
    without full domination yet

Horizon type:
    - "open"
    - "false_horizon"
    - "true_horizon"

Canonical reading:
    good extraction lowers distance_to_horizon by removing false alternatives
    without driving dominant_fraction to 1.0
"""

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


def clamp01(series: pd.Series) -> pd.Series:
    """Clip values into [0, 1]."""
    return series.clip(lower=0.0, upper=1.0)


def safe_normalize(
    series: pd.Series,
    *,
    max_value: Optional[float] = None,
) -> pd.Series:
    """
    Normalize a nonnegative series into [0, 1].

    If max_value is not provided, uses the observed max.
    If max is 0 or NaN, returns zeros.
    """
    if max_value is None:
        max_value = float(series.max(skipna=True))

    if not np.isfinite(max_value) or max_value <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    out = series.astype(float) / max_value
    return clamp01(out)


def transform_curvature(
    curvature: pd.Series,
    *,
    use_log: bool = False,
) -> pd.Series:
    """
    Transform curvature before inverse mapping.

    Parameters
    ----------
    use_log:
        If True, use log1p(curvature) to reduce dynamic-range compression
        when curvature spans multiple orders of magnitude.
    """
    c = curvature.astype(float).clip(lower=0.0)
    if use_log:
        c = np.log1p(c)
    return c


def inverse_curvature(
    curvature: pd.Series,
    *,
    use_log: bool = False,
) -> pd.Series:
    """
    K(x) = 1 / (1 + C(x))

    Optionally applies log1p to curvature first.
    """
    c = transform_curvature(curvature, use_log=use_log)
    return 1.0 / (1.0 + c)


def log_product_terms(*terms: pd.Series) -> pd.Series:
    """
    Stable product in log space:
        exp(sum(log(term_i + eps)))
    """
    acc = pd.Series(np.zeros(len(terms[0])), index=terms[0].index, dtype=float)
    for term in terms:
        acc = acc + np.log(term.astype(float) + EPS)
    return np.exp(acc)


def distance_to_horizon(
    entropy_norm: pd.Series,
    diversity: pd.Series,
    mode_norm: pd.Series,
    inv_curvature: pd.Series,
) -> pd.Series:
    """
    D_H = E * U * B * K

    where:
        E = normalized entropy
        U = outcome diversity
        B = normalized mode count
        K = inverse curvature
    """
    return log_product_terms(entropy_norm, diversity, mode_norm, inv_curvature)


def horizon_components(
    entropy_norm: pd.Series,
    diversity: pd.Series,
    mode_norm: pd.Series,
    inv_curvature: pd.Series,
) -> pd.DataFrame:
    """
    Expose log-space contributions for diagnostics.
    More negative => stronger collapse contribution.
    """
    return pd.DataFrame(
        {
            "log_entropy_term": np.log(entropy_norm + EPS),
            "log_diversity_term": np.log(diversity + EPS),
            "log_mode_term": np.log(mode_norm + EPS),
            "log_inverse_curvature_term": np.log(inv_curvature + EPS),
        }
    )


def horizon_pressure(
    diversity: pd.Series,
    curvature: pd.Series,
    dominant_fraction: pd.Series,
) -> pd.Series:
    """
    P_H = C * (1 - U) * (1 - D)

    High when:
        - curvature is high
        - diversity is already shrinking
        - but no full domination yet
    """
    u = clamp01(diversity.astype(float))
    d = clamp01(dominant_fraction.astype(float))
    c = curvature.astype(float).clip(lower=0.0)
    return c * (1.0 - u) * (1.0 - d)


def classify_horizon_type(
    diversity: pd.Series,
    dominant_fraction: pd.Series,
    *,
    diversity_threshold: float = 0.2,
    dominance_threshold: float = 0.85,
) -> pd.Series:
    """
    Distinguish:
        - false_horizon: diversity low, but not dominated
        - true_horizon: diversity low, and strongly dominated
        - open: otherwise
    """
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


def plot_trace(
    df: pd.DataFrame,
    spec: ColumnSpec,
    outdir: Path,
    *,
    max_trajectories: int = 5,
    plot_components: bool = True,
) -> None:
    """
    Plot distance, pressure, dominance for up to `max_trajectories` trajectories.
    If no trajectory_id is present, plot the whole dataset as one sequence.
    """
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

        # Main trace plot
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, g["distance_to_horizon"], label="distance_to_horizon")
        ax.plot(x, g["horizon_pressure"], label="horizon_pressure")
        ax.plot(x, g["dominant_fraction"], label="dominant_fraction")
        ax.set_title(f"Horizon trace — {name}")
        ax.set_xlabel(x_col if x_col is not None else "index")
        ax.set_ylabel("value")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"horizon_trace_{name}.png", dpi=160)
        plt.close(fig)

        # Optional diagnostic components
        if plot_components:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

            axes[0].plot(x, g["entropy_norm"], label="entropy_norm")
            axes[0].plot(x, g["outcome_diversity_norm"], label="outcome_diversity_norm")
            axes[0].plot(x, g["mode_count_norm"], label="mode_count_norm")
            axes[0].plot(x, g["inverse_curvature"], label="inverse_curvature")
            axes[0].set_title(f"Normalized horizon factors — {name}")
            axes[0].set_ylabel("factor value")
            axes[0].legend()

            axes[1].plot(x, g["log_entropy_term"], label="log_entropy_term")
            axes[1].plot(x, g["log_diversity_term"], label="log_diversity_term")
            axes[1].plot(x, g["log_mode_term"], label="log_mode_term")
            axes[1].plot(
                x,
                g["log_inverse_curvature_term"],
                label="log_inverse_curvature_term",
            )
            axes[1].set_title(f"Log component contributions — {name}")
            axes[1].set_xlabel(x_col if x_col is not None else "index")
            axes[1].set_ylabel("log contribution")
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(outdir / f"horizon_components_{name}.png", dpi=160)
            plt.close(fig)

            if "curvature_transformed" in g.columns:
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(x, g["curvature_raw"], label="curvature_raw")
                ax.plot(x, g["curvature_transformed"], label="curvature_transformed")
                ax.set_title(f"Curvature transform — {name}")
                ax.set_xlabel(x_col if x_col is not None else "index")
                ax.set_ylabel("curvature")
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / f"curvature_transform_{name}.png", dpi=160)
                plt.close(fig)


def compute_horizon_metrics(
    df: pd.DataFrame,
    spec: ColumnSpec,
    *,
    entropy_max: Optional[float] = None,
    mode_count_max: Optional[float] = None,
    diversity_threshold: float = 0.2,
    dominance_threshold: float = 0.85,
    log_scale_curvature: bool = False,
) -> pd.DataFrame:
    """
    Add canonical horizon metrics to the dataframe.
    """
    required = [
        spec.entropy_col,
        spec.diversity_col,
        spec.mode_count_col,
        spec.curvature_col,
        spec.dominant_col,
    ]
    require_columns(df, required)

    out = df.copy()

    # Normalize core observables.
    out["entropy_norm"] = safe_normalize(
        out[spec.entropy_col],
        max_value=entropy_max,
    )
    out["outcome_diversity_norm"] = clamp01(out[spec.diversity_col].astype(float))
    out["mode_count_norm"] = safe_normalize(
        out[spec.mode_count_col],
        max_value=mode_count_max,
    )

    out["curvature_raw"] = out[spec.curvature_col].astype(float).clip(lower=0.0)
    out["curvature_transformed"] = transform_curvature(
        out["curvature_raw"],
        use_log=log_scale_curvature,
    )
    out["inverse_curvature"] = 1.0 / (1.0 + out["curvature_transformed"])

    out["dominant_fraction"] = clamp01(out[spec.dominant_col].astype(float))

    # Composite metrics.
    out["distance_to_horizon"] = distance_to_horizon(
        out["entropy_norm"],
        out["outcome_diversity_norm"],
        out["mode_count_norm"],
        out["inverse_curvature"],
    )
    out["horizon_pressure"] = horizon_pressure(
        out["outcome_diversity_norm"],
        out["curvature_raw"],
        out["dominant_fraction"],
    )
    out["horizon_type"] = classify_horizon_type(
        out["outcome_diversity_norm"],
        out["dominant_fraction"],
        diversity_threshold=diversity_threshold,
        dominance_threshold=dominance_threshold,
    )

    # Diagnostic components.
    comps = horizon_components(
        out["entropy_norm"],
        out["outcome_diversity_norm"],
        out["mode_count_norm"],
        out["inverse_curvature"],
    )
    out = pd.concat([out, comps], axis=1)

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Canonical toy horizon-trace experiment (v2).")

    p.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG plots.",
    )

    # Column mapping.
    p.add_argument("--id-col", default="id")
    p.add_argument("--trajectory-col", default="trajectory_id")
    p.add_argument("--step-col", default="step")
    p.add_argument("--entropy-col", default="entropy_joint")
    p.add_argument("--diversity-col", default="outcome_diversity")
    p.add_argument("--mode-count-col", default="mode_count")
    p.add_argument("--curvature-col", default="curvature_proxy")
    p.add_argument("--dominant-col", default="dominant_fraction")

    # Optional normalization controls.
    p.add_argument(
        "--entropy-max",
        type=float,
        default=None,
        help="Override entropy normalization maximum.",
    )
    p.add_argument(
        "--mode-count-max",
        type=float,
        default=None,
        help="Override mode count normalization maximum.",
    )

    # Thresholds.
    p.add_argument("--diversity-threshold", type=float, default=0.2)
    p.add_argument("--dominance-threshold", type=float, default=0.85)

    # v2 additions
    p.add_argument(
        "--log-scale-curvature",
        action="store_true",
        help="Apply log1p transform to curvature before inverse mapping.",
    )
    p.add_argument(
        "--no-component-plots",
        action="store_true",
        help="Disable diagnostic component plots.",
    )

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
        log_scale_curvature=args.log_scale_curvature,
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
    print(f"Curvature log-scaling: {'enabled' if args.log_scale_curvature else 'disabled'}")
    if args.plot_dir is not None:
        print(f"Wrote plots to: {args.plot_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
