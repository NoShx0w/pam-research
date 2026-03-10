from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/index.csv")
DEFAULT_OUTPUT = Path("outputs/phase_summary.csv")
DEFAULT_LONG_OUTPUT = Path("outputs/phase_summary_long.csv")
DEFAULT_COVERAGE_OUTPUT = Path("outputs/seed_coverage.csv")


# Metrics we expect or would strongly like to summarize if present.
PREFERRED_METRICS = [
    "piF_tail",
    "piF_mean",
    "H_joint_mean",
    "H_joint_tail",
    "H_mean",
    "K_max",
    "K_mean",
    "corr0",
    "best_corr",
    "best_lag",
    "delta_r2_freeze",
    "delta_r2_entropy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PAM batch results from outputs/index.csv by (r, alpha)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to wide summary CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--long-output",
        type=Path,
        default=DEFAULT_LONG_OUTPUT,
        help=f"Path to long/tidy summary CSV (default: {DEFAULT_LONG_OUTPUT})",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=DEFAULT_COVERAGE_OUTPUT,
        help=f"Path to seed coverage CSV (default: {DEFAULT_COVERAGE_OUTPUT})",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=6,
        help="Decimal places for output rounding.",
    )
    return parser.parse_args()


def stderr(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(n))


def pick_metric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "r",
        "alpha",
        "seed",
        "run_id",
        "path",
        "json_path",
        "deep_path",
        "outfile",
        "timestamp",
    }

    numeric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    preferred_present = [c for c in PREFERRED_METRICS if c in numeric_cols]
    remaining = [c for c in numeric_cols if c not in preferred_present]

    return preferred_present + remaining


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            name = "_".join(str(part) for part in col if part != "")
        else:
            name = str(col)
        flat_cols.append(name.rstrip("_"))
    out = df.copy()
    out.columns = flat_cols
    return out


def build_seed_coverage(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["r", "alpha"], dropna=False)

    coverage = grouped.agg(
        n_rows=("alpha", "size"),
        n_unique_seeds=("seed", "nunique") if "seed" in df.columns else ("alpha", "size"),
    ).reset_index()

    if "seed" in df.columns:
        seeds = (
            grouped["seed"]
            .apply(lambda s: ",".join(str(x) for x in sorted(pd.unique(s.dropna()))))
            .reset_index(name="seeds")
        )
        coverage = coverage.merge(seeds, on=["r", "alpha"], how="left")

    return coverage.sort_values(["r", "alpha"]).reset_index(drop=True)


def build_phase_summary(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(["r", "alpha"], dropna=False)
    summary_frames = []

    for metric in metrics:
        g = grouped[metric].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
        g[f"{metric}_sem"] = grouped[metric].apply(stderr).values
        g = g.rename(
            columns={
                "count": f"{metric}_n",
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "median": f"{metric}_median",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
            }
        )
        summary_frames.append(g)

    if not summary_frames:
        return df[["r", "alpha"]].drop_duplicates().sort_values(["r", "alpha"]).reset_index(drop=True)

    out = summary_frames[0]
    for frame in summary_frames[1:]:
        out = out.merge(frame, on=["r", "alpha"], how="outer")

    return out.sort_values(["r", "alpha"]).reset_index(drop=True)


def build_long_summary(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(["r", "alpha"], dropna=False)
    rows: list[dict] = []

    for metric in metrics:
        counts = grouped[metric].count()
        means = grouped[metric].mean()
        stds = grouped[metric].std()
        medians = grouped[metric].median()
        mins = grouped[metric].min()
        maxs = grouped[metric].max()
        sems = grouped[metric].apply(stderr)

        for key in counts.index:
            r, alpha = key
            rows.append(
                {
                    "r": r,
                    "alpha": alpha,
                    "metric": metric,
                    "n": counts.loc[key],
                    "mean": means.loc[key],
                    "std": stds.loc[key],
                    "sem": sems.loc[key],
                    "median": medians.loc[key],
                    "min": mins.loc[key],
                    "max": maxs.loc[key],
                }
            )

    return pd.DataFrame(rows).sort_values(["metric", "r", "alpha"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)

    required = {"r", "alpha"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # Normalize numerics where possible.
    for col in ["r", "alpha", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metric_cols = pick_metric_columns(df)

    if not metric_cols:
        raise ValueError(
            "No numeric metric columns found to aggregate. "
            "Check the schema of outputs/index.csv."
        )

    coverage = build_seed_coverage(df)
    wide_summary = build_phase_summary(df, metric_cols)
    long_summary = build_long_summary(df, metric_cols)

    # Join coverage into the wide summary so every (r, alpha) row includes sample info.
    wide_summary = coverage.merge(wide_summary, on=["r", "alpha"], how="left")

    # Round for cleaner CSVs.
    wide_summary = wide_summary.round(args.round)
    long_summary = long_summary.round(args.round)
    coverage = coverage.round(args.round)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide_summary.to_csv(args.output, index=False)
    long_summary.to_csv(args.long_output, index=False)
    coverage.to_csv(args.coverage_output, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Detected metric columns: {', '.join(metric_cols)}")
    print(f"Wrote wide summary:   {args.output}")
    print(f"Wrote long summary:   {args.long_output}")
    print(f"Wrote seed coverage:  {args.coverage_output}")


if __name__ == "__main__":
    main()
