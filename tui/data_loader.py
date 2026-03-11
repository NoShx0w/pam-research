from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from tui.models import Snapshot, SweepSpec


DEFAULT_SWEEP_DICT = {
    "r_values": [0.10, 0.15, 0.20, 0.25, 0.30],
    "alpha_values": [
        0.03,
        0.03857142857142857,
        0.04714285714285714,
        0.055714285714285716,
        0.06428571428571428,
        0.07285714285714286,
        0.08142857142857143,
        0.09,
        0.09857142857142857,
        0.10714285714285714,
        0.11571428571428571,
        0.12428571428571428,
        0.13285714285714287,
        0.14142857142857143,
        0.15,
    ],
    "seeds_per_cell": 10,
}


def load_or_create_sweep_spec(path: Path) -> SweepSpec:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        path.write_text(json.dumps(DEFAULT_SWEEP_DICT, indent=2))
        data = DEFAULT_SWEEP_DICT
    else:
        data = json.loads(path.read_text())

    return SweepSpec(
        r_values=[float(x) for x in data["r_values"]],
        alpha_values=[float(x) for x in data["alpha_values"]],
        seeds_per_cell=int(data["seeds_per_cell"]),
    )


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sorted_unique_numeric(series: pd.Series) -> list[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    return sorted(float(v) for v in vals)


def display_float(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.rstrip("0").rstrip(".")


def build_sweep_spec_text(spec: SweepSpec) -> str:
    return (
        f"r count         {len(spec.r_values)}\n"
        f"r min/max       {display_float(min(spec.r_values), 3)} → {display_float(max(spec.r_values), 3)}\n"
        f"α range         {display_float(min(spec.alpha_values), 3)} → {display_float(max(spec.alpha_values), 3)}\n"
        f"α count         {len(spec.alpha_values)}\n"
        f"seeds / cell    {spec.seeds_per_cell}\n"
        f"intended total  {spec.expected_total}"
    )


def build_observed_grid_text(df: pd.DataFrame, spec: SweepSpec) -> str:
    if df.empty or not {"r", "alpha"}.issubset(df.columns):
        return (
            f"observed r      0 / {len(spec.r_values)}\n"
            f"observed α      0 / {len(spec.alpha_values)}"
        )

    work = _safe_numeric(df, ["r", "alpha"]).dropna(subset=["r", "alpha"])
    observed_r = _sorted_unique_numeric(work["r"])
    observed_alpha = _sorted_unique_numeric(work["alpha"])

    return (
        f"observed r      {len(observed_r)} / {len(spec.r_values)}\n"
        f"observed α      {len(observed_alpha)} / {len(spec.alpha_values)}"
    )


def build_latest_metrics_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data yet."

    latest = df.iloc[-1]

    preferred = [
        ("r", "r"),
        ("alpha", "α"),
        ("seed", "seed"),
        ("piF_tail", "πF_tail"),
        ("H_joint_mean", "H_joint"),
        ("best_corr", "best_corr"),
        ("corr0", "corr0"),
        ("delta_r2_freeze", "ΔR²_freeze"),
        ("delta_r2_entropy", "ΔR²_entropy"),
        ("K_max", "K_max"),
    ]

    lines = []
    for col, label in preferred:
        if col not in df.columns:
            continue
        value = latest[col]
        if pd.isna(value):
            continue

        if col in {"r", "alpha"}:
            lines.append(f"{label:<12} {display_float(float(value), 6)}")
        elif isinstance(value, float):
            lines.append(f"{label:<12} {value:.6g}")
        else:
            lines.append(f"{label:<12} {value}")

    return "\n".join(lines) if lines else "No known metric columns found."


def build_coverage_lookup(df: pd.DataFrame) -> dict[tuple[float, float], int]:
    if df.empty or not {"r", "alpha"}.issubset(df.columns):
        return {}

    work = _safe_numeric(df, ["r", "alpha", "seed"]).dropna(subset=["r", "alpha"])

    if "seed" in work.columns:
        grouped = (
            work.groupby(["r", "alpha"])["seed"]
            .nunique()
            .reset_index(name="n")
        )
    else:
        grouped = (
            work.groupby(["r", "alpha"])
            .size()
            .reset_index(name="n")
        )

    return {
        (round(float(row.r), 12), round(float(row.alpha), 12)): int(row.n)
        for row in grouped.itertuples(index=False)
    }


def load_snapshot(index_path: Path, spec: SweepSpec) -> tuple[Snapshot, dict[tuple[float, float], int]]:
    if not index_path.exists():
        snap = Snapshot(
            row_count=0,
            completed=0,
            expected_total=spec.expected_total,
            percent=0.0,
            last_modified="missing",
            latest_metrics_text="index.csv not found.",
            coverage_heatmap_text="",
            sweep_spec_text=build_sweep_spec_text(spec),
            observed_grid_text=(
                f"observed r      0 / {len(spec.r_values)}\n"
                f"observed α      0 / {len(spec.alpha_values)}"
            ),
        )
        return snap, {}

    try:
        df = pd.read_csv(index_path)
    except Exception as exc:
        snap = Snapshot(
            row_count=0,
            completed=0,
            expected_total=spec.expected_total,
            percent=0.0,
            last_modified="unreadable",
            latest_metrics_text=f"Failed to read CSV:\n{exc}",
            coverage_heatmap_text="",
            sweep_spec_text=build_sweep_spec_text(spec),
            observed_grid_text=(
                f"observed r      0 / {len(spec.r_values)}\n"
                f"observed α      0 / {len(spec.alpha_values)}"
            ),
        )
        return snap, {}

    completed = len(df)
    percent = 100.0 * completed / spec.expected_total if spec.expected_total else 0.0
    mtime = pd.Timestamp(index_path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    lookup = build_coverage_lookup(df)

    snap = Snapshot(
        row_count=len(df),
        completed=completed,
        expected_total=spec.expected_total,
        percent=percent,
        last_modified=mtime,
        latest_metrics_text=build_latest_metrics_text(df),
        coverage_heatmap_text="",
        sweep_spec_text=build_sweep_spec_text(spec),
        observed_grid_text=build_observed_grid_text(df, spec),
    )
    return snap, lookup
