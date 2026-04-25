#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import pandas as pd


@dataclass(frozen=True)
class Config:
    root: str = "outputs"
    scales: tuple[str, ...] = ("base", "10", "100", "1000", "10000", "100000")
    outdir: str = "outputs/toy_scale_dynamic_persistence"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).copy()


def extract_predictive(scale_root: Path) -> dict[str, object]:
    path = scale_root / "fim_ops_scaled" / "scaled_probe_predictive_summary.csv"
    df = load_csv(path)
    out: dict[str, object] = {}
    if df.empty or "lazarus_group" not in df.columns:
        return out

    for group in ["high", "low"]:
        sub = df[df["lazarus_group"] == group]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for col in [
            "seam_cross_rate",
            "mean_phase_flip_count",
            "mean_min_distance_to_seam",
            "mean_path_length",
            "mean_constraint_strength",
        ]:
            if col in row.index:
                out[f"predictive_{group}_{col}"] = row[col]

    if "predictive_high_mean_min_distance_to_seam" in out and "predictive_low_mean_min_distance_to_seam" in out:
        out["predictive_delta_min_distance_to_seam"] = (
            out["predictive_high_mean_min_distance_to_seam"]
            - out["predictive_low_mean_min_distance_to_seam"]
        )
    if "predictive_high_seam_cross_rate" in out and "predictive_low_seam_cross_rate" in out:
        out["predictive_delta_seam_cross_rate"] = (
            out["predictive_high_seam_cross_rate"]
            - out["predictive_low_seam_cross_rate"]
        )
    return out


def extract_transition(scale_root: Path) -> dict[str, object]:
    path = scale_root / "fim_transition_rate" / "transition_rate_summary.csv"
    df = load_csv(path)
    out: dict[str, object] = {}
    if df.empty or "lazarus_group" not in df.columns:
        return out

    for group in ["high", "low"]:
        sub = df[df["lazarus_group"] == group]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for col in [
            "transition_rate",
            "mean_lag_to_next_transition",
            "mean_distance_to_seam",
            "mean_curvature",
        ]:
            if col in row.index:
                out[f"transition_{group}_{col}"] = row[col]

    if "transition_high_transition_rate" in out and "transition_low_transition_rate" in out:
        out["transition_delta_rate"] = (
            out["transition_high_transition_rate"]
            - out["transition_low_transition_rate"]
        )
    if "transition_high_mean_distance_to_seam" in out and "transition_low_mean_distance_to_seam" in out:
        out["transition_delta_distance_to_seam"] = (
            out["transition_high_mean_distance_to_seam"]
            - out["transition_low_mean_distance_to_seam"]
        )
    return out


def extract_temporal(scale_root: Path) -> dict[str, object]:
    path = scale_root / "fim_lazarus_temporal" / "lazarus_temporal_summary.csv"
    df = load_csv(path)
    out: dict[str, object] = {}
    if df.empty:
        return out

    row = df.iloc[0]
    for col in [
        "n_paths",
        "share_lazarus_precedes_seam",
        "share_lazarus_precedes_flip",
        "mean_lag_lazarus_to_seam",
        "mean_lag_lazarus_to_flip",
        "median_lag_lazarus_to_seam",
        "median_lag_lazarus_to_flip",
    ]:
        if col in row.index:
            out[f"temporal_{col}"] = row[col]
    return out


def summarize(df: pd.DataFrame) -> str:
    lines = ["=== Scale Dynamic Persistence Summary ===", ""]
    for _, row in df.iterrows():
        lines.append(
            f"scale={row['scale']}: "
            f"delta_seam_cross={row.get('predictive_delta_seam_cross_rate', float('nan')):.4f}, "
            f"delta_min_seam={row.get('predictive_delta_min_distance_to_seam', float('nan')):.4f}, "
            f"delta_transition_rate={row.get('transition_delta_rate', float('nan')):.4f}, "
            f"delta_transition_distance={row.get('transition_delta_distance_to_seam', float('nan')):.4f}, "
            f"share_precedes_seam={row.get('temporal_share_lazarus_precedes_seam', float('nan')):.4f}, "
            f"share_precedes_flip={row.get('temporal_share_lazarus_precedes_flip', float('nan')):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-scale dynamic persistence summary.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--scales", default="10,100,1000,10000,100000")
    parser.add_argument("--outdir", default="outputs/toy_scale_dynamic_persistence")
    args = parser.parse_args()

    cfg = Config(
        root=args.root,
        scales=tuple(x.strip() for x in args.scales.split(",") if x.strip()),
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = Path(cfg.root)
    rows: list[dict[str, object]] = []

    for scale in cfg.scales:
        sroot = base / "scales" / scale
        row: dict[str, object] = {"scale": scale}
        row.update(extract_predictive(sroot))
        row.update(extract_transition(sroot))
        row.update(extract_temporal(sroot))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "scale_dynamic_persistence_summary.csv", index=False)
    (outdir / "scale_dynamic_persistence_summary.txt").write_text(
        summarize(df),
        encoding="utf-8",
    )

    print(outdir / "scale_dynamic_persistence_summary.csv")
    print(outdir / "scale_dynamic_persistence_summary.txt")


if __name__ == "__main__":
    main()
