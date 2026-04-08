#!/usr/bin/env python3
"""
Validate old vs refactored directional-field outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    old_csv: str = "outputs/toy_identity_transport_alignment/node_transport_alignment.csv"
    new_csv: str = "outputs/obs023_transport_misalignment/obs023_transport_misalignment_nodes.csv"
    outdir: str = "outputs/validate_directional_field_refactor"
    key: str = "node_id"
    cols: tuple[str, ...] = (
        "transport_align_mean_deg",
        "transport_align_max_deg",
        "distance_to_seam",
    )


def compare_frames(old_df: pd.DataFrame, new_df: pd.DataFrame, key: str, cols: tuple[str, ...]) -> pd.DataFrame:
    merged = old_df[[key, *cols]].merge(
        new_df[[key, *cols]], on=key, suffixes=("_old", "_new"), how="inner"
    )

    rows = []
    for col in cols:
        a = pd.to_numeric(merged[f"{col}_old"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_new"], errors="coerce")
        diff = b - a
        rows.append(
            {
                "column": col,
                "n": int((a.notna() & b.notna()).sum()),
                "mean_abs_diff": float(diff.abs().mean()),
                "max_abs_diff": float(diff.abs().max()),
                "corr": float(a.corr(b)) if int((a.notna() & b.notna()).sum()) >= 3 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate refactored directional-field outputs.")
    parser.add_argument("--old-csv", default=Config.old_csv)
    parser.add_argument("--new-csv", default=Config.new_csv)
    parser.add_argument("--outdir", default=Config.outdir)
    args = parser.parse_args()

    cfg = Config(old_csv=args.old_csv, new_csv=args.new_csv, outdir=args.outdir)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    old_df = pd.read_csv(cfg.old_csv)
    new_df = pd.read_csv(cfg.new_csv)

    summary = compare_frames(old_df, new_df, cfg.key, cfg.cols)
    summary_csv = outdir / "directional_field_refactor_validation.csv"
    summary_txt = outdir / "directional_field_refactor_validation.txt"

    summary.to_csv(summary_csv, index=False)
    lines = ["=== Directional Field Refactor Validation ===", ""]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['column']}: n={int(row['n'])}, mean_abs_diff={float(row['mean_abs_diff']):.6f}, "
            f"max_abs_diff={float(row['max_abs_diff']):.6f}, corr={float(row['corr']):.6f}"
        )
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(summary_csv)
    print(summary_txt)


if __name__ == "__main__":
    main()
