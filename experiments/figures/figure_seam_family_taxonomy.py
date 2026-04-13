#!/usr/bin/env python3
"""
Figure — Seam Family Taxonomy

Purpose
-------
Render a polished synthesis figure from OBS-042.

The figure presents the three canonical seam families as columns and the key
regime dimensions as rows:

- dominant regime
- predictive locus
- effective memory horizon
- compression regime
- gateway signature
- canonical interpretation

Inputs
------
outputs/obs042_family_temporal_regimes_synthesis/family_temporal_regimes_summary.csv

Outputs
-------
outputs/figures/seam_family_taxonomy.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = "outputs/obs042_family_temporal_regimes_synthesis/family_temporal_regimes_summary.csv"
OUTPUT_PNG = "outputs/figures/seam_family_taxonomy.png"

FAMILY_ORDER = [
    "branch_exit",
    "stable_seam_corridor",
    "reorganization_heavy",
]

DISPLAY_NAMES = {
    "branch_exit": "Branch Exit",
    "stable_seam_corridor": "Stable Seam Corridor",
    "reorganization_heavy": "Reorganization Heavy",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order"] = df["route_class"].map(lambda x: FAMILY_ORDER.index(x) if x in FAMILY_ORDER else 999)
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)


def fmt_horizon(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "—"
    if v.is_integer():
        return f"k = {int(v)}"
    return f"k = {v:.1f}"


def build_matrix(df: pd.DataFrame) -> tuple[list[str], list[str], list[list[str]]]:
    columns = [DISPLAY_NAMES.get(x, x) for x in df["route_class"].tolist()]

    row_labels = [
        "Dominant regime",
        "Predictive locus",
        "Memory regime",
        "Effective horizon",
        "Compression",
        "Top compression state",
        "Core→escape share",
        "Escape-internal share",
        "Canonical reading",
    ]

    data = []
    for _, row in df.iterrows():
        data.append(
            [
                str(row.get("canonical_regime", "—")),
                str(row.get("predictive_locus", "—")),
                str(row.get("memory_interpretation", "—")),
                fmt_horizon(row.get("best_horizon_k", "—")),
                str(row.get("compression_interpretation", "—")),
                str(row.get("top_compression_state", "—")),
                f"{float(row.get('core_to_escape_share', float('nan'))):.3f}" if pd.notna(row.get("core_to_escape_share")) else "—",
                f"{float(row.get('escape_internal_share', float('nan'))):.3f}" if pd.notna(row.get("escape_internal_share")) else "—",
                str(row.get("canonical_interpretation", "—")),
            ]
        )

    matrix = list(map(list, zip(*data)))
    return columns, row_labels, matrix


def render_figure(df: pd.DataFrame, outpath: Path) -> None:
    columns, row_labels, matrix = build_matrix(df)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.18, 0.57, 0.25])

    ax_title = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_footer = fig.add_subplot(gs[2])

    # Title block
    ax_title.axis("off")
    ax_title.text(
        0.00, 0.90,
        "PAM Observatory — Seam Family Taxonomy",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_title.text(
        0.00, 0.52,
        "Canonical synthesis from OBS-042",
        fontsize=14,
        ha="left",
        va="top",
    )
    ax_title.text(
        0.00, 0.16,
        (
            "The seam resolves into three family-specific temporal regimes: "
            "an immediate directed/downstream family, a one-step local gateway family, "
            "and an extended-memory path-context family with strong compression bottlenecks."
        ),
        fontsize=11.5,
        ha="left",
        va="top",
        wrap=True,
    )

    # Main table-like panel
    ax_main.axis("off")

    n_rows = len(row_labels)
    n_cols = len(columns)

    # tighter, unified table geometry
    left_label_w = 0.21
    table_left = 0.24
    table_right = 0.98
    table_top = 0.95
    table_bottom = 0.05

    table_width = table_right - table_left
    table_height = table_top - table_bottom

    total_rows = n_rows + 1  # include header row inside the table
    col_w = table_width / n_cols
    row_h = table_height / total_rows

    # draw full grid first
    for i in range(total_rows):
        for j in range(n_cols):
            x0 = table_left + j * col_w
            y0 = table_top - (i + 1) * row_h
            rect = plt.Rectangle(
                (x0, y0),
                col_w,
                row_h,
                fill=False,
                linewidth=0.9,
                edgecolor="0.70",
                transform=ax_main.transAxes,
            )
            ax_main.add_patch(rect)

    # outer border
    outer = plt.Rectangle(
        (table_left, table_bottom),
        table_width,
        table_height,
        fill=False,
        linewidth=1.2,
        edgecolor="0.35",
        transform=ax_main.transAxes,
    )
    ax_main.add_patch(outer)

    # column headers INSIDE header row
    for j, col_name in enumerate(columns):
        x = table_left + j * col_w + col_w / 2
        y = table_top - row_h / 2
        ax_main.text(
            x,
            y,
            col_name,
            ha="center",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            transform=ax_main.transAxes,
        )

    # row labels aligned to row centers
    for i, label in enumerate(row_labels):
        y = table_top - (i + 1.5) * row_h
        ax_main.text(
            left_label_w,
            y,
            label,
            ha="right",
            va="center",
            fontsize=10.4,
            fontweight="bold",
            transform=ax_main.transAxes,
        )

    # cell text
    for i in range(n_rows):
        for j in range(n_cols):
            x = table_left + j * col_w + col_w / 2
            y = table_top - (i + 1.5) * row_h

            txt = matrix[i][j]
            fontsize = 9.6
            if i == n_rows - 1:
                fontsize = 8.8

            ax_main.text(
                x,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=fontsize,
                wrap=True,
                transform=ax_main.transAxes,
            )

    # Footer / takeaway block
    ax_footer.axis("off")
    footer_lines = [
        "Branch Exit: immediate, directed/downstream, weakly compressive.",
        "Stable Seam Corridor: local gateway, short-memory, rapidly compressive.",
        "Reorganization Heavy: path-context, extended-memory, strongly compressive through core/escape bottlenecks.",
    ]

    ax_footer.text(
        0.00, 0.92,
        "Takeaway",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top",
    )
    y = 0.70
    for line in footer_lines:
        ax_footer.text(
            0.02, y, line,
            fontsize=11,
            ha="left",
            va="top",
        )
        y -= 0.24

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the seam family taxonomy figure.")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-png", default=OUTPUT_PNG)
    args = parser.parse_args()

    df = load_summary(args.input_csv)
    render_figure(df, Path(args.output_png))
    print(args.output_png)


if __name__ == "__main__":
    main()
