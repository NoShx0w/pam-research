#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


OUTCOME_ORDER = ["in_basin", "seam_graze", "seam_cross", "lazarus", "collapse"]
OUTCOME_PRIORITY = {
    "in_basin": 0,
    "seam_graze": 1,
    "seam_cross": 2,
    "lazarus": 3,
    "collapse": 4,
}
OUTCOME_TO_CODE = {name: i for i, name in enumerate(OUTCOME_ORDER)}


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def choose_dominant_outcome(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values(by=["rate", "outcome_priority"], ascending=[False, False])
    return g.iloc[0]


def build_dominant_table(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    work["outcome_priority"] = work["outcome_class"].map(OUTCOME_PRIORITY)

    dominant = (
        work.groupby(["corpus", "r", "alpha"], dropna=False, group_keys=False)
        .apply(choose_dominant_outcome)
        .reset_index(drop=True)
    )

    pooled = (
        work.groupby(["r", "alpha", "outcome_class"], dropna=False, as_index=False)
        .agg(n=("n", "sum"), n_total=("n_total", "sum"))
    )
    pooled["rate"] = pooled["n"] / pooled["n_total"]
    pooled["outcome_priority"] = pooled["outcome_class"].map(OUTCOME_PRIORITY)

    pooled_dom = (
        pooled.groupby(["r", "alpha"], dropna=False, group_keys=False)
        .apply(choose_dominant_outcome)
        .reset_index(drop=True)
    )
    pooled_dom["corpus"] = "all"

    out = pd.concat([dominant, pooled_dom], ignore_index=True)
    out["outcome_code"] = out["outcome_class"].map(OUTCOME_TO_CODE)
    return out.sort_values(["corpus", "r", "alpha"]).reset_index(drop=True)


def draw_panel(ax, panel_df: pd.DataFrame, title: str, cmap, norm) -> None:
    if len(panel_df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    panel_df = panel_df.sort_values(["r", "alpha"])
    ax.scatter(
        panel_df["alpha"],
        panel_df["r"],
        c=panel_df["outcome_code"],
        cmap=cmap,
        norm=norm,
        s=120,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_xlim(panel_df["alpha"].min() - 0.01, panel_df["alpha"].max() + 0.01)
    ax.set_ylim(panel_df["r"].min() - 0.01, panel_df["r"].max() + 0.01)
    ax.grid(alpha=0.2, linestyle=":")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render phase selection diagram.")
    parser.add_argument(
        "--summary-csv",
        default="outputs/fim_initial_conditions/initial_conditions_outcome_summary.csv",
    )
    parser.add_argument("--outdir", default="outputs/fim_initial_conditions")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = safe_numeric(df, ["r", "alpha", "n", "n_total", "rate"])

    dominant = build_dominant_table(df)
    dominant_path = outdir / "phase_selection_dominant_outcomes.csv"
    dominant.to_csv(dominant_path, index=False)

    families = [c for c in sorted(dominant["corpus"].dropna().unique()) if c != "all"]
    n_panels = len(families) + 1
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    cmap = plt.cm.get_cmap("viridis", len(OUTCOME_ORDER))
    norm = plt.Normalize(vmin=-0.5, vmax=len(OUTCOME_ORDER) - 0.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.3 * nrows))
    axes = np.array(axes).reshape(-1)

    draw_panel(
        axes[0],
        dominant[dominant["corpus"] == "all"],
        "A. Overall dominant outcome",
        cmap,
        norm,
    )

    for i, family in enumerate(families, start=1):
        label = chr(ord("A") + i)
        draw_panel(
            axes[i],
            dominant[dominant["corpus"] == family],
            f"{label}. {family}",
            cmap,
            norm,
        )

    for j in range(len(families) + 1, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mpatches.Patch(color=cmap(norm(OUTCOME_TO_CODE[name])), label=name)
        for name in OUTCOME_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(OUTCOME_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle("Phase selection diagram — dominant outcome over initial conditions", fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    fig_path = outdir / "phase_selection_diagram.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(fig_path)
    print(dominant_path)


if __name__ == "__main__":
    main()
