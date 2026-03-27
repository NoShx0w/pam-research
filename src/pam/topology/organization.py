"""Canonical organizational topology stage for the PAM manifold."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

__all__ = [
    "OUTCOME_ORDER",
    "OUTCOME_PRIORITY",
    "OUTCOME_TO_CODE",
    "run_organization",
]


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_dominant_table(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    work["outcome_priority"] = work["outcome_class"].map(OUTCOME_PRIORITY)

    dominant = (
        work.sort_values(
            by=["corpus", "r", "alpha", "rate", "outcome_priority"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(subset=["corpus", "r", "alpha"], keep="first")
        .copy()
    )

    pooled = (
        work.groupby(["r", "alpha", "outcome_class"], dropna=False, as_index=False)
        .agg(
            n=("n", "sum"),
            n_total=("n_total", "sum"),
        )
    )
    pooled["rate"] = pooled["n"] / pooled["n_total"]
    pooled["outcome_priority"] = pooled["outcome_class"].map(OUTCOME_PRIORITY)

    pooled_dom = (
        pooled.sort_values(
            by=["r", "alpha", "rate", "outcome_priority"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["r", "alpha"], keep="first")
        .copy()
    )
    pooled_dom["corpus"] = "all"

    out = pd.concat([dominant, pooled_dom], ignore_index=True, sort=False)
    out["outcome_code"] = out["outcome_class"].map(OUTCOME_TO_CODE)

    keep_cols = [
        "corpus",
        "r",
        "alpha",
        "outcome_class",
        "n",
        "n_total",
        "rate",
        "outcome_code",
    ]
    out = out[keep_cols]

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


def run_organization(
    summary_csv,
    outdir,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
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

    return dominant
