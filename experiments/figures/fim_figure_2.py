import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def render_figure_2(
    transition_summary: pd.DataFrame,
    transition_labeled: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    outpath: Path,
    within_k: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))

    # A. Transition rate by Lazarus group
    ax = axes[0, 0]
    ax.bar(transition_summary["lazarus_group"], transition_summary["transition_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P(transition within {within_k} steps)")
    ax.set_title("A. Transition rate by Lazarus exposure")

    # B. Transition lag vs Lazarus score
    ax = axes[0, 1]
    plot_df = transition_labeled[["lazarus_score", "lag_to_next_transition"]].copy()
    plot_df["lazarus_score"] = pd.to_numeric(plot_df["lazarus_score"], errors="coerce")
    plot_df["lag_to_next_transition"] = pd.to_numeric(plot_df["lag_to_next_transition"], errors="coerce")
    plot_df = plot_df.dropna()
    ax.scatter(plot_df["lazarus_score"], plot_df["lag_to_next_transition"], alpha=0.25)
    ax.set_xlabel("lazarus_score")
    ax.set_ylabel("lag_to_next_transition")
    ax.set_title("B. Transition lag vs Lazarus score")

    # C. Boundary interaction by pre-collapse regime
    ax = axes[1, 0]
    if "group" in horizon_summary.columns and "seam_cross_rate" in horizon_summary.columns:
        ax.bar(horizon_summary["group"], horizon_summary["seam_cross_rate"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(boundary interaction)")
    else:
        ax.text(0.5, 0.5, "Missing horizon summary columns", ha="center", va="center")
    ax.set_title("C. Boundary interaction by pre-collapse regime")

    # D. Temporal ordering summary
    ax = axes[1, 1]
    if len(temporal_summary) > 0:
        row = temporal_summary.iloc[0]
        labels = [
            "share\nprecedes seam",
            "share\nprecedes flip",
            "median lag\nto seam",
            "median lag\nto flip",
        ]
        values = [
            float(row.get("share_lazarus_precedes_seam", 0.0)),
            float(row.get("share_lazarus_precedes_flip", 0.0)),
            float(row.get("median_lag_lazarus_to_seam", 0.0)),
            float(row.get("median_lag_lazarus_to_flip", 0.0)),
        ]
        ax.bar(labels, values)
        ax.set_title("D. Temporal ordering summary")
    else:
        ax.text(0.5, 0.5, "Missing temporal summary", ha="center", va="center")
        ax.set_title("D. Temporal ordering summary")

    fig.suptitle("Figure 2 — Transition-rate law", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render Figure 2 — Transition-rate law.")
    parser.add_argument("--transition-summary-csv", default="outputs/fim_transition_rate/transition_rate_summary.csv")
    parser.add_argument("--transition-labeled-csv", default="outputs/fim_transition_rate/transition_rate_labeled.csv")
    parser.add_argument("--horizon-summary-csv", default="outputs/fim_horizon/horizon_predictive_summary_from_probes.csv")
    parser.add_argument("--temporal-summary-csv", default="outputs/fim_lazarus_temporal/lazarus_temporal_summary.csv")
    parser.add_argument("--outdir", default="outputs/fim_figure_2")
    parser.add_argument("--within-k", type=int, default=2)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transition_summary = load_csv(args.transition_summary_csv)
    transition_labeled = load_csv(args.transition_labeled_csv)
    horizon_summary = load_csv(args.horizon_summary_csv)
    temporal_summary = load_csv(args.temporal_summary_csv)

    outpath = outdir / "figure_2_transition_rate_law.png"
    render_figure_2(
        transition_summary,
        transition_labeled,
        horizon_summary,
        temporal_summary,
        outpath,
        within_k=args.within_k,
    )
    print(outpath)


if __name__ == "__main__":
    main()
