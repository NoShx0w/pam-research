import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def render_figure_4(df: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))

    ax = axes[0, 0]
    ax.plot(df["probe_count"], df["transition_rate_high"], marker="o", label="high Lazarus")
    ax.plot(df["probe_count"], df["transition_rate_low"], marker="o", label="low Lazarus")
    ax.set_xscale("log")
    ax.set_xlabel("probe count")
    ax.set_ylabel("transition rate")
    ax.set_title("A. Transition-rate stability")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["probe_count"], df["transition_rate_ratio"], marker="o", label="rate ratio")
    ax.plot(df["probe_count"], df["transition_rate_diff"], marker="o", label="rate difference")
    ax.set_xscale("log")
    ax.set_xlabel("probe count")
    ax.set_ylabel("effect size")
    ax.set_title("B. Effect-size stabilization")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["probe_count"], df["median_lag_lazarus_to_seam"], marker="o", label="median lag to seam")
    ax.plot(df["probe_count"], df["median_lag_lazarus_to_flip"], marker="o", label="median lag to flip")
    ax.plot(df["probe_count"], df["mean_lag_lazarus_to_seam"], marker="o", linestyle="--", label="mean lag to seam")
    ax.plot(df["probe_count"], df["mean_lag_lazarus_to_flip"], marker="o", linestyle="--", label="mean lag to flip")
    ax.set_xscale("log")
    ax.set_xlabel("probe count")
    ax.set_ylabel("lag (steps)")
    ax.set_title("C. Temporal-order stability")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(df["probe_count"], df["seam_cross_rate_precollapse"], marker="o", label="precollapse")
    ax.plot(df["probe_count"], df["seam_cross_rate_no_precollapse"], marker="o", label="no precollapse")
    ax.set_xscale("log")
    ax.set_xlabel("probe count")
    ax.set_ylabel("boundary interaction rate")
    ax.set_title("D. Boundary-interaction stability")
    ax.legend()

    fig.suptitle("Figure 4 — Scaling robustness", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(outpath, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render Figure 4 — Scaling robustness.")
    parser.add_argument("--scaling-summary-csv", default="outputs/fim_scaling/scaling_summary.csv")
    parser.add_argument("--outdir", default="outputs/fim_figure_4")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.scaling_summary_csv).sort_values("probe_count")
    outpath = outdir / "figure_4_scaling_robustness.png"
    render_figure_4(df, outpath)

    print(outpath)


if __name__ == "__main__":
    main()
