import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Predictive test: does Lazarus exposure predict seam crossing?"
    )
    parser.add_argument("--paths-csv", default="outputs/fim_ops/operator_S_paths.csv")
    parser.add_argument("--metrics-csv", default="outputs/fim_ops/operator_S_metrics.csv")
    parser.add_argument("--lazarus-csv", default="outputs/fim_lazarus/lazarus_scores.csv")
    parser.add_argument("--outdir", default="outputs/fim_lazarus")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(args.paths_csv)
    metrics = pd.read_csv(args.metrics_csv)
    laz = pd.read_csv(args.lazarus_csv)

    joined = paths.merge(
        laz[["r", "alpha", "lazarus_score", "lazarus_hit"]],
        on=["r", "alpha"],
        how="left",
    )

    agg = (
        joined.groupby("path_id", as_index=False)
        .agg(
            lazarus_max=("lazarus_score", "max"),
            lazarus_mean=("lazarus_score", "mean"),
            lazarus_hit_any=("lazarus_hit", "max"),
        )
    )

    test_df = metrics.merge(agg, on="path_id", how="left")

    # median split
    med = test_df["lazarus_max"].median()
    test_df["lazarus_group"] = test_df["lazarus_max"].apply(
        lambda x: "high" if x >= med else "low"
    )

    summary = (
        test_df.groupby("lazarus_group", as_index=False)
        .agg(
            n_paths=("path_id", "count"),
            seam_cross_rate=("crosses_seam", "mean"),
            mean_phase_flip_count=("phase_flip_count", "mean"),
            mean_min_distance_to_seam=("min_distance_to_seam", "mean"),
            mean_max_curvature=("max_curvature_along_path", "mean"),
            mean_path_length=("path_length_fisher", "mean"),
        )
    )

    summary.to_csv(outdir / "lazarus_predictive_summary.csv", index=False)
    test_df.to_csv(outdir / "lazarus_predictive_test.csv", index=False)

    # simple bar plot
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.bar(summary["lazarus_group"], summary["seam_cross_rate"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(crosses_seam)")
    ax.set_title("Seam crossing rate by Lazarus exposure")
    fig.tight_layout()
    fig.savefig(outdir / "lazarus_predictive_bar.png", dpi=220)
    plt.close(fig)

    print(outdir / "lazarus_predictive_test.csv")
    print(outdir / "lazarus_predictive_summary.csv")
    print(outdir / "lazarus_predictive_bar.png")


if __name__ == "__main__":
    main()
