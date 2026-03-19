#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Orthogonality test: Lazarus vs seam distance.")
    parser.add_argument("--input", default="outputs/fim_transition_rate/transition_rate_labeled.csv")
    parser.add_argument("--outdir", default="outputs/fim_orthogonality")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    df["lazarus_score"] = pd.to_numeric(df["lazarus_score"], errors="coerce")
    df["distance_to_seam"] = pd.to_numeric(df["distance_to_seam"], errors="coerce")
    df["transition_within_k"] = pd.to_numeric(df["transition_within_k"], errors="coerce")

    df = df.dropna(subset=["lazarus_score", "distance_to_seam"])

    # Correlation
    corr = df["lazarus_score"].corr(df["distance_to_seam"])

    with open(outdir / "orthogonality_summary.txt", "w") as f:
        f.write(f"pearson_correlation_lazarus_vs_distance: {corr}\n")

    # Scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(df["distance_to_seam"], df["lazarus_score"], alpha=0.2)
    plt.xlabel("distance_to_seam")
    plt.ylabel("lazarus_score")
    plt.title("Orthogonality test: Lazarus vs seam distance")
    plt.tight_layout()
    plt.savefig(outdir / "lazarus_vs_distance.png", dpi=200)
    plt.close()

    # Conditional test (bin by distance)
    df["distance_bin"] = pd.qcut(df["distance_to_seam"], q=5, duplicates="drop")

    rows = []
    for b, g in df.groupby("distance_bin"):
        if len(g) < 10:
            continue
        thr = g["lazarus_score"].median()
        high = g[g["lazarus_score"] >= thr]
        low = g[g["lazarus_score"] < thr]

        high_rate = high["transition_within_k"].mean()
        low_rate = low["transition_within_k"].mean()

        rows.append({
            "distance_bin": str(b),
            "n": len(g),
            "high_rate": high_rate,
            "low_rate": low_rate,
            "diff": high_rate - low_rate
        })

    pd.DataFrame(rows).to_csv(outdir / "conditional_lazarus_effect.csv", index=False)

    print(outdir)


if __name__ == "__main__":
    main()
