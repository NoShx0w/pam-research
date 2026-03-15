import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--curvature", default="outputs/fim_curvature/curvature_surface.csv")
    parser.add_argument("--outdir", default="outputs/fim_phase")
    parser.add_argument("--threshold", type=float, default=10)

    args = parser.parse_args()

    df = pd.read_csv(args.curvature)

    # magnitude of curvature
    df["absK"] = np.abs(df.scalar_curvature)

    # ridge candidates
    ridge = df[df.absK > args.threshold].copy()

    # sort along r
    ridge = ridge.sort_values("r")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    ridge.to_csv(Path(args.outdir) / "phase_boundary_points.csv", index=False)

    # plot in parameter space
    fig, ax = plt.subplots()

    ax.scatter(df.alpha, df.r, s=20, alpha=0.3)

    ax.scatter(
        ridge.alpha,
        ridge.r,
        s=80,
        color="red"
    )

    ax.plot(
        ridge.alpha,
        ridge.r,
        linewidth=2
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("r")

    ax.set_title("PAM phase boundary (curvature ridge)")

    fig.tight_layout()

    fig.savefig(Path(args.outdir) / "phase_boundary.png", dpi=180)

    plt.close()

    print("boundary extracted")


if __name__ == "__main__":
    main()
