import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--traj-dir", default="outputs/trajectories")
    parser.add_argument("--outdir", default="outputs/manifests")

    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    r_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    alpha_values = np.linspace(0.03, 0.15, 15)
    seeds = range(10)

    rows = []

    for r in r_values:
        for a in alpha_values:
            for s in seeds:

                fname = f"traj_r{r:.2f}_a{a:.6f}_seed{s}.npz"

                path = traj_dir / fname

                rows.append({
                    "r": r,
                    "alpha": a,
                    "seed": s,
                    "filename": fname,
                    "exists": path.exists()
                })

    df = pd.DataFrame(rows)

    missing = df[~df.exists]

    df.to_csv(outdir / "expected_trajectories.csv", index=False)
    missing.to_csv(outdir / "missing_trajectories.csv", index=False)

    print("Expected:", len(df))
    print("Missing:", len(missing))


if __name__ == "__main__":
    main()
