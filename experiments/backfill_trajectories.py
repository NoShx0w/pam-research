import argparse
from pathlib import Path
import pandas as pd

from experiments.exp_batch import run_one_quench   # reuse existing code


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--traj-dir", default="outputs/trajectories")

    parser.add_argument("--filter-r", nargs="*", type=float)

    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    if args.filter_r:
        df = df[df.r.isin(args.filter_r)]

    for _, row in df.iterrows():

        r = row.r
        alpha = row.alpha
        seed = int(row.seed)

        fname = f"traj_r{r:.2f}_a{alpha:.6f}_seed{seed}.npz"
        path = traj_dir / fname

        if path.exists():
            continue

        print("Backfilling", fname)

        run_one_quench(
            r=r,
            alpha=alpha,
            seed=seed,
            traj_path=path
        )


if __name__ == "__main__":
    main()
