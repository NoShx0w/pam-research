import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--traj-dir", default="outputs/trajectories")

    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    traj_dir = Path(args.traj_dir)

    bad = []

    for _, row in df.iterrows():

        path = traj_dir / row.filename

        if not path.exists():
            bad.append(row.filename)
            continue

        try:
            data = np.load(path)

            if "freeze" not in data:
                bad.append(row.filename)

        except Exception:
            bad.append(row.filename)

    print("Invalid trajectories:", len(bad))

    if bad:
        print(bad[:10])


if __name__ == "__main__":
    main()
