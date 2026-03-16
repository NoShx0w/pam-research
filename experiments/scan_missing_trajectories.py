import argparse
from pathlib import Path

import pandas as pd


def trajectory_filename(corpus: str, r: float, alpha: float, seed: int) -> str:
    return f"traj_{corpus}_r{r:.3f}_a{alpha:.6f}_seed{seed}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="outputs/index.csv")
    parser.add_argument("--traj-dir", default="outputs/trajectories")
    parser.add_argument("--outdir", default="outputs/manifests")
    args = parser.parse_args()

    index_csv = Path(args.index_csv)
    traj_dir = Path(args.traj_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)

    required = ["corpus", "r", "alpha", "seed"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in index.csv: {missing_cols}")

    manifest = (
        df[required]
        .drop_duplicates()
        .sort_values(["corpus", "r", "alpha", "seed"])
        .reset_index(drop=True)
        .copy()
    )

    manifest["filename"] = manifest.apply(
        lambda row: trajectory_filename(
            str(row["corpus"]),
            float(row["r"]),
            float(row["alpha"]),
            int(row["seed"]),
        ),
        axis=1,
    )

    manifest["exists"] = manifest["filename"].apply(lambda x: (traj_dir / x).exists())
    manifest["needs_backfill"] = ~manifest["exists"]

    expected_path = outdir / "expected_trajectories.csv"
    missing_path = outdir / "missing_trajectories.csv"

    manifest.to_csv(expected_path, index=False)
    manifest[manifest["needs_backfill"]].to_csv(missing_path, index=False)

    print(f"Expected: {len(manifest)}")
    print(f"Missing: {int(manifest['needs_backfill'].sum())}")
    print(expected_path)
    print(missing_path)


if __name__ == "__main__":
    main()
