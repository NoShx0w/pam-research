
import argparse
from pathlib import Path
from typing import Callable, Dict, Any

import pandas as pd
import numpy as np


def trajectory_filename(corpus: str, r: float, alpha: float, seed: int) -> str:
    return f"traj_{corpus}_r{r:.3f}_a{alpha:.6f}_seed{seed}.npz"


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["corpus", "r", "alpha", "seed", "filename"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return df


def filter_manifest(
    df: pd.DataFrame,
    filter_r: list[float] | None = None,
    corpus: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if corpus is not None:
        out = out[out["corpus"] == corpus]
    if filter_r:
        out = out[out["r"].isin(filter_r)]
    if "needs_backfill" in out.columns:
        out = out[out["needs_backfill"]]
    elif "exists" in out.columns:
        out = out[~out["exists"]]
    if limit is not None:
        out = out.head(limit)
    return out.reset_index(drop=True)


def save_trajectory_npz(path: Path, payload: Dict[str, Any]) -> None:
    """
    Save trajectory payload in a repo-compatible .npz container.

    Expected keys usually include:
      F_raw, H_joint, K, pi, Hj_sm, lags, corrs

    This function is deliberately generic so it can be reused after the
    repo-specific run function is wired in.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def default_stub_runner(*, corpus: str, r: float, alpha: float, seed: int) -> Dict[str, Any]:
    """
    Placeholder runner.

    Replace this wiring with the actual single-run trajectory producer from the repo,
    for example by importing a helper extracted from experiments/exp_batch.py.

    It must return a dict of arrays suitable for np.savez().
    """
    raise NotImplementedError(
        "Wire backfill_trajectories.py to the real single-run trajectory generator "
        "used by the Observatory batch pipeline."
    )


def resolve_runner() -> Callable[..., Dict[str, Any]]:
    """
    Try to resolve a repo-native single-run function if available.
    Fallback is an explicit stub with a clear error message.

    Adjust this function once the repository exposes a stable reusable API.
    """
    # Future-friendly hook point:
    # try:
    #     from experiments.exp_batch import run_quench_trajectory_only
    #     return run_quench_trajectory_only
    # except Exception:
    #     pass
    return default_stub_runner


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing PAM trajectory .npz files from a manifest."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/missing_trajectories.csv",
        help="CSV manifest produced by scan_missing_trajectories.py",
    )
    parser.add_argument(
        "--traj-dir",
        default="outputs/trajectories",
        help="Directory where trajectory .npz files are stored",
    )
    parser.add_argument(
        "--filter-r",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of r values to backfill, e.g. --filter-r 0.15 0.10",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional corpus filter, e.g. C",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backfilled without writing files",
    )
    parser.add_argument(
        "--log-csv",
        default="outputs/manifests/backfilled_trajectories.csv",
        help="CSV log of attempted backfills",
    )
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)
    todo = filter_manifest(df, filter_r=args.filter_r, corpus=args.corpus, limit=args.limit)

    print(f"Rows selected for backfill: {len(todo)}")
    if len(todo) == 0:
        return

    if args.dry_run:
        print(todo[["corpus", "r", "alpha", "seed", "filename"]].to_string(index=False))
        return

    runner = resolve_runner()

    logs = []
    for _, row in todo.iterrows():
        corpus = str(row["corpus"])
        r = float(row["r"])
        alpha = float(row["alpha"])
        seed = int(row["seed"])

        filename = row["filename"]
        if pd.isna(filename) or not str(filename).strip():
            filename = trajectory_filename(corpus, r, alpha, seed)

        outpath = traj_dir / str(filename)

        status = "ok"
        error = ""
        try:
            if outpath.exists():
                status = "exists"
            else:
                payload = runner(corpus=corpus, r=r, alpha=alpha, seed=seed)
                if not isinstance(payload, dict):
                    raise TypeError("Runner must return a dict of arrays for np.savez")
                save_trajectory_npz(outpath, payload)
                status = "written"
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        logs.append(
            {
                "corpus": corpus,
                "r": r,
                "alpha": alpha,
                "seed": seed,
                "filename": str(filename),
                "status": status,
                "error": error,
            }
        )
        print(f"[{status}] {filename}" + (f" :: {error}" if error else ""))

    log_df = pd.DataFrame(logs)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print(f"Wrote log: {log_path}")
    print(log_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
