import argparse
from pathlib import Path
import time

import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing PAM trajectory .npz files from a manifest using a warm per-corpus run context."
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

    from experiments.exp_batch import (
        build_run_context,
        run_one_trajectory_only_with_context,
    )

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

    contexts = {}
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

        t0 = time.perf_counter()
        try:
            if outpath.exists():
                status = "exists"
            else:
                if corpus not in contexts:
                    t_ctx0 = time.perf_counter()
                    print(f"[init] building warm context for corpus={corpus}")
                    contexts[corpus] = build_run_context(corpus_key=corpus)
                    t_ctx = time.perf_counter() - t_ctx0
                    print(f"[init] corpus={corpus} context ready in {t_ctx:.2f}s")

                result = run_one_trajectory_only_with_context(
                    ctx=contexts[corpus],
                    r=r,
                    alpha=alpha,
                    seed=seed,
                )

                written_path = Path(result["trajectory_path"])
                if not written_path.exists():
                    raise FileNotFoundError(f"Backfill runner did not create {written_path}")

                status = "written"
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        elapsed_sec = time.perf_counter() - t0

        logs.append(
            {
                "corpus": corpus,
                "r": r,
                "alpha": alpha,
                "seed": seed,
                "filename": str(filename),
                "status": status,
                "elapsed_sec": elapsed_sec,
                "error": error,
            }
        )
        print(f"[{status}] {filename} ({elapsed_sec:.2f}s)" + (f" :: {error}" if error else ""))

    log_df = pd.DataFrame(logs)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print(f"Wrote log: {log_path}")
    print(log_df["status"].value_counts(dropna=False).to_string())
    if len(log_df) > 0:
        print(f"Mean elapsed_sec: {log_df['elapsed_sec'].mean():.2f}s")


if __name__ == "__main__":
    main()
