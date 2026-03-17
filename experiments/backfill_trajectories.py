import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time

import pandas as pd


# ---------------------------
# shared filename / manifest
# ---------------------------

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


# ---------------------------
# worker state
# ---------------------------

_WORKER_CONTEXTS = {}
_WORKER_READY = False


def _worker_init():
    """
    Per-process initializer.
    Import heavy modules once inside each worker process.
    """
    global _WORKER_READY
    if _WORKER_READY:
        return

    # Imported lazily so the parent process stays light.
    from experiments.exp_batch import build_run_context, run_one_trajectory_only_with_context

    globals()["_build_run_context"] = build_run_context
    globals()["_run_one_trajectory_only_with_context"] = run_one_trajectory_only_with_context
    _WORKER_READY = True


def _get_context(corpus: str):
    """
    Cache one warm run context per corpus, per worker process.
    """
    global _WORKER_CONTEXTS

    if corpus not in _WORKER_CONTEXTS:
        t0 = time.perf_counter()
        _WORKER_CONTEXTS[corpus] = _build_run_context(corpus_key=corpus)
        dt = time.perf_counter() - t0
        print(f"[worker {os.getpid()}] context for corpus={corpus} ready in {dt:.2f}s", flush=True)

    return _WORKER_CONTEXTS[corpus]


def _worker_run(row: dict) -> dict:
    """
    Run one backfill task in a worker process.
    """
    corpus = str(row["corpus"])
    r = float(row["r"])
    alpha = float(row["alpha"])
    seed = int(row["seed"])
    out_dir = str(row["out_dir"])
    filename = str(row["filename"])

    outpath = Path(out_dir) / "trajectories" / filename

    t0 = time.perf_counter()
    status = "ok"
    error = ""

    try:
        if outpath.exists():
            status = "exists"
        else:
            ctx = _get_context(corpus)
            result = _run_one_trajectory_only_with_context(
                ctx=ctx,
                r=r,
                alpha=alpha,
                seed=seed,
                out_dir=out_dir,
            )
            written_path = Path(result["trajectory_path"])
            if not written_path.exists():
                raise FileNotFoundError(f"Backfill runner did not create {written_path}")
            status = "written"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    elapsed_sec = time.perf_counter() - t0

    return {
        "corpus": corpus,
        "r": r,
        "alpha": alpha,
        "seed": seed,
        "filename": filename,
        "status": status,
        "elapsed_sec": elapsed_sec,
        "error": error,
        "pid": os.getpid(),
    }


# ---------------------------
# main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing PAM trajectory .npz files using a warm worker pool."
    )
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/missing_trajectories.csv",
        help="CSV manifest produced by scan_missing_trajectories.py",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Repository outputs directory",
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
        "--workers",
        type=int,
        default=6,
        help="Number of worker processes",
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

    out_dir = Path(args.out_dir)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)
    todo = filter_manifest(df, filter_r=args.filter_r, corpus=args.corpus, limit=args.limit)

    print(f"Rows selected for backfill: {len(todo)}")
    if len(todo) == 0:
        return

    if args.dry_run:
        print(todo[["corpus", "r", "alpha", "seed", "filename"]].to_string(index=False))
        return

    rows = []
    for _, row in todo.iterrows():
        corpus = str(row["corpus"])
        r = float(row["r"])
        alpha = float(row["alpha"])
        seed = int(row["seed"])
        filename = row["filename"]
        if pd.isna(filename) or not str(filename).strip():
            filename = trajectory_filename(corpus, r, alpha, seed)

        rows.append(
            {
                "corpus": corpus,
                "r": r,
                "alpha": alpha,
                "seed": seed,
                "filename": str(filename),
                "out_dir": str(out_dir),
            }
        )

    print(f"Launching worker pool with workers={args.workers}")
    logs = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        futures = [ex.submit(_worker_run, row) for row in rows]

        for fut in as_completed(futures):
            result = fut.result()
            logs.append(result)

            status = result["status"]
            filename = result["filename"]
            elapsed_sec = result["elapsed_sec"]
            pid = result["pid"]
            error = result["error"]

            print(
                f"[{status}] {filename} ({elapsed_sec:.2f}s, pid={pid})"
                + (f" :: {error}" if error else ""),
                flush=True,
            )

    log_df = pd.DataFrame(logs).sort_values(["corpus", "r", "alpha", "seed"]).reset_index(drop=True)
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print(f"Wrote log: {log_path}")
    print(log_df["status"].value_counts(dropna=False).to_string())
    if len(log_df) > 0:
        print(f"Mean elapsed_sec: {log_df['elapsed_sec'].mean():.2f}s")


if __name__ == "__main__":
    main()
