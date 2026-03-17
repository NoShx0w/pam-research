import csv
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pam.types import RunParams

from common_quench_metrics import (
    CORPORA,
    build_tip,
    build_injector,
    run_one_summary,
)

OUT_DIR = Path("outputs")


INDEX_FIELDNAMES = [
    "filename",
    "corpus",
    "alpha",
    "r",
    "iters",
    "W",
    "seed",
    "run_id",
    "piF_mean",
    "piF_tail",
    "H_joint_mean",
    "var_H_joint",
    "H_min",
    "H_max",
    "K_min",
    "K_max",
    "corr0",
    "delta_r2_freeze",
    "delta_r2_entropy",
    "best_lag",
    "best_corr",
]


def load_completed_keys(index_path):
    if not index_path.exists():
        return set()

    keys = set()
    with index_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["corpus"],
                float(row["r"]),
                float(row["alpha"]),
                int(row["iters"]),
                int(row["W"]),
                int(row["seed"]),
            )
            keys.add(key)
    return keys


def build_meta(corpus_key, r, alpha, iters, W, seed):
    return {
        "corpus": corpus_key,
        "r": r,
        "alpha": alpha,
        "iters": iters,
        "W": W,
        "seed": seed,
    }


def append_summary_index_row(meta, summary, out_dir="outputs", filename=""):
    index_path = Path(out_dir) / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "filename": filename,
        "corpus": meta["corpus"],
        "alpha": meta["alpha"],
        "r": meta["r"],
        "iters": meta["iters"],
        "W": meta["W"],
        "seed": meta["seed"],
        "run_id": "",
        "piF_mean": summary["piF_mean"],
        "piF_tail": summary["piF_tail"],
        "H_joint_mean": summary["H_joint_mean"],
        "var_H_joint": summary["var_H_joint"],
        "H_min": summary["H_min"],
        "H_max": summary["H_max"],
        "K_min": summary["K_min"],
        "K_max": summary["K_max"],
        "corr0": summary["corr0"],
        "delta_r2_freeze": summary["delta_r2_freeze"],
        "delta_r2_entropy": summary["delta_r2_entropy"],
        "best_lag": summary["best_lag"],
        "best_corr": summary["best_corr"],
    }

    write_header = not index_path.exists()

    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one_job(job):
    corpus_key, r, alpha, iters, W, seed = job

    texts0 = CORPORA[corpus_key]
    tip = build_tip()
    mix_inj = build_injector(tip, texts0)

    params = RunParams(
        alpha=alpha,
        r=r,
        seed=seed,
        iters=iters,
        anchor_set_size=10,
        store_snapshots=True,
        store_every=1,
    )

    traj_name = build_trajectory_filename(corpus_key, r, alpha, seed)
    traj_path = OUT_DIR / "trajectories" / traj_name

    summary = run_one_summary(
        texts0=texts0,
        tip=tip,
        mix_inj=mix_inj,
        params=params,
        alpha=alpha,
        W=W,
        save_trajectory=True,
        trajectory_path=str(traj_path),
    )

    meta = build_meta(corpus_key, r, alpha, iters, W, seed)
    return meta, summary


def iter_jobs(corpus_key, rs, alphas, seeds, iters, W, completed):
    for r, alpha, seed in itertools.product(rs, alphas, seeds):
        key = (
            corpus_key,
            float(r),
            float(alpha),
            iters,
            W,
            seed,
        )
        if key in completed:
            continue
        yield (corpus_key, r, alpha, iters, W, seed)


def build_trajectory_filename(corpus_key, r, alpha, seed):
    return f"traj_{corpus_key}_r{r:.3f}_a{alpha:.6f}_seed{seed}.npz"


def run_one_trajectory_only(
    *,
    corpus_key: str,
    r: float,
    alpha: float,
    seed: int,
    iters: int = 300,
    W: int = 30,
    out_dir: str = "outputs",
):
    ctx = build_run_context(corpus_key=corpus_key)
    return run_one_trajectory_only_with_context(
        ctx=ctx,
        r=r,
        alpha=alpha,
        seed=seed,
        iters=iters,
        W=W,
        out_dir=out_dir,
    )


def build_run_context(*, corpus_key: str):
    """
    Build reusable heavy objects for repeated runs on the same corpus.
    """
    texts0 = CORPORA[corpus_key]
    tip = build_tip()
    mix_inj = build_injector(tip, texts0)

    return {
        "corpus_key": corpus_key,
        "texts0": texts0,
        "tip": tip,
        "mix_inj": mix_inj,
    }


def run_one_trajectory_only_with_context(
    *,
    ctx,
    r: float,
    alpha: float,
    seed: int,
    iters: int = 300,
    W: int = 30,
    out_dir: str = "outputs",
):
    """
    Run one Observatory quench using a prebuilt reusable context.
    """
    corpus_key = ctx["corpus_key"]
    texts0 = ctx["texts0"]
    tip = ctx["tip"]
    mix_inj = ctx["mix_inj"]

    params = RunParams(
        alpha=alpha,
        r=r,
        seed=seed,
        iters=iters,
        anchor_set_size=10,
        store_snapshots=True,
        store_every=1,
    )

    traj_name = build_trajectory_filename(corpus_key, r, alpha, seed)
    traj_path = Path(out_dir) / "trajectories" / traj_name

    summary = run_one_summary(
        texts0=texts0,
        tip=tip,
        mix_inj=mix_inj,
        params=params,
        alpha=alpha,
        W=W,
        save_trajectory=True,
        trajectory_path=str(traj_path),
    )

    return {
        "filename": traj_name,
        "trajectory_path": str(traj_path),
        "summary": summary,
    }


def main():
    corpus_key = "C"

    rs = [0.10, 0.15, 0.20, 0.25, 0.30]
    alphas = np.linspace(0.03, 0.15, 15)
    seeds = range(10)

    iters = 300
    W = 30
    max_workers = 6
    max_in_flight = max_workers * 2

    index_path = OUT_DIR / "index.csv"
    completed = load_completed_keys(index_path)

    jobs = list(iter_jobs(corpus_key, rs, alphas, seeds, iters, W, completed))
    total = len(jobs)

    print("Jobs remaining:", total)

    if not jobs:
        print("All jobs already complete.")
        return

    done = 0
    job_iter = iter(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        in_flight = {}

        # Prime the queue with a bounded number of tasks.
        for _ in range(min(max_in_flight, total)):
            job = next(job_iter, None)
            if job is None:
                break
            fut = pool.submit(run_one_job, job)
            in_flight[fut] = job

        while in_flight:
            for fut in as_completed(list(in_flight)):
                job = in_flight.pop(fut)

                meta, summary = fut.result()
                append_summary_index_row(meta, summary, out_dir=str(OUT_DIR))

                done += 1
                print(f"progress {done}/{total}", end="\r")

                # Submit one replacement job to keep the queue bounded.
                next_job = next(job_iter, None)
                if next_job is not None:
                    new_fut = pool.submit(run_one_job, next_job)
                    in_flight[new_fut] = next_job

                # Break so we re-enter as_completed with the updated in_flight set.
                break

    print()
    print("Batch complete.")


if __name__ == "__main__":
    main()
