import argparse
import csv
import itertools
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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

MANIFEST_FIELDNAMES = [
    "job_id",
    "corpus",
    "r",
    "alpha",
    "iters",
    "W",
    "seed",
    "trajectory_filename",
    "status",
    "started_at",
    "finished_at",
    "duration_sec",
    "error",
]


@dataclass(frozen=True)
class BatchPaths:
    root: Path
    run_name: str

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def trajectories_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def index_csv(self) -> Path:
        return self.root / "index.csv"

    @property
    def run_spec_path(self) -> Path:
        return self.root / "run_spec.json"

    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / f"{self.run_name}_manifest.csv"

    @property
    def progress_path(self) -> Path:
        return self.manifests_dir / f"{self.run_name}_progress.json"

    @property
    def events_path(self) -> Path:
        return self.logs_dir / f"{self.run_name}_events.jsonl"


def campaign_root(base_out_dir: str | Path, corpus_key: str, campaign: str) -> Path:
    return Path(base_out_dir) / "corpora" / corpus_key / "campaigns" / campaign


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_job_id(corpus_key: str, r: float, alpha: float, iters: int, W: int, seed: int) -> str:
    return f"{corpus_key}|r={r:.3f}|a={alpha:.6f}|iters={iters}|W={W}|seed={seed}"


def build_trajectory_filename(corpus_key: str, r: float, alpha: float, seed: int) -> str:
    return f"traj_{corpus_key}_r{r:.3f}_a{alpha:.6f}_seed{seed}.npz"


def ensure_runtime_dirs(paths: BatchPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.trajectories_dir.mkdir(parents=True, exist_ok=True)


def append_event(paths: BatchPaths, event_type: str, payload: dict) -> None:
    ensure_runtime_dirs(paths)
    record = {
        "ts": utc_now_iso(),
        "event": event_type,
        **payload,
    }
    with paths.events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest_rows(paths: BatchPaths, rows: list[dict]) -> None:
    ensure_runtime_dirs(paths)
    with paths.manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_rows(paths: BatchPaths) -> list[dict]:
    if not paths.manifest_path.exists():
        return []
    with paths.manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_manifest_row(paths: BatchPaths, job_id: str, **updates) -> None:
    rows = load_manifest_rows(paths)
    changed = False
    for row in rows:
        if row["job_id"] == job_id:
            for k, v in updates.items():
                row[k] = "" if v is None else str(v)
            changed = True
            break
    if changed:
        write_manifest_rows(paths, rows)


def bootstrap_manifest(
    paths: BatchPaths,
    jobs: list[tuple[str, float, float, int, int, int]],
    iters: int,
    W: int,
) -> None:
    rows = load_manifest_rows(paths)
    existing = {row["job_id"] for row in rows}

    for corpus_key, r, alpha, _, _, seed in jobs:
        job_id = build_job_id(corpus_key, r, alpha, iters, W, seed)
        if job_id in existing:
            continue
        rows.append(
            {
                "job_id": job_id,
                "corpus": corpus_key,
                "r": r,
                "alpha": alpha,
                "iters": iters,
                "W": W,
                "seed": seed,
                "trajectory_filename": build_trajectory_filename(corpus_key, r, alpha, seed),
                "status": "pending",
                "started_at": "",
                "finished_at": "",
                "duration_sec": "",
                "error": "",
            }
        )
    write_manifest_rows(paths, rows)


def write_progress_snapshot(paths: BatchPaths, start_time: float) -> None:
    rows = load_manifest_rows(paths)

    total = len(rows)
    done = sum(r["status"] == "done" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    running = sum(r["status"] == "running" for r in rows)
    pending = sum(r["status"] == "pending" for r in rows)

    elapsed = max(time.time() - start_time, 1e-9)
    throughput_per_min = done / elapsed * 60.0
    remaining = pending + running
    eta_sec = (remaining / max(done / elapsed, 1e-9)) if done > 0 else None

    finished_rows = [r for r in rows if r["status"] in {"done", "failed"} and r["finished_at"]]
    finished_rows.sort(key=lambda x: x["finished_at"])
    last_completed = finished_rows[-1]["job_id"] if finished_rows else None

    failed_rows = [r for r in rows if r["status"] == "failed" and r["error"]]
    last_error = failed_rows[-1]["error"] if failed_rows else None

    snapshot = {
        "run_name": paths.run_name,
        "root": str(paths.root),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": utc_now_iso(),
        "total": total,
        "done": done,
        "failed": failed,
        "running": running,
        "pending": pending,
        "percent": (done + failed) / total if total else 1.0,
        "elapsed_sec": elapsed,
        "throughput_jobs_per_min": throughput_per_min,
        "eta_sec": eta_sec,
        "last_completed": last_completed,
        "last_error": last_error,
    }

    ensure_runtime_dirs(paths)
    with paths.progress_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def load_completed_keys(index_path: Path) -> set[tuple[str, float, float, int, int, int]]:
    if not index_path.exists():
        return set()

    keys: set[tuple[str, float, float, int, int, int]] = set()
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


def build_meta(corpus_key: str, r: float, alpha: float, iters: int, W: int, seed: int) -> dict:
    return {
        "corpus": corpus_key,
        "r": r,
        "alpha": alpha,
        "iters": iters,
        "W": W,
        "seed": seed,
    }


def append_summary_index_row(meta: dict, summary: dict, index_path: Path, filename: str = "") -> None:
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


def build_run_spec(
    *,
    corpus_key: str,
    campaign: str,
    rs,
    alphas,
    seeds,
    iters: int,
    W: int,
) -> dict:
    return {
        "corpus": corpus_key,
        "campaign": campaign,
        "rs": [float(x) for x in rs],
        "alphas": [float(x) for x in alphas],
        "seeds": [int(x) for x in seeds],
        "iters": int(iters),
        "W": int(W),
    }


def ensure_run_spec(paths: BatchPaths, spec: dict) -> None:
    ensure_runtime_dirs(paths)

    if not paths.run_spec_path.exists():
        with paths.run_spec_path.open("w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        return

    with paths.run_spec_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)

    if existing != spec:
        raise ValueError(
            f"Campaign spec mismatch for {paths.root}. "
            "Refusing to mix incompatible jobs into the same campaign."
        )


def run_one_job(job: tuple[str, float, float, int, int, int], root_dir: str):
    corpus_key, r, alpha, iters, W, seed = job
    started = time.time()
    job_id = build_job_id(corpus_key, r, alpha, iters, W, seed)

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
    traj_path = Path(root_dir) / "trajectories" / traj_name
    traj_path.parent.mkdir(parents=True, exist_ok=True)

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
    meta["job_id"] = job_id
    meta["trajectory_filename"] = traj_name
    meta["duration_sec"] = time.time() - started
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


def run_one_trajectory_only(
    *,
    corpus_key: str,
    r: float,
    alpha: float,
    seed: int,
    iters: int = 300,
    W: int = 30,
    out_dir: str = "outputs",
    campaign: str = "manual_v1",
):
    ctx = build_run_context(corpus_key=corpus_key)
    campaign_out_dir = campaign_root(out_dir, corpus_key, campaign)
    return run_one_trajectory_only_with_context(
        ctx=ctx,
        r=r,
        alpha=alpha,
        seed=seed,
        iters=iters,
        W=W,
        out_dir=str(campaign_out_dir),
    )


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
    traj_path.parent.mkdir(parents=True, exist_ok=True)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-key", default="Cp")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--base-out-dir", default="outputs")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--W", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    corpus_key = args.corpus_key
    campaign = args.campaign
    run_name = f"texts_{corpus_key}_{campaign}"

    paths = BatchPaths(
        root=campaign_root(args.base_out_dir, corpus_key, campaign),
        run_name=run_name,
    )
    ensure_runtime_dirs(paths)

    # Full intended campaign spec
    rs = [0.10, 0.15, 0.20, 0.25, 0.30]
    alphas = np.linspace(0.03, 0.15, 15)
    seeds = range(10)

    iters = args.iters
    W = args.W
    max_workers = args.max_workers
    max_in_flight = args.max_in_flight or (max_workers * 2)

    spec = build_run_spec(
        corpus_key=corpus_key,
        campaign=campaign,
        rs=rs,
        alphas=alphas,
        seeds=seeds,
        iters=iters,
        W=W,
    )
    ensure_run_spec(paths, spec)

    completed = load_completed_keys(paths.index_csv)
    jobs = list(iter_jobs(corpus_key, rs, alphas, seeds, iters, W, completed))

    # Smoke tests should be subset launches of the same campaign.
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    total = len(jobs)

    bootstrap_manifest(paths, jobs, iters, W)
    write_progress_snapshot(paths, start_time)

    print("Corpus:", corpus_key)
    print("Campaign:", campaign)
    print("Output root:", paths.root)
    print("Jobs remaining in this launch:", total)

    if not jobs:
        print("All jobs already complete for this launch.")
        return

    done = 0
    job_iter = iter(jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        in_flight = {}

        for _ in range(min(max_in_flight, total)):
            job = next(job_iter, None)
            if job is None:
                break

            j_corpus, r, alpha, j_iters, j_W, seed = job
            job_id = build_job_id(j_corpus, r, alpha, j_iters, j_W, seed)

            update_manifest_row(paths, job_id, status="running", started_at=utc_now_iso())
            append_event(paths, "job_submitted", {"job_id": job_id})
            append_event(paths, "job_started", {"job_id": job_id})

            fut = pool.submit(run_one_job, job, str(paths.root))
            in_flight[fut] = job

        while in_flight:
            for fut in as_completed(list(in_flight)):
                job = in_flight.pop(fut)
                j_corpus, r, alpha, j_iters, j_W, seed = job
                job_id = build_job_id(j_corpus, r, alpha, j_iters, j_W, seed)

                try:
                    meta, summary = fut.result()
                    append_summary_index_row(
                        meta,
                        summary,
                        index_path=paths.index_csv,
                        filename=meta["trajectory_filename"],
                    )

                    update_manifest_row(
                        paths,
                        job_id,
                        status="done",
                        finished_at=utc_now_iso(),
                        duration_sec=f"{meta['duration_sec']:.3f}",
                        error="",
                    )
                    append_event(
                        paths,
                        "job_done",
                        {
                            "job_id": job_id,
                            "trajectory_filename": meta["trajectory_filename"],
                            "duration_sec": meta["duration_sec"],
                        },
                    )
                except Exception as e:
                    update_manifest_row(
                        paths,
                        job_id,
                        status="failed",
                        finished_at=utc_now_iso(),
                        error=repr(e),
                    )
                    append_event(paths, "job_failed", {"job_id": job_id, "error": repr(e)})

                done += 1
                write_progress_snapshot(paths, start_time)
                print(f"progress {done}/{total}", end="\r")

                next_job = next(job_iter, None)
                if next_job is not None:
                    n_corpus, n_r, n_alpha, n_iters, n_W, n_seed = next_job
                    next_job_id = build_job_id(n_corpus, n_r, n_alpha, n_iters, n_W, n_seed)

                    update_manifest_row(paths, next_job_id, status="running", started_at=utc_now_iso())
                    append_event(paths, "job_submitted", {"job_id": next_job_id})
                    append_event(paths, "job_started", {"job_id": next_job_id})

                    new_fut = pool.submit(run_one_job, next_job, str(paths.root))
                    in_flight[new_fut] = next_job

                break

    write_progress_snapshot(paths, start_time)
    print()
    print("Batch complete.")


if __name__ == "__main__":
    main()