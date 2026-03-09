import csv
import traceback
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from pam.types import RunParams
from pam.utils.progress import progress_bar
from common_quench_metrics import CORPORA, build_tip, build_injector, run_one
from exp_quench import write_deep_run_json, append_index_row


OUT_DIR = Path("outputs")
INDEX_CSV = OUT_DIR / "index.csv"
ERROR_LOG = OUT_DIR / "errors.log"


def load_completed_keys(index_csv: Path):
    """
    Read outputs/index.csv and return a set of completed run keys:
    (corpus, r, alpha, iters, W, seed)
    """
    done = set()
    if not index_csv.exists():
        return done

    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    str(row["corpus"]),
                    round(float(row["r"]), 6),
                    round(float(row["alpha"]), 6),
                    int(float(row["iters"])),
                    int(float(row["W"])),
                    int(float(row["seed"])),
                )
                done.add(key)
            except Exception:
                continue
    return done


def run_key(corpus, r, alpha, iters, W, seed):
    return (
        str(corpus),
        round(float(r), 6),
        round(float(alpha), 6),
        int(iters),
        int(W),
        int(seed),
    )


def log_error(meta, exc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] ERROR\n")
        f.write(f"meta: {meta}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "=" * 80 + "\n\n")


def build_meta(corpus_key, r, alpha, iters, W, seed):
    return {
        "corpus": corpus_key,
        "alpha": alpha,
        "r": r,
        "iters": iters,
        "W": W,
        "seed": seed,
        "invariants": [
            {"name": "reflective", "threshold": 0.6},
            {"name": "coherent", "threshold": 0.6},
            {"name": "playful_serious", "threshold": 0.6},
            {"name": "geometric", "threshold": 0.6},
        ],
        "engine": {
            "source_window": 10,
            "anchor_set_size": 10,
        },
    }


def run_one_job(job):
    """
    Worker-safe pure compute function.
    Returns (meta, out) and does NOT write files.
    """
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

    out = run_one(
        texts0=texts0,
        tip=tip,
        mix_inj=mix_inj,
        params=params,
        alpha=alpha,
        W=W,
    )

    meta = build_meta(corpus_key, r, alpha, iters, W, seed)
    return meta, out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Batch spec
    # ---------------------------
    corpus_keys = ["C"]
    rs = [0.20, 0.30, 0.40]
    alphas = [0.050, 0.060, 0.070, 0.075, 0.080, 0.090]
    seeds = list(range(10))

    iters = 300
    W = 30

    # Tune this for your M4 Pro
    max_workers = 10

    # ---------------------------
    # Resume support
    # ---------------------------
    completed = load_completed_keys(INDEX_CSV)

    jobs = []
    for corpus_key in corpus_keys:
        for r in rs:
            for alpha in alphas:
                for seed in seeds:
                    key = run_key(corpus_key, r, alpha, iters, W, seed)
                    if key not in completed:
                        jobs.append((corpus_key, r, alpha, iters, W, seed))

    total = len(jobs)
    if total == 0:
        print("No pending jobs. Batch is already complete.")
        return

    print(f"Starting batch: {total} pending runs with {max_workers} workers")

    done_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_job = {ex.submit(run_one_job, job): job for job in jobs}

        for fut in as_completed(future_to_job):
            job = future_to_job[fut]
            corpus_key, r, alpha, iters, W, seed = job
            meta = build_meta(corpus_key, r, alpha, iters, W, seed)

            try:
                meta_out, out = fut.result()

                filename = write_deep_run_json(out, out_dir=str(OUT_DIR), meta=meta_out)
                append_index_row(filename, meta_out, out, out_dir=str(OUT_DIR))

            except Exception as exc:
                print()
                print(f"[error] corpus={corpus_key} r={r:.2f} alpha={alpha:.3f} seed={seed}: {exc}")
                log_error(meta, exc)

            done_count += 1
            status = f" corpus={corpus_key} r={r:.2f} alpha={alpha:.3f} seed={seed}"
            print(progress_bar(done_count, total, prefix="batch "), status, end="\r")

    print()
    print("Batch finished.")


if __name__ == "__main__":
    main()