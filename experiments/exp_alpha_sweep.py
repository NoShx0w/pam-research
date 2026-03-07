import numpy as np
import csv
import datetime as dt
import hashlib, json

from pathlib import Path
from pam.types import RunParams
from pam.utils.progress import progress_bar
from common_quench_metrics import CORPORA, build_tip, build_injector, run_one_seed


SWEEPS_INDEX_FIELDNAMES = [
    "filename",
    "corpus",
    "alpha",
    "r",
    "iters",
    "W",
    "seeds",
    "n_seeds",
    "run_id",
    "corr0_mean",
    "corr0_std",
    "best_corr_mean",
    "best_corr_std",
    "delta_freeze_mean",
    "delta_freeze_std",
    "delta_entropy_mean",
    "delta_entropy_std",
    "K_min_overall",
    "K_max_overall",
]

SWEEP_ROW_FIELDNAMES = [
    # meta first
    "meta_corpus",
    "meta_alpha",
    "meta_r",
    "meta_iters",
    "meta_W",
    "meta_seeds",
    "meta_n_seeds",
    "meta_run_id",
    # row fields (from run_one_seed)
    "seed",
    "corr0",
    "best_lag",
    "best_corr",
    "delta_r2_freeze",
    "delta_r2_entropy",
    "var_H",
    "K_min",
    "K_max",
]


def write_rows_csv(rows, out_dir="outputs", filename=None, meta=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("[csv] no rows to write")
        return None

    meta = meta or {}

    if filename is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        corpus = meta.get("corpus", "X")
        r = meta.get("r", "X")
        alpha = meta.get("alpha", "X")
        iters = meta.get("iters", "X")
        W = meta.get("W", "X")
        n_seeds = meta.get("n_seeds", len(rows))

        # format floats nicely if present
        r_s = f"{r:.2f}" if isinstance(r, (float, int)) else str(r)
        a_s = f"{alpha:.3f}" if isinstance(alpha, (float, int)) else str(alpha)

        filename = f"sweep_{corpus}_r{r_s}_a{a_s}_it{iters}_W{W}_S{n_seeds}_{stamp}.csv"

    fp = out_path / filename

    meta_pref = {f"meta_{k}": v for k, v in meta.items()}
    flat_rows = [{**meta_pref, **r} for r in rows]

    # keep stable ordering; allow extra keys without breaking
    fieldnames = SWEEP_ROW_FIELDNAMES[:]
    extras = sorted(set(flat_rows[0].keys()) - set(fieldnames))
    fieldnames += extras

    with fp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in flat_rows:
            w.writerow({k: rr.get(k, "") for k in fieldnames})

    print(f"[csv] wrote {len(rows)} rows to {fp}")
    return filename


def append_sweep_index(filename, meta, rows, out_dir="outputs"):
    index_path = Path(out_dir) / "sweeps_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()

    def col(name): return np.array([r.get(name, np.nan) for r in rows], dtype=float)

    row = {
        "filename": filename,
        "corpus": meta.get("corpus"),
        "alpha": meta.get("alpha"),
        "r": meta.get("r"),
        "iters": meta.get("iters"),
        "W": meta.get("W"),
        "seeds": meta.get("seeds"),
        "n_seeds": meta.get("n_seeds"),
        "run_id": meta.get("run_id"),
        "corr0_mean": float(np.nanmean(col("corr0"))),
        "corr0_std": float(np.nanstd(col("corr0"))),
        "best_corr_mean": float(col("best_corr").mean()),
        "best_corr_std": float(col("best_corr").std()),
        "delta_freeze_mean": float(col("delta_r2_freeze").mean()),
        "delta_freeze_std": float(col("delta_r2_freeze").std()),
        "delta_entropy_mean": float(col("delta_r2_entropy").mean()),
        "delta_entropy_std": float(col("delta_r2_entropy").std()),
        "K_min_overall": float(col("K_min").min()),
        "K_max_overall": float(col("K_max").max()),
    }

    safe_row = {k: row.get(k, "") for k in SWEEPS_INDEX_FIELDNAMES}
    with index_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SWEEPS_INDEX_FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(safe_row)

    print(f"[sweep-index] appended to {index_path}")


def main():
    corpus_key = "C"
    texts0 = CORPORA[corpus_key]

    tip = build_tip()

    r = 0.30
    alpha = 0.075
    iters = 300
    W = 30

    params = RunParams(alpha=alpha, r=r, seed=0, iters=iters, anchor_set_size=10, store_snapshots=True, store_every=1)
    mix_inj = build_injector(tip, texts0)

    seeds = list(range(5))
    rows = []
    for j, seed in enumerate(seeds, start=1):
        print(progress_bar(j, len(seeds), prefix="sweep "), end="\r")
        rows.append(run_one_seed(seed=seed, texts0=texts0, tip=tip, mix_inj=mix_inj, params=params, alpha=alpha, W=W))
    print()


    meta = {
        "corpus": corpus_key,
        "alpha": alpha,
        "r": r,
        "iters": iters,
        "W": W,
        "seeds": ",".join(map(str, seeds)),
        "n_seeds": len(seeds),
    }

    run_id = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]
    meta["run_id"] = run_id


    csv_name = write_rows_csv(rows, out_dir="outputs", meta=meta)

    if csv_name is not None:
        append_sweep_index(csv_name, meta, rows, out_dir="outputs")


    def col(name): return np.array([r[name] for r in rows], dtype=float)


    print("\nSeed sweep summary")
    print("------------------")
    print("corr0 mean±std:", float(col("corr0").mean()), float(col("corr0").std()))
    print("best_corr mean±std:", float(col("best_corr").mean()), float(col("best_corr").std()))
    print("best_lag values:", [r["best_lag"] for r in rows])
    print("ΔR²_freeze mean±std:", float(col("delta_r2_freeze").mean()), float(col("delta_r2_freeze").std()))
    print("ΔR²_entropy mean±std:", float(col("delta_r2_entropy").mean()), float(col("delta_r2_entropy").std()))
    print("K range overall:", (float(col("K_min").min()), float(col("K_max").max())))


if __name__ == "__main__":
    main()