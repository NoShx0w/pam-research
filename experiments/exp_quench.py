import numpy as np
import hashlib, json
import csv
import datetime as dt

from pathlib import Path
from pam.types import RunParams
from common_quench_metrics import CORPORA, build_tip, build_injector, run_one


def to_jsonable(x):
    # numpy scalars
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    # containers
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def write_deep_run_json(out, out_dir="outputs", filename=None, meta=None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta = meta or {}
    if filename is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        corpus = meta.get("corpus", "X")
        r = meta.get("r", "X")
        alpha = meta.get("alpha", "X")
        r_s = f"{r:.2f}" if isinstance(r, (float, int)) else str(r)
        a_s = f"{alpha:.3f}" if isinstance(alpha, (float, int)) else str(alpha)
        iters = meta.get("iters", "X")
        W = meta.get("W", "X")
        seed = meta.get("seed", "X")
        filename = f"deep_{corpus}_r{r_s}_a{a_s}_it{iters}_W{W}_seed{seed}_{stamp}.json"

    fp = out_path / filename

    payload = {"meta": meta, "data": to_jsonable(out)}

    with fp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[json] wrote deep run to {fp}")
    return filename


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


def append_index_row(filename, meta, out, out_dir="outputs"):
    """
    Append a one-line summary to outputs/index.csv
    """

    index_path = Path(out_dir) / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # --- extract series
    H_joint = np.array(out["H_joint"], dtype=float)
    K = np.array(out["K"], dtype=float)
    states = out["states"]

    # --- phase observables
    F_raw = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)

    piF_mean = float(np.mean(F_raw))

    tail_frac = 0.2
    tail_n = max(1, int(round(len(F_raw) * tail_frac)))
    piF_tail = float(np.mean(F_raw[-tail_n:]))

    H_joint_mean = float(np.mean(H_joint))
    var_H = float(np.var(H_joint))
    H_min = float(np.min(H_joint))
    H_max = float(np.max(H_joint))

    K_min = float(np.min(K))
    K_max = float(np.max(K))

    delta_freeze = float(out["deltas_H"]["freeze_delta_r2"])
    delta_entropy = float(out["deltas_H"]["entropy_delta_r2"])

    best_lag = int(out["lag"]["best_lag"])
    best_corr = float(out["lag"]["best_corr"])
    corr0 = float(out["lag"].get("corr0", best_corr))

    row = {
        "filename": filename,
        "corpus": meta.get("corpus"),
        "alpha": meta.get("alpha"),
        "r": meta.get("r"),
        "iters": meta.get("iters"),
        "W": meta.get("W"),
        "seed": meta.get("seed"),
        "run_id": meta.get("run_id", ""),
        "piF_mean": piF_mean,
        "piF_tail": piF_tail,
        "H_joint_mean": H_joint_mean,
        "var_H_joint": var_H,
        "H_min": H_min,
        "H_max": H_max,
        "K_min": K_min,
        "K_max": K_max,
        "corr0": corr0,
        "delta_r2_freeze": delta_freeze,
        "delta_r2_entropy": delta_entropy,
        "best_lag": best_lag,
        "best_corr": best_corr,
    }

    write_header = not index_path.exists()
    safe_row = {k: row.get(k, "") for k in INDEX_FIELDNAMES}

    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)

    print(f"[index] appended summary to {index_path}")


def main():
    corpus_key = "C"
    texts0 = CORPORA[corpus_key]

    tip = build_tip()

    r = 0.30
    alpha = 0.075
    iters = 300
    W = 30
    seed = 0

    params = RunParams(alpha=alpha, r=r, seed=seed, iters=iters, anchor_set_size=10, store_snapshots=True, store_every=1)
    mix_inj = build_injector(tip, texts0)

    out = run_one(texts0=texts0, tip=tip, mix_inj=mix_inj, params=params, alpha=alpha, W=W)

    meta = {
        "corpus": corpus_key,
        "alpha": alpha,
        "r": r,
        "iters": iters,
        "W": W,
        "seed": seed,
    }

    meta["invariants"] = [
        {"name": "reflective", "threshold": 0.6},
        {"name": "coherent", "threshold": 0.6},
        {"name": "playful_serious", "threshold": 0.6},
        {"name": "geometric", "threshold": 0.6},
    ]

    meta["engine"] = {"source_window": 10, "anchor_set_size": 10}
    run_id = hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:10]
    meta["run_id"] = run_id

    filename = write_deep_run_json(out, out_dir="outputs", meta=meta)
    append_index_row(filename, meta, out)

    H_joint = out["H_joint"]
    H_marg = out["H_marginal"]
    K = out["K"]
    MI = out["MI"]

    print(f"var(H_joint_raw): {np.var(H_joint)}")
    print(f"var(H_marginal_raw): {np.var(H_marg)}")
    print(f"K range: {(float(np.min(K)), float(np.max(K)))}")
    print(f"MI_proxy range: {(float(np.min(MI)), float(np.max(MI)))}")

    print("RAW Model A:", out["models_H"]["model_A"])
    print("RAW Model B:", out["models_H"]["model_B"])
    print("RAW Freeze ΔR2:", out["deltas_H"]["freeze_delta_r2"])
    print("RAW Entropy ΔR2:", out["deltas_H"]["entropy_delta_r2"])

    print("RAW(MI) Model A:", out["models_MI"]["model_A"])
    print("RAW(MI) Model B:", out["models_MI"]["model_B"])
    print("RAW(MI) Freeze ΔR2:", out["deltas_MI"]["freeze_delta_r2"])
    print("RAW(MI) Entropy ΔR2:", out["deltas_MI"]["entropy_delta_r2"])

    print("best_lag (pi vs H_joint_sm):", out["lag"]["best_lag"], "best_corr:", out["lag"]["best_corr"])


if __name__ == "__main__":
    main()