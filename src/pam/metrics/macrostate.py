# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

import numpy as np
from typing import Sequence, Any
from collections import Counter

def sliding_piF(states: Sequence[Any], W: int = 30) -> np.ndarray:
    y = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)
    return np.convolve(y, np.ones(W) / W, mode="valid")

def compute_transition_rates(states):
    c_FM = c_MF = 0
    n_F = n_M = 0

    for i in range(len(states) - 1):
        if states[i] == "F":
            n_F += 1
            if states[i+1] == "M":
                c_FM += 1
        else:
            n_M += 1
            if states[i+1] == "F":
                c_MF += 1

    p_FM = c_FM / n_F if n_F else float("nan")
    p_MF = c_MF / n_M if n_M else float("nan")

    return {
        "c_FM": c_FM,
        "c_MF": c_MF,
        "n_F": n_F,
        "n_M": n_M,
        "p_FM": p_FM,
        "p_MF": p_MF
    }

def sig_to_key(sig: dict) -> tuple:
    # canonical, hashable
    return tuple(sorted(sig.items()))

def dominant_signature_for_window(tip, window_texts):
    keys = [sig_to_key(tip.predict(t)[1]) for t in window_texts]
    return Counter(keys).most_common(1)[0][0]

def label_windows_by_signature(tip, texts, window=5, step=1):
    labels = []
    windows = []
    for i in range(0, len(texts) - window + 1, step):
        w = texts[i:i+window]
        windows.append((i, i+window))
        labels.append(dominant_signature_for_window(tip, w))
    return windows, labels

def run_lengths(labels):
    if not labels:
        return []
    runs = []
    cur = labels[0]
    length = 1
    for x in labels[1:]:
        if x == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = x
            length = 1
    runs.append((cur, length))
    return runs

def grain_stats(labels):
    runs = run_lengths(labels)
    lengths = np.array([L for _, L in runs], dtype=float) if runs else np.array([])
    boundaries = max(0, len(labels) - 1)
    n_changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
    boundary_density = (n_changes / boundaries) if boundaries > 0 else 0.0
    return {
        "n_windows": len(labels),
        "n_phases": len(set(labels)),
        "n_grains": len(runs),
        "mean_grain": float(lengths.mean()) if len(lengths) else 0.0,
        "median_grain": float(np.median(lengths)) if len(lengths) else 0.0,
        "max_grain": float(lengths.max()) if len(lengths) else 0.0,
        "boundary_density": float(boundary_density),
        "runs": runs,
    }

def boundary_density_from_labels(labels):
    if len(labels) <= 1:
        return 0.0
    changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
    return changes / (len(labels) - 1)

def microstructure_metrics_for_snapshot(tip, texts_snapshot, window=5, step=1):
    _, labels = label_windows_by_signature(tip, texts_snapshot, window=window, step=step)
    stats = grain_stats(labels)
    return {
        "n_phases": stats["n_phases"],
        "mean_grain": stats["mean_grain"],
        "max_grain": stats["max_grain"],
        "boundary_density": stats["boundary_density"],
        "labels": labels,
        "runs": stats["runs"],
    }

def analyze_quench_microstructure(
    corpora,
    tip,
    title="",
    window=6,
    step=1,
    sample_every=5,
):
    # compute microstructure metrics across time (subsample to keep it light)
    ts = []
    bd = []
    mg = []
    npg = []
    for it in range(0, len(corpora), sample_every):
        m = microstructure_metrics_for_snapshot(tip, corpora[it], window=window, step=step)
        ts.append(it)
        bd.append(m["boundary_density"])
        mg.append(m["mean_grain"])
        npg.append(m["n_phases"])
    '''
    # plot
    plt.figure()
    plt.plot(ts, bd, marker="o")
    plt.title(f"Boundary density over time {title}".strip())
    plt.xlabel("Iteration")
    plt.ylabel("Boundary density")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ts, mg, marker="o")
    plt.title(f"Mean grain size over time {title}".strip())
    plt.xlabel("Iteration")
    plt.ylabel("Mean grain size")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ts, npg, marker="o")
    plt.title(f"Number of phases over time {title}".strip())
    plt.xlabel("Iteration")
    plt.ylabel("n_phases")
    plt.tight_layout()
    plt.show()
    '''
    return {"iters": ts, "boundary_density": bd, "mean_grain": mg, "n_phases": npg}

def label_state(bd: float, mg: float, *, bd_thresh: float, mg_thresh: float) -> str:
    """
    Classify macrostate from microstructure scalars.
    Convention (as you’ve been using):
      - Frozen (F): low boundary density AND large grains
      - Mixed  (M): otherwise
    """
    return "F" if (bd <= bd_thresh and mg >= mg_thresh) else "M"

def states_from_microstructure(out_dict, bd_thresh=0.05, mg_thresh=15.0):
    # out_dict from analyze_quench_microstructure
    states = []
    for bd, mg in zip(out_dict["boundary_density"], out_dict["mean_grain"]):
        states.append(label_state(bd, mg, bd_thresh=bd_thresh, mg_thresh=mg_thresh))
    return states

def macrostate_from_microstructure(
    corp_snapshots,
    tip,
    params,
    *,
    window: int = 6,
    step: int = 1,
    sample_every: int = 1,
    bd_thresh: float = 0.05,
    mg_thresh: float = 15.0,
):
    out = analyze_quench_microstructure(
        corp_snapshots, tip,
        title="",
        window=window,
        step=step,
        sample_every=sample_every,
    )
    return states_from_microstructure(out, bd_thresh=bd_thresh, mg_thresh=mg_thresh)
