# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class EntropyReport:
    H_joint: float
    H_marginal: float
    K: int
    n: int


def compute_entropy_series(
    corp_snapshots,
    tip,
    anchor_set_size: int,
    sample_every: int = 1,
) -> List[EntropyReport]:
    out = []
    for i, snap in enumerate(corp_snapshots):
        if i % sample_every != 0:
            continue
        out.append(signature_entropy_pool(snap, tip, anchor_set_size))
    return out


def signature_entropy_pool(snapshot_texts, tip, anchor_set_size: int) -> EntropyReport:
    signatures = _extract_signatures(snapshot_texts, tip, anchor_set_size)
    hj, k = signature_entropy_joint(signatures)
    hm = signature_entropy_marginal(signatures)
    return EntropyReport(H_joint=hj, H_marginal=hm, K=k, n=len(signatures))


def _extract_signatures(snapshot_texts, tip, anchor_set_size: int) -> List[Dict[str, bool]]:
    texts = snapshot_texts[anchor_set_size:]
    return [tip.signature(txt) for txt in texts]


def signature_entropy_joint(signatures: List[Dict[str, bool]]) -> Tuple[float, int]:
    keys = [tuple(sorted(d.items())) for d in signatures]
    c = Counter(keys)
    total = sum(c.values())
    ps = np.array([v / total for v in c.values()], dtype=float)
    h = -np.sum(ps * np.log2(ps + 1e-12))
    return float(h), len(c)


def signature_entropy_marginal(signatures: List[Dict[str, bool]]) -> float:
    if not signatures:
        return 0.0

    counts = {}
    for sig in signatures:
        for k, v in sig.items():
            counts[k] = counts.get(k, 0) + int(v)

    total = len(signatures)
    h = 0.0
    for c in counts.values():
        p = c / total
        h += _binary_entropy(p)
    return float(h)


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def sliding_piF(states: Sequence[Any], W: int = 30) -> np.ndarray:
    y = np.array([1.0 if s == "F" else 0.0 for s in states], dtype=float)
    return np.convolve(y, np.ones(W) / W, mode="valid")


def compute_transition_rates(states):
    c_FM = c_MF = 0
    n_F = n_M = 0

    for i in range(len(states) - 1):
        if states[i] == "F":
            n_F += 1
            if states[i + 1] == "M":
                c_FM += 1
        else:
            n_M += 1
            if states[i + 1] == "F":
                c_MF += 1

    p_FM = c_FM / n_F if n_F else float("nan")
    p_MF = c_MF / n_M if n_M else float("nan")

    return {
        "c_FM": c_FM,
        "c_MF": c_MF,
        "n_F": n_F,
        "n_M": n_M,
        "p_FM": p_FM,
        "p_MF": p_MF,
    }


def sig_to_key(sig: dict) -> tuple:
    return tuple(sorted(sig.items()))


def dominant_signature_for_window(tip, window_texts):
    keys = [sig_to_key(tip.predict(t)[1]) for t in window_texts]
    return Counter(keys).most_common(1)[0][0]


def label_windows_by_signature(tip, texts, window=5, step=1):
    labels = []
    windows = []
    for i in range(0, len(texts) - window + 1, step):
        w = texts[i : i + window]
        windows.append((i, i + window))
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
    n_changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
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
    changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
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
    return {"iters": ts, "boundary_density": bd, "mean_grain": mg, "n_phases": npg}


def label_state(bd: float, mg: float, *, bd_thresh: float, mg_thresh: float) -> str:
    """
    Classify macrostate from microstructure scalars.

    Convention:
      - Frozen (F): low boundary density AND large grains
      - Mixed  (M): otherwise
    """
    return "F" if (bd <= bd_thresh and mg >= mg_thresh) else "M"


def states_from_microstructure(out_dict, bd_thresh=0.05, mg_thresh=15.0):
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
        corp_snapshots,
        tip,
        title="",
        window=window,
        step=step,
        sample_every=sample_every,
    )
    return states_from_microstructure(out, bd_thresh=bd_thresh, mg_thresh=mg_thresh)


__all__ = [
    "EntropyReport",
    "compute_entropy_series",
    "signature_entropy_pool",
    "signature_entropy_joint",
    "signature_entropy_marginal",
    "sliding_piF",
    "compute_transition_rates",
    "sig_to_key",
    "dominant_signature_for_window",
    "label_windows_by_signature",
    "run_lengths",
    "grain_stats",
    "boundary_density_from_labels",
    "microstructure_metrics_for_snapshot",
    "analyze_quench_microstructure",
    "label_state",
    "states_from_microstructure",
    "macrostate_from_microstructure",
]
