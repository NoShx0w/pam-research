# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class EntropyReport:
    H_joint: float          # Shannon entropy over joint signatures (bits)
    H_marginal: float       # Sum of marginal Bernoulli entropies (bits)
    K: int                  # number of unique joint signatures
    n: int                  # number of samples (mutable texts used)


def compute_entropy_series(corp_snapshots, tip, anchor_set_size: int, sample_every: int = 1) -> List[EntropyReport]:
    out = []
    for i, snap in enumerate(corp_snapshots):
        if i % sample_every != 0:
            continue
        out.append(signature_entropy_pool(snap, tip, anchor_set_size))
    return out

def signature_entropy_pool(snapshot_texts, tip, anchor_set_size: int) -> EntropyReport:
    signatures = _extract_signatures(snapshot_texts, tip, anchor_set_size)
    Hj, K = signature_entropy_joint(signatures)
    Hm = signature_entropy_marginal(signatures)
    return EntropyReport(H_joint=Hj, H_marginal=Hm, K=K, n=len(signatures))

def _extract_signatures(snapshot_texts, tip, anchor_set_size: int) -> List[Dict[str, bool]]:
    texts = snapshot_texts[anchor_set_size:]  # mutable only
    return [tip.signature(txt) for txt in texts]

def signature_entropy_joint(signatures: List[Dict[str, bool]]) -> Tuple[float, int]:
    """
    Shannon entropy (bits) of the empirical distribution over joint signatures.
    """
    keys = [tuple(sorted(d.items())) for d in signatures]
    c = Counter(keys)
    total = sum(c.values())
    ps = np.array([v / total for v in c.values()], dtype=float)
    H = -np.sum(ps * np.log2(ps + 1e-12))
    return float(H), len(c)

def signature_entropy_marginal(signatures: List[Dict[str, bool]]) -> float:
    """
    Sum of binary entropies of each invariant marginal: Σ_k h(p_k).
    """
    if not signatures:
        return 0.0

    counts = {}
    for sig in signatures:
        for k, v in sig.items():
            counts[k] = counts.get(k, 0) + int(v)

    total = len(signatures)
    H = 0.0
    for c in counts.values():
        p = c / total
        H += _binary_entropy(p)
    return float(H)

def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
