# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class TIMReport:
    """
    Time(scale)-Invariant Metric diagnostics for a set of samples.

    - tim_score: 0..1 (higher = more time/scale invariant)
    - signature_stability: fraction of rescalings that preserve TIP signature
    - path_dist_mean: mean DTW-like path distance across rescalings (lower is better)
    """
    tim_score: float
    signature_stability: float
    path_dist_mean: float


class TIM:
    """
    TIM for text: compare a base text to its time-rescaled views (truncations / chunkings)
    using a path metric on sentence-embedding sequences (DTW-lite),
    plus invariant-signature stability from The Invariant Perceptron (TIP).
    """

    def __init__(
        self,
        tip: Any,  # InvariantPerceptron
        max_sentences: int = 12,
    ):
        if not hasattr(tip, "embedder"):
            raise ValueError("TIP must expose .embedder (SentenceTransformer)")
        self.tip = tip
        self.max_sentences = max_sentences

    # -------------------------
    # Public API
    # -------------------------

    def evaluate_text(
        self,
        text: str,
        rescale_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> TIMReport:
        """
        rescale_specs: list of dicts describing time-rescalings (views).
        If None, uses a sensible default set.
        """
        if rescale_specs is None:
            rescale_specs = self.default_rescales()

        base_sig = self.tip.predict(text)[1]
        base_seq = self._embed_sequence(self._to_sequence(text))

        stabilities = []
        dists = []

        for spec in rescale_specs:
            view_text = self._make_view(text, spec)
            view_sig = self.tip.predict(view_text)[1]
            view_seq = self._embed_sequence(self._to_sequence(view_text))

            stabilities.append(1.0 if view_sig == base_sig else 0.0)
            dists.append(self._dtw_lite_distance(base_seq, view_seq))

        signature_stability = float(np.mean(stabilities)) if stabilities else 1.0
        path_dist_mean = float(np.mean(dists)) if dists else 0.0

        # Turn into a 0..1 score: stable signatures + small path distance
        # (scale the distance with a soft squashing function)
        dist_term = float(np.exp(-path_dist_mean))  # 1 when dist=0, decays smoothly
        tim_score = float(np.clip(0.6 * signature_stability + 0.4 * dist_term, 0.0, 1.0))

        return TIMReport(
            tim_score=tim_score,
            signature_stability=signature_stability,
            path_dist_mean=path_dist_mean,
        )

    def evaluate_batch(
        self,
        texts: List[str],
        rescale_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> TIMReport:
        reports = [self.evaluate_text(t, rescale_specs=rescale_specs) for t in texts]
        return TIMReport(
            tim_score=float(np.mean([r.tim_score for r in reports])),
            signature_stability=float(np.mean([r.signature_stability for r in reports])),
            path_dist_mean=float(np.mean([r.path_dist_mean for r in reports])),
        )

    # -------------------------
    # Rescalings ("time transforms")
    # -------------------------

    @staticmethod
    def default_rescales() -> List[Dict[str, Any]]:
        """
        Views that simulate different 'time scales':
        - truncation by sentence count
        - keep head/tail slices
        - downsample sentences (every k)
        """
        return [
            {"type": "truncate_head", "sentences": 3},
            {"type": "truncate_head", "sentences": 5},
            {"type": "truncate_tail", "sentences": 3},
            {"type": "truncate_tail", "sentences": 5},
            {"type": "downsample", "step": 2},
            {"type": "downsample", "step": 3},
            {"type": "window", "start": 0, "length": 4},
            {"type": "window", "start": 1, "length": 4},
        ]

    def _make_view(self, text: str, spec: Dict[str, Any]) -> str:
        sents = self._split_sentences(text)
        if not sents:
            return text

        t = spec.get("type")
        if t == "truncate_head":
            n = int(spec["sentences"])
            return " ".join(sents[:n])
        if t == "truncate_tail":
            n = int(spec["sentences"])
            return " ".join(sents[-n:])
        if t == "downsample":
            step = int(spec["step"])
            return " ".join(sents[::step])
        if t == "window":
            start = int(spec["start"])
            length = int(spec["length"])
            return " ".join(sents[start : start + length])
        # fallback
        return text

    # -------------------------
    # Sequence representation
    # -------------------------

    def _split_sentences(self, text: str) -> List[str]:
        # Lightweight; upgrade later if you want spaCy/nltk.
        parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
        # Cap to avoid pathological long texts
        return parts[: self.max_sentences]

    def _to_sequence(self, text: str) -> List[str]:
        sents = self._split_sentences(text)
        return sents if sents else [text.strip()]

    def _embed_sequence(self, seq: List[str]) -> np.ndarray:
        E = self.tip.embedder.encode(seq, normalize_embeddings=True)
        return np.asarray(E, dtype=np.float32)

    # -------------------------
    # Time(scale)-invariant path metric (DTW-lite)
    # -------------------------

    def _dtw_lite_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        Dynamic Time Warping on cosine distances between embedding sequences.
        Time-scale invariant because it aligns sequences of different lengths.

        Returns average path cost (lower is better).
        """
        n, m = A.shape[0], B.shape[0]
        if n == 0 or m == 0:
            return 1.0

        # cosine distance matrix: D[i,j] = 1 - dot(A_i, B_j)
        D = 1.0 - (A @ B.T)
        D = np.clip(D, 0.0, 2.0)

        # DP
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = D[i - 1, j - 1]
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

        # Normalize by an approximate path length
        path_len = n + m
        return float(dp[n, m] / max(1, path_len))

# -------------------------
# Glue: TIM × PAM observable for a report you already compute
# -------------------------

def alignment_observable_TIMxPAM(
    pam_volume_entropy_bits: float,
    curvature_proxy_shift: float,
    tim_score: float,
    eps: float = 1e-6,
) -> float:
    """
    One simple scalar:
      A = (PAM volume proxy) * (inverse curvature proxy) * (TIM)

    - pam_volume_entropy_bits: signature entropy (bits) or dispersion-based proxy
    - curvature_proxy_shift: shift-from-base (higher = more curvature), invert it
    - tim_score: 0..1
    """
    inv_curv = 1.0 / (curvature_proxy_shift + eps)
    return float(pam_volume_entropy_bits * inv_curv * tim_score)

