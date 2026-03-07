# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# Optional: comment these out if you don't want embedding dependency yet
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


@dataclass
class InvariantSpec:
    name: str
    threshold: float = 0.6


class InvariantPerceptron:
    """
    The Invariant Perceptron (TIP): maps a sample x -> invariant scores + signature.

    Modes:
      - heuristic: no training; uses hand-built scoring functions
      - learned: per-invariant linear models over embeddings (logistic regression)
    """

    def __init__(
        self,
        invariants: List[InvariantSpec],
        mode: str = "heuristic",
        embedding_model: str = "all-MiniLM-L6-v2",
        random_state: int = 0,
    ):
        assert mode in {"heuristic", "learned"}
        self.invariants = invariants
        self.mode = mode
        self.random_state = random_state

        self.embedder = SentenceTransformer(embedding_model)
        self.models: Dict[str, LogisticRegression] = {}

    # ---------- public API ----------

    def predict(self, text: str):
        scores = self._score(text)            # dict[str, float]
        sig = self._signature(scores)         # dict[str, bool]
        score = float(sum(sig.values())) / max(1, len(sig))  # optional aggregate
        return score, sig
    
    def batch_predict(self, texts: List[str]) -> List[Tuple[Dict[str, float], Dict[str, bool]]]:
        return [self.predict(t) for t in texts]

    def fit(self, texts: List[str], labels: Dict[str, List[int]]) -> None:
        """
        Train a linear classifier per invariant.

        labels: dict invariant_name -> list of 0/1 labels aligned with texts
        """
        if self.mode != "learned":
            raise ValueError("fit() requires mode='learned'")

        X = self.embedder.encode(texts, normalize_embeddings=True)
        for inv in self.invariants:
            y = np.asarray(labels[inv.name], dtype=int)
            clf = LogisticRegression(
                solver="liblinear",
                random_state=self.random_state,
                class_weight="balanced",
            )
            clf.fit(X, y)
            self.models[inv.name] = clf

    # ---------- core scoring ----------

    def _score(self, text: str):
        if self.mode == "heuristic":
            return self._heuristic_scores(text)
        ...
        # must return dict[str, float]

    def _signature(self, scores: dict):
        out = {}
        for inv in self.invariants:
            out[inv.name] = scores[inv.name] >= inv.threshold
        return out

    # optional convenience if you want:
    def signature(self, text: str):
        return self._signature(self._score(text))

    # ---------- heuristic detectors (v0) ----------

    def _heuristic_scores(self, text: str) -> Dict[str, float]:
        t = text.lower()

        reflective_markers = [
            "i think", "it seems", "we can see", "one might",
            "in other words", "this suggests", "we notice",
            "reflect", "observe", "meta"
        ]
        geometric_markers = [
            "manifold", "space", "dimension", "structure",
            "relation", "mapping", "curvature", "geometry",
            "topology", "boundary", "surface"
        ]
        serious_markers = [
            "structure", "invariant", "system", "model",
            "information", "geometry", "alignment", "metric"
        ]
        playful_markers = [
            "imagine", "like", "almost", "kind of",
            "playful", "wink", "✨", "😄", "😎"
        ]

        def capped_hits(markers, cap):
            return min(1.0, sum(m in t for m in markers) / cap)

        # Coherence proxy: average cosine similarity between sentence embeddings
        coherence = self._coherence_score(text)

        reflective = capped_hits(reflective_markers, cap=3)
        geometric = capped_hits(geometric_markers, cap=2)

        serious = sum(m in t for m in serious_markers)
        playful = sum(m in t for m in playful_markers)
        playful_serious = min(1.0, min(serious, playful) / 2)

        # Return only invariants we declared; default 0 for unknown names
        raw = {
            "reflective": reflective,
            "coherent": coherence,
            "playful_serious": playful_serious,
            "geometric": geometric,
        }
        return {inv.name: float(raw.get(inv.name, 0.0)) for inv in self.invariants}

    def _coherence_score(self, text: str) -> float:
        # Simple segmentation; upgrade later with nltk/spacy if desired
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
        if len(sentences) < 2:
            return 0.5  # ambiguous, not a fail

        E = self.embedder.encode(sentences, normalize_embeddings=True)
        sims = E @ E.T
        n = sims.shape[0]
        avg_offdiag = (np.sum(sims) - n) / (n * (n - 1))
        return float(np.clip(avg_offdiag, 0.0, 1.0))

    # ---------- learned scoring ----------

    def _learned_scores(self, text: str) -> Dict[str, float]:
        if not self.models:
            raise ValueError("No models trained. Call fit() first.")

        x = self.embedder.encode([text], normalize_embeddings=True)
        scores = {}
        for inv in self.invariants:
            clf = self.models.get(inv.name)
            if clf is None:
                raise ValueError(f"Missing trained model for invariant '{inv.name}'")
            # probability of class 1
            p = float(clf.predict_proba(x)[0, 1])
            scores[inv.name] = p
        return scores

