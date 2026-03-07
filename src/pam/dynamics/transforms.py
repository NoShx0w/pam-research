import re
import numpy as np
from typing import List, Dict


# -------------------------
# Tiny synonym map (safe, controlled)
# Keep it small: big maps cause semantic drift.
# -------------------------
SYNONYMS: Dict[str, List[str]] = {
    "structure": ["form", "scaffold"],
    "invariant": ["constant", "preserved pattern"],
    "metric": ["measure", "distance"],
    "manifold": ["space", "surface"],
    "geometry": ["shape", "structure"],
    "coherence": ["consistency", "continuity"],
    "curvature": ["bend", "shape-change"],
    "drift": ["wander", "shift"],
    "alignment": ["fit", "attunement"],
    "compress": ["condense", "pack"],
    "phase": ["regime", "stage"],
    "trajectory": ["path", "curve"],
    "detect": ["sense", "identify"],
    "meaning": ["sense", "content"],
}


def split_sentences(text: str) -> List[str]:
    # lightweight sentence split
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


def join_sentences(sents: List[str]) -> str:
    out = " ".join(sents).strip()
    return out


def swap_some_synonyms(rng: np.random.Generator, text: str, p: float = 0.25) -> str:
    t = text
    for key, options in SYNONYMS.items():
        if rng.random() < p:
            repl = options[int(rng.integers(0, len(options)))]
            t = re.sub(rf"\b{re.escape(key)}\b", repl, t, count=1, flags=re.IGNORECASE)
    return t


def reorder_sentences(rng: np.random.Generator, text: str) -> str:
    sents = split_sentences(text)
    if len(sents) < 3:
        return text
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    return join_sentences([sents[i] for i in idx])


def mild_clause_shuffle(rng: np.random.Generator, text: str) -> str:
    sents = split_sentences(text)
    out = []
    for s in sents:
        clauses = [c.strip() for c in s.split(",") if c.strip()]
        if len(clauses) >= 3 and rng.random() < 0.5:
            idx = np.arange(len(clauses))
            rng.shuffle(idx)
            out.append(", ".join([clauses[i] for i in idx]))
        else:
            out.append(s)
    return join_sentences(out)


def lens_toggle(rng: np.random.Generator, text: str) -> str:
    lenses = [
        "Geometrically: ",
        "In invariance terms: ",
        "Metaphorically: ",
        "Operationally: ",
        "Phenomenologically: ",
    ]
    prefix = lenses[int(rng.integers(0, len(lenses)))]
    if text.strip().lower().startswith(tuple(l.lower() for l in lenses)):
        return text
    return prefix + text.strip()