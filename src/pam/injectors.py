# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from collections import Counter
from pam.dynamics.mutation import mutate_once

def self_resample_generator(rng, src_texts, n):
    idx = rng.integers(0, len(src_texts), size=n)
    return [src_texts[i] for i in idx]

def signature_key(sig_dict: dict) -> tuple:
    return tuple(sorted(sig_dict.items()))

def top_k_signatures(tip, texts, k=2):
    sig_keys = [signature_key(tip.predict(t)[1]) for t in texts]
    counts = Counter(sig_keys).most_common(k)
    return [dict(items) for items, _ in counts]

def mutation_injector_multi_sig_factory(tip, target_signatures, attempts_per_sample=12):
    target_set = {tuple(sorted(s.items())) for s in target_signatures}

    def injector(rng, anchor_texts, n):
        out = []
        for _ in range(n):
            base = anchor_texts[int(rng.integers(0, len(anchor_texts)))]
            chosen = None
            for _try in range(attempts_per_sample):
                cand = mutate_once(rng, base)
                sig = tuple(sorted(tip.predict(cand)[1].items()))
                if sig in target_set:
                    chosen = cand
                    break
            out.append(chosen if chosen is not None else base)
        return out

    return injector

def mixture_injector_factory(anchor_injector, self_generator):
    """
    Returns inj(rng, anchor_texts, src_texts, n, alpha) -> list[str]
    """
    def inj(rng, anchor_texts, src_texts, n, alpha):
        out = []
        for _ in range(n):
            if rng.random() < alpha:
                out.extend(anchor_injector(rng, anchor_texts, 1))
            else:
                out.extend(self_generator(rng, src_texts, 1))
        return out
    return inj
