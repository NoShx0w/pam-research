import numpy as np

from pam.dynamics.transforms import (
    swap_some_synonyms,
    mild_clause_shuffle,
    reorder_sentences,
    lens_toggle,
)

def mutate_once(rng: np.random.Generator, text: str) -> str:
    """
    Compose a couple of low-risk transforms.
    """
    t = text
    transforms = [swap_some_synonyms, mild_clause_shuffle, reorder_sentences, lens_toggle]
    rng.shuffle(transforms)
    k = 2
    for f in transforms[:k]:
        if f is swap_some_synonyms:
            t = f(rng, t, p=0.30)
        else:
            if rng.random() < 0.8:
                t = f(rng, t)
    return t