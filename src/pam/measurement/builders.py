# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

from pam.measurement.tip import InvariantPerceptron, InvariantSpec
from pam.engine.injectors import (
    mixture_injector_factory,
    mutation_injector_multi_sig_factory,
    self_resample_generator,
    top_k_signatures,
)
from pam.observables.core import macrostate_from_microstructure


def build_tip() -> InvariantPerceptron:
    invariants = [
        InvariantSpec("reflective", threshold=0.6),
        InvariantSpec("coherent", threshold=0.6),
        InvariantSpec("playful_serious", threshold=0.6),
        InvariantSpec("geometric", threshold=0.6),
    ]
    return InvariantPerceptron(invariants=invariants, mode="heuristic")


def build_injector(tip, texts0, k: int = 2, attempts_per_sample: int = 12):
    targets = top_k_signatures(tip, texts0, k=k)
    anchor_inj = mutation_injector_multi_sig_factory(
        tip,
        targets,
        attempts_per_sample=attempts_per_sample,
    )
    return mixture_injector_factory(anchor_inj, self_resample_generator)


def macro_fn_factory(bd_thresh: float = 0.05, mg_thresh: float = 15.0):
    return lambda corp_snaps, tip_, p: macrostate_from_microstructure(
        corp_snaps,
        tip_,
        p,
        window=6,
        step=1,
        sample_every=1,
        bd_thresh=bd_thresh,
        mg_thresh=mg_thresh,
    )
