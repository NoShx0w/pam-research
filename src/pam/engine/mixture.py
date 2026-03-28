# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

import numpy as np


def run_mixture_quench(
    *,
    texts0,
    mixture_injector,
    replace_frac=0.30,
    anchor_set_size=10,
    source_window=10,
    iters=100,
    seed=0,
    alpha_schedule=None,
):
    if alpha_schedule is None:
        raise ValueError("Provide alpha_schedule(it)->alpha")

    rng = np.random.default_rng(seed)

    texts0 = list(texts0)
    n_total = len(texts0)

    anchor = texts0[:anchor_set_size]
    mutable = texts0[anchor_set_size:]

    corp_snapshots = []
    alphas_used = []

    for it in range(iters + 1):
        full = anchor + mutable
        corp_snapshots.append(full.copy())

        if it == iters:
            break

        alpha = float(alpha_schedule(it))
        alphas_used.append(alpha)

        n_rep = max(1, int(round(n_total * replace_frac)))
        n_rep = min(n_rep, len(mutable))

        src = (
            mutable[-source_window:]
            if len(mutable) >= source_window
            else mutable[:]
        )
        if not src:
            src = anchor[:]

        injected = mixture_injector(rng, anchor, src, n_rep, alpha)
        mutable = mutable[n_rep:] + injected

    return corp_snapshots, alphas_used
