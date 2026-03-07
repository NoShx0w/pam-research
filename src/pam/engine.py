# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from dataclasses import asdict
from typing import Callable, List, Optional, Any, Dict, Tuple
from pam.types import RunParams, RunResult
from pam.dynamics_mixture import run_mixture_quench

MacrostateFn = Callable[[List[List[str]], Any, RunParams], List[Any]]

def run_dynamics_quench(
    texts0: List[str],
    *,
    tip,
    mixture_injector,
    alpha_schedule,
    params: RunParams,
    source_window: int = 10,
    macrostate_fn: Optional[MacrostateFn] = None,
) -> RunResult:
    """
    Engine wrapper around run_mixture_quench:
    - Returns corpus snapshots
    - Optionally returns macrostate sequence via injected macrostate_fn
    """

    corp_snapshots, dashboard = run_mixture_quench(
        texts0=texts0,
        tip=tip,
        tim=None,  # keep TIM out of the engine; inject later if needed
        compute_dashboard=None,
        mixture_injector=mixture_injector,
        replace_frac=params.r,
        anchor_set_size=params.anchor_set_size,
        source_window=source_window,
        iters=params.iters,
        seed=params.seed,
        alpha_schedule=alpha_schedule,
    )

    states = []
    if macrostate_fn is not None:
        states = macrostate_fn(corp_snapshots, tip, params)

    return RunResult(
        params=params,
        states=states,
        corp_snapshots=corp_snapshots if params.store_snapshots else None,
        notes={
            "params": asdict(params),
            "source_window": source_window,
            "dashboard_available": dashboard is not None,
        },
    )

def run_dynamics(
    corp: List[str],
    params: RunParams,
    *,
    mutate_step_fn,
    macrostate_fn,
) -> RunResult:
    """
    Core engine: evolves corpus over time.

    - corp: initial corpus list[str]
    - mutate_step_fn: function(corp, rng, params) -> new_corp
    - macrostate_fn: function(corp, params) -> state label (e.g., "F"/"M")

    Returns RunResult with states and optional corpus snapshots.
    """

    rng = np.random.default_rng(params.seed)

    corp_cur = list(corp)
    states: List[Any] = []
    snapshots: Optional[List[List[str]]] = [] if params.store_snapshots else None

    for t in range(params.iters):
        # 1) record macrostate *before* mutation (or after, but pick one convention and keep it)
        st = macrostate_fn(corp_cur, params)
        states.append(st)

        # 2) optionally store snapshot
        if snapshots is not None and (t % params.store_every == 0):
            snapshots.append(list(corp_cur))

        # 3) evolve corpus by one step
        corp_cur = mutate_step_fn(corp_cur, rng, params)

    notes = {
        "final_corp_size": len(corp_cur),
        "params": asdict(params),
    }

    return RunResult(
        params=params,
        states=states,
        corp_snapshots=snapshots,
        notes=notes,
    )

def run_quench(
    *,
    texts0: List[str],
    tip: Any,
    mixture_injector: Any,
    params: RunParams,
    alpha_schedule,
    source_window: int = 10,
    macrostate_fn: Optional[MacrostateFn] = None,
) -> RunResult:

    corp_snaps, alphas_used = run_mixture_quench(
        texts0=texts0,
        mixture_injector=mixture_injector,
        replace_frac=params.r,
        anchor_set_size=params.anchor_set_size,
        source_window=source_window,
        iters=params.iters,
        seed=params.seed,
        alpha_schedule=alpha_schedule,
    )

    states = macrostate_fn(corp_snaps, tip, params) if macrostate_fn else []

    return RunResult(
        params=params,
        states=states,
        corp_snapshots=corp_snaps if params.store_snapshots else None,
        notes={
            "params": asdict(params),
            "source_window": source_window,
            "alphas_used": alphas_used,
            "snapshot_convention": "pre-mutation, includes terminal snapshot",
        },
    )
