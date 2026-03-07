# PAM – Phase Analysis of Meaning
# Copyright (c) 2026 Rik van Lent
# Licensed under the MIT License

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RunParams:
    alpha: float
    r: float
    seed: int
    iters: int

    anchor_set_size: int = 10
    window_size: int = 6

    # storage control
    store_snapshots: bool = True
    store_every: int = 1


@dataclass
class RunResult:
    params: RunParams
    states: List[Any]
    corp_snapshots: Optional[List[List[str]]]
    notes: Dict[str, Any]
