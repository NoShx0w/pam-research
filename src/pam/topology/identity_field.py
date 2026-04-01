from __future__ import annotations

"""
PAM Identity Field

Discrete field construction over a 2D grid of IdentityGraph objects.

Defines:
- IdentityFieldResult
- compute_identity_field(...)

This module intentionally stays minimal:
it computes directional identity-change fields, their magnitude,
and a simple discrete spin/curl-like quantity.
"""

from dataclasses import dataclass

import numpy as np

from pam.topology.identity import IdentityGraph, identity_distance


@dataclass(frozen=True)
class IdentityFieldResult:
    vx: np.ndarray
    vy: np.ndarray
    magnitude: np.ndarray
    spin: np.ndarray


def compute_identity_field(
    identity_grid: list[list[IdentityGraph]],
    *,
    normalized: bool = True,
) -> IdentityFieldResult:
    """
    Compute a discrete identity field over a rectangular 2D grid of IdentityGraph objects.

    Conventions
    -----------
    - vx[i, j] compares (i, j) -> (i, j+1)
    - vy[i, j] compares (i, j) -> (i+1, j)
    - magnitude = sqrt(vx^2 + vy^2)
    - spin uses a simple discrete curl-like quantity:
        dVy/dx - dVx/dy

    Notes
    -----
    This is a directional finite-difference field over identity-distance changes.
    It is not yet a continuous embedded vector field in an identity manifold.
    """
    n_rows = len(identity_grid)
    if n_rows == 0:
        raise ValueError("identity_grid must be non-empty")

    n_cols = len(identity_grid[0])
    if n_cols == 0:
        raise ValueError("identity_grid rows must be non-empty")

    for row in identity_grid:
        if len(row) != n_cols:
            raise ValueError("identity_grid must be rectangular")

    vx = np.zeros((n_rows, n_cols), dtype=float)
    vy = np.zeros((n_rows, n_cols), dtype=float)

    for i in range(n_rows):
        for j in range(n_cols):
            if j < n_cols - 1:
                vx[i, j] = identity_distance(
                    identity_grid[i][j],
                    identity_grid[i][j + 1],
                    normalized=normalized,
                )

            if i < n_rows - 1:
                vy[i, j] = identity_distance(
                    identity_grid[i][j],
                    identity_grid[i + 1][j],
                    normalized=normalized,
                )

    magnitude = np.sqrt(vx ** 2 + vy ** 2)

    spin = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            dvy_dx = vy[i, j + 1] - vy[i, j]
            dvx_dy = vx[i + 1, j] - vx[i, j]
            spin[i, j] = dvy_dx - dvx_dy

    return IdentityFieldResult(
        vx=vx,
        vy=vy,
        magnitude=magnitude,
        spin=spin,
    )
