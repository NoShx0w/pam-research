#!/usr/bin/env python3
from __future__ import annotations

"""
Identity Coordinate Invariance Toy

Goal
----
Test whether identity spin / holonomy structure survives a coordinate
reparameterization.

This is a *chart-change* toy, not a new-physics toy.

Key principle
-------------
We keep the same underlying physical lattice points fixed and only
change the coordinates used to represent them.

That means:
- identity is evaluated on the same physical points
- only the local coordinate displacements used in the discrete transport
  / spin proxy are changed

Outputs
-------
- base and warped holonomy fields
- base and warped spin fields
- singularity overlap metrics
- correlation metrics

Run
---
PYTHONPATH=src .venv/bin/python experiments/toy/identity_coordinate_invariance.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pam.topology.identity import Edge, IdentityGraph, Node, identity_distance


# ---------------------------------------------------------------------
# Coordinate charts
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalPoint:
    i: int
    j: int
    r: float
    alpha: float


def make_physical_grid(n_r: int = 18, n_a: int = 18) -> list[list[PhysicalPoint]]:
    r_vals = np.linspace(0.05, 0.30, n_r)
    a_vals = np.linspace(0.02, 0.15, n_a)

    grid: list[list[PhysicalPoint]] = []
    for i, r in enumerate(r_vals):
        row = []
        for j, a in enumerate(a_vals):
            row.append(PhysicalPoint(i=i, j=j, r=float(r), alpha=float(a)))
        grid.append(row)
    return grid


def base_chart(p: PhysicalPoint) -> tuple[float, float]:
    return p.r, p.alpha


def warped_chart(p: PhysicalPoint) -> tuple[float, float]:
    """
    Nonlinear reparameterization of the same physical point.

    This changes the coordinate chart, not the underlying state.
    """
    r = p.r
    a = p.alpha
    u = r
    v = a + 0.30 * r * a
    return float(u), float(v)


# ---------------------------------------------------------------------
# Identity builder on PHYSICAL points
# ---------------------------------------------------------------------


def build_identity_from_physical_point(p: PhysicalPoint) -> IdentityGraph:
    """
    Toy structural identity defined on the underlying physical coordinates.

    Important:
    This depends on the physical point, not the chart.
    """
    r = p.r
    a = p.alpha

    if r < 0.14:
        nodes = {
            "A": Node("A", "basin", {}),
            "B": Node("B", "basin", {}),
        }
        edges = (
            Edge("A", "B", "adjacent"),
        )

    elif a < 0.07:
        nodes = {
            "C": Node("C", "basin", {}),
            "A": Node("A", "basin", {}),
            "B": Node("B", "basin", {}),
        }
        edges = (
            Edge("C", "A", "adjacent"),
            Edge("C", "B", "adjacent"),
        )

    elif r > 0.22 and a > 0.11:
        nodes = {
            "A": Node("A", "basin", {}),
            "B": Node("B", "basin", {}),
            "S": Node("S", "saddle", {}),
        }
        edges = (
            Edge("A", "B", "adjacent"),
            Edge("S", "B", "transition"),
        )

    else:
        nodes = {
            "A": Node("A", "basin", {}),
            "B": Node("B", "basin", {}),
            "C": Node("C", "basin", {}),
            "S": Node("S", "saddle", {}),
        }
        edges = (
            Edge("A", "B", "adjacent"),
            Edge("B", "C", "adjacent"),
            Edge("S", "B", "transition"),
        )

    return IdentityGraph(nodes=nodes, edges=tuple(edges))


def build_identity_field(grid: list[list[PhysicalPoint]]) -> list[list[IdentityGraph]]:
    return [
        [build_identity_from_physical_point(p) for p in row]
        for row in grid
    ]


# ---------------------------------------------------------------------
# Coordinate-aware local transport / spin proxies
# ---------------------------------------------------------------------


def edge_distance(g1: IdentityGraph, g2: IdentityGraph) -> float:
    return float(identity_distance(g1, g2, normalized=True))


def compute_cell_holonomy(
    identities: list[list[IdentityGraph]],
) -> np.ndarray:
    """
    Chart-independent path residual over elementary loops.

    Since this toy uses path totals from graph distances alone,
    this quantity should be more invariant than chart-dependent local
    differential approximations.
    """
    n_i = len(identities)
    n_j = len(identities[0])

    hol = np.zeros((n_i - 1, n_j - 1), dtype=float)

    for i in range(n_i - 1):
        for j in range(n_j - 1):
            A = identities[i][j]
            B = identities[i][j + 1]
            C = identities[i + 1][j + 1]
            D = identities[i + 1][j]

            d_ab = edge_distance(A, B)
            d_bc = edge_distance(B, C)
            d_ad = edge_distance(A, D)
            d_dc = edge_distance(D, C)

            hol[i, j] = (d_ab + d_bc) - (d_ad + d_dc)

    return hol


def compute_chart_aware_spin(
    identities: list[list[IdentityGraph]],
    grid: list[list[PhysicalPoint]],
    chart_fn,
) -> np.ndarray:
    """
    A small chart-aware local spin proxy.

    We compute edge differences and normalize by local coordinate step sizes
    in the active chart. This is still a proxy, but it is now explicitly chart-based.

    For each cell:
      spin ~ (dVy/dx - dVx/dy)

    where x,y are the active chart coordinates.
    """
    n_i = len(identities)
    n_j = len(identities[0])

    vx = np.zeros((n_i, n_j), dtype=float)
    vy = np.zeros((n_i, n_j), dtype=float)

    # local directional "rates"
    for i in range(n_i):
        for j in range(n_j):
            x0, y0 = chart_fn(grid[i][j])

            if j < n_j - 1:
                x1, y1 = chart_fn(grid[i][j + 1])
                ds = edge_distance(identities[i][j], identities[i][j + 1])
                dx = np.hypot(x1 - x0, y1 - y0)
                vx[i, j] = ds / dx if dx > 0 else 0.0

            if i < n_i - 1:
                x1, y1 = chart_fn(grid[i + 1][j])
                ds = edge_distance(identities[i][j], identities[i + 1][j])
                dy = np.hypot(x1 - x0, y1 - y0)
                vy[i, j] = ds / dy if dy > 0 else 0.0

    spin = np.zeros((n_i - 1, n_j - 1), dtype=float)

    for i in range(n_i - 1):
        for j in range(n_j - 1):
            x00, y00 = chart_fn(grid[i][j])
            x01, y01 = chart_fn(grid[i][j + 1])
            x10, y10 = chart_fn(grid[i + 1][j])

            dx_local = np.hypot(x01 - x00, y01 - y00)
            dy_local = np.hypot(x10 - x00, y10 - y00)

            dvy_dx = (vy[i, j + 1] - vy[i, j]) / dx_local if dx_local > 0 else 0.0
            dvx_dy = (vx[i + 1, j] - vx[i, j]) / dy_local if dy_local > 0 else 0.0

            spin[i, j] = dvy_dx - dvx_dy

    return spin


# ---------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------


def threshold_singularities(field: np.ndarray, quantile: float = 0.90) -> set[tuple[int, int]]:
    mag = np.abs(field)
    thr = float(np.quantile(mag, quantile))
    coords = set()
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if abs(field[i, j]) >= thr:
                coords.add((i, j))
    return coords


def jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    import pandas as pd
    df = pd.DataFrame({"x": x.ravel(), "y": y.ravel()}).dropna()
    if len(df) < 3:
        return float("nan")
    return float(df["x"].corr(df["y"], method="spearman"))


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------


def render_panel(
    hol_base: np.ndarray,
    hol_warp: np.ndarray,
    spin_base: np.ndarray,
    spin_warp: np.ndarray,
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    hol_abs_max = float(max(np.max(np.abs(hol_base)), np.max(np.abs(hol_warp)), 1e-9))
    spin_abs_max = float(max(np.max(np.abs(spin_base)), np.max(np.abs(spin_warp)), 1e-9))

    im0 = axes[0, 0].imshow(hol_base, origin="lower", cmap="coolwarm",
                            vmin=-hol_abs_max, vmax=hol_abs_max, aspect="auto")
    axes[0, 0].set_title("Base holonomy")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)

    im1 = axes[0, 1].imshow(hol_warp, origin="lower", cmap="coolwarm",
                            vmin=-hol_abs_max, vmax=hol_abs_max, aspect="auto")
    axes[0, 1].set_title("Warped holonomy")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)

    im2 = axes[1, 0].imshow(spin_base, origin="lower", cmap="coolwarm",
                            vmin=-spin_abs_max, vmax=spin_abs_max, aspect="auto")
    axes[1, 0].set_title("Base spin")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)

    im3 = axes[1, 1].imshow(spin_warp, origin="lower", cmap="coolwarm",
                            vmin=-spin_abs_max, vmax=spin_abs_max, aspect="auto")
    axes[1, 1].set_title("Warped spin")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.85)

    fig.suptitle("Identity coordinate invariance toy", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def run_experiment(
    *,
    n_r: int = 18,
    n_a: int = 18,
    singular_quantile: float = 0.90,
    outdir: str | Path = "outputs/toy_identity_coordinate_invariance",
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    physical_grid = make_physical_grid(n_r=n_r, n_a=n_a)
    identities = build_identity_field(physical_grid)

    # same identities, different charts
    hol_base = compute_cell_holonomy(identities)
    hol_warp = compute_cell_holonomy(identities)

    spin_base = compute_chart_aware_spin(identities, physical_grid, base_chart)
    spin_warp = compute_chart_aware_spin(identities, physical_grid, warped_chart)

    sing_hol_base = threshold_singularities(hol_base, quantile=singular_quantile)
    sing_hol_warp = threshold_singularities(hol_warp, quantile=singular_quantile)

    sing_spin_base = threshold_singularities(spin_base, quantile=singular_quantile)
    sing_spin_warp = threshold_singularities(spin_warp, quantile=singular_quantile)

    summary = {
        "holonomy_spearman": safe_spearman(np.abs(hol_base), np.abs(hol_warp)),
        "spin_spearman": safe_spearman(np.abs(spin_base), np.abs(spin_warp)),
        "holonomy_singularity_jaccard": jaccard(sing_hol_base, sing_hol_warp),
        "spin_singularity_jaccard": jaccard(sing_spin_base, sing_spin_warp),
        "n_hol_base": len(sing_hol_base),
        "n_hol_warp": len(sing_hol_warp),
        "n_spin_base": len(sing_spin_base),
        "n_spin_warp": len(sing_spin_warp),
    }

    render_panel(
        hol_base=hol_base,
        hol_warp=hol_warp,
        spin_base=spin_base,
        spin_warp=spin_warp,
        outpath=outdir / "identity_coordinate_invariance_panel.png",
    )

    with open(outdir / "identity_coordinate_invariance_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\n=== Identity Coordinate Invariance Toy ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(outdir / "identity_coordinate_invariance_panel.png")
    print(outdir / "identity_coordinate_invariance_summary.txt")

    return {
        "hol_base": hol_base,
        "hol_warp": hol_warp,
        "spin_base": spin_base,
        "spin_warp": spin_warp,
        "summary": summary,
    }


if __name__ == "__main__":
    run_experiment()
