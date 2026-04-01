from __future__ import annotations

"""
Identity Distance Toy Experiment

Demonstrates:
- IdentityGraph construction
- identity-distance field over a toy control grid
- identity-change magnitude
- identity-change directional field
- identity spin / curl-like signal

Run:
    python experiments/toy/identity_distance_toy.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pam.topology.identity import Edge, IdentityGraph, Node
from pam.topology.identity_field import compute_identity_field


def two_basin_graph() -> IdentityGraph:
    nodes = {
        "A": Node("A", "basin"),
        "B": Node("B", "basin"),
    }
    edges = (
        Edge("A", "B", "adjacent"),
    )
    return IdentityGraph(nodes=nodes, edges=edges)


def three_basin_with_saddle() -> IdentityGraph:
    nodes = {
        "A": Node("A", "basin"),
        "B": Node("B", "basin"),
        "C": Node("C", "basin"),
        "S": Node("S", "saddle"),
    }
    edges = (
        Edge("A", "B", "adjacent"),
        Edge("B", "C", "adjacent"),
        Edge("S", "B", "transition"),
    )
    return IdentityGraph(nodes=nodes, edges=edges)


def star_graph() -> IdentityGraph:
    nodes = {
        "C": Node("C", "basin"),
        "A": Node("A", "basin"),
        "B": Node("B", "basin"),
        "D": Node("D", "basin"),
    }
    edges = (
        Edge("C", "A", "adjacent"),
        Edge("C", "B", "adjacent"),
        Edge("C", "D", "adjacent"),
    )
    return IdentityGraph(nodes=nodes, edges=edges)


def toy_identity(r: float, alpha: float) -> IdentityGraph:
    if r < 0.4:
        return two_basin_graph()
    if r < 0.7:
        return star_graph() if alpha < 0.5 else two_basin_graph()
    return three_basin_with_saddle()


def build_identity_grid(grid_size: int = 40) -> tuple[np.ndarray, np.ndarray, list[list[IdentityGraph]]]:
    r_vals = np.linspace(0.0, 1.0, grid_size)
    a_vals = np.linspace(0.0, 1.0, grid_size)

    identity_grid = [
        [toy_identity(r, a) for a in a_vals]
        for r in r_vals
    ]
    return r_vals, a_vals, identity_grid


def plot_results(
    r_vals: np.ndarray,
    a_vals: np.ndarray,
    magnitude: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    spin: np.ndarray,
) -> None:
    extent = [float(a_vals.min()), float(a_vals.max()), float(r_vals.min()), float(r_vals.max())]

    plt.figure()
    plt.imshow(magnitude, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="|identity change|")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Change Magnitude")
    plt.tight_layout()

    plt.figure()
    x, y = np.meshgrid(a_vals, r_vals)

    us = vx
    vs = vy

    mag = np.sqrt(us**2 + vs**2)
    mask = mag > 1e-12

    plt.quiver(
        x[mask],
        y[mask],
        us[mask],
        vs[mask],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        pivot="mid",
    )

    plt.xlim(float(a_vals.min()), float(a_vals.max()))
    plt.ylim(float(r_vals.min()), float(r_vals.max()))
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Change Field")
    plt.tight_layout()

    plt.figure()
    plt.imshow(spin, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="identity spin")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Spin Field")
    plt.tight_layout()

    plt.show()


def main() -> None:
    r_vals, a_vals, identity_grid = build_identity_grid(grid_size=40)
    field = compute_identity_field(identity_grid, normalized=True)
    plot_results(
        r_vals=r_vals,
        a_vals=a_vals,
        magnitude=field.magnitude,
        vx=field.vx,
        vy=field.vy,
        spin=field.spin,
    )


if __name__ == "__main__":
    main()
