"""
Identity Distance Toy Experiment (Extended)

Demonstrates:
- IdentityGraph construction
- Identity distance scalar field
- Identity gradient (vector field)
- Identity spin (curl)

Run:
    python identity_distance_toy.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pam.topology.identity import (
    Node,
    Edge,
    IdentityGraph,
    identity_distance,
)


def two_basin_graph():
    nodes = {
        "A": Node("A", "basin"),
        "B": Node("B", "basin"),
    }
    edges = (
        Edge("A", "B", "adjacent"),
    )
    return IdentityGraph(nodes, edges)


def three_basin_with_saddle():
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
    return IdentityGraph(nodes, edges)


def star_graph():
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
    return IdentityGraph(nodes, edges)


def toy_identity(r, alpha):
    if r < 0.4:
        return two_basin_graph()
    elif r < 0.7:
        if alpha < 0.5:
            return star_graph()
        else:
            return two_basin_graph()
    else:
        return three_basin_with_saddle()


def run_experiment(grid_size=40):
    r_vals = np.linspace(0.0, 1.0, grid_size)
    a_vals = np.linspace(0.0, 1.0, grid_size)

    identities = [[toy_identity(r, a) for a in a_vals] for r in r_vals]

    Vx = np.zeros((grid_size, grid_size))
    Vy = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):

            if j < grid_size - 1:
                Vx[i, j] = identity_distance(
                    identities[i][j],
                    identities[i][j + 1]
                )

            if i < grid_size - 1:
                Vy[i, j] = identity_distance(
                    identities[i][j],
                    identities[i + 1][j]
                )

    magnitude = np.sqrt(Vx**2 + Vy**2)

    curl = np.zeros((grid_size, grid_size))

    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            dVy_dx = Vy[i, j+1] - Vy[i, j]
            dVx_dy = Vx[i+1, j] - Vx[i, j]
            curl[i, j] = dVy_dx - dVx_dy

    return r_vals, a_vals, magnitude, Vx, Vy, curl


def plot_results(r_vals, a_vals, magnitude, Vx, Vy, curl):

    plt.figure()
    plt.imshow(magnitude, origin="lower", extent=[0,1,0,1])
    plt.colorbar(label="|∇ Identity|")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Gradient Magnitude")
    plt.tight_layout()

    plt.figure()
    plt.quiver(Vx, Vy)
    plt.title("Identity Gradient Field")

    plt.figure()
    plt.imshow(curl, origin="lower", extent=[0,1,0,1])
    plt.colorbar(label="Spin (curl)")
    plt.xlabel("alpha")
    plt.ylabel("r")
    plt.title("Identity Spin Field")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    r_vals, a_vals, magnitude, Vx, Vy, curl = run_experiment()
    plot_results(r_vals, a_vals, magnitude, Vx, Vy, curl)
