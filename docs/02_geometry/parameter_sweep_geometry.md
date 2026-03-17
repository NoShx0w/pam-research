# Parameter Sweep Geometry

This document explains how the PAM parameter sweep forms the foundation of the observatory.

Before constructing geometry, we must understand what the parameter space represents.

## 1. The Parameter Manifold

The system is defined over a two-dimensional parameter space:

θ = (r, α)

where:
- **r** — controls recursive strength
- **α** — controls update sensitivity / coupling

The parameter space is discretized as:

r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
α ∈ linspace(0.03, 0.15, 15)

Each point corresponds to one experimental configuration.

## 2. The Grid as a Sampling of a Continuous Space

Although the parameter space is evaluated on a grid, it is conceptually Θ ⊂ ℝ², a continuous domain.

The grid should therefore be interpreted as a discrete sampling of an underlying continuous system.

## 3. Each Point as a System

For each parameter configuration θ:
- a trajectory is generated
- the system evolves over time
- observables are extracted

Thus, each point in parameter space corresponds to:

θ → dynamical system → observable signature

## 4. From Grid to Field

The collection of observables defines a field x : Θ → ℝⁿ.

Each observable becomes a scalar field over the parameter domain.

## 5. Structure in Parameter Space

In PAM:
- observables vary nonlinearly
- distinct regions emerge
- transitions occur between regimes

This implies the parameter space contains latent structure.

## 6. From Parameter Space to Geometry

Instead of treating Θ as a flat grid, we construct:
- a metric (Fisher Information)
- a distance structure (geodesics)
- an embedding (manifold)

This transforms Θ → 𝓜, where 𝓜 is an intrinsic geometric representation.

## 7. Final Perspective

The role of the parameter sweep is to provide:
- coverage of the system
- resolution of structure
- input to geometry

The final object of interest is the manifold 𝓜 and its phase structure.
