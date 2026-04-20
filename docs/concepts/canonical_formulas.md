# Canonical Formulas

## Purpose

This document collects a compact set of **canonical defining formulas** for core PAM observatory objects.

It is not intended to be a full mathematical monograph. Its role is to give the repository a stable definition layer that sits between:

- implementation details in `src/pam/`
- prose explanations in the architecture and concept docs
- and public-facing scientific writing such as the preprint

Where the repository uses a finite, graph-based, or otherwise operational approximation, that is stated explicitly.

---

## Conventions

Let the control state be

```math
\theta = (r,\alpha) \in \Theta \subset \mathbb{R}^2
```

with:

- $r$: primary control parameter
- $\alpha$: secondary control parameter

Let the observable map be

```math
\mathbf{O}(\theta) = \bigl(O_1(\theta),\dots,O_m(\theta)\bigr)^\top
```

where the components $O_i$ are the observables used to build the Fisher / response geometry.

When the repository works on a sampled control lattice, all smooth derivatives and distances should be read as operational approximations over that finite sampled surface unless otherwise stated.

---

## 1. Fisher / response metric

The canonical response-metric form is

```math
g_{ab}(\theta)
=
\partial_a \mathbf{O}(\theta)^\top
\Sigma(\theta)^{-1}
\partial_b \mathbf{O}(\theta),
\qquad a,b\in\{r,\alpha\}
```

where:

- $\partial_a \mathbf{O}$ is the sensitivity of the observable vector to control coordinate $a$
- $\Sigma(\theta)$ is the local or pooled observable-noise covariance used to uncertainty-weight those sensitivities

Interpretation:

two nearby control states are far apart when small parameter motion produces large, noise-weighted observable change.

### Finite-difference implementation

On a sampled lattice, derivatives are estimated by finite difference, for example

```math
\partial_a O_i(\theta)
\approx
\frac{O_i(\theta+\Delta_a)-O_i(\theta-\Delta_a)}{2\Delta_a}
```

Operational note:

the exact covariance regularization and stencil details are implementation-level concerns. The canonical object is the covariance-weighted local response metric.

---

## 2. Intrinsic path length

For a smooth path $\gamma:[0,1]\to\Theta$, the induced path length is

```math
L(\gamma)
=
\int_0^1
\sqrt{\dot{\gamma}(t)^\top g(\gamma(t))\,\dot{\gamma}(t)}
\,dt
```

This is the ideal continuous object whose finite graph approximation is used throughout the observatory.

---

## 3. Graph edge cost and geodesic distance

Because PAM works on a sampled control lattice, intrinsic distance is operationalized through a weighted neighbor graph.

For neighboring nodes $u,v$ with control coordinates $\theta_u,\theta_v$, the edge cost is approximated by

```math
w(u,v)
\approx
\sqrt{(\theta_v-\theta_u)^\top \bar g_{uv} (\theta_v-\theta_u)}
```

where $\bar g_{uv}$ is a midpoint or endpoint-averaged local metric.

Then the graph geodesic distance is

```math
d_g(u,v)
=
\min_{\pi:u\leadsto v}
\sum_{(i,j)\in \pi} w(i,j)
```

Interpretation:

this is the repository’s canonical intrinsic distance between sampled states.

---

## 4. MDS embedding

Let $D=(d_g(i,j))$ be the pairwise graph geodesic distance matrix.

The MDS chart places points $x_i\in\mathbb{R}^2$ so that

```math
\|x_i-x_j\| \approx d_g(i,j)
```

Interpretation:

the MDS embedding is a low-dimensional visualization chart for intrinsic distance structure.

Important note:

primary observables are defined before dimensionality reduction. The embedding is a representation layer, not the source of the geometry.

---

## 5. Signed phase coordinate

Let

```math
\phi(\theta) \in \mathbb{R}
```

denote the signed phase coordinate.

Its sign determines the local phase polarity, while its zero set defines the ideal seam / phase-boundary object.

Interpretation:

$\phi$ is the canonical oriented phase field.

---

## 6. Phase boundary / seam

The ideal phase boundary is the zero set of the signed phase field:

```math
\mathcal{B}
=
\{\theta \in \Theta : \phi(\theta)=0\}
```

Operationally, on a sampled manifold, the seam is estimated from the sign structure and associated geometric transition cues.

Interpretation:

the seam is the primary transition-separating object of the observatory.

---

## 7. Distance to phase boundary

The intrinsic distance to the phase boundary is

```math
d_{\mathcal B}(\theta)
=
\inf_{\theta' \in \mathcal B} d_g(\theta,\theta')
```

This is one of the repository’s main organizing coordinates.

Interpretation:

$d_{\mathcal B}$ measures how far a state sits from the estimated transition structure in intrinsic manifold distance, not raw Euclidean control-space distance.

---

## 8. Curvature proxy

Let

```math
K(\theta)
```

denote the ideal scalar curvature of the response metric $g(\theta)$.

In practice, the repository works with a finite-difference curvature estimate or curvature proxy

```math
\widehat K(\theta) \approx K(\theta)
```

Interpretation:

$\widehat K$ is a local geometric stress or concentration signal derived from the manifold.

Operational note:

unless the exact estimator is being discussed, “curvature proxy” is usually the most honest public wording.

---

## 9. Lazarus score

The Lazarus score is the canonical trajectory-level compression observable.

For a trajectory $\pi=(\theta_0,\dots,\theta_T)$, the abstract canonical form is

```math
L(\pi)=\mathcal{C}(\pi)
```

where $\mathcal{C}$ is the repository’s chosen compression functional.

A useful expanded schematic form is

```math
L(\pi)
=
\frac{1}{T+1}
\sum_{t=0}^{T}
C(\theta_t)
```

where $C(\theta_t)$ is the local compression contribution at trajectory point $\theta_t$.

Interpretation:

higher Lazarus values indicate stronger trajectory-level compression or recovery-like concentration.

Operational note:

the exact composition of $C(\theta_t)$ is implementation-level and may involve stabilized combinations of local observables. The canonical object is the trajectory-compression summary, not any one prose nickname.

---

## 10. Finite-horizon transition indicator

Let $R_t$ denote the regime label at time $t$, or in the sign-based case let $s_t=\operatorname{sign}\phi(\theta_t)$.

The $k$-step transition indicator is

```math
Y_t^{(k)}
=
\mathbf{1}
\left[
\exists j\in\{1,\dots,k\}
\text{ such that }
R_{t+j}\neq R_t
\right]
```

In the sign-based case, this becomes

```math
Y_t^{(k)}
=
\mathbf{1}
\left[
\exists j\in\{1,\dots,k\}
\text{ such that }
s_{t+j}\neq s_t
\right]
```

Interpretation:

$Y_t^{(k)}=1$ means a regime change occurs within horizon $k$.

---

## 11. Empirical transition probability

The empirical transition probability at a state or stratum is

```math
P_k(\theta)
\approx
\mathbb{P}\!\left(Y_t^{(k)}=1 \mid \theta_t \approx \theta \right)
```

For binned geometric analysis, one often works with a stratum-level form such as

```math
P_k(d,\widehat K)
\approx
\mathbb{P}\!\left(
Y_t^{(k)}=1
\mid
d_{\mathcal B}(\theta_t)\in d,\;
\widehat K(\theta_t)\in \widehat K
\right)
```

Interpretation:

transition probability is treated as a geometric field over distance-to-boundary and curvature strata.

---

## 12. Boundary approach direction

A simple approach-direction observable is the early slope of distance-to-boundary along a trajectory:

```math
A_t
=
\frac{d_{\mathcal B}(\theta_{t+\ell})-d_{\mathcal B}(\theta_t)}{\ell}
```

Negative values indicate approach toward the boundary.

Interpretation:

this is a local directional summary of whether a path is moving toward or away from the seam.

---

## 13. Conditional geometric law

The central transition-law form of the early PAM preprint can be written schematically as

```math
\mathbb{P}\!\left(Y_t^{(k)}=1 \mid \theta_t\right)
\approx
F\!\left(
d_{\mathcal B}(\theta_t),
\widehat K(\theta_t)
\right)
```

Interpretive shorthand:

```math
d_{\mathcal B}\text{ gates access},\qquad
\widehat K\text{ modulates within accessible strata}
```

Interpretation:

distance to the boundary determines whether transition-like behavior is accessible at all, while curvature shapes local expression within that accessible region.

---

## 14. Conditional Lazarus refinement

A stronger conditional statement used in the transition-law program is:

for near-boundary geometric strata $(I,J)$,

```math
\mathbb{P}\!\left(
Y_t^{(k)}=1
\mid
d_{\mathcal B}\in I,\;
\widehat K\in J,\;
L\text{ high}
\right)
>
\mathbb{P}\!\left(
Y_t^{(k)}=1
\mid
d_{\mathcal B}\in I,\;
\widehat K\in J,\;
L\text{ low}
\right)
```

Interpretation:

Lazarus retains predictive structure within fixed geometric strata, especially near the critical regime.

---

## 15. Selection regime

Let $\mathcal{C}$ be a set of outcome classes such as

```math
\mathcal{C}
=
\{
\text{in\_basin},
\text{seam\_graze},
\text{seam\_cross},
\text{lazarus},
\text{collapse},
\text{recovered}
\}
```

Then the selection regime for class $c$ can be written as

```math
\mathcal{S}_c
=
\left\{
\theta \in \Theta :
c = \arg\max_{c' \in \mathcal{C}}
P(c' \mid \theta,\text{IC-family})
\right\}
```

Interpretation:

the manifold partitions into regions where different outcomes dominate under a chosen initial-condition family.

---

## 16. Identity distance

Let $I(\theta)$ denote the local identity object, such as an `IdentityGraph`.

The structural identity distance is schematically

```math
d_I(\theta,\theta')
=
D\!\left(I(\theta), I(\theta')\right)
```

where $D$ compares structural summaries such as:

- node-signature multisets
- adjacency signatures
- transition signatures

Interpretation:

identity distance measures difference in local relational organization rather than difference in a single scalar field.

---

## 17. Identity field magnitude

Once neighboring identity distances are available, identity-field magnitude can be represented schematically as

```math
M_I(\theta)
=
\sqrt{v_x(\theta)^2 + v_y(\theta)^2}
```

where $v_x,v_y$ are local directional identity-difference summaries.

Interpretation:

identity magnitude measures how rapidly local structural organization changes across the manifold.

---

## 18. Loop transport residual

For a small loop with two local paths from $A$ to $C$, the identity-transport residual can be written schematically as

```math
H_{\square}
=
\Delta\!\left(
T_{A\to B\to C},
T_{A\to D\to C}
\right)
```

where $T$ denotes transported identity evolution along a path and $\Delta$ is the chosen residual comparison.

Interpretation:

$H_{\square}$ is a holonomy-like local loop residual. Nonzero values indicate path dependence of identity transport.

---

## 19. Obstruction

Unsigned obstruction is the localized magnitude of nearby loop residual concentration:

```math
\Omega(\theta)
=
\operatorname{Agg}_{\square \ni \theta}
|H_{\square}|
```

where the aggregation runs over nearby loop cells incident to $\theta$.

Signed obstruction retains orientation:

```math
\Omega_{\pm}(\theta)
=
\operatorname{Agg}_{\square \ni \theta}
H_{\square}
```

Interpretation:

obstruction is not postulated. It is constructed from local loopwise identity-transport residual.

---

## 20. Response-guided direction field

Let $T(\theta)$ denote the local response tensor. Its dominant response direction can be represented by the leading eigenvector

```math
v_{\mathrm{rsp}}(\theta)
=
\operatorname{eigvec}_{\max}\!\left(T(\theta)\right)
```

Interpretation:

the response-guided flow layer is built from the dominant local response direction field over the manifold.

Operational note:

because eigenvectors are axial objects, direction comparisons are often taken modulo $\pi$, not $2\pi$.

---

## 21. Continuous response-flow interpolation

In the continuous reconstruction line, the local response direction used by the integrator is an interpolation over nearby anchor directions:

```math
\tilde v(x)
=
\mathcal{I}\!\left(
\{(x_i,v_i)\}_{i\in \mathcal{N}(x)}
\right)
```

where:

- $x$ is an embedded manifold position
- $x_i$ are nearby support points
- $v_i$ are their local response directions
- $\mathcal{I}$ is the interpolation rule, such as nearest-anchor, top2-blend, or local averaging

Interpretation:

continuous response flow is a geometric reconstruction of the discrete response field, not an independent dynamical substrate.

---

## 22. Signed versus unsigned fields

Several observatory fields have both signed and unsigned forms.

A generic signed field is:

```math
f_{\pm}(\theta) \in \mathbb{R}
```

and its unsigned magnitude form is:

```math
|f_{\pm}(\theta)|
```

Interpretation:

- the signed form preserves polarity or orientation
- the unsigned form preserves concentration only

This distinction is especially important for:

- signed phase
- signed obstruction
- holonomy-like residuals
- TUI rendering semantics

---

## Summary

These formulas define the repository’s main observatory objects at a compact canonical level.

They are intended to stabilize the distinction between:

- ideal continuous objects
- operational graph- or lattice-based approximations
- and interface- or prose-level interpretations

The governing principle is:

**PAM should define its observatory objects clearly enough to be scientifically legible, while remaining honest about where the code uses finite, graph-based, or proxy constructions.**
