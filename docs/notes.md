# PAM Research Notes

> This document records the developmental arc of PAM.  
> For the current implemented repository structure, see [`docs/architecture.md`](architecture.md).

This document preserves the conceptual development of the PAM project.

PAM did not begin as a conventional software project. It emerged from an extended exploratory dialogue investigating how meaning evolves under recursive generation.

The goal of these notes is to preserve the key insights and turning points that led to the current experimental framework and, eventually, to the layered observatory architecture now implemented in the repository.

---

## Prelude

The research began with a simple intuition:

> Meaning might behave like a geometric object rather than a static property of text.

Instead of treating language purely as sequences of tokens, the working hypothesis was that semantic structures might exhibit **invariants**, **curvature**, and **phase-like behavior** when subjected to recursive transformation.

The early conversations explored ideas from:

- dynamical systems
- information geometry
- invariance detection
- recursive generative processes

These ideas gradually converged into a concrete experimental question:

**What happens when language systems recursively transform their own outputs?**

---

## Early Concept: Invariants

A central idea emerged early in the project:

Certain semantic properties appear to remain stable under paraphrasing or structural variation.

These properties were referred to as **semantic invariants**.

Examples included:

- reflective tone
- geometric metaphor usage
- coherent conceptual structure
- playful-serious rhetorical balance

To test this idea computationally, the **TIP (Text Invariant Perceptron)** was introduced.

TIP acts as a lightweight detector of invariant semantic signatures.

Each text can therefore be mapped to a signature-like representation of invariant structure.

---

## Corpus Dynamics

Once invariant signatures could be detected, the next question was how they behaved under recursive generation.

This led to the construction of the core experimental loop.

Each iteration:

1. A subset of the corpus is replaced.
2. Replacement samples are generated either by:
   - mutation of anchor texts
   - resampling from recent outputs
3. A fixed anchor subset is periodically reintroduced.

This produced a simple but expressive control system defined by two parameters:

- $r$ = replacement fraction
- $\alpha$ = anchor injection probability

The evolving corpus therefore became a **discrete-time dynamical system**.

---

## Discovery of Freeze States

While observing early runs, a striking structural pattern appeared.

The corpus did not drift randomly.

Instead it organized itself into regions of:

- structural stability
- structural mutation

These regions appeared and disappeared over time.

This led to the concept of **freeze occupancy** $(\pi_F)$:

the fraction of windows exhibiting structural convergence.

Freeze detection relied on microstructure analysis using quantities such as:

- boundary density
- mean grain size

---

## Entropy Dynamics

To track diversity of semantic signatures, a second observable was introduced:

**signature entropy** $(H)$

Entropy measures the diversity of invariant signatures within the mutable corpus.

Two entropy formulations were explored:

- marginal entropy of individual invariants
- joint entropy of full signature patterns

The joint entropy proved to capture structural regime shifts more clearly.

---

## A Surprising Empirical Result

Once freeze occupancy and entropy were tracked together, a strong empirical relationship emerged:

$\mathrm{corr}(\pi_F, H) \approx -0.90 \text{ to } -0.97$

Across many runs and smoothing scales, freeze and entropy exhibited strong anticorrelation.

At first glance this suggested a simple causal story:

entropy collapse might cause structural freezing.

However, further analysis revealed a more subtle picture.

---

## Minimal Dynamical Models

To test causal interaction, minimal autoregressive models were constructed.

Two coupled equations were fit:

$F_{t+1} = a + bF_t + cH_t$

$H_{t+1} = d + eH_t + fF_t$

The key test was whether adding the cross-variable significantly improved prediction.

Results consistently showed:

- $\Delta R^2_{\text{freeze}} \approx 0.08–0.12$
- $\Delta R^2_{\text{entropy}} \approx 0.002–0.005$

Despite strong correlations, direct one-step causal coupling appeared weak.

---

## Interpretation: A Latent Regime Variable

The most plausible explanation was that freeze and entropy were not driving each other directly.

Instead they appeared to be **co-manifestations of an underlying latent regime variable**.

The system seemed to evolve along a **slow manifold** in semantic state space.

Freeze and entropy were therefore interpreted as projections of the same underlying system state.

This was one of the key bridges from descriptive measurement toward a geometric interpretation.

---

## Phase Structure

Sweeps over parameter space revealed distinct behavioral regimes:

| Regime | Description |
|------|------|
| Entropy-dominated | rapid structural drift |
| Mixed | intermittent freeze and mutation |
| Freeze | stable invariant structure |

These regimes formed an early **phase diagram in $(r, \alpha)$ parameter space**.

This phase-oriented interpretation later developed into the full geometry → phase → operators → topology observatory stack.

---

## Trajectory Invariance

A third observable was introduced:

**TIM — Trajectory Invariance Metric**

TIM was intended to measure robustness of semantic trajectories under perturbations such as:

- sentence truncation
- time-axis distortion
- partial sampling

TIM helped extend the project from snapshot invariants toward invariance of transformation paths.

---

## Robustness Testing

Several stress tests were introduced:

- seed sweeps
- smoothing-window variation
- sampling perturbations
- nested regression tests

The freeze–entropy relationship remained stable across these perturbations.

This contributed to the broader methodological rule that observables should survive moderate representational variation if they are to anchor structural analysis.

---

## Toward a Phase Discovery Protocol

The project gradually converged toward a more general methodology for studying recursive generative systems.

A working protocol emerged:

1. introduce a tunable control parameter
2. measure orthogonal macroscopic observables
3. sweep parameter space
4. detect regime transitions
5. test causal coupling
6. stress-test robustness
7. compress invariant structure into geometric form

This protocol became one of the methodological foundations of PAM.

---

## Meta Observation

An unusual feature of the project is that the research process itself involved recursive dialogue with a language model.

The system used to study recursive language dynamics was itself developed through recursive language interaction.

This introduced a form of **meta-recursion** into the development process:

the instrument for studying recursive systems was itself partially shaped through recursive linguistic exchange.

---

## From Early PAM to the Observatory

What began as an inquiry into invariants, freeze states, and entropy dynamics gradually expanded into a larger observatory architecture.

The project moved through a sequence like:

```text
recursive corpus dynamics
↓
invariant signatures
↓
entropy / macrostate observables
↓
phase structure over parameter space
↓
information geometry
↓
seam structure and signed phase
↓
operator probing
↓
topological organization
```

This progression transformed PAM from an exploratory research thread into a layered computational instrument.

---

## Current Status

The repository now supports substantially more than the early PAM framework described above.

It now includes:

- externalized corpora under `observatory/corpora/`
- canonical engine logic under `src/pam/engine/`
- measurement modules under `src/pam/measurement/`
- observable layers under `src/pam/observables/`
- canonical geometry, phase, operators, and topology packages under `src/pam/`
- stage-based orchestration under `src/pam/pipeline/`
- a canonical full-pipeline entrypoint via:

```bash
bash scripts/run_full_pipeline.sh
```

So these notes should be read as the developmental path that led into the current observatory, not as a complete description of the repository as it exists today.

---

## Next Steps

Immediate directions now include:

- refinement of canonical documentation
- stronger figure-level scientific communication
- deeper operator-path analysis
- richer response-field and topology integration
- possible continuous geometric extensions such as connection-based analysis

Longer-term directions may include:

- nonlinear regime switching
- richer operator composition
- more expressive invariant detectors
- broader capability / interaction manifold studies
- epistemic observatories for language models

---

## Final Perspective

These notes document the path by which PAM became possible.

They preserve the conceptual development of the project:

- from invariants
- to recursive dynamics
- to observables
- to phase structure
- to geometry
- to the observatory

They are intended not as the final word on PAM, but as a record of how the current instrument came into being.
