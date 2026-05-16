# 059 — Proto-Groupoid Returnability Design

## Status

Design note. Not yet a canonical OBS result.

This note defines a PAM-native way to relax strict algebraic invertibility into stochastic, lossy, bounded-cost returnability over reduced symbolic state spaces. It is intended as a future design basis for an OBS-style study, likely `OBS-068`, after the current trajectory and corpus campaign lane stabilizes.

## Purpose

PAM has already measured survival and drift across proto, gateway, canonical-family, and anchor-family layers. OBS-065 and OBS-066 tested survival under route-origin decoy replacement; OBS-067 joined those outputs into a cross-layer coupling ledger.

The next algebraic question is not only whether a symbolic label survives, but whether its structural role survives.

A move may retain its label while becoming harder to reverse. A fine proto-relation may fail while the system still returns to the same canonical family. A gateway may be cheap in one direction and expensive or absent in the other. These are not strict groupoid behaviors, but they are proto-groupoid behaviors.

This note defines that structure through returnability.

## Guardrail

PAM should not claim strict groupoid invertibility.

A strict inverse would require a move

```text
m: A -> B
```

with inverse

```text
m^-1: B -> A
```

such that both compositions recover identity exactly:

```text
m^-1 ∘ m = 1_A
m ∘ m^-1 = 1_B
```

Language dynamics are directed, lossy, stochastic, projected through reduced symbolic layers, and frequently absorb fine differences at coarser family levels. Therefore PAM uses a weaker operational definition:

> A move has a proto-inverse when the system can return to an equivalent structural class under bounded cost, bounded path length, and bounded residual drift.

## 1. Reduced State Space and Family Projection

Let `S` be a reduced symbolic state space. A state may represent a route state, sector state, gateway state, canonical-family state, anchor-family state, or another PAM-defined compressed continuation context.

Let `F` be a set of canonical or anchor families. Define a family assignment map:

```text
[·]: S -> F
```

For a state `s`, `[s]` denotes the canonical or anchor family to which `s` belongs.

The family map is intentionally coarser than exact state identity. This preserves the OBS-067 lesson: downstream layers may absorb fine proto-relation failure through coarser projection.

## 2. Symbolic Moves and Cost Field

A symbolic move is a directed edge:

```text
m: s_i -> s_j
```

where `m` may correspond to a generator, typed edge, route transition, gateway event, canonical-family transition, or another reduced symbolic transition available in PAM artifacts.

The affordance landscape assigns a cost to symbolic moves:

```text
c_pi(m) >= 0
```

where `pi` denotes the prompt, corpus, campaign, or decoy-conditioned context.

### Cost estimator guardrail

Do not assume that `c_pi(m)` is a direct LM negative log-probability unless direct state-transition probabilities are actually available.

Current PAM artifacts generally provide empirical counts, survival rates, rank behavior, matching distances, TV distances, and drift summaries. Therefore `c_pi(m)` should be treated as an operational edge-cost estimator.

Possible estimators include:

```text
-log empirical transition frequency
rank-derived cost
decoy matching-distance cost
distribution TV-distance cost
survival-failure penalty
composite normalized edge score
```

Every artifact produced from this design should include:

```text
cost_estimator
cost_estimator_version
```

This prevents later comparisons from silently mixing incompatible meanings of cost.

## 3. Path Cost

For a path

```text
rho = (e_1, e_2, ..., e_n)
```

use bottleneck cost as the first design target:

```text
C_pi(rho) = max_e c_pi(e)
```

The bottleneck definition preserves the gateway intuition: a path is only as accessible as its most expensive required transition.

An additive path energy may later be useful, but bottleneck cost is the safest first choice for thresholded returnability.

## 4. Proto-Inverse Path

Given a move

```text
m: s_i -> s_j
```

a proto-inverse is not a single reverse edge. It is a valid path

```text
rho: s_j -> s_k
```

such that

```text
[s_k] = [s_i]
```

That is: the path returns to the source state’s canonical or anchor family.

Exact source-state recovery is a special case. Family-level recovery is the general PAM-native case.

## 5. Return Metrics

For each forward move `m`, preserve three decoupled return metrics.

### 5.1 Return bottleneck cost

```text
R_pi(m) = inf_{rho: target(m) -> s_k, [s_k] = [source(m)]} C_pi(rho)
```

Interpretation:

```text
low R_pi(m): cheap return path exists
high R_pi(m): return requires an expensive gateway
infinite R_pi(m): no return observed in the reduced graph under the chosen search limits
```

### 5.2 Return path length

```text
len(rho*)
```

where `rho*` is the selected best return path.

A short high-cost return and a long low-cost return represent different regimes, so length should not be collapsed into the cost too early.

### 5.3 Residual return drift

Return drift measures how much structure is lost even after family-level recovery.

A first version should use a tiered symbolic drift class rather than a single embedding metric:

```text
0 = exact_return
1 = sector_return
2 = family_projected_return
3 = weak_family_return
infinite = no_return_observed
```

Possible components:

```text
exact_state_match
sector_match
canonical_family_match
proto_relation_overlap
distribution_tv_distance, if available
```

This tiered structure matches OBS-067 better than a premature continuous distance, because coarse projection can survive fine proto failure.

## 6. Directional Asymmetry

Define move asymmetry:

```text
A_pi(m) = R_pi(m) - C_pi(m)
```

where `C_pi(m)` is the forward move cost.

Interpretation:

```text
A_pi(m) approx 0:
  quasi-reversible core

A_pi(m) > 0:
  easy to leave, hard to return; directed escape or gateway behavior

A_pi(m) < 0:
  costly entry into a state that cheaply relaxes back to the origin family

A_pi(m) = infinite:
  forward path exists but no return is observed in the reduced graph
```

This asymmetry is one of the core proto-groupoid diagnostics.

## 7. Correct Minimax Return Search

The return search is performed over the observed reduced symbolic transition graph, not over the full latent language-model state space.

Therefore, `directed_escape` means:

> no return was observed in the reduced artifact graph under the selected graph construction, cost estimator, depth constraints, and search policy.

It does not prove that no return exists in the model.

### 7.1 Algorithmic note

A minimax path search can be implemented with a Dijkstra-like priority queue because bottleneck cost is monotone non-decreasing along a path:

```text
new_cost = max(current_cost, edge.cost)
```

However, the search must preserve path length as a secondary tie-breaker. A tracking dictionary that stores only the best bottleneck cost per node can discard equal-bottleneck but shorter paths. Since return path length is a first-class diagnostic, the best state must be stored as a tuple:

```text
best[node] = (best_bottleneck_cost, best_path_length)
```

A relaxation update should be accepted if it improves bottleneck cost, or if it preserves bottleneck cost while shortening the path:

```text
(new_cost, new_length) < best[next_node]
```

lexicographically.

### 7.2 Reference implementation

```python
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any, Callable


@dataclass(frozen=True)
class Edge:
    source: Any
    target: Any
    cost: float
    label: str = ""


class SymbolicTransitionGraph:
    def __init__(self) -> None:
        self.adj: dict[Any, list[Edge]] = {}

    def add_edge(self, source: Any, target: Any, cost: float, label: str = "") -> None:
        self.adj.setdefault(source, [])
        self.adj.setdefault(target, [])
        self.adj[source].append(Edge(source=source, target=target, cost=float(cost), label=label))

    def out_edges(self, node: Any) -> list[Edge]:
        return self.adj.get(node, [])


def minimax_return_path(
    graph: SymbolicTransitionGraph,
    start: Any,
    goal_family: Any,
    family_of: Callable[[Any], Any],
    max_depth: int | None = None,
) -> tuple[float, int | None, list[Edge] | None, Any | None]:
    """
    Find the minimum-bottleneck return path from `start` to any node whose
    family equals `goal_family`.

    Path length is used as a secondary tie-breaker.

    Returns:
        return_bottleneck, return_path_len, return_edges, return_target_state
    """
    heap: list[tuple[float, int, Any, list[Edge]]] = [(0.0, 0, start, [])]
    best: dict[Any, tuple[float, int]] = {start: (0.0, 0)}

    while heap:
        cost, length, node, path = heappop(heap)

        current_best = best.get(node, (inf, inf))
        if (cost, length) > current_best:
            continue

        if length > 0 and family_of(node) == goal_family:
            return cost, length, path, node

        if max_depth is not None and length >= max_depth:
            continue

        for edge in graph.out_edges(node):
            nxt = edge.target
            new_cost = max(cost, edge.cost)
            new_length = length + 1
            candidate = (new_cost, new_length)

            if candidate < best.get(nxt, (inf, inf)):
                best[nxt] = candidate
                heappush(heap, (new_cost, new_length, nxt, path + [edge]))

    return inf, None, None, None
```

## 8. Proto-Inverse Classification

The first classifier should remain conservative and inspectable. It should preserve raw metrics in the CSV and only assign a coarse class for summary use.

Recommended classes:

```text
quasi_inverse
bounded_return
costly_return
family_projected_return
directed_escape
no_return_observed
```

In the first version, `directed_escape` and `no_return_observed` may be separated only if graph density and search coverage justify the distinction.

### Reference classifier

```python
def classify_proto_inverse(
    return_found: bool,
    asymmetry: float,
    length: int | None,
    drift_class: int,
    tolerance: float = 0.2,
) -> str:
    if not return_found or length is None:
        return "directed_escape"

    if drift_class <= 1 and length <= 2 and abs(asymmetry) <= tolerance:
        return "quasi_inverse"

    # Check projected returns before bounded returns so that low-cost but
    # high-drift family recovery is not mislabeled as locally bounded.
    if drift_class >= 2:
        return "family_projected_return"

    if asymmetry > tolerance:
        return "costly_return"

    if asymmetry <= tolerance:
        return "bounded_return"

    return "bounded_return"
```

Thresholds such as `tolerance` should not be universal constants in the first study. Prefer empirical quantiles per corpus, band, route class, or cost estimator.

## 9. Survival Modes Under Decoy Replacement

Under a decoy replacement context change:

```text
pi -> pi_decoy
```

compare both label survival and returnability survival.

Important deltas:

```text
Delta R = |R_baseline(m) - R_decoy(m)|
Delta A = |A_baseline(m) - A_decoy(m)|
```

Suggested survival profiles:

```text
strong_structural_survival:
  label survives and returnability profile remains stable

algebraic_mutation:
  label survives but returnability or asymmetry changes strongly

topological_absorption_or_projection:
  fine label fails but family-level returnability survives

true_structural_break:
  label fails and return path disappears in the reduced graph
```

This distinction extends OBS-067 directly. OBS-067 showed that proto survival does not simply determine gateway/canonical survival. A returnability study would ask whether the algebraic role of a surviving or failing relation also survives.

## 10. Proposed Artifact Schema

A first OBS-style implementation could emit:

```text
outputs/obs068_proto_inverse_returnability_<CORPUS>/
  obs068_proto_inverse_edges.csv
  obs068_returnability_summary.csv
  obs068_proto_inverse_report.md
```

Suggested edge-level CSV columns:

```csv
corpus,band,replacement_route_class,context_id,
cost_estimator,cost_estimator_version,
source_state,target_state,source_family,target_family,move_label,move_cost,
return_found,return_bottleneck,return_path_len,return_target_state,return_target_family,
return_drift_class,return_drift,
asymmetry_score,proto_inverse_class
```

Optional columns:

```csv
max_depth,graph_n_nodes,graph_n_edges,edge_count,edge_frequency,edge_rank,
baseline_label_survived,delta_return_bottleneck,delta_asymmetry,
source_artifact,source_artifact_row_id
```

## 11. Relation to Existing OBS Chain

```text
OBS-065:
  proto-groupoid survival under route-origin decoy replacement

OBS-066:
  gateway, canonical-family, and anchor-family survival under the same regimes

OBS-067:
  coupling ledger between proto survival and downstream survival

Future OBS-068 candidate:
  proto-inverse returnability and algebraic-role survival
```

OBS-068 should not rescan raw language-model traces unless necessary. The first version should consume compact symbolic path/count artifacts, build an observed reduced transition graph, and compute returnability over that graph.

## 12. Open Design Choices

Before implementation, decide:

```text
1. Which reduced graph to build first:
   proto_edge graph, proto_relation graph, gateway graph, or canonical-family graph.

2. Which first cost estimator to use:
   empirical edge frequency is likely the simplest first estimator.

3. How to define family_of(node):
   canonical family, anchor family, sector family, or route class.

4. How to bound search:
   max_depth = 4 or 5 is a conservative first default.

5. How to classify drift:
   exact, sector, family-projected, weak-family, no-return.

6. How to compare baseline vs decoy:
   label-matched only, family-matched fallback, or both.
```

## 13. Minimal Scientific Claim

The safe claim is:

> PAM can weaken strict invertibility into bounded-cost returnability over reduced symbolic states. This defines a proto-groupoid diagnostic: moves with cheap, short, low-drift returns belong to a quasi-reversible core; moves with costly, long, projected, or absent returns mark directed escape, gateway behavior, or structural break.

The unsafe claim, avoided here, is:

> The language model forms a true groupoid over tokens or hidden states.

## 14. Summary

The returnability design gives PAM an algebraic diagnostic that sits between symbolic survival and topological persistence.

It asks:

```text
When the system leaves a symbolic family, can it return?
How expensive is that return?
How long is the return path?
How much structure is lost in the loop?
Does decoy replacement preserve the label but mutate the role?
Does fine proto failure still preserve coarse family returnability?
```

This is a natural continuation of the OBS-065/066/067 line, but it should remain a design note until the active trajectory campaigns and runtime infrastructure are stable.
