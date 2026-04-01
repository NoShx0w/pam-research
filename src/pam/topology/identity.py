from __future__ import annotations

"""
PAM Topological Identity

Defines:
- IdentityGraph: minimal relational/topological representation
- extract_identity(...): construct an IdentityGraph from topology-like state
- identity_distance(...): compare two identity graphs structurally

Upgrades in this version:
- degree-aware node signatures
- label-invariant structural comparison
- multiset-based edge comparison
- optional normalized distance
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    attributes: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str  # "adjacent" or "transition"
    weight: float = 1.0


@dataclass(frozen=True)
class IdentityGraph:
    nodes: dict[str, Node]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class IdentityDistanceReport:
    node_cost: float
    adjacency_cost: float
    transition_cost: float
    raw_distance: float
    normalized_distance: float


def _validate_graph(graph: IdentityGraph) -> None:
    node_ids = set(graph.nodes.keys())

    for node_id, node in graph.nodes.items():
        if node_id != node.id:
            raise ValueError(
                f"IdentityGraph node key/id mismatch: key={node_id!r}, node.id={node.id!r}"
            )
        if not node.kind:
            raise ValueError(f"Node {node_id!r} has empty kind.")

    for edge in graph.edges:
        if not edge.kind:
            raise ValueError("Edge kind must be non-empty.")
        if edge.source not in node_ids:
            raise ValueError(f"Edge source {edge.source!r} missing from graph nodes.")
        if edge.target not in node_ids:
            raise ValueError(f"Edge target {edge.target!r} missing from graph nodes.")


def _coerce_attributes(obj: Any) -> dict[str, float]:
    attrs = getattr(obj, "attributes", None)
    if attrs is None:
        return {}

    if not isinstance(attrs, Mapping):
        raise TypeError(f"Expected mapping-like attributes, got {type(attrs).__name__}")

    out: dict[str, float] = {}
    for k, v in attrs.items():
        out[str(k)] = float(v)
    return out


def _adjacent_degree(graph: IdentityGraph, node_id: str) -> int:
    return sum(
        1
        for e in graph.edges
        if e.kind == "adjacent" and (e.source == node_id or e.target == node_id)
    )


def _transition_out_degree(graph: IdentityGraph, node_id: str) -> int:
    return sum(1 for e in graph.edges if e.kind == "transition" and e.source == node_id)


def _transition_in_degree(graph: IdentityGraph, node_id: str) -> int:
    return sum(1 for e in graph.edges if e.kind == "transition" and e.target == node_id)


def node_signature(graph: IdentityGraph, node_id: str) -> tuple[str, int, int, int]:
    node = graph.nodes[node_id]
    return (
        node.kind,
        _adjacent_degree(graph, node_id),
        _transition_out_degree(graph, node_id),
        _transition_in_degree(graph, node_id),
    )


def node_signature_histogram(graph: IdentityGraph) -> Counter[tuple[str, int, int, int]]:
    return Counter(node_signature(graph, node_id) for node_id in graph.nodes)


def adjacency_signature_counter(graph: IdentityGraph) -> Counter[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]]:
    counter: Counter[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]] = Counter()

    for e in graph.edges:
        if e.kind != "adjacent":
            continue
        s1 = node_signature(graph, e.source)
        s2 = node_signature(graph, e.target)
        pair = tuple(sorted((s1, s2)))
        counter[pair] += 1

    return counter


def transition_signature_counter(graph: IdentityGraph) -> Counter[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]]:
    counter: Counter[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]] = Counter()

    for e in graph.edges:
        if e.kind != "transition":
            continue
        s1 = node_signature(graph, e.source)
        s2 = node_signature(graph, e.target)
        counter[(s1, s2)] += 1

    return counter


def _counter_l1_distance(c1: Counter[Any], c2: Counter[Any]) -> int:
    keys = set(c1) | set(c2)
    return sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in keys)


def _counter_total_mass(c: Counter[Any]) -> int:
    return sum(c.values())


def extract_identity(
    state: Any | None = None,
    *,
    basins: Iterable[Any] | None = None,
    critical_points: Iterable[Any] | None = None,
    adjacency: Iterable[tuple[str, str]] | None = None,
) -> IdentityGraph:
    """
    Convert topology-like state into an IdentityGraph.

    You may supply either:
    - state with .basins, .critical_points, .adjacency
    - or explicit basins=..., critical_points=..., adjacency=...
    """
    if state is not None:
        if basins is None:
            basins = getattr(state, "basins", [])
        if critical_points is None:
            critical_points = getattr(state, "critical_points", [])
        if adjacency is None:
            adjacency = getattr(state, "adjacency", [])

    basins = [] if basins is None else list(basins)
    critical_points = [] if critical_points is None else list(critical_points)
    adjacency = [] if adjacency is None else list(adjacency)

    nodes: dict[str, Node] = {}

    for basin in basins:
        basin_id = str(getattr(basin, "id"))
        attrs = _coerce_attributes(basin)
        if "size" not in attrs and hasattr(basin, "size"):
            attrs["size"] = float(getattr(basin, "size"))
        nodes[basin_id] = Node(id=basin_id, kind="basin", attributes=attrs)

    for cp in critical_points:
        cp_id = str(getattr(cp, "id"))
        cp_kind = getattr(cp, "type", getattr(cp, "kind", None))
        if cp_kind is None:
            raise ValueError(f"Critical point {cp_id!r} is missing .type/.kind")
        nodes[cp_id] = Node(
            id=cp_id,
            kind=str(cp_kind),
            attributes=_coerce_attributes(cp),
        )

    edges: list[Edge] = []

    for pair in adjacency:
        if len(pair) != 2:
            raise ValueError(f"Adjacency entry must have length 2, got {pair!r}")
        a, b = str(pair[0]), str(pair[1])
        edges.append(Edge(a, b, kind="adjacent"))

    for cp in critical_points:
        cp_id = str(getattr(cp, "id"))
        for basin in getattr(cp, "connected_basins", []):
            basin_id = str(getattr(basin, "id"))
            edges.append(Edge(cp_id, basin_id, kind="transition"))

    graph = IdentityGraph(nodes=nodes, edges=tuple(edges))
    _validate_graph(graph)
    return graph


def identity_distance_report(
    g1: IdentityGraph,
    g2: IdentityGraph,
    *,
    w_nodes: float = 1.0,
    w_adjacent: float = 1.0,
    w_transition: float = 1.0,
) -> IdentityDistanceReport:
    _validate_graph(g1)
    _validate_graph(g2)

    node_hist_1 = node_signature_histogram(g1)
    node_hist_2 = node_signature_histogram(g2)
    node_cost = float(_counter_l1_distance(node_hist_1, node_hist_2))

    adj_1 = adjacency_signature_counter(g1)
    adj_2 = adjacency_signature_counter(g2)
    adjacency_cost = float(_counter_l1_distance(adj_1, adj_2))

    trans_1 = transition_signature_counter(g1)
    trans_2 = transition_signature_counter(g2)
    transition_cost = float(_counter_l1_distance(trans_1, trans_2))

    raw_distance = (
        w_nodes * node_cost
        + w_adjacent * adjacency_cost
        + w_transition * transition_cost
    )

    max_mass = (
        w_nodes * (_counter_total_mass(node_hist_1) + _counter_total_mass(node_hist_2))
        + w_adjacent * (_counter_total_mass(adj_1) + _counter_total_mass(adj_2))
        + w_transition * (_counter_total_mass(trans_1) + _counter_total_mass(trans_2))
    )

    normalized_distance = 0.0 if max_mass == 0 else float(raw_distance / max_mass)

    return IdentityDistanceReport(
        node_cost=node_cost,
        adjacency_cost=adjacency_cost,
        transition_cost=transition_cost,
        raw_distance=float(raw_distance),
        normalized_distance=normalized_distance,
    )


def identity_distance(
    g1: IdentityGraph,
    g2: IdentityGraph,
    *,
    w_nodes: float = 1.0,
    w_adjacent: float = 1.0,
    w_transition: float = 1.0,
    normalized: bool = True,
) -> float:
    report = identity_distance_report(
        g1,
        g2,
        w_nodes=w_nodes,
        w_adjacent=w_adjacent,
        w_transition=w_transition,
    )
    return report.normalized_distance if normalized else report.raw_distance


def identity_from_state(state: Any) -> IdentityGraph:
    return extract_identity(state)


def identity_distance_from_states(
    s1: Any,
    s2: Any,
    *,
    w_nodes: float = 1.0,
    w_adjacent: float = 1.0,
    w_transition: float = 1.0,
    normalized: bool = True,
) -> float:
    g1 = extract_identity(s1)
    g2 = extract_identity(s2)
    return identity_distance(
        g1,
        g2,
        w_nodes=w_nodes,
        w_adjacent=w_adjacent,
        w_transition=w_transition,
        normalized=normalized,
    )