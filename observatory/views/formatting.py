from __future__ import annotations


def fmt_value(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
    except Exception:
        pass
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


OVERLAY_LABELS: dict[str, str] = {
    "coverage": "Coverage",
    "completion": "Completion",
    "missing": "Missing",
    "curvature": "Scalar Curvature",
    "determinant": "FIM Determinant",
    "condition_number": "FIM Condition #",
    "signed_phase": "Signed Phase",
    "distance_to_seam": "Distance to Seam",
    "criticality": "Criticality",
    "lazarus": "Lazarus",
    "identity_magnitude": "Identity Magnitude",
    "absolute_holonomy": "Absolute Holonomy",
    "unsigned_local_obstruction": "Unsigned Obstruction",
    "signed_local_obstruction": "Signed Obstruction",
    "legacy_spin": "Legacy Spin (cmp)",
    "mean_lambda_local": "Mean λ Local",
    "bounded_share": "Bounded Share",
    "recovering_landings": "Recovering Landings",
    "attractor_score": "Attractor Score",
}


MODE_LABELS: dict[str, str] = {
    "Run": "Run",
    "Geometry": "Geometry",
    "Phase": "Phase",
    "Topology": "Topology",
    "Operators": "Operators",
    "Identity": "Identity",
    "Transitions": "Transitions",
}


OVERLAY_META: dict[str, dict[str, str]] = {
    "coverage": {
        "kind": "unsigned",
        "encoding": "low→high occupancy",
        "meaning": "Run coverage across the sweep lattice.",
    },
    "curvature": {
        "kind": "unsigned",
        "encoding": "low→high magnitude",
        "meaning": "Scalar curvature on the manifold.",
    },
    "determinant": {
        "kind": "unsigned",
        "encoding": "low→high magnitude",
        "meaning": "Local Fisher metric determinant.",
    },
    "condition_number": {
        "kind": "unsigned",
        "encoding": "low→high magnitude",
        "meaning": "Local Fisher metric conditioning.",
    },
    "signed_phase": {
        "kind": "signed",
        "encoding": "blue↔red sign / magnitude",
        "meaning": "Signed phase coordinate across the seam structure.",
    },
    "distance_to_seam": {
        "kind": "unsigned",
        "encoding": "low→high seam distance",
        "meaning": "Distance from the phase boundary.",
    },
    "criticality": {
        "kind": "unsigned",
        "encoding": "low→high criticality",
        "meaning": "Topological / structural criticality signal.",
    },
    "lazarus": {
        "kind": "unsigned",
        "encoding": "low→high Lazarus score",
        "meaning": "Recovery / rebound-associated operator signal.",
    },
    "identity_magnitude": {
        "kind": "unsigned",
        "encoding": "low→high magnitude",
        "meaning": "Local strength of identity change.",
    },
    "absolute_holonomy": {
        "kind": "unsigned",
        "encoding": "low→high holonomy",
        "meaning": "Node-local absolute transport obstruction summary.",
    },
    "unsigned_local_obstruction": {
        "kind": "unsigned",
        "encoding": "low→high obstruction",
        "meaning": "Unsigned local obstruction derived from transport.",
    },
    "signed_local_obstruction": {
        "kind": "signed",
        "encoding": "blue↔red sign / magnitude",
        "meaning": "Signed local transport-derived obstruction.",
    },
    "legacy_spin": {
        "kind": "signed",
        "encoding": "blue↔red sign / magnitude",
        "meaning": "Legacy local comparison proxy, not primary.",
    },
    "mean_lambda_local": {
        "kind": "signed",
        "encoding": "blue↔red sign / magnitude",
        "meaning": "OBS-051 local divergence estimate projected onto node visitation.",
    },
    "bounded_share": {
        "kind": "unsigned",
        "encoding": "low→high bounded fraction",
        "meaning": "OBS-051 fraction of local comparisons that remained bounded.",
    },
    "recovering_landings": {
        "kind": "unsigned",
        "encoding": "low→high landing count",
        "meaning": "OBS-052 count of recovery-like landings accumulated at the node.",
    },
    "attractor_score": {
        "kind": "signed",
        "encoding": "blue↔red signed basin score",
        "meaning": "OBS-052 provisional composite basin score.",
    },
}


def overlay_label(name: str) -> str:
    return OVERLAY_LABELS.get(name, name)


def mode_label(name: str) -> str:
    return MODE_LABELS.get(name, name)


def overlay_meta(name: str) -> dict[str, str]:
    return OVERLAY_META.get(
        name,
        {
            "kind": "unknown",
            "encoding": "n/a",
            "meaning": "No overlay description available.",
        },
    )