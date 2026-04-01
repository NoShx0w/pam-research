from __future__ import annotations

MODES: list[str] = [
    "Run",
    "Geometry",
    "Phase",
    "Topology",
    "Operators",
    "Identity",
]

DEFAULT_OVERLAY_BY_MODE: dict[str, str] = {
    "Run": "coverage",
    "Geometry": "curvature",
    "Phase": "signed_phase",
    "Topology": "criticality",
    "Operators": "lazarus",
    "Identity": "signed_local_obstruction",
}

OVERLAYS_BY_MODE: dict[str, list[str]] = {
    "Run": ["coverage", "completion", "missing"],
    "Geometry": ["curvature", "determinant", "condition_number"],
    "Phase": ["signed_phase", "distance_to_seam"],
    "Topology": ["criticality"],
    "Operators": ["lazarus", "transition_rate"],
    "Identity": [
        "identity_magnitude",
        "absolute_holonomy",
        "unsigned_local_obstruction",
        "signed_local_obstruction",
        "legacy_spin",
    ],
}
