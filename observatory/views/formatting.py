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
}


MODE_LABELS: dict[str, str] = {
    "Run": "Run",
    "Geometry": "Geometry",
    "Phase": "Phase",
    "Topology": "Topology",
    "Operators": "Operators",
    "Identity": "Identity",
}


def overlay_label(name: str) -> str:
    return OVERLAY_LABELS.get(name, name)


def mode_label(name: str) -> str:
    return MODE_LABELS.get(name, name)
