from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

from pam.observatory.state import CellValue, ObservatoryState


@dataclass(slots=True)
class AdapterConfig:
    index_csv: Path
    dataset_total: int = 750


def _safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _normalize_map(values: dict[tuple[float, float], float]) -> dict[tuple[float, float], float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax <= vmin:
        return {k: 1.0 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def load_state_from_index(config: AdapterConfig) -> ObservatoryState:
    rows: list[dict[str, str]] = []
    with config.index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    r_values = sorted({_safe_float(row.get("r")) for row in rows})
    alpha_values = sorted({_safe_float(row.get("alpha") or row.get("α")) for row in rows})

    grouped: dict[tuple[float, float], list[dict[str, str]]] = {}
    for row in rows:
        r = _safe_float(row.get("r"))
        alpha = _safe_float(row.get("alpha") or row.get("α"))
        grouped.setdefault((r, alpha), []).append(row)

    raw_curvature: dict[tuple[float, float], float] = {}
    raw_pif: dict[tuple[float, float], float] = {}
    raw_hjoint: dict[tuple[float, float], float] = {}

    for key, group in grouped.items():
        # Mean across seeds / repeated rows
        raw_curvature[key] = sum(_safe_float(r.get("K_max")) for r in group) / len(group)
        raw_pif[key] = sum(_safe_float(r.get("piF_tail")) for r in group) / len(group)
        raw_hjoint[key] = sum(_safe_float(r.get("H_joint_mean") or r.get("H_joint")) for r in group) / len(group)

    norm_curvature = _normalize_map(raw_curvature)
    norm_pif = _normalize_map(raw_pif)
    norm_hjoint = _normalize_map(raw_hjoint)

    expected_per_cell = max(len({int(_safe_float(row.get("seed"))) for row in rows if row.get("seed") not in (None, "")}), 1)

    cells: dict[tuple[float, float], CellValue] = {}
    for r in r_values:
        for alpha in alpha_values:
            key = (r, alpha)
            present_rows = grouped.get(key, [])
            coverage = min(len(present_rows) / expected_per_cell, 1.0) if expected_per_cell else 0.0
            cells[key] = CellValue(
                coverage=coverage,
                curvature=norm_curvature.get(key, 0.0),
                piF_tail=norm_pif.get(key, 0.0),
                h_joint_mean=norm_hjoint.get(key, 0.0),
                present=bool(present_rows),
            )

    selected_r = r_values[0] if r_values else 0.0
    selected_alpha = alpha_values[0] if alpha_values else 0.0

    return ObservatoryState(
        dataset_progress=len(rows),
        dataset_total=config.dataset_total,
        selected_r=selected_r,
        selected_alpha=selected_alpha,
        color_mode="coverage",
        embedding_mode="r",
        probe_mode="fan",
        r_values=r_values,
        alpha_values=alpha_values,
        cells=cells,
    )
