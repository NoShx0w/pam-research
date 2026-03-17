from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np


@dataclass(slots=True)
class TrajectorySeries:
    f_raw: list[float]
    h_joint: list[float]
    k_series: list[float]
    pif_smooth: list[float]
    source_path: Path | None = None
    source_kind: str = "placeholder"


@dataclass(slots=True)
class TrajectoryAdapterConfig:
    trajectories_dir: Path
    max_points: int = 64


def _downsample(values: np.ndarray, max_points: int) -> list[float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return []
    if values.size <= max_points:
        return values.tolist()

    idx = np.linspace(0, values.size - 1, max_points).astype(int)
    return values[idx].tolist()


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _candidate_paths(trajectories_dir: Path, r: float, alpha: float) -> list[Path]:
    """
    Flexible filename search so the adapter can survive naming changes.

    Matches examples such as:
    - traj_r0.15_a0.064_seed3.npz
    - trajectory_r0.15_alpha0.064_seed3.npz
    - r0.15_a0.064_seed3.npz

    Also supports a fallback wildcard on seed if the caller omits it.
    """
    r_tok = f"{r:.2f}"
    a_tok = f"{alpha:.3f}"

    patterns = [
        f"*r{r_tok}*a{a_tok}*seed*.npz",
        f"*r{r_tok}*alpha{a_tok}*seed*.npz",
        f"*{r_tok}*{a_tok}*seed*.npz",
    ]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(trajectories_dir.glob(pattern)))

    seen = set()
    unique: list[Path] = []
    for path in matches:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _extract_named_array(payload: Any, candidates: list[str]) -> np.ndarray | None:
    for name in candidates:
        if name in payload:
            arr = np.asarray(payload[name], dtype=float).reshape(-1)
            if arr.size:
                return arr
    return None


def _build_k_series(f_raw: np.ndarray, h_joint: np.ndarray) -> np.ndarray:
    """
    Placeholder local structure proxy until true K(t) is available.

    Uses the magnitude of the discrete derivative vector:
        sqrt((ΔF)^2 + (ΔH)^2)
    """
    if f_raw.size == 0 or h_joint.size == 0:
        return np.array([], dtype=float)

    n = min(f_raw.size, h_joint.size)
    f_raw = f_raw[:n]
    h_joint = h_joint[:n]

    df = np.diff(f_raw, prepend=f_raw[0])
    dh = np.diff(h_joint, prepend=h_joint[0])
    return np.sqrt(df**2 + dh**2)


def _placeholder_series(r: float, alpha: float, max_points: int) -> TrajectorySeries:
    """
    Deterministic fallback so the TUI never goes blank during integration.
    """
    n = max(16, min(max_points, 48))
    t = np.linspace(0, 1, n)

    seam_boost = 1.0 if abs(r - 0.15) < 1e-9 else 0.3
    alpha_phase = alpha * 10.0

    f_raw = 0.20 + 0.60 * np.exp(-3 * t) + 0.10 * np.sin(2 * math.pi * (t + alpha_phase))
    h_joint = 0.10 + 0.80 * (t ** (0.8 + seam_boost * 0.15))
    k_series = 0.15 + seam_boost * 0.35 * np.exp(-((t - 0.55) ** 2) / 0.03)
    pif_smooth = _moving_average(f_raw, window=5)

    return TrajectorySeries(
        f_raw=_normalize(f_raw).tolist(),
        h_joint=_normalize(h_joint).tolist(),
        k_series=_normalize(k_series).tolist(),
        pif_smooth=_normalize(pif_smooth).tolist(),
        source_path=None,
        source_kind="placeholder",
    )


def load_trajectory_series(
    config: TrajectoryAdapterConfig,
    *,
    r: float,
    alpha: float,
) -> TrajectorySeries:
    """
    Load trajectory series for the selected `(r, alpha)` cell.

    Current behavior:
    - searches for an `.npz` file by flexible filename patterns
    - extracts likely arrays for F_raw and H_joint
    - derives:
        - K(t): local discrete derivative magnitude proxy
        - πF_smooth(t): moving-average smoothing of F_raw
    - falls back to deterministic placeholder data if no file is found
    """
    candidates = _candidate_paths(config.trajectories_dir, r=r, alpha=alpha)
    if not candidates:
        return _placeholder_series(r=r, alpha=alpha, max_points=config.max_points)

    path = candidates[0]
    with np.load(path, allow_pickle=False) as payload:
        f_raw = _extract_named_array(payload, ["F_raw", "F", "freeze", "freeze_series"])
        h_joint = _extract_named_array(payload, ["H_joint", "H", "entropy", "entropy_series"])
        k_series = _extract_named_array(payload, ["K", "K_series", "curvature", "k_series"])
        pif_smooth = _extract_named_array(payload, ["piF_smooth", "pif_smooth", "piF", "piF_tail_series"])

    if f_raw is None:
        return _placeholder_series(r=r, alpha=alpha, max_points=config.max_points)

    if h_joint is None:
        h_joint = np.zeros_like(f_raw)

    if k_series is None:
        k_series = _build_k_series(f_raw, h_joint)

    if pif_smooth is None:
        pif_smooth = _moving_average(f_raw, window=5)

    return TrajectorySeries(
        f_raw=_downsample(_normalize(f_raw), config.max_points),
        h_joint=_downsample(_normalize(h_joint), config.max_points),
        k_series=_downsample(_normalize(k_series), config.max_points),
        pif_smooth=_downsample(_normalize(pif_smooth), config.max_points),
        source_path=path,
        source_kind="npz",
    )
