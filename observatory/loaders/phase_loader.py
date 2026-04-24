from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PhaseData:
    phase_df: pd.DataFrame
    signed_phase_mtime: float | None
    seam_distance_mtime: float | None


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_phase_data(
    outputs_root: str | Path = "outputs",
    observatory_root: str | Path = "observatory",
) -> PhaseData:
    outputs_root = Path(outputs_root)
    observatory_root = Path(observatory_root)

    canonical_signed_phase_csv = observatory_root / "derived" / "phase" / "signed_phase.csv"
    canonical_seam_distance_csv = observatory_root / "derived" / "phase" / "distance_to_seam.csv"

    legacy_signed_phase_csv = outputs_root / "fim_phase" / "signed_phase_coords.csv"
    legacy_seam_distance_csv = outputs_root / "fim_phase" / "phase_distance_to_seam.csv"

    signed_phase_csv = (
        canonical_signed_phase_csv if canonical_signed_phase_csv.exists() else legacy_signed_phase_csv
    )
    seam_distance_csv = (
        canonical_seam_distance_csv if canonical_seam_distance_csv.exists() else legacy_seam_distance_csv
    )

    signed = pd.read_csv(signed_phase_csv).copy() if signed_phase_csv.exists() else pd.DataFrame()
    seam = pd.read_csv(seam_distance_csv).copy() if seam_distance_csv.exists() else pd.DataFrame()

    if not signed.empty:
        for col in ["r", "alpha", "signed_phase"]:
            if col in signed.columns:
                signed[col] = pd.to_numeric(signed[col], errors="coerce")

    if not seam.empty:
        for col in ["r", "alpha", "distance_to_seam"]:
            if col in seam.columns:
                seam[col] = pd.to_numeric(seam[col], errors="coerce")

    if signed.empty and seam.empty:
        merged = pd.DataFrame(columns=["r", "alpha", "signed_phase", "distance_to_seam"])
    elif signed.empty:
        merged = seam.copy()
    elif seam.empty:
        merged = signed.copy()
    else:
        merged = signed.merge(
            seam[["r", "alpha", "distance_to_seam"]],
            on=["r", "alpha"],
            how="outer",
        )

    return PhaseData(
        phase_df=merged.sort_values(["r", "alpha"]).reset_index(drop=True) if not merged.empty else merged,
        signed_phase_mtime=_safe_mtime(signed_phase_csv),
        seam_distance_mtime=_safe_mtime(seam_distance_csv),
    )