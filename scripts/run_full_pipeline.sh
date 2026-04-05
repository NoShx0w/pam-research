#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
OBSERVATORY_ROOT="${OBSERVATORY_ROOT:-observatory}"
CORPUS="${CORPUS:-}"

GEOMETRY_OBSERVABLES="${GEOMETRY_OBSERVABLES:-piF_tail,H_joint_mean}"
GEOMETRY_RIDGE_EPS="${GEOMETRY_RIDGE_EPS:-1e-8}"
GEOMETRY_NEIGHBOR_MODE="${GEOMETRY_NEIGHBOR_MODE:-4}"
GEOMETRY_COST_MODE="${GEOMETRY_COST_MODE:-midpoint}"
GEOMETRY_COLOR_BY="${GEOMETRY_COLOR_BY:-fim_det}"

GEOMETRY_RUN_SINGLE_GEODESIC="${GEOMETRY_RUN_SINGLE_GEODESIC:-0}"
GEOMETRY_GEODESIC_START_R="${GEOMETRY_GEODESIC_START_R:-}"
GEOMETRY_GEODESIC_START_ALPHA="${GEOMETRY_GEODESIC_START_ALPHA:-}"
GEOMETRY_GEODESIC_END_R="${GEOMETRY_GEODESIC_END_R:-}"
GEOMETRY_GEODESIC_END_ALPHA="${GEOMETRY_GEODESIC_END_ALPHA:-}"

GEOMETRY_RUN_GEODESIC_FAN="${GEOMETRY_RUN_GEODESIC_FAN:-0}"
GEOMETRY_FAN_START_R="${GEOMETRY_FAN_START_R:-}"
GEOMETRY_FAN_START_ALPHA="${GEOMETRY_FAN_START_ALPHA:-}"
GEOMETRY_FAN_TARGET_R="${GEOMETRY_FAN_TARGET_R:-}"

PHASE_SEAM_THRESHOLD="${PHASE_SEAM_THRESHOLD:-10.0}"
PHASE_SEAM_SAMPLES="${PHASE_SEAM_SAMPLES:-100}"

OPERATORS_LAZARUS_THRESHOLD_QUANTILE="${OPERATORS_LAZARUS_THRESHOLD_QUANTILE:-0.85}"
OPERATORS_SCALED_N_PAIRS="${OPERATORS_SCALED_N_PAIRS:-100}"
OPERATORS_SCALED_SEED="${OPERATORS_SCALED_SEED:-42}"
OPERATORS_SCALED_MAX_DRAW="${OPERATORS_SCALED_MAX_DRAW:-25}"
OPERATORS_TRANSITION_WITHIN_K="${OPERATORS_TRANSITION_WITHIN_K:-2}"

TOPOLOGY_CRITICAL_TOP_K="${TOPOLOGY_CRITICAL_TOP_K:-5}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

OUTPUTS_ROOT="$OUTPUTS_ROOT" \
OBSERVATORY_ROOT="$OBSERVATORY_ROOT" \
CORPUS="$CORPUS" \
GEOMETRY_OBSERVABLES="$GEOMETRY_OBSERVABLES" \
GEOMETRY_RIDGE_EPS="$GEOMETRY_RIDGE_EPS" \
GEOMETRY_NEIGHBOR_MODE="$GEOMETRY_NEIGHBOR_MODE" \
GEOMETRY_COST_MODE="$GEOMETRY_COST_MODE" \
GEOMETRY_COLOR_BY="$GEOMETRY_COLOR_BY" \
GEOMETRY_RUN_SINGLE_GEODESIC="$GEOMETRY_RUN_SINGLE_GEODESIC" \
GEOMETRY_GEODESIC_START_R="$GEOMETRY_GEODESIC_START_R" \
GEOMETRY_GEODESIC_START_ALPHA="$GEOMETRY_GEODESIC_START_ALPHA" \
GEOMETRY_GEODESIC_END_R="$GEOMETRY_GEODESIC_END_R" \
GEOMETRY_GEODESIC_END_ALPHA="$GEOMETRY_GEODESIC_END_ALPHA" \
GEOMETRY_RUN_GEODESIC_FAN="$GEOMETRY_RUN_GEODESIC_FAN" \
GEOMETRY_FAN_START_R="$GEOMETRY_FAN_START_R" \
GEOMETRY_FAN_START_ALPHA="$GEOMETRY_FAN_START_ALPHA" \
GEOMETRY_FAN_TARGET_R="$GEOMETRY_FAN_TARGET_R" \
PHASE_SEAM_THRESHOLD="$PHASE_SEAM_THRESHOLD" \
PHASE_SEAM_SAMPLES="$PHASE_SEAM_SAMPLES" \
OPERATORS_LAZARUS_THRESHOLD_QUANTILE="$OPERATORS_LAZARUS_THRESHOLD_QUANTILE" \
OPERATORS_SCALED_N_PAIRS="$OPERATORS_SCALED_N_PAIRS" \
OPERATORS_SCALED_SEED="$OPERATORS_SCALED_SEED" \
OPERATORS_SCALED_MAX_DRAW="$OPERATORS_SCALED_MAX_DRAW" \
OPERATORS_TRANSITION_WITHIN_K="$OPERATORS_TRANSITION_WITHIN_K" \
TOPOLOGY_CRITICAL_TOP_K="$TOPOLOGY_CRITICAL_TOP_K" \
"$PYTHON_BIN" - <<'PY'
import os

from pam.pipeline.runner import run_pipeline


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float_or_none(name: str):
    raw = os.environ.get(name, "").strip()
    return None if raw == "" else float(raw)


corpus = os.environ.get("CORPUS") or None
geometry_observables = [
    x.strip() for x in os.environ["GEOMETRY_OBSERVABLES"].split(",") if x.strip()
]

run_pipeline(
    outputs_root=os.environ["OUTPUTS_ROOT"],
    observatory_root=os.environ["OBSERVATORY_ROOT"],
    corpus=corpus,
    geometry_observables=geometry_observables,
    geometry_ridge_eps=float(os.environ["GEOMETRY_RIDGE_EPS"]),
    geometry_neighbor_mode=os.environ["GEOMETRY_NEIGHBOR_MODE"],
    geometry_cost_mode=os.environ["GEOMETRY_COST_MODE"],
    geometry_color_by=os.environ["GEOMETRY_COLOR_BY"],
    geometry_run_single_geodesic=env_bool("GEOMETRY_RUN_SINGLE_GEODESIC"),
    geometry_geodesic_start_r=env_float_or_none("GEOMETRY_GEODESIC_START_R"),
    geometry_geodesic_start_alpha=env_float_or_none("GEOMETRY_GEODESIC_START_ALPHA"),
    geometry_geodesic_end_r=env_float_or_none("GEOMETRY_GEODESIC_END_R"),
    geometry_geodesic_end_alpha=env_float_or_none("GEOMETRY_GEODESIC_END_ALPHA"),
    geometry_run_geodesic_fan=env_bool("GEOMETRY_RUN_GEODESIC_FAN"),
    geometry_fan_start_r=env_float_or_none("GEOMETRY_FAN_START_R"),
    geometry_fan_start_alpha=env_float_or_none("GEOMETRY_FAN_START_ALPHA"),
    geometry_fan_target_r=env_float_or_none("GEOMETRY_FAN_TARGET_R"),
    phase_seam_threshold=float(os.environ["PHASE_SEAM_THRESHOLD"]),
    phase_seam_samples=int(os.environ["PHASE_SEAM_SAMPLES"]),
    operators_lazarus_threshold_quantile=float(os.environ["OPERATORS_LAZARUS_THRESHOLD_QUANTILE"]),
    operators_scaled_n_pairs=int(os.environ["OPERATORS_SCALED_N_PAIRS"]),
    operators_scaled_seed=int(os.environ["OPERATORS_SCALED_SEED"]),
    operators_scaled_max_draw=int(os.environ["OPERATORS_SCALED_MAX_DRAW"]),
    operators_transition_within_k=int(os.environ["OPERATORS_TRANSITION_WITHIN_K"]),
    topology_critical_top_k=int(os.environ["TOPOLOGY_CRITICAL_TOP_K"]),
)
PY

echo "Full PAM pipeline complete."