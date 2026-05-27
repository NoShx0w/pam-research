#!/usr/bin/env bash
set -euo pipefail

# ensure_default_scales.sh
#
# Ensures a canonical default set of scale outputs exists.
# Missing scales are generated; existing complete scales are left untouched.
#
# Default scales: 10,100,1000,10000,100000
#
# Expected per-scale outputs:
#   <scale_root>/<N>/fim_ops_scaled/scaled_probe_paths.csv
#   <scale_root>/<N>/fim_ops_scaled/scaled_probe_metrics.csv
#   <scale_root>/<N>/fim_transition_rate/transition_rate_summary.csv
#   <scale_root>/<N>/fim_horizon/horizon_predictive_summary_from_probes.csv
#   <scale_root>/<N>/fim_lazarus_temporal/lazarus_temporal_summary.csv
#
# Scope model
# -----------
# Legacy/root mode:
#   OUTPUTS_ROOT=outputs
#   SCALE_ROOT=outputs/scales
#
# Corpus-scoped campaign mode:
#   OUTPUTS_ROOT=outputs/corpora/<corpus>/campaigns/<campaign>/pipeline
#   SCALE_ROOT=$OUTPUTS_ROOT/scales
#
# Scale outputs remain scale-local, but all base geometry/phase/operator inputs
# are read explicitly from OUTPUTS_ROOT. This prevents scoped campaign scale
# runs from silently consuming root-level outputs/fim_* artifacts.

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: no Python interpreter found (.venv/bin/python, python3, or python)." >&2
    exit 1
  fi
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
SCALE_ROOT="${SCALE_ROOT:-$OUTPUTS_ROOT/scales}"
DEFAULT_SCALES="${DEFAULT_SCALES:-10,100,1000,10000,100000}"
WITHIN_K="${WITHIN_K:-2}"

# fim_operator_probe_scale.py controls
OPERATORS_SCALED_SEED="${OPERATORS_SCALED_SEED:-42}"
OPERATORS_SCALED_MAX_DRAW="${OPERATORS_SCALED_MAX_DRAW:-25}"

PYTHONPATH_VALUE="${PYTHONPATH:-./:./src:./experiments}"

# ---------------------------------------------------------------------------
# Scoped base inputs for experiments/fim_operator_probe_scale.py
# ---------------------------------------------------------------------------

MDS_CSV="${MDS_CSV:-$OUTPUTS_ROOT/fim_mds/mds_coords.csv}"
EDGES_CSV="${EDGES_CSV:-$OUTPUTS_ROOT/fim_distance/fisher_edges.csv}"
PHASE_CSV="${PHASE_CSV:-$OUTPUTS_ROOT/fim_phase/signed_phase_coords.csv}"
SEAM_CSV="${SEAM_CSV:-$OUTPUTS_ROOT/fim_phase/phase_distance_to_seam.csv}"
LAZARUS_CSV="${LAZARUS_CSV:-$OUTPUTS_ROOT/fim_lazarus/lazarus_scores.csv}"

IFS=',' read -r -a SCALES <<< "$DEFAULT_SCALES"

log() {
  echo
  echo "==> $1"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]]
}

verify_required_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file missing: $path" >&2
    exit 1
  fi
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required file exists but is empty: $path" >&2
    exit 1
  fi
}

print_config() {
  log "ensure_default_scales configuration"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "PYTHONPATH=$PYTHONPATH_VALUE"
  echo "OUTPUTS_ROOT=$OUTPUTS_ROOT"
  echo "SCALE_ROOT=$SCALE_ROOT"
  echo "DEFAULT_SCALES=$DEFAULT_SCALES"
  echo "WITHIN_K=$WITHIN_K"
  echo "OPERATORS_SCALED_SEED=$OPERATORS_SCALED_SEED"
  echo "OPERATORS_SCALED_MAX_DRAW=$OPERATORS_SCALED_MAX_DRAW"
  echo "MDS_CSV=$MDS_CSV"
  echo "EDGES_CSV=$EDGES_CSV"
  echo "PHASE_CSV=$PHASE_CSV"
  echo "SEAM_CSV=$SEAM_CSV"
  echo "LAZARUS_CSV=$LAZARUS_CSV"
}

verify_base_inputs() {
  log "verifying scoped base inputs"

  verify_required_file "$MDS_CSV"
  verify_required_file "$EDGES_CSV"
  verify_required_file "$PHASE_CSV"
  verify_required_file "$SEAM_CSV"
  verify_required_file "$LAZARUS_CSV"
}

scale_complete() {
  local n="$1"
  local root="$SCALE_ROOT/$n"

  require_file "$root/fim_ops_scaled/scaled_probe_paths.csv" &&
  require_file "$root/fim_ops_scaled/scaled_probe_metrics.csv" &&
  require_file "$root/fim_transition_rate/transition_rate_summary.csv" &&
  require_file "$root/fim_horizon/horizon_predictive_summary_from_probes.csv" &&
  require_file "$root/fim_lazarus_temporal/lazarus_temporal_summary.csv"
}

run_scale_pipeline() {
  local n="$1"
  local root="$SCALE_ROOT/$n"

  echo "==> ensuring scale $n at $root"
  mkdir -p "$root"

  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" experiments/fim_operator_probe_scale.py \
    --n-pairs "$n" \
    --seed "$OPERATORS_SCALED_SEED" \
    --max-draw "$OPERATORS_SCALED_MAX_DRAW" \
    --edges-csv "$EDGES_CSV" \
    --mds-csv "$MDS_CSV" \
    --signed-phase-csv "$PHASE_CSV" \
    --lazarus-csv "$LAZARUS_CSV" \
    --seam-csv "$SEAM_CSV" \
    --outdir "$root/fim_ops_scaled"

  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" experiments/fim_transition_rate.py \
    --paths-csv "$root/fim_ops_scaled/scaled_probe_paths.csv" \
    --outdir "$root/fim_transition_rate" \
    --within-k "$WITHIN_K"

  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" experiments/studies/fim_horizon_from_probes.py \
    --input-csv "$root/fim_ops_scaled/scaled_probe_metrics.csv" \
    --outdir "$root/fim_horizon"

  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" experiments/studies/fim_lazarus_temporal.py \
    --paths-csv "$root/fim_ops_scaled/scaled_probe_paths.csv" \
    --outdir "$root/fim_lazarus_temporal"
}

main() {
  print_config
  verify_base_inputs

  echo
  echo "==> checking canonical default scales: ${DEFAULT_SCALES}"

  for n in "${SCALES[@]}"; do
    if scale_complete "$n"; then
      echo "   ok: scale $n already complete"
    else
      echo "   missing or incomplete: scale $n"
      run_scale_pipeline "$n"
    fi
  done

  echo
  echo "==> default scales ensured"
}

main "$@"