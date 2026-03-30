#!/usr/bin/env bash
set -euo pipefail

# ensure_default_scales.sh
#
# Ensures a canonical default set of scale outputs exists.
# Missing scales are generated; existing complete scales are left untouched.
#
# Default scales: 10,100,1000,10000
#
# Expected per-scale outputs:
#   <scale_root>/<N>/fim_ops_scaled/scaled_probe_paths.csv
#   <scale_root>/<N>/fim_transition_rate/transition_rate_summary.csv
#   <scale_root>/<N>/fim_horizon/horizon_predictive_summary_from_probes.csv
#   <scale_root>/<N>/fim_lazarus_temporal/lazarus_temporal_summary.csv
#
# This script assumes the canonical experiment scripts already exist.

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

SCALE_ROOT="${SCALE_ROOT:-outputs/scales}"
DEFAULT_SCALES="${DEFAULT_SCALES:-10,100,1000,10000}"
WITHIN_K="${WITHIN_K:-2}"

IFS=',' read -r -a SCALES <<< "$DEFAULT_SCALES"

require_file() {
  local path="$1"
  [[ -f "$path" ]]
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

  PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/fim_operator_probe_scale.py \
    --n-pairs "$n" \
    --outdir "$root/fim_ops_scaled"

  PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/fim_transition_rate.py \
    --paths-csv "$root/fim_ops_scaled/scaled_probe_paths.csv" \
    --outdir "$root/fim_transition_rate" \
    --within-k "$WITHIN_K"

  PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/studies/fim_horizon_from_probes.py \
    --input-csv "$root/fim_ops_scaled/scaled_probe_metrics.csv" \
    --outdir "$root/fim_horizon"

  PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/studies/fim_lazarus_temporal.py \
    --paths-csv "$root/fim_ops_scaled/scaled_probe_paths.csv" \
    --outdir "$root/fim_lazarus_temporal"
}

echo "==> checking canonical default scales: ${DEFAULT_SCALES}"
for n in "${SCALES[@]}"; do
  if scale_complete "$n"; then
    echo "   ok: scale $n already complete"
  else
    echo "   missing or incomplete: scale $n"
    run_scale_pipeline "$n"
  fi
done

echo "==> default scales ensured"
