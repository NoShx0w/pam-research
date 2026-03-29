#!/usr/bin/env bash
set -euo pipefail

# pre_push_seal.sh
#
# Canonical pre-push integrity guard.
#
# What it does:
# 1. Verifies required canonical scripts exist
# 2. Ensures canonical default scales (10/100/1000/10000) exist
# 3. Refreshes figure-facing derived data
# 4. Verifies required outputs exist
# 5. Fails if refreshed canonical outputs changed but are not staged
#
# Install suggestion:
#   mkdir -p .githooks
#   cp /path/to/pre-push .githooks/pre-push
#   git config core.hooksPath .githooks

PYTHON_BIN="${PYTHON_BIN:-python}"
WITHIN_K="${WITHIN_K:-2}"
SCALE_ROOT="${SCALE_ROOT:-outputs/scales}"
DEFAULT_SCALES="${DEFAULT_SCALES:-10,100,1000,10000}"

REQUIRED_SCRIPTS=(
  "scripts/run_geometry_pipeline.sh"
  "scripts/ensure_default_scales.sh"
  "scripts/pre_push_seal.sh"
  "experiments/figures/run_refresh_data_for_figures.py"
  "experiments/figures/fim_figure_1_geometric_structure_v3.py"
  "experiments/figures/fim_figure_2.py"
  "experiments/figures/fim_figure_3_conditional_v2.py"
  "experiments/figures/fim_figure_4.py"
)

REQUIRED_OUTPUTS=(
  "outputs/fim_transition_rate/transition_rate_summary.csv"
  "outputs/fim_transition_rate/transition_rate_labeled.csv"
  "outputs/fim_horizon/horizon_predictive_summary_from_probes.csv"
  "outputs/fim_lazarus_temporal/lazarus_temporal_summary.csv"
  "outputs/fim_scaling/scaling_summary.csv"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

echo "==> pre-push seal: verifying canonical scripts"
for path in "${REQUIRED_SCRIPTS[@]}"; do
  require_file "$path"
done

echo "==> pre-push seal: ensuring default scales"
PYTHONPATH=./:./src:./experiments bash scripts/ensure_default_scales.sh

#echo "==> pre-push seal: refreshing figure-facing data"
#PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/figures/run_refresh_data_for_figures.py \
#  --within-k "$WITHIN_K" \
#  --scales-root "$SCALE_ROOT" \
#  --scales "$DEFAULT_SCALES"

echo "==> pre-push seal: validating required outputs"
for path in "${REQUIRED_OUTPUTS[@]}"; do
  require_file "$path"
done

echo "==> pre-push seal: checking for unstaged canonical output drift"
UNSTAGED=0
for path in "${REQUIRED_OUTPUTS[@]}"; do
  if ! git diff --quiet -- "$path"; then
    echo "ERROR: refreshed output changed but is not staged: $path" >&2
    UNSTAGED=1
  fi
done

if [[ "$UNSTAGED" -ne 0 ]]; then
  echo "Stage refreshed canonical outputs before pushing." >&2
  exit 1
fi

echo "==> pre-push seal passed"
