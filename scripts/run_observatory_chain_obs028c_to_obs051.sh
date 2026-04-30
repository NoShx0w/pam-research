et -euo pipefail

# run_observatory_chain_obs028c_to_obs051.sh
#
# Canonical runner for the observatory chain from OBS-028c through OBS-051.
#
# Scope:
#   1. OBS-028c prerequisite studies
#   2. OBS-028c canonical seam bundle
#   3. Scale family substrate build
#   4. OBS-022 scene input preparation
#   5. OBS-050 structural coupling persistence
#   6. OBS-051 local divergence (all/core/near)
#
# Notes:
# - OBS-050 is treated as canonical / cross-corpus qualitatively stable
# - OBS-051 is treated as provisional / corpus-sensitive
# - This script is intentionally explicit rather than clever

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: no Python interpreter found." >&2
    exit 1
  fi
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PYTHONPATH:-src}"

OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
SCALE_ROOT="${SCALE_ROOT:-outputs/scales/100000}"

RUN_OBS028C_PREREQS="${RUN_OBS028C_PREREQS:-1}"
RUN_OBS028C="${RUN_OBS028C:-1}"
RUN_FAMILY_SUBSTRATE="${RUN_FAMILY_SUBSTRATE:-1}"
RUN_SCENE_PREP="${RUN_SCENE_PREP:-1}"
RUN_OBS050="${RUN_OBS050:-1}"
RUN_OBS051="${RUN_OBS051:-1}"

OBS051_MIN_INITIAL_DISTANCE="${OBS051_MIN_INITIAL_DISTANCE:-0.05}"

log() {
  echo
  echo "==> $1"
}

run_py() {
  echo "+ $PYTHON_BIN $*"
  "$PYTHON_BIN" "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file missing: $path" >&2
    exit 1
  fi
}

verify_nonempty() {
  local path="$1"
  require_file "$path"
  if [[ ! -s "$path" ]]; then
    echo "ERROR: file exists but is empty: $path" >&2
    exit 1
  fi
}

print_config() {
  log "runner configuration"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "PYTHONPATH=$PYTHONPATH"
  echo "OUTPUTS_ROOT=$OUTPUTS_ROOT"
  echo "SCALE_ROOT=$SCALE_ROOT"
  echo "RUN_OBS028C_PREREQS=$RUN_OBS028C_PREREQS"
  echo "RUN_OBS028C=$RUN_OBS028C"
  echo "RUN_FAMILY_SUBSTRATE=$RUN_FAMILY_SUBSTRATE"
  echo "RUN_SCENE_PREP=$RUN_SCENE_PREP"
  echo "RUN_OBS050=$RUN_OBS050"
  echo "RUN_OBS051=$RUN_OBS051"
  echo "OBS051_MIN_INITIAL_DISTANCE=$OBS051_MIN_INITIAL_DISTANCE"
}

run_obs028c_prereqs() {
  log "OBS-028c prerequisite chain"
  run_py experiments/studies/fim_response_complex_compatibility.py
  run_py experiments/studies/fim_response_operator_decomposition.py
  run_py experiments/studies/obs025_anisotropy_vs_relational_obstruction.py
  run_py experiments/studies/obs025_two_field_seam_panel.py
  run_py experiments/studies/obs026_family_two_field_occupancy.py
  run_py experiments/studies/obs028_embedding_comparison.py
  run_py experiments/studies/obs028b_diffusion_mode_analysis.py

  verify_nonempty "$OUTPUTS_ROOT/fim_response_complex_compatibility/response_complex_compatibility_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/fim_response_operator_decomposition/response_operator_decomposition_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs025_anisotropy_vs_relational_obstruction/obs025_anisotropy_vs_relational_obstruction_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs025_two_field_seam_panel/obs025_two_field_seam_panel_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs026_family_two_field_occupancy/obs026_family_two_field_occupancy_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs028_embedding_comparison/obs028_embedding_comparison_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs028b_diffusion_mode_analysis/obs028b_diffusion_mode_analysis_summary.txt"
}

run_obs028c() {
  log "OBS-028c canonical seam bundle"
  run_py experiments/studies/obs028c_export_canonical_seam_bundle.py

  verify_nonempty "$OUTPUTS_ROOT/obs028c_canonical_seam_bundle/seam_nodes.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs028c_canonical_seam_bundle/seam_family_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs028c_canonical_seam_bundle/seam_embedding_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs028c_canonical_seam_bundle/seam_metadata.txt"
}

run_family_substrate() {
  log "scale-conditioned family substrate"
  run_py experiments/toy/build_scale_family_substrate.py \
    --scale-root "$SCALE_ROOT" \
    --outputs-root "$OUTPUTS_ROOT"

  verify_nonempty "$SCALE_ROOT/family_substrate/path_nodes_for_family.csv"
  verify_nonempty "$SCALE_ROOT/family_substrate/path_node_diagnostics.csv"
  verify_nonempty "$SCALE_ROOT/family_substrate/path_diagnostics.csv"
  verify_nonempty "$SCALE_ROOT/family_substrate/path_family_assignments.csv"
  verify_nonempty "$SCALE_ROOT/family_substrate/path_family_summary.csv"
  verify_nonempty "$SCALE_ROOT/family_substrate/family_substrate_metadata.txt"
}

run_scene_prep() {
  log "OBS-022 scene input preparation"
  run_py experiments/toy/prepare_obs022_scene_inputs.py \
    --scale-root "$SCALE_ROOT" \
    --outputs-root "$OUTPUTS_ROOT" \
    --run-hotspot-occupancy \
    --run-canonical-seam-bundle \
    --run-pass2-annotations

  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_nodes.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_edges.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_seam.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_hubs.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_routes.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_glyphs.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs022_scene_bundle/scene_metadata.txt"

  verify_nonempty "$OUTPUTS_ROOT/obs024_family_hotspot_occupancy/family_hotspot_occupancy_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs024_family_hotspot_occupancy/obs024_family_hotspot_occupancy_summary.txt"
  verify_nonempty "$OUTPUTS_ROOT/obs028c_canonical_seam_bundle/seam_metadata.txt"
}

run_obs050() {
  log "OBS-050 structural coupling persistence"
  run_py experiments/studies/obs050_structural_coupling_persistence.py

  verify_nonempty "$OUTPUTS_ROOT/obs050_structural_coupling_persistence/structural_coupling_segments.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs050_structural_coupling_persistence/structural_coupling_seam_band_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs050_structural_coupling_persistence/structural_coupling_coupled_vs_decoupled_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs050_structural_coupling_persistence/obs050_structural_coupling_persistence_summary.txt"
}

run_obs051_pass() {
  local seam_band="$1"
  log "OBS-051 local divergence (${seam_band})"
  run_py experiments/studies/obs051_local_divergence_in_coupled_windows.py \
    --seam-band-filter "$seam_band" \
    --min-initial-distance "$OBS051_MIN_INITIAL_DISTANCE"

  verify_nonempty "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_window_divergence.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_outcome_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_family_summary.csv"
  verify_nonempty "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_local_divergence_summary.txt"

  cp \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_window_divergence.csv" \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_window_divergence_${seam_band}.csv"

  cp \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_outcome_summary.csv" \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_outcome_summary_${seam_band}.csv"

  cp \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_family_summary.csv" \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_family_summary_${seam_band}.csv"

  cp \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_local_divergence_summary.txt" \
    "$OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows/obs051_local_divergence_summary_${seam_band}.txt"
}

main() {
  print_config

  if [[ "$RUN_OBS028C_PREREQS" == "1" ]]; then
    run_obs028c_prereqs
  fi

  if [[ "$RUN_OBS028C" == "1" ]]; then
    run_obs028c
  fi

  if [[ "$RUN_FAMILY_SUBSTRATE" == "1" ]]; then
    run_family_substrate
  fi

  if [[ "$RUN_SCENE_PREP" == "1" ]]; then
    run_scene_prep
  fi

  if [[ "$RUN_OBS050" == "1" ]]; then
    run_obs050
  fi

  if [[ "$RUN_OBS051" == "1" ]]; then
    run_obs051_pass all
    run_obs051_pass core
    run_obs051_pass near
  fi

  log "observatory chain complete"
  echo "OBS-028c outputs: $OUTPUTS_ROOT/obs028c_canonical_seam_bundle"
  echo "Family substrate:   $SCALE_ROOT/family_substrate"
  echo "OBS-022 bundle:     $OUTPUTS_ROOT/obs022_scene_bundle"
  echo "OBS-024 outputs:    $OUTPUTS_ROOT/obs024_family_hotspot_occupancy"
  echo "OBS-050 outputs:    $OUTPUTS_ROOT/obs050_structural_coupling_persistence"
  echo "OBS-051 outputs:    $OUTPUTS_ROOT/obs051_local_divergence_in_coupled_windows"
}

main "$@"
