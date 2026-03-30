#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "==> PAM geometry pipeline starting"

echo "==> 1/18 Fisher metric"
"$PYTHON_BIN" experiments/fim.py \
  --index-csv outputs/index.csv \
  --outdir outputs/fim

echo "==> 2/18 Fisher distance graph"
"$PYTHON_BIN" experiments/fim_distance.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_distance

echo "==> 3/18 MDS embedding"
"$PYTHON_BIN" experiments/fim_mds.py \
  --distance-csv outputs/fim_distance/fisher_distance_matrix.csv \
  --nodes-csv outputs/fim_distance/fisher_nodes.csv \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_mds \
  --color-by fim_det

echo "==> 4/18 Scalar curvature"
"$PYTHON_BIN" experiments/fim_curvature_scalar.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_curvature

echo "==> 5/18 Single geodesic"
"$PYTHON_BIN" experiments/fim_geodesic.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 --a1 0.15 \
  --outdir outputs/fim_geodesic

echo "==> 6/18 Geodesic fan"
"$PYTHON_BIN" experiments/fim_geodesic_fan.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 \
  --outdir outputs/fim_geodesic

echo "==> 7/18 Phase seam extraction"
"$PYTHON_BIN" experiments/fim_phase_boundary.py \
  --curvature outputs/fim_curvature/curvature_surface.csv \
  --outdir outputs/fim_phase

echo "==> 8/18 Seam fit in MDS"
"$PYTHON_BIN" experiments/fim_phase_boundary_mds.py \
  --boundary-csv outputs/fim_phase/phase_boundary_points.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --outdir outputs/fim_phase

echo "==> 9/18 Distance to seam + critical points"
"$PYTHON_BIN" experiments/fim_phase_distance.py \
  --distance-csv outputs/fim_distance/fisher_distance_matrix.csv \
  --nodes-csv outputs/fim_distance/fisher_nodes.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_phase

"$PYTHON_BIN" experiments/fim_critical_points.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --phase-distance-csv outputs/fim_phase/phase_distance_to_seam.csv \
  --outdir outputs/fim_critical

echo "==> 10/18 Signed phase field"
"$PYTHON_BIN" experiments/fim_signed_phase.py \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --phase-distance-csv outputs/fim_phase/phase_distance_to_seam.csv \
  --outdir outputs/fim_phase

echo "==> 11/18 Canonical manifold figure"
"$PYTHON_BIN" experiments/fim_canonical_figure.py \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --critical-csv outputs/fim_critical/critical_points.csv \
  --outdir outputs/fim_report

echo "==> 12/18 Operator S"
"$PYTHON_BIN" experiments/fim_operator_S.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops

echo "==> 13/18 Canonical GE probes"
"$PYTHON_BIN" experiments/fim_operator_probes.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops

echo "==> 14/18 Lazarus regime"
"$PYTHON_BIN" experiments/fim_lazarus.py \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_lazarus

echo "==> 15/18 Scaled operator probe experiment"
"$PYTHON_BIN" experiments/fim_operator_probe_scale.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --lazarus-csv outputs/fim_lazarus/lazarus_scores.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops_scaled \
  --n-pairs 100

echo "==> 16/18 Final report"
"$PYTHON_BIN" experiments/fim_mds_seam_critical_overlay.py \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --critical-csv outputs/fim_critical/critical_points.csv \
  --outdir outputs/fim_critical

"$PYTHON_BIN" experiments/fim_phase_report.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --critical-csv outputs/fim_critical/critical_points.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --phase-distance-csv outputs/fim_phase/phase_distance_to_seam.csv \
  --outdir outputs/fim_report

REFRESH_PUBLICATION_LAYER="${REFRESH_PUBLICATION_LAYER:-1}"

if [ "$REFRESH_PUBLICATION_LAYER" -eq 1 ]; then
  echo "==> [Stage 17/18] Ensure canonical default scales"
  PYTHONPATH=./:./src:./experiments bash scripts/ensure_default_scales.sh

  echo "==> [Stage 18/18] Refresh figure-facing data products"
  PYTHONPATH=./:./src:./experiments "$PYTHON_BIN" experiments/figures/run_refresh_data_for_figures.py
fi

echo "==> PAM geometry pipeline complete"
