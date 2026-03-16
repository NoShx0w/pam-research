i#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "==> PAM geometry pipeline starting"

echo "==> 1/10 Fisher metric"
"$PYTHON_BIN" experiments/fim.py \
  --index-csv outputs/index.csv \
  --outdir outputs/fim

echo "==> 2/10 Fisher distance graph"
"$PYTHON_BIN" experiments/fim_distance.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_distance

echo "==> 3/10 MDS embedding"
"$PYTHON_BIN" experiments/fim_mds.py \
  --distance-csv outputs/fim_distance/fisher_distance_matrix.csv \
  --nodes-csv outputs/fim_distance/fisher_nodes.csv \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_mds \
  --color-by fim_det

echo "==> 4/10 Scalar curvature"
"$PYTHON_BIN" experiments/fim_curvature_scalar.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_curvature

echo "==> 5/10 Single geodesic"
"$PYTHON_BIN" experiments/fim_geodesic.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 --a1 0.15 \
  --outdir outputs/fim_geodesic

echo "==> 6/10 Geodesic fan"
"$PYTHON_BIN" experiments/fim_geodesic_fan.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 \
  --outdir outputs/fim_geodesic

echo "==> 7/10 Phase seam extraction"
"$PYTHON_BIN" experiments/fim_phase_boundary.py \
  --curvature outputs/fim_curvature/curvature_surface.csv \
  --outdir outputs/fim_phase

echo "==> 8/10 Seam fit in MDS"
"$PYTHON_BIN" experiments/fim_phase_boundary_mds.py \
  --boundary-csv outputs/fim_phase/phase_boundary_points.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --outdir outputs/fim_phase

echo "==> 9/10 Distance to seam + critical points"
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

echo "==> 10/10 Final report"
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

echo "==> PAM geometry pipeline complete"
