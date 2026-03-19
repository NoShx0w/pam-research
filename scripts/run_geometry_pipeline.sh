#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "==> PAM geometry pipeline starting"

echo "==> 1/16 Fisher metric"
"$PYTHON_BIN" experiments/fim.py \
  --index-csv outputs/index.csv \
  --outdir outputs/fim

echo "==> 2/16 Fisher distance graph"
"$PYTHON_BIN" experiments/fim_distance.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_distance

echo "==> 3/16 MDS embedding"
"$PYTHON_BIN" experiments/fim_mds.py \
  --distance-csv outputs/fim_distance/fisher_distance_matrix.csv \
  --nodes-csv outputs/fim_distance/fisher_nodes.csv \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_mds \
  --color-by fim_det

echo "==> 4/16 Scalar curvature"
"$PYTHON_BIN" experiments/fim_curvature_scalar.py \
  --fim-csv outputs/fim/fim_surface.csv \
  --outdir outputs/fim_curvature

echo "==> 5/16 Single geodesic"
"$PYTHON_BIN" experiments/fim_geodesic.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 --a1 0.15 \
  --outdir outputs/fim_geodesic

echo "==> 6/16 Geodesic fan"
"$PYTHON_BIN" experiments/fim_geodesic_fan.py \
  --nodes outputs/fim_distance/fisher_nodes.csv \
  --edges outputs/fim_distance/fisher_edges.csv \
  --coords outputs/fim_mds/mds_coords.csv \
  --r0 0.10 --a0 0.03 \
  --r1 0.25 \
  --outdir outputs/fim_geodesic

echo "==> 7/16 Phase seam extraction"
"$PYTHON_BIN" experiments/fim_phase_boundary.py \
  --curvature outputs/fim_curvature/curvature_surface.csv \
  --outdir outputs/fim_phase

echo "==> 8/16 Seam fit in MDS"
"$PYTHON_BIN" experiments/fim_phase_boundary_mds.py \
  --boundary-csv outputs/fim_phase/phase_boundary_points.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --outdir outputs/fim_phase

echo "==> 9/16 Distance to seam + critical points"
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

echo "==> 10/16 Signed phase field"
"$PYTHON_BIN" experiments/fim_signed_phase.py \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --phase-distance-csv outputs/fim_phase/phase_distance_to_seam.csv \
  --outdir outputs/fim_phase

echo "==> 11/16 Canonical manifold figure"
"$PYTHON_BIN" experiments/fim_canonical_figure.py \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --critical-csv outputs/fim_critical/critical_points.csv \
  --outdir outputs/fim_report

echo "==> 12/16 Operator S"
"$PYTHON_BIN" experiments/fim_operator_S.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops

echo "==> 13/16 Canonical GE probes"
"$PYTHON_BIN" experiments/fim_operator_probes.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops

echo "==> 14/16 Lazarus regime"
"$PYTHON_BIN" experiments/fim_lazarus.py \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_lazarus

echo "==> 15/16 Scaled operator probe experiment"
"$PYTHON_BIN" experiments/fim_operator_probe_scale.py \
  --edges-csv outputs/fim_distance/fisher_edges.csv \
  --mds-csv outputs/fim_mds/mds_coords.csv \
  --signed-phase-csv outputs/fim_phase/signed_phase_coords.csv \
  --curvature-csv outputs/fim_curvature/curvature_surface.csv \
  --lazarus-csv outputs/fim_lazarus/lazarus_scores.csv \
  --seam-csv outputs/fim_phase/phase_boundary_mds_backprojected.csv \
  --outdir outputs/fim_ops_scaled \
  --n-pairs 100

echo "==> 16/16 Final report"
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
