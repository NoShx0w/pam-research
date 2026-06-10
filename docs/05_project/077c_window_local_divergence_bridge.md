# OBS-077c — Window-Local Divergence Bridge

## Status

Implemented and run for:

text C Cp2 Cp3 

under the matched shared-14 MDS-pilot scale-space contract.

OBS-077c bridges:

text OBS-077b path/support occupancy   → OBS-051 coupled-window local divergence diagnostics 

This completes the first three-layer empirical chain:

text scale-space structural supports   → path-label cohorts   → local window divergence / boundedness 

---

## Purpose

OBS-077c asks:

text Do paths occupying OBS-076/OBS-077 scale-space supports and pinch cohorts show distinct OBS-051 window-local divergence / boundedness structure? 

The motivating question is whether the support-transition cohorts found in OBS-077b are merely label-enriched, or whether they also differ in local coupled-window dynamics.

OBS-077c therefore links structural supports to window-level diagnostics such as:

text mean_lambda_local mean_delta_d bounded_share seam_band path_family outcome_group 

---

## Scope Guardrail

The initial OBS-077c goal was to bridge toward OBS-075 transfer/asymmetry labels such as:

text coupled_outcome_group recovery_channel coupling_class 

However, current Cp2/Cp3 path-level artifacts do not expose coupled_outcome_group or recovery_channel as path/window-joinable columns.

The available OBS-051 window-level bridge fields are:

text path_id path_family outcome_group coupling_class seam_band mean_lambda_local mean_delta_d bounded_share 

OBS-077c v2 detects that:

text coupling_class = coupled 

for all joined OBS-051 window paths in C, Cp2, and Cp3.

Therefore:

text OBS-077c is not a categorical coupling_class enrichment test. 

The canonical interpretation is:

text OBS-077c is a window-local divergence / boundedness bridge over coupled OBS-051 windows. 

This is subset-specific: OBS-077c speaks only about paths that both occupy OBS-077b supports and have OBS-051 window-divergence rows.

---

## Inputs

For each corpus, OBS-077c consumes:

text OBS-077b path-weighted support membership:   obs077b_path_object_membership_path_weighted.csv  OBS-051 window divergence table:   obs051_window_divergence_all.csv  OBS-077a pinch candidates:   obs077a_pinch_point_candidates.csv 

Canonical output directories:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs077c_window_coupling_bridge_shared14_mds_pilot_v2/  outputs/corpora/Cp2/campaigns/full_v2/pipeline/   obs077c_window_coupling_bridge_shared14_mds_pilot_v2/  outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs077c_window_coupling_bridge_shared14_mds_pilot_v2/ 

---

## Join Contract

The bridge join is:

text OBS-077b support membership   path_id  OBS-051 coupled-window divergence table   path_id 

The membership table is path-weighted:

text one path contributes at most once per object × scale support 

The joined OBS-051 rows remain window-weighted for numeric diagnostics.

OBS-077c reports both:

text n_paths n_windows 

for every support/cohort summary.

---

## Join Audit

### C

text membership_paths        = 99,940 window_paths            = 10,295 path_id_overlap         = 10,295 support_membership_rows = 3,261,752 window_rows             = 10,739 support_joined_rows     = 492,116 

### Cp2

text membership_paths        = 99,980 window_paths            = 6,472 path_id_overlap         = 6,472 support_membership_rows = 4,117,630 window_rows             = 6,527 support_joined_rows     = 332,263 

### Cp3

text membership_paths        = 99,928 window_paths            = 6,108 path_id_overlap         = 6,108 support_membership_rows = 4,033,211 window_rows             = 6,159 support_joined_rows     = 293,364 

Interpretation:

text OBS-077c is a valid bridge, but subset-specific. It applies to the OBS-051 coupled-window subset, not all 100,000 paths. 

---

## Label Degeneracy

For all three corpora:

text coupling_class:   n_values = 1   top_value = coupled   top_path_share = 1.0   top_window_share = 1.0 

Thus coupling_class is degenerate and excluded from primary enrichment interpretation.

Non-degenerate categorical fields:

text seam_band path_family outcome_group 

---

## Numeric Fields

Primary numeric diagnostics:

text mean_lambda_local:   local divergence / separation-rate proxy  mean_delta_d:   local change in distance over coupled windows  bounded_share:   share of windows remaining bounded under the local divergence criterion 

OBS-077c v2 also computes:

text mean_lambda_local_z mean_delta_d_z bounded_share_z divergence_z_sum 

where:

text divergence_z_sum =   mean_lambda_local_z   + mean_delta_d_z   - bounded_share_z 

Higher divergence_z_sum means:

text higher local divergence higher distance growth lower boundedness 

relative to the corpus-specific OBS-051 window baseline.

---

## Global Numeric Baselines

### C

text mean_lambda_local = -0.2658 mean_delta_d      =  1.5295 bounded_share     =  0.2234 

### Cp2

text mean_lambda_local = 0.4493 mean_delta_d      = 1.9288 bounded_share     = 0.0388 

### Cp3

text mean_lambda_local = 0.4329 mean_delta_d      = 2.0104 bounded_share     = 0.0373 

These baselines are corpus-specific. Cross-corpus comparisons should therefore emphasize relative cohort contrast within each corpus, not raw metric magnitude alone.

---

# Results

## C — Bounded Recovery Recruitment

### Top OBS-077a candidate

text object          = density_core transition      = 5 → 6 dominant_family = intrinsic_dimension 

OBS-077b found that the entered cohort was enriched for:

text recovering stable_seam_corridor 

and depleted for:

text nonrecovering off_seam_reorganizing 

OBS-077c v2 shows that this same entered cohort is low-divergence and comparatively bounded:

text C density_core 5→6 entered:   mean_lambda_local = -0.2925   mean_lambda_z     = -0.024   mean_delta_d      =  0.8896   mean_delta_d_z    = -0.548   bounded_share     =  0.3172   bounded_share_z   = +0.257   divergence_z_sum  = -0.829 

The corresponding exited cohort is much more locally divergent and less bounded:

text C density_core 5→6 exited:   mean_lambda_local = -0.0376   mean_lambda_z     = +0.206   mean_delta_d      =  1.9975   mean_delta_d_z    = +0.401   bounded_share     =  0.1090   bounded_share_z   = -0.313   divergence_z_sum  = +0.920 

Canonical C read:

text C recruits recovery/stability into density_core through a low-divergence, more bounded entrant cohort, while shedding a higher-divergence, less-bounded population. 

Compact:

text C = bounded recovery recruitment 

---

## Cp2 — High-Divergence Recovery Sorting

### Top OBS-077a candidate

text object          = response_ridge transition      = 4 → 5 dominant_family = shape 

OBS-077b found that the entered cohort was enriched for:

text recovering reorganization_heavy seam_contact 

OBS-077c v2 shows that this entered cohort is locally high-divergence relative to the Cp2 baseline:

text Cp2 response_ridge 4→5 entered:   mean_lambda_local = 0.5416   mean_lambda_z     = +0.239   mean_delta_d      = 2.1028   mean_delta_d_z    = +0.211   bounded_share     = 0.0355   bounded_share_z   = -0.029   divergence_z_sum  = +0.480 

The exited cohort is less favorable under the divergence composite:

text Cp2 response_ridge 4→5 exited:   divergence_z_sum = -0.330 

Canonical Cp2 read:

text Cp2 recruits recovery-compatible paths through a high-divergence response-ridge entry event. 

This means recovery in Cp2 is not calm or low-change. It is achieved through a locally divergent reorganization channel.

Compact:

text Cp2 = high-divergence recovery sorting 

---

## Cp3 — Earlier Divergence, Later Nonrecovering Settlement

### Top OBS-077a candidate

text object          = energy_ridge transition      = 6 → 7 dominant_family = intrinsic_dimension 

OBS-077b found that the late energy_ridge transition recruits or preserves paths enriched for:

text nonrecovering off_seam_reorganizing seam_distant 

and depleted for:

text stable_seam_corridor recovering 

OBS-077c v2 shows that the entered cohort of this top late energy transition is not locally divergence-maximal:

text Cp3 energy_ridge 6→7 entered:   mean_lambda_local = 0.3798   mean_lambda_z     = -0.119   mean_delta_d      = 1.9307   mean_delta_d_z    = -0.097   bounded_share     = 0.0503   bounded_share_z   = +0.098   divergence_z_sum  = -0.314 

The persisted cohort is more numerically relevant, but still not the strongest Cp3 divergence event:

text Cp3 energy_ridge 6→7 persisted:   divergence_z_sum = +0.193 

The strongest Cp3 local-divergence cohorts occur earlier or in other objects:

text Cp3 coupling_negative 4→5 entered:   divergence_z_sum = +1.437  Cp3 seam_proxy 2→3 entered:   divergence_z_sum = +1.425  Cp3 energy_ridge 4→5 exited:   divergence_z_sum = +0.521  Cp3 response_ridge 4→5 entered:   divergence_z_sum = +0.409  Cp3 frobenius_ridge 4→5 entered:   divergence_z_sum = +0.409 

Canonical Cp3 read:

text Cp3 separates local divergence from late settlement. Its strongest divergence occurs earlier or in other structural supports, while the late energy-ridge transition settles nonrecovering/off-seam paths. 

Compact:

text Cp3 = earlier divergence, later nonrecovering settlement 

---

# Three-Way Comparison

OBS-077c establishes the following three-way mechanistic signature:

text C:   bounded recovery recruitment  Cp2:   high-divergence recovery sorting  Cp3:   earlier divergence, later nonrecovering settlement 

Expanded:

text C recruits recovery/stability into its density core through a relatively bounded, low-divergence entrant cohort.  Cp2 recruits recovery-compatible paths through a high-divergence response-ridge entry event.  Cp3 does not align late energy-ridge settlement with maximal local divergence; instead, high-divergence events occur earlier or in other supports, while the late energy transition settles nonrecovering/off-seam structure. 

This is the first clean three-way bridge from:

text scale-space support structure   → path-label population   → local coupled-window dynamics 

---

## Relation to OBS-075

OBS-075 showed Cp2/Cp3 transfer/asymmetry effects under low-complexity controls, especially around path/outcome structure.

OBS-077c does not directly retest OBS-075 labels such as coupled_outcome_group, because those labels are not currently path/window-joinable in the discovered artifacts.

However, OBS-077c provides a plausible geometric/dynamical substrate for the transfer asymmetry:

text C and Cp2 both recruit recovery-compatible populations, but through different local divergence regimes.  C:   recovery/stability recruitment is bounded.  Cp2:   recovery sorting is high-divergence and response-ridge mediated.  Cp3:   divergence and settlement are separated; late energy settlement is   nonrecovering/off-seam rather than recovery-compatible. 

This suggests that Cp2/Cp3 asymmetry may arise not simply from one corpus being “more divergent,” but from a different placement of divergence relative to recovery and settlement events.

Guardrail:

text This is a substrate explanation, not a causal proof of transfer asymmetry. A direct OBS-075 ↔ OBS-077 bridge requires path/window-joinable OBS-075 target labels. 

---

## Output Artifacts

Each per-corpus OBS-077c v2 run writes:

text obs077c_input_manifest.csv obs077c_join_audit.csv obs077c_label_degeneracy_audit.csv obs077c_global_numeric_baseline.csv obs077c_support_window_coupling_summary.csv obs077c_support_numeric_contrast.csv obs077c_support_coupling_class_enrichment.csv obs077c_support_seam_band_enrichment.csv obs077c_support_path_family_enrichment.csv obs077c_support_outcome_group_enrichment.csv obs077c_pinch_cohort_window_coupling_summary.csv obs077c_pinch_cohort_numeric_contrast.csv obs077c_top_pinch_numeric_contrast.csv obs077c_pinch_cohort_coupling_class_enrichment.csv obs077c_pinch_cohort_seam_band_enrichment.csv obs077c_pinch_cohort_path_family_enrichment.csv obs077c_pinch_cohort_outcome_group_enrichment.csv obs077c_report.md 

Recommended comparison artifact:

text outputs/comparisons/obs077c_C_Cp2_Cp3_window_coupling_bridge_shared14_mds_pilot_v2/   obs077c_C_Cp2_Cp3_window_divergence_bridge_summary.md 

---

## Guardrails

OBS-077c does not establish:

text causal transfer mechanism generated-text semantics formal attractors topological defects direct coupled_outcome_group localization 

OBS-077c does establish:

text a subset-specific bridge from scale-space support transitions to path-label cohorts and OBS-051 local divergence/boundedness diagnostics. 

---

## Canonical Summary

text OBS-077c establishes that the C/Cp2/Cp3 distinction is visible not only in scale-space support geometry and path-label occupancy, but also in how those support transitions relate to local coupled-window divergence.  C:   bounded recovery recruitment  Cp2:   high-divergence recovery sorting  Cp3:   earlier divergence, later nonrecovering settlement 

---

END OBS-077c
