# OBS-077b — Path-Label Projection onto Scale-Space Supports

## Status

Implemented and run for:

text C Cp2 Cp3 

under the matched shared-14 MDS-pilot scale-space contract.

OBS-077b extends the OBS-076/077a scale-space branch by projecting instrument-level path labels onto structural supports and pinch-point transitions.

This is the first layer where scale-space geometry is connected back to path-level semantics.

---

## Purpose

OBS-077b asks:

text Do scale-space structural supports and pinch-point candidates correspond to known path-level instrument labels? 

The analysis does not use generated text. It uses existing path-level labels derived from the structural coupling pipeline.

The primary goal is to bridge:

text OBS-076c:   structural-object factorization across diffusion scale  OBS-077a:   candidate support reorganization / pinch-point events  OBS-077b:   path-label populations occupying those supports and transitions 

The resulting chain is:

text field factor   → scale transition   → path-label cohort 

---

## Inputs

OBS-077b uses three main artifact families per corpus.

### Path-node spine

text scales/100000/family_substrate/path_nodes_for_family.csv 

Required columns:

text path_id step node_id r alpha mds1 mds2 

This file connects paths to manifold nodes.

### Path-label table

text obs050_structural_coupling_persistence/structural_coupling_path_summary.csv 

Primary categorical labels:

text path_family outcome_group seam_class 

Primary numeric path descriptors include:

text n_escalation_windows n_compression n_graze n_dissipation near_fraction mid_fraction far_fraction mean_criticality max_criticality mean_unsigned_obstruction max_unsigned_obstruction mean_absolute_holonomy mean_angle_jump_deg max_angle_jump_deg n_sector_changes core_windows near_windows far_windows 

### Scale-space object memberships

text obs076c_structural_object_persistence_shared14_mds_pilot_v2/   obs076c_object_membership_by_scale.csv 

Tracked structural objects include:

text energy_ridge density_core seam_proxy phase_band_positive phase_band_negative lazarus_concentration response_ridge frobenius_ridge coupling_positive coupling_negative 

### Pinch-point candidates

text obs077a_pinch_point_geometry_shared14_mds_pilot_v2/   obs077a_pinch_point_candidates.csv 

This provides candidate scale-local transitions with:

text object scale_index_from scale_index_to dominant_family dominant_reason support_score overlap_score shape_score id_score pinch_score_total 

---

## Join Contract

OBS-077b performs two joins.

First:

text path_nodes_for_family.csv   ⨝ structural_coupling_path_summary.csv   on path_id 

Second:

text path-node rows   ⨝ obs076c_object_membership_by_scale.csv   on node_id / id after PAM node-id normalization 

The normalization maps integer node IDs to canonical OBS node IDs:

text 0  → node_0000 1  → node_0001 10 → node_0010 

This was necessary because path_nodes_for_family.csv stores node IDs as integers, while OBS-076c stores IDs in canonical node_XXXX form.

---

## Join Audit

The final OBS-077b joins are valid.

### C

text path_id overlap:     100000 / 100000 node_id overlap:     73 step-weighted rows:  8,741,882 path-weighted rows:  3,261,752 

### Cp2

text path_id overlap:     100000 / 100000 node_id overlap:     72 step-weighted rows:  9,887,528 path-weighted rows:  4,117,630 

### Cp3

text path_id overlap:     100000 / 100000 node_id overlap:     73 step-weighted rows:  10,046,707 path-weighted rows:  4,033,211 

The node overlap is below 75 because the OBS-076c object membership table contains only nodes appearing in tracked top-quantile structural supports. This is expected.

---

## Weighting Modes

OBS-077b computes two occupancy modes.

### Path-weighted

text Each path_id contributes at most once per object/scale. 

This is the primary scientific read because it prevents long paths from dominating the counts.

### Step-weighted

text Every path-node-step occupancy contributes. 

This measures occupancy intensity but can be biased by path length.

Unless otherwise noted, the interpretation below uses path-weighted enrichment.

---

## Enrichment Definition

For a label value L inside object O at scale t:

text object_share(L | O, t)   = count of paths with label L touching O at t     / count of paths touching O at t 

Global share:

text global_share(L)   = count of paths with label L     / total path count 

Enrichment:

text enrichment(L | O, t)   = object_share(L | O, t) / global_share(L) 

Values greater than 1 indicate over-representation. Values below 1 indicate depletion.

---

## OBS-077b Global Structural-Support Results

### C

C’s strongest path-weighted enrichments are concentrated in the coalesced response/energy/Lazarus/coupling ridge.

Representative rows:

text lazarus_concentration, scale 4:   path_family = reorganization_heavy   enrichment = 2.278  coupling_positive / lazarus / response / frobenius / energy, scales 5–7:   path_family = reorganization_heavy   enrichment ≈ 2.202 

Interpretation:

text C’s coalesced coarse ridge is enriched for reorganization_heavy paths. 

This is consistent with OBS-076c, where C coalesces energy, response, Frobenius response, and Lazarus into a single dominant ridge.

---

### Cp2

Cp2’s strongest path-weighted enrichments are dominated by the relation between negative phase / energy and seam-distant paths.

Representative rows:

text phase_band_negative, scales 0–3:   seam_class = seam_distant   enrichment = 2.831  energy_ridge, scales 5 and 7:   seam_class = seam_distant   enrichment = 2.773  energy_ridge / phase_band_negative, scale 6:   seam_class = seam_distant   enrichment = 2.758 

Interpretation:

text Cp2’s energy and negative-phase supports are strongly enriched for seam-distant paths. 

This is consistent with OBS-076c, where Cp2 energy aligns with negative phase.

---

### Cp3

Cp3’s strongest path-weighted enrichments show a factor split.

Representative rows:

text lazarus_concentration, scales 5–6:   path_family = stable_seam_corridor   enrichment = 2.423  response_ridge / frobenius_ridge, scale 7:   path_family = stable_seam_corridor   enrichment = 2.423  phase_band_positive, scales 3–5:   seam_class = seam_distant   enrichment ≈ 2.116–2.156 

Interpretation:

text Cp3 response/Frobenius/Lazarus supports project onto stable seam-corridor paths.  Cp3 positive-phase supports project onto seam-distant paths. 

This is consistent with OBS-076c, where Cp3 response/Frobenius/Lazarus form a distinct factor, while energy aligns with positive phase and seam aligns with negative phase.

---

## Pinch-Candidate Path Projection

OBS-077b also projects path labels onto the top OBS-077a pinch candidates.

For each candidate transition, paths are divided into cohorts:

text before:   paths touching the object at scale_from  after:   paths touching the object at scale_to  entered:   paths newly entering the object at scale_to  exited:   paths leaving the object after scale_from  persisted:   paths present at both sides  union:   paths present at either side 

The entered, exited, and persisted cohorts are the most informative.

---

## C Top Pinch Candidate

Top candidate:

text object: density_core transition: scale 5 → 6 dominant_family: intrinsic_dimension 

The entered cohort is enriched for:

text recovering:   enrichment = 1.796  stable_seam_corridor:   enrichment = 2.665  seam_distant:   enrichment = 1.808 

and depleted for:

text nonrecovering:   enrichment = 0.557  settled_distant:   enrichment = 0.480  off_seam_reorganizing:   enrichment = 0.664 

Interpretation:

text C’s top transition recruits recovering and stable-corridor paths while depleting nonrecovering and off-seam paths. 

This gives C a recovery/stability-recruitment signature.

---

## Cp2 Top Pinch Candidate

Top candidate:

text object: response_ridge transition: scale 4 → 5 dominant_family: shape 

The entered cohort is sharply enriched for:

text recovering:   enrichment = 3.390  reorganization_heavy:   enrichment = 4.628  seam_contact:   share = 1.000 

and strongly depleted for:

text nonrecovering:   enrichment = 0.006  off_seam_reorganizing:   enrichment = 0.0135 

The exited cohort shows the opposite tendency:

text nonrecovering:   enrichment = 1.226  off_seam_reorganizing:   enrichment = 1.343  seam_grazing:   enrichment = 2.221  recovering:   enrichment = 0.457  stable_seam_corridor:   enrichment = 0.106 

Interpretation:

text Cp2’s response-ridge transition sorts path populations.  Entered paths are recovering, reorganization-heavy, seam-contact paths.  Exited paths are nonrecovering, off-seam, seam-grazing paths. 

This gives Cp2 a response-ridge recovery-sorting signature.

---

## Cp3 Top Pinch Candidate

Top candidate:

text object: energy_ridge transition: scale 6 → 7 dominant_family: intrinsic_dimension 

The entered cohort is enriched for:

text seam_distant:   enrichment = 2.689  off_seam_reorganizing:   enrichment = 1.569  nonrecovering:   enrichment = 1.328 

and depleted for:

text recovering:   enrichment = 0.445  stable_seam_corridor:   enrichment = 0.0719 

The persisted cohort is also off-seam and nonrecovering biased:

text off_seam_reorganizing:   enrichment = 1.577  nonrecovering:   enrichment = 1.165  stable_seam_corridor:   enrichment = 0.0276  recovering:   enrichment = 0.721 

The exited cohort is comparatively more neutral or opposite:

text settled_distant:   enrichment = 1.174  seam_grazing:   enrichment = 1.469  recovering:   enrichment ≈ 1.009  nonrecovering:   enrichment ≈ 0.995 

Interpretation:

text Cp3’s late energy-ridge transition recruits and preserves off-seam, seam-distant, nonrecovering paths while excluding stable/recovering paths. 

This gives Cp3 a nonrecovering off-seam energy-settlement signature.

---

## Comparative Result

OBS-077b establishes a strong contrast among C, Cp2, and Cp3.

text C:   top transition recruits recovering / stable-corridor paths  Cp2:   response-ridge transition sorts paths:     entered = recovering + reorganization-heavy + seam-contact     exited  = nonrecovering + off-seam + seam-grazing  Cp3:   late energy-ridge transition recruits/preserves nonrecovering +   off-seam + seam-distant paths and excludes stable/recovering paths 

Compact form:

text C   = recovery/stability recruitment Cp2 = response-ridge recovery sorting Cp3 = nonrecovering off-seam energy settlement 

This is the main OBS-077b finding.

---

## Relation to OBS-076

OBS-076c established final-scale structural factorization:

text C:   energy = response = Frobenius = Lazarus  Cp2:   energy ≈ negative phase   response ≈ Lazarus   energy disjoint from response/Frobenius  Cp3:   energy ≈ positive phase   response ≈ Lazarus   seam ≈ negative phase   energy disjoint from response/Frobenius/Lazarus 

OBS-077a then located candidate transition modes:

text C:   support-stable shape deformation of a coalesced ridge  Cp2:   response/Frobenius/Lazarus shape transition and energy/phase relocation  Cp3:   mid-late response/Frobenius support transition and late energy intrinsic-dimension transition 

OBS-077b adds path-label meaning:

text C:   coalesced transition recruits recovering/stable-corridor paths  Cp2:   response-ridge transition admits recovering seam-contact paths and sheds   nonrecovering off-seam/seam-grazing paths  Cp3:   energy-ridge transition recruits nonrecovering off-seam/seam-distant paths 

Together:

text scale-space factor   → transition mode   → path-label cohort 

---

## Relation to OBS-075

OBS-075 showed that Cp3→Cp2 directional transfer asymmetry survives several low-complexity controls, especially around coupled outcome structure.

OBS-077b does not directly retest OBS-075 transfer asymmetry.

However, it identifies a plausible geometric substrate for why Cp2 and Cp3 differ:

text Cp2:   response-ridge transition is organized around recovering seam-contact   reorganization.  Cp3:   late energy-ridge transition is organized around nonrecovering off-seam   settlement. 

This suggests that Cp2/Cp3 differences are expressed in dynamic response/phase/coupling fields and their path-label occupancy, rather than only in endpoint labels.

This remains an inference until a direct OBS-075 ↔ OBS-077 joined transfer test is implemented.

---

## Guardrails

OBS-077b should not be overread.

It does not establish:

text generated-text semantics syntax or grammar categories human-interpreted reasoning branches causal mechanisms formal attractors topological defects direct transfer-asymmetry proof 

Current evidence supports:

text instrument-level path-label enrichment over scale-space structural supports path-label sorting across candidate pinch transitions corpus-specific path-label occupancy signatures 

The phrase “basin” may be used only cautiously as an occupancy shorthand:

text path-label occupancy basin 

It should not yet be interpreted as a formal dynamical attractor.

---

## Output Artifacts

Per corpus:

text obs077b_path_label_projection_shared14_mds_pilot/   obs077b_input_manifest.csv   obs077b_join_audit.csv   obs077b_path_object_membership_step_weighted.csv   obs077b_path_object_membership_path_weighted.csv   obs077b_label_enrichment_by_object_scale.csv   obs077b_numeric_summary_by_object_scale.csv   obs077b_pinch_label_projection.csv   obs077b_report.md 

Implemented script:

text experiments/studies/obs077b_path_label_projection.py 

---

## Next Steps

Recommended next branch:

text OBS-077c — Direct transfer bridge between OBS-075 labels and OBS-077 scale-space supports 

Candidate questions:

text Do Cp3→Cp2 asymmetric transfer targets map onto specific OBS-077b path-label cohorts?  Does coupled_outcome_group concentrate in the Cp2 response-ridge recovery-sorting transition or the Cp3 energy-ridge off-seam settlement transition?  Can scale-space support occupancy improve or explain low-complexity transfer asymmetry? 

Additional future branch:

text OBS-077d — Text/provenance projection 

This requires stable response/path/text joins.

---

## Canonical Summary

OBS-077b projects operational path labels onto OBS-076/077 scale-space supports and pinch-point candidates.

The main result is that C, Cp2, and Cp3 do not merely differ by abstract geometric factorization. Their structural transitions recruit different path populations.

text C:   recovery/stability recruitment  Cp2:   response-ridge recovery sorting  Cp3:   nonrecovering off-seam energy settlement 

This completes the first empirical chain from scale-space field factors to support-transition events to path-label cohorts.

OBS-077b therefore establishes path-label projection as a meaningful bridge between the multiscale geometry layer and the earlier transfer/asymmetry branch.

