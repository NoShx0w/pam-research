# 076c — Structural Object Persistence

Status: observatory note  
Role: scale-space structural-object persistence  
Scope: OBS-076c v2, observable-space structural proxies  
Provenance: builds on OBS-076a and OBS-076b

## Purpose

OBS-076c tracks named structural-object supports across diffusion scale.

OBS-076a established the observable diffusion substrate:

text X(t) = exp(-t L_sym) X 

OBS-076b rebuilt observable-space geometry from each X(t).

OBS-076c asks:

text When observable fields are diffused and observable-space geometry is rebuilt, which structural supports persist, migrate, overlap, or separate? 

The goal is to move beyond generic geometry drift and ask whether named structures co-move or factorize across scale.

## Scope

OBS-076c is a structural-object proxy study.

It does not claim canonical Fisher geometry persistence.

It does not claim canonical seam, gateway, attractor, or route-family persistence.

It tracks proxy supports over OBS-076b observable-space geometry under the matched shared-14 MDS-pilot scale-space contract.

The central objects are:

text energy_ridge density_core seam_proxy phase_band_positive phase_band_negative lazarus_concentration response_ridge frobenius_ridge coupling_positive coupling_negative 

Each object is represented as a fixed-size high-support set at each scale, usually the top quantile of a scale-dependent field.

The default quantile is:

text q = 0.85 

For N = 75, this yields approximately 12 nodes per object.

## v1 to v2 Patch

OBS-076c v1 consumed only the OBS-076b node geometry table.

That table contained dynamic OBS-076b fields such as:

text energy density_score seam_proxy_score is_seam_proxy 

but many context fields were inherited from the original node table:

text signed_phase lazarus_score response_strength frobenius_T signed_coupling 

This meant that some v1 objects were static reference supports rather than diffused structural supports.

OBS-076c v2 fixes this ambiguity.

When an OBS-076a diffusion bundle is provided, v2 injects diffused observable columns into the OBS-076b node table as:

text dyn__<observable_name> 

Examples:

text dyn__signed_phase dyn__lazarus_score dyn__response_strength dyn__frobenius_T dyn__signed_coupling 

Dynamic-eligible objects then use the diffused columns by default.

This makes OBS-076c v2 a genuine dynamic structural-support tracker.

## Implementation

Script:

text experiments/studies/obs076c_structural_object_persistence.py 

v2 command pattern:

bash PYTHONPATH=src .venv/bin/python experiments/studies/obs076c_structural_object_persistence.py \   --obs076b-node-geometry <obs076b_node_geometry_by_scale.csv> \   --obs076a-bundle <obs076a_diffusion_bundle.npz> \   --outdir <obs076c_output_dir> 

Primary outputs:

text obs076c_input_manifest.csv obs076c_object_manifest.csv obs076c_object_membership_by_scale.csv obs076c_object_persistence.csv obs076c_object_context_summary.csv obs076c_object_overlap_by_scale.csv obs076c_selected_object_overlap_by_scale.csv obs076c_object_centroid_drift.csv obs076c_report.md 

Comparison artifact:

text outputs/comparisons/obs076c_C_vs_Cp3_shared14_structural_objects_mds_pilot_v2/   obs076c_C_vs_Cp3_final_object_persistence_v2.csv   obs076c_C_vs_Cp3_final_object_overlaps_v2.csv   obs076c_C_vs_Cp3_structural_object_summary_v2.md 

## Object Definitions

### Dynamic Proxy Objects

These objects are already dynamic in OBS-076b:

text energy_ridge:   top quantile by energy  density_core:   top quantile by density_score  seam_proxy:   OBS-076b phase-derived seam-proxy flag 

### Dynamic Observable Objects

When --obs076a-bundle is provided, these objects use injected diffused fields:

text phase_band_positive:   top quantile by dyn__signed_phase  phase_band_negative:   bottom quantile by dyn__signed_phase  lazarus_concentration:   top quantile by dyn__lazarus_score  response_ridge:   top quantile by dyn__response_strength  frobenius_ridge:   top quantile by dyn__frobenius_T  coupling_positive:   top quantile by dyn__signed_coupling  coupling_negative:   bottom quantile by dyn__signed_coupling 

If a dynamic column is missing, the script records whether an object falls back to a static reference column or is unavailable.

In the C/Cp3 shared-14 v2 runs, all dynamic-eligible objects resolved to dynamic columns.

## Runs

### C

OBS-076c v2 output:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076c_structural_object_persistence_shared14_mds_pilot_v2/ 

Inputs:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/     obs076b_node_geometry_by_scale.csv  outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_shared_cp3cols/     obs076a_diffusion_bundle.npz 

### Cp3

OBS-076c v2 output:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076c_structural_object_persistence_shared14_mds_pilot_v2/ 

Inputs:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/     obs076b_node_geometry_by_scale.csv  outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/     obs076a_diffusion_bundle.npz 

Matched conditions:

text nodes = 75 scales = 8 shared observable columns = 14 object quantile = 0.85 object size ≈ 12 nodes graph-distance provenance = canonical_mds_pilot geometry type = observable-space geometry proxy 

## C Result

At final scale, C coalesces several dynamic supports into one dominant ridge.

Final persistence:

text energy_ridge          Jaccard vs base = 0.600 response_ridge        Jaccard vs base = 0.600 frobenius_ridge       Jaccard vs base = 0.600 coupling_positive     Jaccard vs base = 0.600 lazarus_concentration Jaccard vs base = 0.263 

The strongest result is in the final object-overlap table:

text energy_ridge ∩ response_ridge        = 12/12 energy_ridge ∩ frobenius_ridge       = 12/12 energy_ridge ∩ lazarus_concentration = 12/12 lazarus_concentration ∩ response     = 12/12 

So C’s final coarse-scale structure is:

text energy = response = frobenius = lazarus 

under the observable-space proxy contract.

Phase and seam supports are mostly separate from this ridge:

text energy_ridge ∩ phase_band_positive = 2/12 energy_ridge ∩ phase_band_negative = 0/12 energy_ridge ∩ seam_proxy          = 0/12 response_ridge ∩ seam_proxy        = 0/12 lazarus_concentration ∩ seam_proxy = 0/12 

The seam proxy aligns with the positive phase band:

text phase_band_positive ∩ seam_proxy = 8/12 phase_band_negative ∩ seam_proxy = 0/12 

Thus C exhibits a coarse-scale split:

text dominant ridge:   energy / response / frobenius / lazarus  seam-phase support:   positive phase / seam proxy 

## Cp3 Result

Cp3 behaves differently.

Final persistence shows stronger support migration than C:

text energy_ridge          Jaccard vs base = 0.200 response_ridge        Jaccard vs base = 0.0909 frobenius_ridge       Jaccard vs base = 0.0909 seam_proxy            Jaccard vs base = 0.0435 density_core          Jaccard vs base = 0.0435 

Unlike C, Cp3 does not coalesce into one dominant energy-response ridge.

Final overlaps:

text energy_ridge ∩ response_ridge        = 0/12 energy_ridge ∩ frobenius_ridge       = 0/12 energy_ridge ∩ lazarus_concentration = 0/12 

Instead, Cp3 energy aligns with the positive phase band:

text energy_ridge ∩ phase_band_positive = 8/12 energy_ridge ∩ phase_band_negative = 1/12 

The response, Frobenius, and Lazarus supports remain co-located with each other:

text lazarus_concentration ∩ response_ridge = 11/12 

The seam proxy aligns strongly with the negative phase band:

text phase_band_negative ∩ seam_proxy = 10/12 phase_band_positive ∩ seam_proxy = 0/12 response_ridge ∩ seam_proxy      = 0/12 

Thus Cp3 factorizes into distinct coarse-scale supports:

text energy support:   mostly positive phase  response / frobenius / lazarus support:   mutually co-located, but disjoint from energy  seam support:   mostly negative phase 

## C vs Cp3 Final Contrast

The final-scale structural contrast is sharp.

### Energy and Response

text C:   energy ∩ response = 12/12  Cp3:   energy ∩ response = 0/12 

### Energy and Lazarus

text C:   energy ∩ lazarus = 12/12  Cp3:   energy ∩ lazarus = 0/12 

### Energy and Positive Phase

text C:   energy ∩ phase_positive = 2/12  Cp3:   energy ∩ phase_positive = 8/12 

### Seam and Phase Polarity

text C:   seam ∩ phase_positive = 8/12   seam ∩ phase_negative = 0/12  Cp3:   seam ∩ phase_positive = 0/12   seam ∩ phase_negative = 10/12 

This gives the central OBS-076c finding:

text C coalesces. Cp3 factorizes. 

## Interpretation

OBS-076c v2 shows that C and Cp3 have different coarse-scale structural organizations under matched shared-14 MDS-pilot observable-space diffusion.

C coalesces diffused energy, response, Frobenius response, and Lazarus support into one dominant coarse-scale ridge.

Cp3 does not.

Cp3 separates into distinct coarse-scale factors:

text positive-phase energy support response/Frobenius/Lazarus support negative-phase seam support 

This means Cp3 is not merely less stable.

A better description is:

text Cp3 is scale-factorized. 

That is, its fine- and coarse-scale supports do not collapse into a single dominant ridge. Instead, diffusion exposes separable structural factors.

## Relation to OBS-076a and OBS-076b

OBS-076a showed that Cp3 high-energy observable support migrates more strongly than C across diffusion scale.

OBS-076b showed that this migration has an observable-space geometry signature.

OBS-076c v2 identifies the structural destination of that migration.

The sequence is:

text OBS-076a:   Cp3 high-energy support migrates.  OBS-076b:   Cp3 observable-space geometry reorganizes.  OBS-076c:   Cp3 coarse-scale supports factorize:     energy → positive phase     seam → negative phase     response / Lazarus remain co-located but separate from energy. 

For C, the sequence is different:

text OBS-076a:   C high-energy support is more persistent.  OBS-076b:   C observable-space geometry decays more slowly.  OBS-076c:   C coarse-scale energy, response, Frobenius, and Lazarus supports coalesce. 

## Relation to OBS-075

OBS-075/075b/075c tested Cp3 directional transfer asymmetry and low-complexity survival.

OBS-076c does not retest classifiers.

It identifies candidate scale-space structural factors that may underlie those transfer results.

The relevant hypothesis is:

text Cp3 directional asymmetry may depend on interactions between separable coarse supports:   positive-phase energy support   response/Frobenius/Lazarus support   negative-phase seam support 

This hypothesis must be tested explicitly in OBS-076d or later.

OBS-076c alone does not prove a mechanism for OBS-075.

## Limitations

OBS-076c tracks structural-object proxies, not canonical PAM objects.

The inherited graph-distance contract remains:

text canonical_mds_pilot 

This is weaker than canonical Fisher/geodesic distance.

The seam object is a phase-derived seam proxy from OBS-076b, not a canonical seam.

The geometry is observable-space geometry, not Fisher geometry.

The structural objects are quantile supports, not mechanistic entities by themselves.

Overlap indicates co-location, not causation.

The C/Cp3 comparison is matched on shared observable columns, node count, scale ladder, and script contract, but upstream campaign provenance still differs.

## Next Steps

### OBS-076d

Retest Cp3 directional asymmetry across diffusion scale.

For each scale:

text X(t)   → scale-dependent feature table   → low-complexity transfer tests   → asymmetry / specificity persistence 

Primary question:

text Does Cp3→Cp2 asymmetry survive when fine-scale response spikes are diffused away, or is it tied to one of the coarse factors revealed by OBS-076c? 

### OBS-076e

Repeat the OBS-076a/b/c pipeline across additional corpora or campaigns:

text C Cp Cp2 Cp3 Cp4 C0_instant 

Primary question:

text Is Cp3 scale-factorization unique, or part of a broader corpus-family pattern? 

### OBS-076b-v2 / OBS-076c-v3

Move closer to canonical geometry rebuild:

text X(t)   → Fisher-like metric reconstruction   → canonical-style distance / embedding   → seam and response-object persistence 

This should remain explicitly labeled as scale-space rebuild, not canonical replacement.

## Current Status

OBS-076c v2 is complete as a structural-object persistence checkpoint.

Preferred comparison artifact:

text outputs/comparisons/obs076c_C_vs_Cp3_shared14_structural_objects_mds_pilot_v2/   obs076c_C_vs_Cp3_structural_object_summary_v2.md 

Preferred conclusion:

text Under matched shared-14 MDS-pilot observable-space scale diffusion, C coalesces diffused energy, response, Frobenius response, and Lazarus supports into one dominant coarse-scale ridge.  Cp3 factorizes into distinct coarse-scale supports: energy aligns with positive phase, seam proxy aligns with negative phase, and response/Frobenius/Lazarus remain mutually co-located but separate from energy.  Cp3 is therefore not merely less stable; it is scale-factorized. 
