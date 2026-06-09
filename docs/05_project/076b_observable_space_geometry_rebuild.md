# 076b — Observable-Space Geometry Rebuild

Status: observatory note  
Role: scale-space geometry-rebuild proxy  
Scope: OBS-076b, not canonical Fisher rebuild  
Provenance: builds on OBS-076a and design note 057

## Purpose

OBS-076b consumes the OBS-076a scale-space diffusion substrate and rebuilds observable-space geometry at each diffusion scale.

OBS-076a established:

text X(t) = exp(-t L_sym) X 

OBS-076b asks:

text When observable fields are diffused across scale, do geometry-level proxies persist, migrate, or collapse? 

The purpose is not to claim a new canonical geometry.

The purpose is to test whether the OBS-076a scale-space support migration has a geometry-level signature.

## Scope

OBS-076b is an observable-space geometry rebuild.

It does not recompute canonical Fisher geometry.

It does not replace the canonical PAM pipeline.

It does not claim canonical seam, attractor, gateway, or route-family persistence.

Instead, for each scale level t, it rebuilds a simple observable-space geometry:

text X(t)   → robust standardized Z(t)   → pairwise observable-space distance D_X(t)   → 2D classical-MDS embedding E(t)   → proxy geometry summaries   → proxy seam / phase / density / energy persistence 

All seam, density, and geometry objects in OBS-076b are proxy objects.

## Implementation

Script:

text experiments/studies/obs076b_rebuild_geometry_from_scale_space.py 

Inputs:

text obs076a_diffusion_bundle.npz optional node-context CSV 

Primary outputs:

text obs076b_input_manifest.csv obs076b_node_geometry_by_scale.csv obs076b_scale_geometry_summary.csv obs076b_topk_geometry_persistence.csv obs076b_seam_proxy_persistence.csv obs076b_phase_proxy_summary.csv obs076b_report.md 

The script reads the OBS-076a diffusion bundle and, for every scale, computes:

text observable_distance_corr_vs_base embedding_distance_corr_vs_base embedding_fit_corr mean pairwise observable distance mean kNN observable distance energy top-k persistence density top-k persistence phase proxy persistence seam proxy persistence 

The embedding used in this checkpoint is deterministic classical MDS.

## Proxy Definitions

### Observable-Space Distance

For each scale, the diffused observable matrix is robust-standardized columnwise:

text Z(t) = robust_scale(X(t)) 

Then pairwise Euclidean distance is computed in observable space:

text D_X(t) = pairwise_distance(Z(t)) 

This is not Fisher distance.

It is a scale-dependent observable-space distance used only for robustness diagnostics.

### Embedding Proxy

Each D_X(t) is embedded into two dimensions using classical MDS:

text D_X(t) → E(t) 

Embedding-distance correlation against the base scale is used as a proxy for geometry persistence.

### Energy Proxy

Node energy is computed as:

text energy_i(t) = ||Z_i(t)|| 

Top-k energy persistence asks whether high-energy nodes at base scale remain high-energy after diffusion.

### Density Proxy

Density is defined from observable-space kNN distance:

text density_i(t) = - mean_kNN_distance_i(t) 

Higher density means a node lies in a locally denser observable-space neighborhood.

Top-k density persistence asks whether dense regions remain dense across scale.

### Seam Proxy

When signed_phase is available, OBS-076b constructs a seam proxy from:

text low absolute signed_phase high local phase contrast 

This produces a scale-dependent seam-proxy score and a fixed-top-quantile seam-proxy set.

This is not the canonical seam.

It is a phase-derived observable-space seam proxy.

## Runs

OBS-076b was run on the matched shared-14 OBS-076a substrates for C and Cp3.

### C

Input bundle:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_shared_cp3cols/     obs076a_diffusion_bundle.npz 

Node context:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs069_inputs/     obs069_canonical_node_table_pilot_log_robust_scaled.csv 

Output:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/ 

### Cp3

Input bundle:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/     obs076a_diffusion_bundle.npz 

Node context:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076a_inputs/     obs076a_canonical_node_table_pilot_log_robust_scaled.csv 

Output:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/ 

Matched conditions:

text nodes = 75 observables = 14 scales = 8 embedding = classical MDS k_density = 7 seam_quantile = 0.85 graph-distance provenance inherited from OBS-076a: canonical_mds_pilot 

## C Result

C preserves observable-space geometry more strongly across diffusion scale.

Distance correlation against base:

text observable_distance_corr_vs_base: 1.000 → 0.435777  embedding_distance_corr_vs_base: 1.000 → 0.422749 

Energy support remains substantially persistent:

text energy top-5 final Jaccard vs base:  0.428571 energy top-10 final Jaccard vs base: 0.666667 energy top-20 final Jaccard vs base: 0.428571 

Density support does not remain tied to the original base support at the largest scale:

text density top-5 final Jaccard vs base:  0 density top-10 final Jaccard vs base: 0 density top-20 final Jaccard vs base: 0 

This indicates that C’s high-energy support is more persistent than its density support.

The seam proxy migrates gradually:

text seam_proxy_jaccard_vs_base: 1.000 0.846 0.846 0.600 0.412 0.200 0.091 0.000 

C phase remains coherent through low and intermediate scales, then weakens at large scale:

text phase_corr_vs_base: 1.000 → 0.274604  phase_sign_agreement_vs_base: 1.000 → 0.666667 

## Cp3 Result

Cp3 shows faster observable-space geometry reorganization.

Distance correlation against base:

text observable_distance_corr_vs_base: 1.000 → 0.246762  embedding_distance_corr_vs_base: 1.000 → 0.150134 

Energy support is much less persistent at top-5 and top-10:

text energy top-5 final Jaccard vs base:  0 energy top-10 final Jaccard vs base: 0.111111 energy top-20 final Jaccard vs base: 0.428571 

Density support also migrates strongly:

text density top-5 final Jaccard vs base:  0 density top-10 final Jaccard vs base: 0.0526316 density top-20 final Jaccard vs base: 0.025641 

Cp3 seam-proxy support migrates earlier than C.

At low/intermediate scale:

text t = 0.193: C seam Jaccard vs base   = 0.846154 Cp3 seam Jaccard vs base = 0.600000  t = 0.719: C seam Jaccard vs base   = 0.600000 Cp3 seam Jaccard vs base = 0.333333 

At final scale:

text C seam Jaccard vs base   = 0 Cp3 seam Jaccard vs base = 0.0434783 

Both seam proxies eventually migrate almost completely, but Cp3 begins this migration earlier.

Cp3 phase remains moderately coherent at large scale:

text phase_corr_vs_base: 1.000 → 0.450923  phase_sign_agreement_vs_base: 1.000 → 0.680000 

This is an important nuance. Cp3 geometry reorganizes faster than C, but the phase field does not simply collapse into noise.

## C vs Cp3 Comparison

The central OBS-076b comparison is:

text C:   observable-space geometry decays, but energy support remains relatively stable.  Cp3:   observable-space geometry decays faster, and energy/density/seam-proxy support migrates more strongly. 

Key final-scale comparison:

| metric | C | Cp3 |
|---|---:|---:|
| observable distance corr vs base | 0.435777 | 0.246762 |
| embedding distance corr vs base | 0.422749 | 0.150134 |
| energy top-5 Jaccard vs base | 0.428571 | 0 |
| energy top-10 Jaccard vs base | 0.666667 | 0.111111 |
| energy top-20 Jaccard vs base | 0.428571 | 0.428571 |
| density top-10 Jaccard vs base | 0 | 0.0526316 |
| seam-proxy final Jaccard vs base | 0 | 0.0434783 |
| final phase corr vs base | 0.274604 | 0.450923 |
| final phase sign agreement | 0.666667 | 0.680000 |

## Interpretation

OBS-076b confirms that the OBS-076a Cp3 support migration has an observable-space geometry signature.

Under matched shared-14 MDS-pilot scale-space rebuilds:

text C preserves observable-space geometry and energy support more strongly.  Cp3 reorganizes observable-space geometry, energy support, density support, and seam-proxy support more rapidly. 

However, Cp3 does not collapse into noise.

The phase field remains moderately coherent at large scale, and the OBS-076a analysis showed that Cp3 high-energy support migrates into structured coarse bands.

Thus the provisional read is:

text Cp3 is more scale-stratified than C.  Fine-scale Cp3 response-heavy peaks are less co-located with coarse-scale observable-space geometry than in C.  The coarse-scale Cp3 organization remains structured. 

## Relation to OBS-076a

OBS-076a found that Cp3 high-energy observable support migrates more strongly than C under matched shared-14 diffusion.

OBS-076b adds that this migration is not merely an energy-ranking artifact.

It appears in geometry-level proxies:

text pairwise observable-space distance embedding distance energy top-k persistence density top-k persistence seam-proxy persistence 

This strengthens the OBS-076a conclusion.

## Relation to OBS-075

OBS-075/075b/075c tested Cp3 directional transfer asymmetry and low-complexity survival.

OBS-076b does not retest transfer classifiers.

It asks a different question:

text Is the Cp3 observable substrate scale-stable or scale-stratified? 

The provisional bridge is:

text OBS-075:   Cp3 directional asymmetry survives selected low-complexity controls.  OBS-076a:   Cp3 high-energy observable support migrates more strongly than C across diffusion scale.  OBS-076b:   Cp3 support migration also appears in observable-space geometry proxies. 

Open question:

text Are OBS-075 survivor channels tied to fine-scale response spikes, coarse-scale phase/seam bands, or interactions between both? 

That question belongs to later OBS-076 scale-space transfer studies.

## Limitations

OBS-076b v1 rebuilds observable-space geometry, not canonical Fisher geometry.

The graph-distance contract inherited from OBS-076a is canonical_mds_pilot.

Seam structures are phase-derived proxies, not canonical seams.

Density structures are observable-space kNN proxies.

The embedding is classical MDS over observable-space distances, not the canonical PAM embedding.

The C and Cp3 comparison is matched at 14 columns, but upstream campaign provenance still differs.

No path-family, gateway, attractor, or classifier persistence claim is made here.

## Next Steps

### OBS-076b-v2

Attempt a closer canonical-geometry rebuild from X(t):

text X(t)   → Fisher-like metric / distance reconstruction   → canonical-style embedding   → phase / seam / Lazarus summaries 

This should remain explicitly marked as scale-space rebuild, not canonical replacement.

### OBS-076c

Test structural persistence of named PAM objects:

text seam persistence Lazarus concentration dynamics response ridge persistence gateway / route-family persistence where compatible 

### OBS-076d

Retest Cp3 directional asymmetry across scale:

text for each t:   build scale-dependent feature table   rerun transfer classifiers   measure asymmetry/specificity persistence 

Primary question:

text Does Cp3→Cp2 directional asymmetry survive at intermediate diffusion scales, or is it concentrated in fine-scale response spikes? 

## Current Status

OBS-076b is complete as an observable-space geometry-rebuild checkpoint.

Preferred C run:

text outputs/corpora/C/campaigns/canonical_legacy/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/ 

Preferred Cp3 run:

text outputs/corpora/Cp3/campaigns/full_v1/pipeline/   obs076b_geometry_rebuild_shared14_mds_pilot/ 

Preferred conclusion:

text C preserves observable-space geometry and energy support more strongly across scale. Cp3 reorganizes observable-space geometry and support more rapidly, but retains moderate phase coherence, indicating structured reorganization rather than collapse into noise. 
