Core question:
```text
Which observable-field structures survive scale,
and which reorganize under diffusion?

OBS-076a does not yet rebuild Fisher geometry, seams, attractors, response operators, or route-family structures at each scale. It establishes the substrate required for OBS-076b.

Contract

Given an observable matrix:

X ∈ R^(N × D)

where rows are nodes and columns are observable fields, and a graph-distance matrix:

D_graph ∈ R^(N × N)

OBS-076a builds a self-tuning affinity graph:

W_ij = exp(-D_graph(i,j)^2 / (2 σ_i σ_j))

using local k-nearest neighborhoods.

The symmetric normalized graph Laplacian is:

L_sym = I - D^{-1/2} W D^{-1/2}

and observable diffusion is computed as:

X(t) = exp(-t L_sym) X

The scale ladder uses geometric spacing:

n_scales = 8
t_min = 0.1
t_max = 10.0

Default graph parameters:

k = 15
sigma_rank = 7

Guardrails

OBS-076a is an observable-field diffusion substrate.

It does not claim that scale-diffused observables define canonical PAM geometry.

It does not claim seam, attractor, path-family, or response-operator persistence directly.

It does not replace the canonical pipeline.

Graph-distance provenance matters. In this checkpoint, the graph distance is declared as:

canonical_mds_pilot

This is weaker than a canonical Fisher/geodesic graph and should be treated as a pilot contract.

Implementation

Script:

experiments/studies/obs076a_observable_scale_space_diffusion.py

Primary outputs:

obs076a_input_manifest.csv
obs076a_scale_ladder.csv
obs076a_graph_diagnostics.json
obs076a_laplacian_spectrum.csv
obs076a_diffusion_bundle.npz
obs076a_observable_drift_summary.csv
obs076a_topk_persistence_summary.csv
obs076a_scale_space_report.md

The script records graph-distance provenance, input artifacts, observable columns, graph diagnostics, Laplacian spectrum, diffused observable fields, drift summaries, and top-k persistence summaries.

C Baseline

The first OBS-076a baseline used the legacy C canonical-MDS pilot substrate.

Initial determinant/condition-inclusive runs were mechanically valid but dominated by extreme-scale determinant / condition channels. A cleaner no-det/cond observable bundle produced interpretable scale-space behavior.

Preferred C baseline:

outputs/corpora/C/campaigns/canonical_legacy/pipeline/
  obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/

The no-det/cond C run showed smooth monotone decay:

variance_retained_ratio:
0.890489 → 0.0752803
laplacian_energy_ratio_vs_base:
0.833023 → 0.00718529
flat_corr_with_base:
0.998996 → 0.549633

High-energy node support remained substantially persistent:

top-5 final Jaccard vs base:  0.428571
top-10 final Jaccard vs base: 0.666667
top-20 final Jaccard vs base: 0.379310

This established a connected, usable OBS-076a substrate for C.

Cp3 Substrate

Cp3 did not yet have OBS-069 prepared inputs, but it did have the response-operator node table:

outputs/corpora/Cp3/campaigns/full_v1/pipeline/
  fim_response_operator/response_operator_nodes.csv

OBS-076a pilot inputs were prepared from that table:

outputs/corpora/Cp3/campaigns/full_v1/pipeline/obs076a_inputs/
  obs076a_canonical_node_table_pilot_log_robust_scaled.csv
  obs076a_mds_distance_pilot.csv

The Cp3 run used the same declared graph-distance class:

canonical_mds_pilot

Cp3 OBS-076a output:

outputs/corpora/Cp3/campaigns/full_v1/pipeline/
  obs076a_scale_space_canonical_mds_pilot_log_robust_scaled_no_detcond/

Cp3 graph diagnostics were healthy:

nodes: 75
connected components: 1
undirected edges: 663
mean binary degree: 17.68
min binary degree: 15
max binary degree: 27
second Laplacian eigenvalue: 0.0400221

Cp3 also diffused smoothly, but its high-energy support was less persistent:

variance_retained_ratio:
0.864287 → 0.0410612
laplacian_energy_ratio_vs_base:
0.826824 → 0.00290375
flat_corr_with_base:
0.999142 → 0.399742

Final top-k persistence:

top-5 final Jaccard vs base:  0
top-10 final Jaccard vs base: 0.111111
top-20 final Jaccard vs base: 0.25

This indicates stronger high-energy support migration in Cp3 than in C under the same pilot graph contract.

Matched C vs Cp3 Comparison

To remove the observable-count confound, C was rerun using the same 14 observable columns available in the Cp3 substrate.

Comparison artifact:

outputs/comparisons/obs076a_C_vs_Cp3_shared14_mds_pilot/
  obs076a_C_vs_Cp3_shared14_summary.csv
  obs076a_C_vs_Cp3_shared14_summary.md

Matched conditions:

N = 75
n_scales = 8
k = 15
sigma_rank = 7
graph_distance_kind = canonical_mds_pilot
observable_count = 14

Matched summary:

case	topk	final variance retained	final Laplacian-energy ratio	final flat corr	final Jaccard vs base	final Jaccard vs previous
C_shared14	5	0.0713481	0.00711298	0.542892	0.666667	1.000000
C_shared14	10	0.0713481	0.00711298	0.542892	0.666667	1.000000
C_shared14	20	0.0713481	0.00711298	0.542892	0.379310	0.904762
Cp3_shared14	5	0.0410612	0.00290375	0.399742	0.000000	0.428571
Cp3_shared14	10	0.0410612	0.00290375	0.399742	0.111111	0.818182
Cp3_shared14	20	0.0410612	0.00290375	0.399742	0.250000	0.818182

High-Energy Support Behavior

The matched C run preserves high-energy support across diffusion scale.

For top-10 nodes:

C retained: 8 / 10
C lost:     2 / 10
C gained:   2 / 10

The retained C support lies mainly on a stable response/Lazarus ridge:

r ≈ 0.20–0.30
alpha ≈ 0.132857–0.150000
nodes: 42–44, 57–59, 71–74

This ridge is already visible at fine scale and remains the coarse-scale high-energy support after diffusion.

Cp3 behaves differently.

For top-10 nodes:

Cp3 retained: 2 / 10
Cp3 lost:     8 / 10
Cp3 gained:   8 / 10

Cp3 base top nodes include several response-heavy local peaks:

node_0010
node_0028
node_0071
node_0008
node_0062
node_0007
node_0009
node_0060
node_0047
node_0000

Cp3 final top nodes form more compact repeated bands:

nodes 15–17
nodes 30–32
nodes 45–47
node 62

This indicates structured migration rather than random collapse.

Cp3 Migration Read

Cp3’s high-energy support changes character across scale.

At early scales, the top support is response-operator dominated:

high response_strength
high frobenius_T
mixed or seam-adjacent signed_phase

At larger scales, the top support shifts toward a coarser phase/seam-band substrate:

higher signed_phase
lower response_strength
lower frobenius_T
more concentrated alpha band

Thus Cp3 appears scale-stratified:

fine scale:
  local response-heavy peaks
coarse scale:
  broader phase/seam-band support

This does not imply that Cp3 structure is noise. The late-scale support is structured and stabilizes across successive scale steps. It indicates that Cp3’s strongest fine-scale peaks are less co-located with its coarse-scale substrate than in C.

Provisional Interpretation

OBS-076a validates the observable-diffusion substrate and reveals a matched C/Cp3 scale-space contrast.

C preserves high-energy support across diffusion scale.

Cp3 shows stronger high-energy support migration across diffusion scale.

The Cp3 migration is structured, not random: fine-scale response-heavy peaks wash into coarser phase/seam-band support.

This result is consistent with the idea that Cp3 contains sharper local response spikes on top of a different coarse-scale organization, but OBS-076a alone does not determine whether this affects Cp3 directional asymmetry.

Relation to OBS-075

OBS-075/075b/075c tested Cp3 directional transfer asymmetry and low-complexity survival.

OBS-076a does not retest those classifiers.

Instead, OBS-076a asks whether the underlying observable field has scale-persistent support.

The provisional bridge is:

OBS-075:
  Cp3 directional asymmetry survives selected controls.
OBS-076a:
  Cp3 observable support is more scale-stratified than C under matched MDS-pilot diffusion.
Open question:
  Does Cp3 directional asymmetry survive when geometry and path/field structures are rebuilt from X(t)?

That question belongs to OBS-076b and later scale-space studies.

Limitations

The graph-distance contract is currently canonical_mds_pilot, not canonical Fisher/geodesic distance.

OBS-076a diffuses observables but does not rebuild geometry.

Top-k energy is a generic substrate metric, not a seam/gateway/path-family metric.

Cp3 inputs were prepared from the available response-operator node table, while the C baseline came from the prior OBS-069 prepared inputs.

The matched 14-column comparison reduces, but does not eliminate, differences in upstream campaign provenance.

No path-family, gateway, attractor, seam, or classifier persistence claim is made here.

Next Step

OBS-076b should rebuild geometry from the scale-diffused observable fields:

X(t)
  → Fisher / distance geometry
  → embedding
  → phase / seam objects
  → Lazarus / response summaries
  → persistence metrics

Primary OBS-076b questions:

Does the C response/Lazarus ridge remain a geometry-level structure across scale?
Does Cp3’s fine-scale response layer separate from its coarse-scale phase/seam substrate after geometry rebuild?
Does Cp3 directional asymmetry survive at intermediate diffusion scales?
Are Cp3 OBS-075 survivor channels tied to fine-scale response spikes, coarse-scale bands, or both?

Current Status

OBS-076a is complete as a substrate checkpoint.

Preferred comparison artifact:

outputs/comparisons/obs076a_C_vs_Cp3_shared14_mds_pilot/
  obs076a_C_vs_Cp3_shared14_summary.md

Preferred conclusion:

C preserves high-energy observable support across scale.
Cp3 reorganizes high-energy observable support across scale, but does so into structured coarse bands rather than noise.
