OBS-080b — Stability Core Scale-Band Sensitivity

Summary

OBS-080b tests whether the OBS-078/079 three-feature local stability core persists when the OBS-078a feature table is restricted to alternate regions of the scale ladder.

OBS-080a showed that the stability core is robust across numeric transform contracts. OBS-080b changes a different part of the measurement contract: which scale transitions are allowed to contribute rows.

Core result:

OBS-080b shows that the local stability core is scale-band stable but
scale-position sensitive.
C vs Cp2 and C vs Cp3 remain strong across all meaningful non-empty bands.
Cp2 vs Cp3 remains above permutation across early, middle, all-below-late,
and mid-to-coarse contracts, with strongest separation in the middle /
mid-to-coarse transition corridor.
The late-only Cp2/Cp3 contract is underpowered and not robust.

Compact interpretation:

C separation is scale-band stable.
Cp2/Cp3 separation is scale-position sensitive, strongest in the transition corridor.

Even more compact:

The stability core persists across scale bands, but its hardest pairwise split
is concentrated in the middle-to-coarse corridor.

⸻

Research Context

OBS-078 found that the C / Cp2 / Cp3 distinction can be compressed to a three-feature local stability core:

mean_lambda_local_mean
mean_delta_d_mean
bounded_share_mean

Operational interpretation:

mean_lambda_local_mean:
  local divergence / expansion tendency
mean_delta_d_mean:
  local distance-growth / displacement tendency
bounded_share_mean:
  local boundedness / containment tendency

OBS-079 then showed that this core is:

structurally robust
bootstrap-stable
pairwise anatomized

OBS-080 begins contract-sensitivity testing.

OBS-080a tested transform contracts:

raw
standard_z
robust_median_iqr
rank_percentile
quantile_normal
minmax
signed_log1p_abs

OBS-080b tests scale-band contracts.

⸻

Core Question

OBS-080b asks:

Does the OBS-078/079 local stability core persist when the feature table is
restricted to different regions of the scale ladder?

This is distinct from OBS-080a:

OBS-080a:
  Does the core survive alternate numeric representations?
OBS-080b:
  Does the core survive alternate scale-selection contracts?

⸻

Input

Primary input:

outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/
  obs078a_feature_table.csv

Output directory:

outputs/comparisons/obs080b_stability_core_scale_band_sensitivity/

Script:

experiments/studies/obs080b_stability_core_scale_band_sensitivity.py

Run command:

PYTHONPATH=src .venv/bin/python experiments/studies/obs080b_stability_core_scale_band_sensitivity.py \
  --feature-table outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv \
  --outdir outputs/comparisons/obs080b_stability_core_scale_band_sensitivity \
  --require-all-classes-in-test \
  --n-permutations 200

⸻

Scale-Band Contracts

OBS-080b derives:

transition_midpoint = (scale_index_from + scale_index_to) / 2

and filters rows into scale-band contracts:

all
early
middle
late
fine_to_mid
mid_to_coarse
adjacent_only
long_jump
all_but_early
all_but_late

Observed scale-band manifest:

all:
  n = 168
  case counts = C:49; Cp2:60; Cp3:59
  transitions = 5
  midpoint range = 2.5–6.5
early:
  n = 94
  case counts = C:17; Cp2:42; Cp3:35
  transitions = 3
  midpoint range = 2.5–4.5
middle:
  n = 42
  case counts = C:12; Cp2:12; Cp3:18
  transitions = 1
  midpoint = 5.5
late:
  n = 32
  case counts = C:20; Cp2:6; Cp3:6
  transitions = 1
  midpoint = 6.5
fine_to_mid:
  n = 94
  case counts = C:17; Cp2:42; Cp3:35
  transitions = 3
  midpoint range = 2.5–4.5
mid_to_coarse:
  n = 115
  case counts = C:32; Cp2:36; Cp3:47
  transitions = 3
  midpoint range = 4.5–6.5
adjacent_only:
  n = 168
  case counts = C:49; Cp2:60; Cp3:59
  transitions = 5
  midpoint range = 2.5–6.5
long_jump:
  n = 0
all_but_early:
  n = 74
  case counts = C:32; Cp2:18; Cp3:24
  transitions = 2
  midpoint range = 5.5–6.5
all_but_late:
  n = 136
  case counts = C:29; Cp2:54; Cp3:53
  transitions = 4
  midpoint range = 2.5–5.5

Important observation:

transition_delta = 1 for all rows

Therefore:

adjacent_only = all
long_jump = empty

OBS-080b is therefore a test of scale-ladder position, not transition jump length.

⸻

Tasks

OBS-080b evaluates:

three_way:
  C / Cp2 / Cp3
pairwise:
  C vs Cp2
  C vs Cp3
  Cp2 vs Cp3

Validation schemes:

stratified_cv
leave_object_out
leave_cohort_out
leave_transition_out

Models:

logreg
tree_depth2
rf_depth2
dummy

Permutation baselines are run on stratified CV.

⸻

Three-Way Results

The three-way classifier remains above chance across every meaningful non-empty scale band.

Best primary-valid balanced accuracy by band:

late:
  BA = 1.000
  best scheme = leave_cohort_out
  stratified CV BA = 0.833
mid_to_coarse:
  BA = 0.963
  best scheme = leave_object_out
  stratified CV BA = 0.899
all_but_early:
  BA = 0.944
  best scheme = leave_object_out
  stratified CV BA = 0.902
middle:
  BA = 0.917
  best scheme = leave_cohort_out
  stratified CV BA = 0.917
all / adjacent_only:
  BA = 0.916
  best scheme = leave_object_out
  stratified CV BA = 0.880
all_but_late:
  BA = 0.874
  best scheme = leave_cohort_out
  stratified CV BA = 0.872
early / fine_to_mid:
  BA = 0.838
  best scheme = stratified_cv
  stratified CV BA = 0.838

Interpretation:

The stability core is visible early,
but strengthens toward middle and coarse scale bands.

This is the first sign that the core is not scale-fragile, but is scale-position sensitive.

⸻

Pairwise Results

C vs Cp2

C vs Cp2 remains strong across all meaningful bands.

all_but_early:
  BA = 1.000
  stratified CV BA = 0.984
late:
  BA = 1.000
  stratified CV BA = 1.000
mid_to_coarse:
  BA = 1.000
  stratified CV BA = 0.984
all / adjacent_only:
  BA = 0.988
  stratified CV BA = 0.980
early / fine_to_mid:
  BA = 0.977
  stratified CV BA = 0.941
all_but_late:
  BA = 0.970
  stratified CV BA = 0.966
middle:
  BA = 0.958
  stratified CV BA = 0.958

Interpretation:

C vs Cp2 is scale-band stable.

The C/Cp2 distinction persists throughout the scale ladder.

⸻

C vs Cp3

C vs Cp3 is also strong across all meaningful bands, and becomes especially clean from middle scale onward.

all_but_early:
  BA = 1.000
  stratified CV BA = 0.984
late:
  BA = 1.000
  stratified CV BA = 1.000
mid_to_coarse:
  BA = 1.000
  stratified CV BA = 0.984
middle:
  BA = 1.000
  stratified CV BA = 1.000
all / adjacent_only:
  BA = 0.990
  stratified CV BA = 0.990
all_but_late:
  BA = 0.985
  stratified CV BA = 0.983
early / fine_to_mid:
  BA = 0.971
  stratified CV BA = 0.971

Interpretation:

C vs Cp3 is scale-band stable and strongest from middle scale onward.

The C/Cp3 boundedness split is visible across the ladder, with strongest separation in middle/coarse contracts.

⸻

Cp2 vs Cp3

Cp2 vs Cp3 is the hardest and most diagnostic pair.

middle:
  BA = 1.000
  stratified CV BA = 0.917
all_but_early:
  BA = 1.000
  stratified CV BA = 0.847
mid_to_coarse:
  BA = 0.972
  stratified CV BA = 0.906
all_but_late:
  BA = 0.900
  stratified CV BA = 0.888
all / adjacent_only:
  BA = 0.883
  stratified CV BA = 0.857
early / fine_to_mid:
  BA = 0.867
  stratified CV BA = 0.838
late:
  BA = 0.833
  stratified CV BA = 0.750

Interpretation:

Cp2 vs Cp3 is not fragile,
but it is scale-position sensitive.

The strongest Cp2/Cp3 separation appears in:

middle
mid_to_coarse
all_but_late

This suggests that the Cp2/Cp3 distinction is not merely a final coarse-scale settlement effect. It is already strong in the transition corridor.

The late-only Cp2/Cp3 contract is weak because it is underpowered:

late Cp2 vs Cp3:
  Cp2 = 6
  Cp3 = 6

Permutation confirms this weakness:

Cp2_vs_Cp3 late:
  logreg BA = 0.750, p = 0.065
  tree_depth2 BA = 0.667, p = 0.190
  rf_depth2 BA = 0.583, p = 0.370

Therefore late-only Cp2/Cp3 should not be treated as a robust negative result.

⸻

Permutation Baselines

Permutation baselines behave as expected.

For binary tasks:

permutation BA ≈ 0.50

For three-way tasks:

permutation BA ≈ 0.33

Meaningful observed scores are far above permutation across most scale bands.

Examples:

C_vs_Cp2 late:
  observed BA = 1.000
  permutation mean BA ≈ 0.49–0.51
  p = 0.000
C_vs_Cp3 middle:
  observed BA = 1.000
  permutation mean BA ≈ 0.51
  p = 0.000
Cp2_vs_Cp3 middle:
  observed BA = 0.917
  permutation mean BA ≈ 0.50
  p = 0.000
three_way middle:
  observed BA = 0.917
  permutation mean BA ≈ 0.33–0.35
  p = 0.000

The only notable weak permutation result is the underpowered late-only Cp2/Cp3 contract.

⸻

Expected Failure Modes

Several recorded failures are expected and reflect contract non-identifiability rather than script failure.

Middle / late leave-transition-out

The middle and late bands each contain only one transition:

middle:
  transition = 5→6
late:
  transition = 6→7

Therefore, leave-transition-out has no training rows:

empty_train_or_test

This is expected.

Late leave-object-out

Some late-band held-out objects exhaust one class from the training set.

Examples:

C_vs_Cp2 late / leave_object_out / lazarus_concentration:
  train has only C
C_vs_Cp3 late / leave_object_out / energy_ridge:
  train has only C
Cp2_vs_Cp3 late / leave_object_out:
  held-out object can remove one of the two classes entirely

These folds are non-identifiable and are not used for primary robustness interpretation.

Long-jump

Long-jump is empty:

transition_delta == 1 for all rows

Therefore all long-jump tasks fail with:

missing_task_classes

This is expected.

⸻

Interpretation

OBS-080b is not a simple pass/fail result. It is a structured result.

The stability core is scale-band stable because:

three-way classification remains above permutation across all meaningful bands
C vs Cp2 remains strong across all meaningful bands
C vs Cp3 remains strong across all meaningful bands
Cp2 vs Cp3 remains above permutation across early, middle, all-below-late,
and mid-to-coarse contracts

The stability core is scale-position sensitive because:

three-way accuracy strengthens toward middle/coarse bands
C vs Cp3 becomes especially clean from middle scale onward
Cp2 vs Cp3 is strongest in the middle / mid-to-coarse transition corridor
late-only Cp2 vs Cp3 is underpowered and not robust

Canonical statement:

OBS-080b shows that the local stability core is scale-band stable but
scale-position sensitive.
C separation is broad across the scale ladder.
The hardest pairwise split, Cp2 vs Cp3, concentrates in the
middle-to-coarse transition corridor.

⸻

Relation to OBS-080a

OBS-080a showed:

The stability core is not a numeric-calibration artifact.

OBS-080b adds:

The stability core is not restricted to a single scale-band contract.

Together:

OBS-080a:
  transform-contract stability
OBS-080b:
  scale-band stability / scale-position sensitivity

This extends OBS-079 from within-table robustness to measurement-contract robustness.

⸻

Relation to OBS-079

OBS-079 established:

The OBS-078 stability core is structurally robust,
bootstrap-stable,
and pairwise anatomized within the OBS-078a feature table.

OBS-080b shows that the same core survives when row inclusion is constrained by scale position.

This strengthens the current robustness ladder:

OBS-078:
  compressed local stability core
OBS-079:
  within-contract robustness
OBS-080a:
  transform-contract stability
OBS-080b:
  scale-band contract stability

⸻

Guardrails

OBS-080b does not claim:

causal proof of OBS-075 transfer asymmetry
universal model-independent generalization
robustness beyond the tested OBS-078a feature table
formal attractor basins
topological defects
skyrmion structure
direct generated-text semantic causality

The correct scope remains:

model-specific
corpus-specific
artifact-specific
matched-contract-specific
scale-band-contract-specific
robust within tested perturbation families
provisional with respect to causal transfer

⸻

Canonical Summary

OBS-080b tests whether the OBS-078/079 local stability core survives
alternate scale-band row-selection contracts.
Result:
  the core is scale-band stable but scale-position sensitive.
C vs Cp2:
  stable across all meaningful bands
C vs Cp3:
  stable across all meaningful bands,
  strongest from middle scale onward
Cp2 vs Cp3:
  not fragile,
  strongest in the middle / mid-to-coarse transition corridor,
  weak and underpowered in late-only form
Overall:
  the stability core persists across scale bands,
  while its hardest pairwise distinction localizes to the transition corridor.

Compact:

C separation is scale-band stable.
Cp2/Cp3 separation is transition-corridor concentrated.

Even more compact:

OBS-080b = scale-band stable, scale-position sensitive.

