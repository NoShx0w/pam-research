Here is a repo-ready note for:

docs/05_project/75c_cp3_low_complexity_control.md

# OBS-075c — Cp3 Low-Complexity Directional-Asymmetry Control
**Date:** 2026-06-06  
**Status:** Observatory note  
**Scope:** Model-specific, corpus-specific, feature-set-specific, provisional  
**Depends on:** OBS-073, OBS-075, OBS-075b  
**Artifacts:** `outputs/comparisons/obs075c_cp3_low_complexity_control/`
---
## Summary
OBS-075c tests whether the Cp3→Cp2 directional-asymmetry signal observed in OBS-075/OBS-075b survives when classifier capacity is reduced.
The result is stronger than expected.
The main finding is that Cp3→Cp2 asymmetry does **not** appear to be only a full Random Forest / flexible-boundary artifact. Several rows survive under shallow Random Forests and, more importantly, under logistic regression.
The strongest survivor is the coupled-outcome target under the anti-shortcut, endpoint/velocity-ablated feature set:
```text
model: logreg
target: coupled_outcome_group_no_direct_seam_no_grid
feature_set: no_direct_seam_no_grid_no_endpoint_velocity
Cp2→Cp3 BA: 0.5182
Cp3→Cp2 BA: 0.9317
asymmetry Cp3−Cp2: 0.4136
specificity_vs_cp: 0.4414
specificity_vs_c: 0.3138
survival_read: survives_vs_cp_and_c

This means the Cp3→Cp2 directional advantage survives:

* direct seam removal,
* absolute grid/location removal,
* endpoint/velocity/path-length removal,
* Cp control comparison,
* C control comparison,
* and low-complexity linear classification.

This weakens the simple “Cp3 is just a broad/noisy boundary corpus” explanation.

⸻

Question

OBS-075c asks:

Does Cp3 directional asymmetry survive when the classifier boundary is forced to be simple?

OBS-075b recomputed endpoint/velocity ablations. OBS-075c compares those OBS-075b runs across model capacity:

* full Random Forest,
* shallow Random Forest, depth 2,
* shallow Random Forest, depth 4,
* logistic regression.

OBS-075c does not recompute geometry, path families, coupling, outcomes, recovery channels, or model predictions. It is a cross-run comparison over already-produced OBS-075b artifacts.

⸻

Inputs

OBS-075c consumed:

outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_v2_smoke
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_rf_depth2
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_rf_depth4
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_logreg

The comparison output is:

outputs/comparisons/obs075c_cp3_low_complexity_control/

Primary generated artifacts:

obs075c_summary.md
obs075c_model_scores_combined.csv
obs075c_pair_asymmetry.csv
obs075c_specificity.csv
obs075c_survival_table.csv
obs075c_model_capacity_matrix.csv
obs075c_run_summary.csv

⸻

Definitions

Directional asymmetry is computed as:

asymmetry Cp3−baseline = BA(Cp3→baseline) − BA(baseline→Cp3)

Cp2 specificity is then tested against controls:

specificity_vs_cp = asymmetry(Cp3→Cp2) − asymmetry(Cp3→Cp)
specificity_vs_c = asymmetry(Cp3→Cp2) − asymmetry(Cp3→C)

A row is strongest when:

asymmetry Cp3→Cp2 > 0
specificity_vs_cp > 0
specificity_vs_c > 0

Rows with unavailable C-control recovery support are marked as Cp-only or partial, not as negative evidence.

⸻

Run-level result

OBS-075c found the following run-level pattern:

run	strict focus rows	survives vs Cp and C	survives vs Cp only	collapsed / near zero	mean Cp3−Cp2 asymmetry
logreg	18	7	6	3	0.1388
rf_depth2	18	2	8	8	0.0608
rf_depth4	18	8	6	4	0.0804
rf_full	18	6	3	8	0.0280

The important feature is not that full Random Forest is strongest. It is not. Logistic regression and shallow Random Forests preserve or amplify several of the directional asymmetries.

This is evidence against interpreting the original OBS-075 effect as merely high-capacity classifier flexibility.

⸻

Main survivors

Coupled-outcome group

The coupled-outcome target is the strongest Cp2-specific survivor.

Best row:

logreg
coupled_outcome_group_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.4136
specificity_vs_cp: 0.4414
specificity_vs_c: 0.3138

This is the cleanest OBS-075c result.

It survives the anti-shortcut feature set and endpoint/velocity removal, while also surviving both Cp and C controls. Because it survives in logistic regression, the separating structure is compatible with a low-complexity field boundary rather than requiring a flexible Random Forest boundary.

Other coupled-outcome rows also survive:

logreg / holonomy_criticality_shape_only:
asymmetry 0.3610
specificity_vs_cp 0.3241
specificity_vs_c 0.3827
logreg / no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity:
asymmetry 0.3610
specificity_vs_cp 0.3241
specificity_vs_c 0.3827
rf_depth4 / no_direct_seam_no_grid_no_endpoint_velocity:
asymmetry 0.1513
specificity_vs_cp 0.2235
specificity_vs_c 0.2005

Interpretation:

Coupled-outcome asymmetry is no longer merely a terminal-trajectory or high-capacity-RF artifact. It survives low-complexity controls.

⸻

Recovery channel

The recovery-channel result is also strong, but with a narrower control scope.

Best unbounded recovery row:

logreg
recovery_channel_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.3362
specificity_vs_cp: 0.4045
specificity_vs_c: NA
survival_read: survives_vs_cp_only

The recovery-channel signal survives:

* no direct seam,
* no grid/location,
* endpoint/velocity removal,
* logistic regression,
* and Cp control.

However, the C-control recovery rows are unavailable or class-fragile because C has very small false-recovery support.

Therefore the correct read is:

Recovery-channel asymmetry is Cp-control-supported and low-complexity-supported, but not fully C-control-certified.

Boundedness-strict recovery also survives across model families, but it has small class-count warnings and should remain secondary.

⸻

Outcome group

Outcome-group rows survive across several settings, including low-complexity models.

Example:

logreg
outcome_group_no_direct_seam_no_grid
no_direct_seam_no_grid
asymmetry Cp3−Cp2: 0.1005
specificity_vs_cp: 0.0525
specificity_vs_c: 0.1229

Outcome-group is supportive but less central than coupled-outcome, because it is broader and less Cp2-specific in mechanism.

⸻

Path family

Path-family rows also show some survival, especially under logistic regression:

logreg
path_family_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.0786
specificity_vs_cp: 0.1089
specificity_vs_c: 0.1903

This suggests that some route-family geometry remains visible under low-complexity controls, but this is not the main OBS-075c claim.

⸻

Coupling class

Coupling class behaves as a useful negative control.

Across the tested feature sets and model capacities, coupling-class directional asymmetry generally collapses or reverses:

coupling_class_no_direct_seam_no_grid:
collapsed_or_near_zero across logreg, rf_depth2, rf_depth4, rf_full

This is important because it shows OBS-075c is not simply finding “everything transfers Cp3→Cp2 better.”

The survival is target-specific.

⸻

Interpretation

The strongest OBS-075c interpretation is:

Cp3→Cp2 directional asymmetry survives low-complexity controls for coupled-outcome and recovery-channel targets, especially after anti-shortcut and endpoint/velocity ablation.

This weakens three critiques:

1. Broad/noisy Cp3 critique
    If Cp3 were merely a generic broad-boundary corpus, the same asymmetry should appear uniformly against Cp and C controls. It does not.
2. High-capacity Random Forest critique
    The best coupled-outcome row is logistic regression, not full Random Forest.
3. Endpoint/velocity shortcut critique
    The strongest row survives no_direct_seam_no_grid_no_endpoint_velocity.

The result does not prove a causal mechanism. It does, however, raise OBS-075 from “fragile transfer oddity” to “target-specific directional structure requiring explanation.”

⸻

Updated OBS-075 read

OBS-075 should no longer be summarized as:

Full Random Forest found a fragile Cp3 asymmetry.

The corrected read is:

Cp3→Cp2 directional asymmetry survives low-complexity controls, including logistic regression, for several anti-shortcut / endpoint-ablated feature sets. The strongest survivor is coupled-outcome. Recovery-channel survives with Cp-control support but remains C-control-limited.

This is materially stronger than the initial OBS-075 interpretation.

⸻

Guardrails

* OBS-075c is still corpus-specific and artifact-specific.
* It does not prove causality.
* It does not recompute labels or geometry.
* It does not include lexical controls.
* Recovery-channel C-control rows are unavailable or class-fragile because C has very small false-recovery support.
* Boundedness-strict recovery rows should be interpreted cautiously due to small class counts.
* Positive Cp specificity without C specificity indicates corpus-family interaction, not strict Cp2 specificity.
* Logistic regression survival is stronger than full-RF-only survival, but still depends on the OBS-073/OBS-075b feature table construction.
* Path-level lexical mediation remains open.

⸻

Recommended next step

The next defense layer is OBS-074 on Cp3 / Cp2 / Cp / C, interpreted as a lexical mediation check.

The key question:

Does the Cp3→Cp2 geometric asymmetry have a lexical-semantic carrier, or does it survive where lexical-field recovery is weak?

If OBS-074 lexical recovery is weak while OBS-075c geometry survives, the field-geometric interpretation strengthens.

If OBS-074 lexical recovery mirrors the OBS-075c survivors, then the asymmetry may be lexically mediated.

Either result is useful.

And the shell write command:
```bash
cat > docs/05_project/75c_cp3_low_complexity_control.md <<'MD'
# OBS-075c — Cp3 Low-Complexity Directional-Asymmetry Control
**Date:** 2026-06-06  
**Status:** Observatory note  
**Scope:** Model-specific, corpus-specific, feature-set-specific, provisional  
**Depends on:** OBS-073, OBS-075, OBS-075b  
**Artifacts:** `outputs/comparisons/obs075c_cp3_low_complexity_control/`
---
## Summary
OBS-075c tests whether the Cp3→Cp2 directional-asymmetry signal observed in OBS-075/OBS-075b survives when classifier capacity is reduced.
The result is stronger than expected.
The main finding is that Cp3→Cp2 asymmetry does **not** appear to be only a full Random Forest / flexible-boundary artifact. Several rows survive under shallow Random Forests and, more importantly, under logistic regression.
The strongest survivor is the coupled-outcome target under the anti-shortcut, endpoint/velocity-ablated feature set:
```text
model: logreg
target: coupled_outcome_group_no_direct_seam_no_grid
feature_set: no_direct_seam_no_grid_no_endpoint_velocity
Cp2→Cp3 BA: 0.5182
Cp3→Cp2 BA: 0.9317
asymmetry Cp3−Cp2: 0.4136
specificity_vs_cp: 0.4414
specificity_vs_c: 0.3138
survival_read: survives_vs_cp_and_c

This means the Cp3→Cp2 directional advantage survives:

* direct seam removal,
* absolute grid/location removal,
* endpoint/velocity/path-length removal,
* Cp control comparison,
* C control comparison,
* and low-complexity linear classification.

This weakens the simple “Cp3 is just a broad/noisy boundary corpus” explanation.

⸻

Question

OBS-075c asks:

Does Cp3 directional asymmetry survive when the classifier boundary is forced to be simple?

OBS-075b recomputed endpoint/velocity ablations. OBS-075c compares those OBS-075b runs across model capacity:

* full Random Forest,
* shallow Random Forest, depth 2,
* shallow Random Forest, depth 4,
* logistic regression.

OBS-075c does not recompute geometry, path families, coupling, outcomes, recovery channels, or model predictions. It is a cross-run comparison over already-produced OBS-075b artifacts.

⸻

Inputs

OBS-075c consumed:

outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_v2_smoke
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_rf_depth2
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_rf_depth4
outputs/comparisons/obs075b_cp3_endpoint_velocity_ablation_logreg

The comparison output is:

outputs/comparisons/obs075c_cp3_low_complexity_control/

Primary generated artifacts:

obs075c_summary.md
obs075c_model_scores_combined.csv
obs075c_pair_asymmetry.csv
obs075c_specificity.csv
obs075c_survival_table.csv
obs075c_model_capacity_matrix.csv
obs075c_run_summary.csv

⸻

Definitions

Directional asymmetry is computed as:

asymmetry Cp3−baseline = BA(Cp3→baseline) − BA(baseline→Cp3)

Cp2 specificity is then tested against controls:

specificity_vs_cp = asymmetry(Cp3→Cp2) − asymmetry(Cp3→Cp)
specificity_vs_c = asymmetry(Cp3→Cp2) − asymmetry(Cp3→C)

A row is strongest when:

asymmetry Cp3→Cp2 > 0
specificity_vs_cp > 0
specificity_vs_c > 0

Rows with unavailable C-control recovery support are marked as Cp-only or partial, not as negative evidence.

⸻

Run-level result

OBS-075c found the following run-level pattern:

run	strict focus rows	survives vs Cp and C	survives vs Cp only	collapsed / near zero	mean Cp3−Cp2 asymmetry
logreg	18	7	6	3	0.1388
rf_depth2	18	2	8	8	0.0608
rf_depth4	18	8	6	4	0.0804
rf_full	18	6	3	8	0.0280

The important feature is not that full Random Forest is strongest. It is not. Logistic regression and shallow Random Forests preserve or amplify several of the directional asymmetries.

This is evidence against interpreting the original OBS-075 effect as merely high-capacity classifier flexibility.

⸻

Main survivors

Coupled-outcome group

The coupled-outcome target is the strongest Cp2-specific survivor.

Best row:

logreg
coupled_outcome_group_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.4136
specificity_vs_cp: 0.4414
specificity_vs_c: 0.3138

This is the cleanest OBS-075c result.

It survives the anti-shortcut feature set and endpoint/velocity removal, while also surviving both Cp and C controls. Because it survives in logistic regression, the separating structure is compatible with a low-complexity field boundary rather than requiring a flexible Random Forest boundary.

Other coupled-outcome rows also survive:

logreg / holonomy_criticality_shape_only:
asymmetry 0.3610
specificity_vs_cp 0.3241
specificity_vs_c 0.3827
logreg / no_direct_seam_no_grid_no_endpoint_velocity_no_tortuosity:
asymmetry 0.3610
specificity_vs_cp 0.3241
specificity_vs_c 0.3827
rf_depth4 / no_direct_seam_no_grid_no_endpoint_velocity:
asymmetry 0.1513
specificity_vs_cp 0.2235
specificity_vs_c 0.2005

Interpretation:

Coupled-outcome asymmetry is no longer merely a terminal-trajectory or high-capacity-RF artifact. It survives low-complexity controls.

⸻

Recovery channel

The recovery-channel result is also strong, but with a narrower control scope.

Best unbounded recovery row:

logreg
recovery_channel_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.3362
specificity_vs_cp: 0.4045
specificity_vs_c: NA
survival_read: survives_vs_cp_only

The recovery-channel signal survives:

* no direct seam,
* no grid/location,
* endpoint/velocity removal,
* logistic regression,
* and Cp control.

However, the C-control recovery rows are unavailable or class-fragile because C has very small false-recovery support.

Therefore the correct read is:

Recovery-channel asymmetry is Cp-control-supported and low-complexity-supported, but not fully C-control-certified.

Boundedness-strict recovery also survives across model families, but it has small class-count warnings and should remain secondary.

⸻

Outcome group

Outcome-group rows survive across several settings, including low-complexity models.

Example:

logreg
outcome_group_no_direct_seam_no_grid
no_direct_seam_no_grid
asymmetry Cp3−Cp2: 0.1005
specificity_vs_cp: 0.0525
specificity_vs_c: 0.1229

Outcome-group is supportive but less central than coupled-outcome, because it is broader and less Cp2-specific in mechanism.

⸻

Path family

Path-family rows also show some survival, especially under logistic regression:

logreg
path_family_no_direct_seam_no_grid
no_direct_seam_no_grid_no_endpoint_velocity
asymmetry Cp3−Cp2: 0.0786
specificity_vs_cp: 0.1089
specificity_vs_c: 0.1903

This suggests that some route-family geometry remains visible under low-complexity controls, but this is not the main OBS-075c claim.

⸻

Coupling class

Coupling class behaves as a useful negative control.

Across the tested feature sets and model capacities, coupling-class directional asymmetry generally collapses or reverses:

coupling_class_no_direct_seam_no_grid:
collapsed_or_near_zero across logreg, rf_depth2, rf_depth4, rf_full

This is important because it shows OBS-075c is not simply finding “everything transfers Cp3→Cp2 better.”

The survival is target-specific.

⸻

Interpretation

The strongest OBS-075c interpretation is:

Cp3→Cp2 directional asymmetry survives low-complexity controls for coupled-outcome and recovery-channel targets, especially after anti-shortcut and endpoint/velocity ablation.

This weakens three critiques:

1. Broad/noisy Cp3 critique
    If Cp3 were merely a generic broad-boundary corpus, the same asymmetry should appear uniformly against Cp and C controls. It does not.
2. High-capacity Random Forest critique
    The best coupled-outcome row is logistic regression, not full Random Forest.
3. Endpoint/velocity shortcut critique
    The strongest row survives no_direct_seam_no_grid_no_endpoint_velocity.

The result does not prove a causal mechanism. It does, however, raise OBS-075 from “fragile transfer oddity” to “target-specific directional structure requiring explanation.”

⸻

Updated OBS-075 read

OBS-075 should no longer be summarized as:

Full Random Forest found a fragile Cp3 asymmetry.

The corrected read is:

Cp3→Cp2 directional asymmetry survives low-complexity controls, including logistic regression, for several anti-shortcut / endpoint-ablated feature sets. The strongest survivor is coupled-outcome. Recovery-channel survives with Cp-control support but remains C-control-limited.

This is materially stronger than the initial OBS-075 interpretation.

⸻

Guardrails

* OBS-075c is still corpus-specific and artifact-specific.
* It does not prove causality.
* It does not recompute labels or geometry.
* It does not include lexical controls.
* Recovery-channel C-control rows are unavailable or class-fragile because C has very small false-recovery support.
* Boundedness-strict recovery rows should be interpreted cautiously due to small class counts.
* Positive Cp specificity without C specificity indicates corpus-family interaction, not strict Cp2 specificity.
* Logistic regression survival is stronger than full-RF-only survival, but still depends on the OBS-073/OBS-075b feature table construction.
* Path-level lexical mediation remains open.

⸻

Recommended next step

The next defense layer is OBS-074 on Cp3 / Cp2 / Cp / C, interpreted as a lexical mediation check.

The key question:

Does the Cp3→Cp2 geometric asymmetry have a lexical-semantic carrier, or does it survive where lexical-field recovery is weak?

If OBS-074 lexical recovery is weak while OBS-075c geometry survives, the field-geometric interpretation strengthens.

If OBS-074 lexical recovery mirrors the OBS-075c survivors, then the asymmetry may be lexically mediated.

Either result is useful.
MD
