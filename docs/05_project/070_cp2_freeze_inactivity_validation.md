### Downstream pipeline follow-up

After adding explicit downstream-safe freeze-status handling, Cp2 `full_v2` was run through the full PAM pipeline using a corpus-scoped output root:

`outputs/corpora/Cp2/campaigns/full_v2/pipeline/`

The run used entropy observables `H_joint_mean,var_H_joint` because `piF_tail` is inactive across Cp2. The pipeline completed and produced 46 CSV artifacts under the scoped Cp2 pipeline root, including FIM, distance graph, MDS, curvature, phase, Lazarus, operator, transition-rate, identity, holonomy, obstruction, and initial-condition outputs.

Guardrail: this does not reactivate the freeze observable. It establishes that Cp2 has downstream entropy-geometry structure despite freeze-channel inactivity.
