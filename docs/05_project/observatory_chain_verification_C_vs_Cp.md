# Observatory Chain Verification — C vs Cp

## Scope

This note records end-to-end verification of the canonical observatory chain from OBS-028c through OBS-051 on both supported corpus environments:

- **C** on the MacBook
- **Cp** on the Mac mini

The purpose of this note is operational and scientific:

- confirm that the chain runs reproducibly on both machines
- distinguish stable from provisional observatory stages
- record which conclusions replicate across corpora and which do not

---

## Verification Date

- Date: 2026-04-27

---

## Environments

### MacBook
- Corpus: **C**
- Role: canonical smaller observatory baseline

### Mac mini
- Corpus: **Cp**
- Role: larger corpus observatory regime

---

## Runner Used

Canonical runner:

```bash
./scripts/run_observatory_chain_obs028c_to_obs051.sh
```

This runner completed successfully on both machines.

---

## Chain Stages Verified

The following stages were executed successfully on both environments:

1. OBS-028c prerequisite chain
2. OBS-028c canonical seam bundle
3. scale-conditioned family substrate build
4. OBS-022 scene input preparation
5. OBS-024 family hotspot occupancy
6. OBS-050 structural coupling persistence
7. OBS-051 local divergence in coupled windows
   - `all`
   - `core`
   - `near`

---

## File-Level Verification

On both machines, the chain produced the expected output directories and summary artifacts for:

- `outputs/obs028c_canonical_seam_bundle/`
- `outputs/scales/100000/family_substrate/`
- `outputs/obs022_scene_bundle/`
- `outputs/obs024_family_hotspot_occupancy/`
- `outputs/obs050_structural_coupling_persistence/`
- `outputs/obs051_local_divergence_in_coupled_windows/`

The runner completed without manual intervention.

---

## Scientific Verification Summary

## OBS-028c

### Status
- Reproducible
- Treated as canonical

### Verification result
- Canonical seam bundle exports successfully on both machines
- No cross-machine instability observed at the operational level

---

## Scale Family Substrate / OBS-022 / OBS-024

### Status
- Reproducible
- Treated as canonical infrastructure layers

### Verification result
- Family substrate builds correctly on both machines
- OBS-022 scene bundle refreshes correctly on both machines
- OBS-024 hotspot occupancy runs correctly on both machines

These stages are currently treated as stable observatory infrastructure.

---

## OBS-050 — Structural Coupling Persistence

### Status
- Reproducible
- Treated as canonical
- Cross-corpus qualitatively replicated

### Shared qualitative result
On both C and Cp:

- recovery-like roughness-escalation windows are much more likely than nonrecovering windows to remain in coupled seam bands
- the seam-coupling persistence result is preserved qualitatively
- coupled-vs-decoupled separation remains the key predictive signal

### Conclusion
OBS-050 is currently treated as the first **cross-corpus robust predictive observatory result**.

### Operational interpretation
- OBS-050 is stable enough to serve as part of the canonical observatory chain
- exact effect sizes differ between corpora, but the direction of the result is preserved

---

## OBS-051 — Local Divergence in Coupled Windows

### Status
- Reproducible as a pipeline stage
- **Not** currently canonical in interpretation
- Treated as provisional / corpus-sensitive

### Verification result
OBS-051 runs successfully on both machines in all refined modes:

- `all`
- `core`
- `near`

However, the scientific conclusion does **not** remain stable across corpora.

### Shared structural observation
- the `core` seam band does **not** behave like a clean bounded recovery regime
- `core` is better interpreted as a high-pressure contact regime

### Corpus-sensitive result
The `near` band, which is the most informative band in the refined OBS-051 instrument, does not preserve the same directional conclusion across C and Cp.

- On corpus **C**, refined OBS-051 suggests stronger boundedness in recovery-like near-band windows
- On corpus **Cp**, refined OBS-051 does not preserve that direction and can reverse it

### Conclusion
OBS-051 is currently treated as:

- operationally reproducible
- scientifically promising
- but **corpus-sensitive and provisional**

It should not yet be cited as a corpus-invariant observatory result.

---

## Current Canonical / Provisional Split

### Canonical / stable enough to rely on
- OBS-028c
- scale family substrate build
- OBS-022 scene input preparation
- OBS-024 family hotspot occupancy
- OBS-050 structural coupling persistence

### Provisional / not yet corpus-invariant
- OBS-051 local divergence in coupled windows

---

## Recommended Current Observatory Reading

The observatory chain currently supports the following distinction:

### Stable
Retained seam coupling during roughness escalation is a robust cross-corpus feature.

### Not yet stable
The boundedness structure inside seam-coupled windows is corpus-sensitive and not yet canonical.

This means the present observatory spine supports:

- seam bundle
- family substrate
- scene preparation
- hotspot occupancy
- predictive seam-coupling persistence

while treating local-divergence interpretation as an active refinement frontier rather than closed observatory fact.

---

## Practical Consequence

Future downstream studies should use the following rule:

- OBS-050 may be treated as part of the canonical observatory foundation
- OBS-051 may be run and inspected, but should be labeled **provisional** until a corpus-stable interpretation is achieved

This distinction should remain explicit in:
- docs
- PR descriptions
- logbook entries
- future observatory summaries

---

## Next-Step Recommendation

The chain from OBS-028c through OBS-051 is now stable enough operationally to support future work.

Recommended next posture:

1. preserve the current canonical/provisional split explicitly
2. avoid building strong new theoretical claims on top of OBS-051 alone
3. treat OBS-052 and later basin-style analyses as downstream of a still-provisional divergence layer
4. use the current chain as the reproducible observatory baseline for future refinement
