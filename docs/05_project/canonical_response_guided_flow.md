Canonical Response-Guided Flow

Purpose

This document defines the current canonical reading of response-guided flow in the PAM observatory.

It covers:

* the original discrete response-flow layer established in OBS-043
* the continuous reconstruction line introduced in OBS-044
* the comparison between discrete and continuous flow
* the negative support-expansion result of OBS-045
* interpolation-model sensitivity from OBS-046
* the stabilized minimal-blend baseline from OBS-047

The discrete response-guided flow layer remains the canonical dynamical object. The continuous line is best understood as a seam-preserving geometric reconstruction and approximation of that layer.

⸻

Canonical discrete response-guided flow (OBS-043)

OBS-043 established the first dynamical flow layer on the observatory manifold.

The core object is a response-guided direction field induced by the dominant eigenvector of the response tensor. Embedded integration over the node graph yields discrete trajectories that reveal coherent dynamical organization beyond static field inspection.

The main stabilized properties of the discrete layer are:

* flow is coherent rather than noise-like
* seam engagement is consistently high across tested regimes
* the seam behaves both as a local capture region and as a launch region for broader routing
* phase partitions are permeable under relaxed integration
* mismatch scalars behave as routing costs rather than absolute barriers

The main limitation of the discrete layer is methodological rather than conceptual. Paths terminate primarily because of forward-neighbor exhaustion in a sparse graph, not because the response field lacks direction or because seam structure blocks motion in principle.

This discrete response-guided flow remains the canonical dynamical layer of the observatory.

⸻

Continuous response-flow reconstruction (OBS-044)

OBS-044 introduced a continuous reconstruction of response-guided flow over the embedded manifold.

The goal of this line was not to replace the discrete layer immediately, but to test whether the response field could be reconstructed as a smoother local flow using interpolation-based integration in embedded space.

The main result is that continuous reconstruction preserves the strongest invariant of the discrete line:

* seam engagement remains high

At the same time, the first continuous reconstruction is more conservative than the discrete relaxed reference:

* mean path extent is reduced
* phase-crossing share is reduced
* paths remain more tightly seam-local

The resulting interpretation is clear:

the continuous reconstruction captures a real and meaningful seam-engaged flow structure, but under-recovers outward release and cross-phase traversal relative to the discrete relaxed baseline.

Thus OBS-044 established the continuous line as informative, but not yet sufficient to replace the discrete canonical flow layer.

⸻

Continuous vs discrete comparison (OBS-044b / OBS-044c)

The first direct comparison between continuous and discrete flow clarified what the continuous line preserved and what it lost.

Relative to the discrete relaxed reference, the continuous reconstruction:

* preserves high seam-contact share
* remains concentrated closer to the seam on average
* shortens path extent in the embedded manifold
* reduces phase-crossing frequency

This comparison also showed that family-like route structure remains visible in the continuous line. Seam-hugging, release-directed, short-trapped, and mixed path families all persist as recognizable categories.

However, the continuous line weakens the most transition-rich part of the discrete dynamical picture:

* release-directed paths are still present
* but their cross-phase realization is reduced
* short-trapped structure remains comparatively strong

The correct reading is therefore:

continuous flow preserves the family-like structure of the discrete dynamical layer, but compresses release behavior and under-recovers cross-phase traversal.

This makes the continuous line useful for geometric reconstruction and seam-local flow analysis, but still incomplete as a surrogate for the discrete family layer.

⸻

Support-radius expansion does not recover release (OBS-045)

OBS-045 tested a simple hypothesis:

perhaps the under-recovery of release behavior in the continuous line is mainly a support problem, and can be corrected by enlarging the support radius.

This did not occur.

Increasing support radius:

* lengthened continuous paths
* preserved strong seam engagement
* did not materially improve overall phase crossing
* did not materially improve release-directed cross-phase recovery

This is a strong negative result.

It shows that the missing release behavior is not primarily caused by too little spatial support in the local interpolator. The main limitation therefore does not disappear simply by making the solver looser.

Canonical interpretation:

support expansion alone does not recover the missing release structure of the continuous line.

This result is important because it rules out a naive geometric fix and forces attention toward the interpolation rule itself.

⸻

Interpolation-model sensitivity (OBS-046)

OBS-046 tested whether the quality of continuous reconstruction depends materially on the local interpolation model.

This comparison showed that interpolation choice matters.

The conservative k-nearest-neighbor averaging baseline does not give the strongest release recovery. Alternative local rules improve different aspects of the reconstruction:

* nearest-anchor improves overall crossing relative to the old conservative baseline
* top2-blend gives the strongest release-directed cross-phase recovery among tested interpolation modes
* seam engagement remains high across the best-performing models

The main conclusion is that the continuous line is interpolation-sensitive.

This is an important refinement. It means the underperformance of the earliest continuous baseline should not be interpreted as a generic failure of continuous response flow, but as evidence that local reconstruction quality depends strongly on the chosen interpolation rule.

Canonical interpretation:

interpolation-model choice matters more than naive support expansion in the continuous reconstruction line.

⸻

Minimal-blend baseline stabilization (OBS-047)

OBS-047 turned the interpolation sensitivity result into a stabilized baseline.

A small, disciplined sweep was performed around the strongest continuous candidates. This established a preferred minimal-blend reconstruction rule:

* interpolation mode: top2_blend
* support radius scale: 3.5
* step size scale: 0.15

This stabilized baseline improves both:

* overall phase crossing
* release-directed cross-phase recovery

while preserving the strongest invariant of the flow program:

* high seam engagement

The resulting continuous baseline is therefore not just another experiment. It is the first stabilized continuous response-flow reference.

Its role is now clear:

* it is the best current seam-preserving continuous reconstruction
* it improves materially over earlier continuous baselines
* it still remains a partial approximation to the discrete canonical layer rather than a replacement for it

Canonical interpretation:

minimal local blending yields the best current continuous response-flow baseline, but continuous flow remains an approximate geometric reconstruction of the canonical discrete dynamical layer.

⸻

Relation to family structure

The response-flow arc now connects naturally to the family layer.

The discrete flow program established the first dynamical manifold layer. The continuous line showed that family-like path structure persists under interpolation-based reconstruction, even when release behavior is only partially recovered.

At the same time, later recoverability analysis clarified that route-family identity is not strongly pointwise recoverable. Stable seam corridor is relatively locally legible, while reorganization-heavy remains poorly recoverable even after short-context enrichment.

This supports an important reading of the flow-family relation:

flow families should not be treated as simple local labels attached to isolated nodes. They are better understood as distributed route objects whose full identity depends on broader structural context.

This strengthens the continuity between:

* seam-family laws
* temporal-depth results
* compression results
* and the flow layer

without collapsing them into a single local classifier picture.

⸻

Current canonical reading

The current canonical reading of response-guided flow is:

* the discrete response-guided flow layer remains the canonical dynamical object
* continuous response flow provides a meaningful seam-preserving geometric reconstruction of that layer
* the first continuous baseline under-recovered outward release and cross-phase traversal
* support expansion alone did not solve this limitation
* interpolation-model choice materially affects reconstruction quality
* minimal local blending provides the best current continuous baseline
* seam engagement remains the strongest invariant across both discrete and continuous regimes
* release recovery in the continuous line remains partial rather than complete

The observatory should therefore treat the discrete flow layer as primary, and the stabilized continuous baseline as the current best geometric approximation.

⸻

What remains open

Several questions remain open.

First, the continuous line still under-recovers the strongest release behavior visible in the discrete relaxed regime.

Second, the best current continuous baseline is stable and useful, but not final.

Third, family interpretation remains stronger in the discrete layer than in the continuous reconstruction.

Future work may improve the continuous line through:

* richer local interpolation rules
* better manifold-web support
* explicit path-thread overlays
* tighter route-family integration
* and improved canonical artifact promotion into the observatory surface

For now, the correct stance is not that continuous flow has failed, but that it has reached a disciplined intermediate stage:

* real
* useful
* stabilized
* and still incomplete.
