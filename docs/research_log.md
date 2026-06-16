# Research Log

## OBS-001

**Date:** 2026-03-16  
**State:** Grid sweep approaching completion (~650/750)

**Claim:**  
The observatory had a stable Fisher manifold, an emerging seam near r≈0.15, and a working geometry pipeline while the grid sweep was nearing completion.

**Summary:**  
At this stage, the parameter sweep had progressed far enough to reveal a coherent manifold structure under the Fisher metric. The embedding showed clear organization, with an emergent transition corridor visible around r≈0.15. This corridor was not yet fully characterized but appeared as a consistent structural feature across sampled configurations.

The geometry pipeline—covering metric construction, graph distances, and embedding—was operational and producing stable outputs. While the dataset was incomplete, the observed structure was already robust enough to support further probing.

**Operational consequence:**  
Enabled continuation to full-grid completion with confidence that the geometry stack was stable and meaningful.

---

## OBS-002

**Date:** 2026-03-17  
**State:** Full grid sweep complete (750/750)

**Claim:**  
Full-grid completion stabilized the manifold and established coherent signed phase structure aligned with the transition region.

**Summary:**  
With the full parameter grid completed, the manifold representation became fully populated and globally consistent. The previously observed transition region around r≈0.15 sharpened into a coherent phase boundary, separating two regimes in the embedded space.

A signed phase coordinate was introduced and shown to vary systematically across the manifold. This provided a clean, data-driven phase structure aligned with the geometric transition region.

**Operational consequence:**  
Established a stable phase diagram and enabled downstream analyses conditioned on phase structure.

---

## OBS-003

**Date:** 2026-03-18  
**State:** Operator layer established on top of stabilized phase geometry

**Claim:**  
The observatory added operator structure and formalized horizon concepts on top of the stabilized phase manifold.

**Summary:**  
An operator layer was introduced to study response and transition behavior across the manifold. This layer included constructs for probing local response properties and identifying regions of structural sensitivity.

The concept of a horizon was operationalized, referring to regions where observables exhibited sharp changes or instability. These regions were found to align with the previously identified phase boundary.

**Operational consequence:**  
Enabled systematic probing of transition dynamics and response structure across the manifold.

---

## OBS-004

**Date:** 2026-03-19  
**State:** Lazarus regime identified and validated under scaled geodesic probing

**Claim:**  
High-Lazarus paths were shown to be seam-adjacent, high-curvature, and predictive of unstable or crossing behavior under geodesic probing.

**Summary:**  
A Lazarus score was defined to capture multiplicative instability across observables. Under geodesic probing, paths with high Lazarus scores concentrated near the transition region and exhibited strong curvature effects.

These paths were associated with increased likelihood of phase transition or instability. The Lazarus measure therefore acted as a proxy for identifying critical regions and transition-prone trajectories.

**Operational consequence:**  
Provided a scalar diagnostic for identifying high-risk or transition-prone regions on the manifold.

---

## OBS-005

**Date:** 2026-03-18  
**State:** Transition-rate law identified under Lazarus conditioning

**Claim:**  
Boundary contact, compression peak, and phase transition were organized into a temporal chain with Lazarus conditioning increasing short-horizon transition probability.

**Summary:**  
Analysis of trajectories revealed a consistent ordering of events: approach to the boundary, increase in compression (Lazarus), and eventual phase transition. This temporal structure suggested a causal progression rather than a purely geometric coincidence.

Conditioning on high Lazarus values increased the probability of observing a transition within a short horizon. This established a predictive link between local compression and imminent phase change.

**Operational consequence:**  
Enabled probabilistic forecasting of transitions based on local compression signals.

---

## OBS-006

**Date:** 2026-03-29  
**State:** Canonical observatory architecture consolidated

**Claim:**  
The repository structure was reorganized so the software architecture matched the scientific layered architecture of the observatory.

**Summary:**  
The codebase was restructured into a layered architecture reflecting the conceptual components of the observatory, including geometry, phase, operators, and topology. This replaced a flatter script-based organization with modular packages.

The pipeline was formalized into a canonical execution flow, and data outputs were standardized under a consistent artifact structure. Documentation was aligned with this new architecture.

**Operational consequence:**  
Improved reproducibility, maintainability, and clarity of the observatory pipeline.

---

## OBS-007

**Date:** 2026-03-31  
**State:** Identity field established as a first-pass topology-layer observable

**Claim:**  
A first-pass local structural identity field and sparse sign-structured singularities were identified as distinct topology-layer observables.

**Summary:**  
An identity field was introduced to capture local structural distinctions beyond geometric distance. This field revealed regions with distinct identity signatures, including sparse singularities with sign structure.

These singularities suggested the presence of topological features not captured by metric geometry alone. The identity field thus opened a new observational layer focused on structural equivalence and distinction.

**Operational consequence:**  
Added a topology-sensitive observable for detecting structural singularities and identity variation.

---

## OBS-008

**Date:** 2026-04-01  
**State:** Identity metric established; identity spin remains a higher-order structural signal

**Claim:**  
Identity magnitude admitted a local metric-like interpretation while identity spin resisted reduction to first-order local metric structure.

**Summary:**  
The identity field was decomposed into magnitude and spin components. The magnitude behaved like a local metric quantity, varying smoothly and supporting distance-like interpretation.

In contrast, the spin component exhibited more complex behavior that could not be reduced to a simple local metric. This suggested that spin captured higher-order or nonlocal structural information.

**Operational consequence:**  
Separated identity into metric-like and higher-order components for targeted analysis.

---

## OBS-009

**Date:** 2026-04-01  
**State:** Identity spin established as an operational measure of connection curvature

**Claim:**  
Loop-based holonomy residuals aligned strongly with identity spin, supporting spin as an operational connection-curvature signal for identity transport.

**Summary:**  
By transporting identity around small loops, holonomy residuals were computed and compared to the spin field. A strong alignment was observed, indicating that spin reflects curvature in the identity connection.

This provided an operational interpretation of spin as a measure of connection curvature, linking local identity behavior to global transport properties.

**Operational consequence:**  
Enabled use of identity spin as a proxy for curvature in transport analyses.

---

## OBS-010

**Date:** 2026-04-01  
**State:** Holonomy stabilizes as the invariant obstruction object; spin is reclassified as a chart-dependent local proxy

**Claim:**  
Coordinate-invariance testing supported holonomy as the intrinsic obstruction object and demoted the current spin field to a chart-sensitive proxy.

**Summary:**  
Testing under coordinate transformations showed that holonomy remained invariant, while the spin field varied with the chosen chart. This distinguished holonomy as the intrinsic object capturing obstruction.

Spin was therefore reinterpreted as a local proxy dependent on representation, while holonomy became the canonical measure of transport obstruction.

**Operational consequence:**  
Shifted focus from spin to holonomy for invariant analysis of obstruction.

---

## OBS-011

**Date:** 2026-04-01  
**State:** Transport-derived local obstruction field established from invariant holonomy

**Claim:**  
A node-local obstruction field derived from invariant holonomy replaced the chart-sensitive spin proxy while preserving strong empirical alignment with it.

**Summary:**  
Holonomy measurements were localized to define a node-level obstruction field. This field retained the empirical patterns previously observed in spin while being invariant under coordinate changes.

The new obstruction field provided a stable basis for analyzing transport difficulty and structural incompatibility across the manifold.

**Operational consequence:**  
Introduced an invariant obstruction field for consistent transport analysis.

---

## OBS-012

**Date:** 2026-04-01  
**State:** Signed local obstruction field established from oriented holonomy

**Claim:**  
A signed, orientation-aware local obstruction field was derived from oriented holonomy and aligned substantially with the historical signed spin signal.

**Summary:**  
By incorporating loop orientation, the obstruction field was extended to include sign information. This produced a richer structure capturing directional asymmetries in transport.

The signed obstruction field aligned with previously observed spin patterns, reinforcing its validity while providing a more principled foundation.

**Operational consequence:**  
Enabled directional analysis of obstruction and asymmetry in transport.

---

## OBS-013

**Date:** 2026-04-03  
**State:** Identity state resolves into angular type plus contextual strength

**Claim:**  
The identity layer compressed cleanly into a robust angular coordinate while structural strength remained better represented by explicit auxiliary axes than by a single radius.

**Summary:**  
Identity states were found to organize naturally around an angular coordinate, capturing type-like distinctions. Attempts to represent strength as a radial coordinate were less effective.

Instead, strength was better captured by separate observables, indicating that identity structure is inherently multi-dimensional.

**Operational consequence:**  
Established angular identity representation with auxiliary axes for strength.

---

## OBS-014

**Date:** 2026-04-04  
**State:** Phase boundary acts as an identity reorganization zone

**Claim:**  
As seam distance decreased, identity angle shifted, dispersed, and roughened under rising transport load and criticality, showing the seam as an identity reorganization zone.

**Summary:**  
Near the phase boundary, identity structure became less stable and more dispersed. Angular identity coordinates shifted and roughened, indicating reorganization under stress.

This behavior was linked to increased transport load and criticality, suggesting that the seam is a region where identity undergoes structural transformation.

**Operational consequence:**  
Identified the seam as a region of identity instability and transformation.

---

## OBS-015

**Date:** 2026-04-04  
**State:** Seam-contact geodesics split into corridor and reorganization route families

**Claim:**  
Seam-contacting geodesics separated into stable seam corridors and reorganization-heavy routes, revealing differentiated routing regimes on the manifold.

**Summary:**  
Geodesics interacting with the seam were found to fall into two distinct families. One group maintained stable, coherent paths along the seam (corridors), while the other exhibited more fragmented, restructuring behavior.

This separation indicated that routing behavior near the seam is not uniform but organized into distinct regimes with different structural characteristics.

**Operational consequence:**  
Established route-family taxonomy for analyzing seam-interacting trajectories.

---

## OBS-016

**Date:** 2026-04-04  
**State:** Seam-contact geodesic families are Lazarus-rich but geometrically split

**Claim:**  
Both major seam-contact geodesic families were Lazarus-rich, showing that seam-mediated routing rather than roughness alone concentrates compression exposure.

**Summary:**  
Analysis of seam-contacting geodesics showed that both stable seam corridors and reorganization-heavy routes exhibit high Lazarus scores. This indicates that compression exposure is not confined to a single routing mode.

Despite similar Lazarus enrichment, the two families remained geometrically distinct in how they traversed the seam region. This demonstrates that compression alone does not determine route structure.

The result separates the roles of compression and geometry: Lazarus identifies seam-relevant regions, while routing behavior determines how those regions are traversed.

**Operational consequence:**  
Supports joint use of Lazarus and route-family classification rather than treating compression as a standalone discriminator.

---

## OBS-017

**Date:** 2026-04-04  
**State:** Stable seam corridors are the highest-response geodesic family

**Claim:**  
Stable seam corridors were identified as the seam-contact family with the highest average response exposure.

**Summary:**  
Response-field measurements along geodesics revealed that stable seam corridors accumulate higher average response than other seam-contacting paths. This distinguishes them as the most response-aligned routes.

This alignment is not incidental but systematic across sampled paths, indicating that corridor routes follow regions of elevated response structure more consistently than reorganization-heavy paths.

The result links routing coherence with response exposure, reinforcing the structural significance of the corridor family.

**Operational consequence:**  
Prioritizes corridor paths for response-aligned analyses and downstream operator studies.

---

## OBS-018

**Date:** 2026-04-04  
**State:** Seam-adjacent Lazarus dynamics persist across scale

**Claim:**  
Above the smallest regime, high-Lazarus dynamics remained seam-adjacent, more seam-crossing, and more transition-prone across scales.

**Summary:**  
Scaling analysis showed that the association between high Lazarus values and seam adjacency persists beyond local neighborhoods. This indicates that the relationship is not a small-scale artifact.

High-Lazarus paths continued to exhibit increased seam crossing and transition likelihood as scale increased. The effect remained stable under broader geodesic sampling.

This establishes Lazarus as a scale-robust indicator of seam-mediated dynamics.

**Operational consequence:**  
Justifies using Lazarus as a consistent diagnostic across different probing scales.

---

## OBS-019

**Date:** 2026-04-05  
**State:** Stable seam corridors emerge as the dominant privileged family at large probe scale

**Claim:**  
As sampled shortest-path scale increased, the route-family taxonomy stabilized and stable seam corridors became the dominant privileged family.

**Summary:**  
At larger geodesic scales, the distinction between route families became clearer and more stable. Stable seam corridors increasingly dominated among high-quality paths.

This dominance reflects both structural coherence and favorable alignment with manifold geometry at scale. Reorganization-heavy routes remained present but less privileged under global sampling.

The result indicates that corridor behavior is not only locally coherent but also globally favored.

**Operational consequence:**  
Supports using corridor routes as canonical representatives of seam-mediated dynamics at scale.

---

## OBS-020

**Date:** 2026-04-04  
**State:** Stable seam corridors are the most eigenvector-aligned route family

**Claim:**  
Stable seam corridors aligned most strongly with both Fisher and response principal directions, especially the response eigenvector field.

**Summary:**  
Comparison between geodesic directions and local principal directions showed that stable seam corridors align closely with dominant eigenvectors of both the Fisher metric and the response tensor.

This alignment was strongest with the response eigenvector, indicating that corridor paths follow preferred directions induced by the response field.

The result connects routing behavior with local linear structure, providing a geometric basis for corridor coherence.

**Operational consequence:**  
Enables interpretation of corridor paths as directionally guided by local response structure.

---

## OBS-021

**Date:** 2026-04-05  
**State:** Seam-contact families share a hub substrate but differ in traversal coherence

**Claim:**  
Stable seam corridors and reorganization-heavy routes used the same seam/Lazarus/critical-rich hub ecology but differed in how coherently they traversed it.

**Summary:**  
Node-level analysis showed that both route families pass through similar high-importance regions characterized by seam proximity, high Lazarus scores, and criticality.

However, their traversal patterns differed: corridor paths maintained continuity through these regions, while reorganization-heavy paths fragmented their traversal.

This indicates that the distinction between families lies in traversal dynamics rather than node selection.

**Operational consequence:**  
Motivates separating node-level importance from path-level coherence in analyses.

---

## OBS-022

**Date:** 2026-04-05  
**State:** Stable seam corridors are the coherent seam-residency mode

**Claim:**  
Reorganization-heavy paths were more seam-immersed, but stable seam corridors traversed the seam region more coherently with fewer fragmented seam episodes.

**Summary:**  
Residency analysis showed that reorganization-heavy paths spend more total time near the seam but do so in a fragmented manner.

In contrast, stable seam corridors maintain longer continuous segments within the seam region, indicating coherent residency.

This distinguishes intensity of contact from quality of traversal, refining the interpretation of seam engagement.

**Operational consequence:**  
Introduces seam-residency coherence as a key metric for route classification.

---

## OBS-023

**Date:** 2026-04-07  
**State:** Transport-aware response-field misalignment localizes at the phase seam

**Claim:**  
Under Fisher transport, response principal directions became markedly misaligned near the seam, establishing a seam-localized directional incompatibility field.

**Summary:**  
Transporting response directions along the manifold revealed increasing misalignment as paths approached the seam. This effect was not present in regions far from the boundary.

The misalignment reflects a breakdown in directional consistency under transport, indicating a form of structural incompatibility localized at the seam.

This establishes the seam as a region of directional mismatch rather than only scalar instability.

**Operational consequence:**  
Adds a directional mismatch field as a diagnostic for seam structure.

---

## OBS-024

**Date:** 2026-04-07  
**State:** Seam obstruction is relational, not pointwise; sustained exit relaxes it; families use obstruction hotspots differently

**Claim:**  
Seam obstruction was shown to be fundamentally relational rather than pointwise, to relax after genuine exit, and to be occupied differently by route families.

**Summary:**  
Analysis of obstruction showed that it arises from relationships between neighboring states rather than from isolated node properties.

When paths exited the seam region and remained outside, obstruction levels decreased, indicating relaxation after sustained departure.

Different route families engaged obstruction hotspots in distinct ways, further supporting the relational interpretation.

**Operational consequence:**  
Shifts obstruction analysis from node-based to relational and path-based frameworks.

---

## OBS-025

**Date:** 2026-04-08  
**State:** Seam resolved as a two-field structural regime

**Claim:**  
The seam was identified as a composite regime in which response anisotropy and relational obstruction are both seam-enriched but only weakly aligned node by node.

**Summary:**  
Two distinct fields—response anisotropy and relational obstruction—were both found to intensify near the seam.

However, their spatial overlap was limited, indicating that they represent different structural aspects rather than a single unified field.

This establishes the seam as a composite regime with internal heterogeneity.

**Operational consequence:**  
Requires multi-field analysis for accurate characterization of seam structure.

---

## OBS-026

**Date:** 2026-04-09  
**State:** Route families resolved by two-field seam occupancy

**Claim:**  
Route families were differentiated by how they occupy the seam’s anisotropy and relational-obstruction hotspot structure.

**Summary:**  
Occupancy analysis showed that different route families preferentially inhabit different combinations of anisotropy and obstruction hotspots.

Stable seam corridors exhibited strong residency in both field types, while other families showed more selective or transient engagement.

This links route classification directly to field structure within the seam.

**Operational consequence:**  
Enables classification of routes based on multi-field occupancy patterns.

---

## OBS-027

**Date:** 2026-04-09  
**State:** Seam regime unified as a canonical multi-field observatory object

**Claim:**  
The seam was synthesized as a multi-field structural regime and family-selective residency landscape.

**Summary:**  
Results from multiple prior observations were consolidated into a unified description of the seam.

The seam was characterized as a region where multiple fields co-exist and where route families are distinguished by their residency patterns within those fields.

This synthesis replaced fragmented interpretations with a single structured model.

**Operational consequence:**  
Provides a canonical framework for interpreting all seam-related analyses.

---

## OBS-028

**Date:** 2026-04-09  
**State:** Embedding policy clarified; MDS retained as canonical, diffusion reinterpreted as slow-mode diagnostic

**Claim:**  
MDS was retained as the canonical observatory embedding, while diffusion was reinterpreted as a diagnostic of the seam’s slow connectivity mode.

**Summary:**  
Comparison between embedding methods showed that MDS preserves the primary geometric structure required for observatory analyses.

Diffusion-based embeddings were found to highlight slow connectivity modes, particularly around the seam, rather than serving as primary embeddings.

This clarified the role of each method within the observatory.

**Operational consequence:**  
Standardizes MDS as the canonical embedding while retaining diffusion for diagnostic use.

---

## OBS-029

**Date:** 2026-04-09  
**State:** Seam departure resolved into family-specific committed escape channels

**Claim:**  
Committed seam exit was shown to occur through coherent, family-specific directional channels rather than through generic loss of seam contact.

**Summary:**  
Analysis of exit behavior revealed that leaving the seam is not a diffuse process but occurs along structured directional pathways.

These pathways differ by route family, indicating that escape dynamics are organized rather than random.

The result reframes seam exit as a structured routing phenomenon.

**Operational consequence:**  
Supports modeling exit behavior as channel-based rather than threshold-based.

---

## OBS-030

**Date:** 2026-04-09  
**State:** Seam dynamics promoted from field observations to a typed transition algebra

**Claim:**  
Seam dynamics admitted a typed transition description, but the key release structures lived at the level of short compositional motifs rather than single arrows.

**Summary:**  
Transitions between seam states were formalized as typed generators, enabling an algebraic description of dynamics.

Analysis showed that meaningful structure arises from short compositions of transitions rather than isolated steps.

This marks the shift from field-based to algebraic representation of seam dynamics.

**Operational consequence:**  
Introduces a transition-algebra framework for analyzing seam behavior.

---

## OBS-031

**Date:** 2026-04-09  
**State:** Seam dynamics promoted to an empirical proto-groupoid with family-specific partial subalgebras

**Claim:**  
Reduced seam states, named generators, and their partial compositions supported an empirical proto-groupoid with family-specific subalgebras.

**Summary:**  
Seam dynamics were formalized in terms of a reduced symbolic state space and a set of named generators representing observed transitions. These generators did not compose freely but admitted partial composition rules constrained by observed dynamics.

The resulting structure was not a full groupoid but a proto-groupoid, where composition exists only for certain admissible sequences. Within this structure, different route families exhibited distinct subsets of generators and compositions.

This established an algebraic layer in which seam dynamics could be described in terms of structured transitions rather than only geometric or field-based observables.

**Operational consequence:**  
Enabled algebraic analysis of seam dynamics using generator composition and family-specific transition structure.

---

## OBS-032

**Date:** 2026-04-09  
**State:** Proto-groupoid shown to be predominantly directed, with only a small quasi-invertible seam-internal core

**Claim:**  
Seam dynamics were shown to be mostly irreversible, with only a small quasi-invertible shuttle remnant inside the seam.

**Summary:**  
Analysis of generator compositions revealed that most transitions do not admit inverses, indicating a strongly directed structure. Only a limited subset of transitions within the seam exhibited approximate reversibility.

This reversible subset formed a small core of shuttle-like behavior, while the majority of transitions led outward into non-reversible dynamics.

The result clarified that seam dynamics are not symmetric but are dominated by directed flow away from the seam.

**Operational consequence:**  
Refined the algebraic model to distinguish reversible core dynamics from dominant directed behavior.

---

## OBS-033

**Date:** 2026-04-09  
**State:** Seam proto-groupoid decomposed into a small reversible core and a dominant directed escape sector

**Claim:**  
The seam proto-groupoid was explicitly partitioned into a small reversible shuttle core and a dominant directed escape sector.

**Summary:**  
Building on the directed nature of the proto-groupoid, the structure was decomposed into two sectors. The reversible core consisted of quasi-invertible shuttle transitions confined to the seam.

The directed escape sector contained transitions that led away from the seam and did not admit return under observed dynamics.

This decomposition provided a clear structural partition of seam dynamics into core and escape regimes.

**Operational consequence:**  
Enabled sector-based analysis separating reversible seam behavior from directed escape dynamics.

---

## OBS-034

**Date:** 2026-04-09  
**State:** Gateway between reversible core and directed escape identified; core leakage is real and asymmetric

**Claim:**  
A biased gateway from the reversible core into directed escape was identified, with forward crossing occurring more often than return.

**Summary:**  
Transitions between the reversible core and directed escape sector were examined and found to occur through specific compositions acting as a gateway.

This gateway was not symmetric: transitions from core to escape were more frequent than reverse transitions, indicating leakage.

The gateway was characterized compositionally rather than by a single transition, linking it to specific generator sequences.

**Operational consequence:**  
Introduced the concept of a gateway interface between sectors for analyzing boundary crossing behavior.

---

## OBS-035

**Date:** 2026-04-09  
**State:** Gateway prediction attempted; leakage removed; modest launch-side signal recovered at instance level

**Claim:**  
After rejecting leakage-prone predictors, the gateway was found to be weakly but genuinely predictable from launch-side typed motion with modest anisotropy modulation.

**Summary:**  
Initial predictive models appeared highly accurate but were found to rely on downstream information, constituting leakage. After removing such features, predictive performance decreased substantially.

A corrected instance-level predictor using only launch-side information showed modest but real predictive power. The strongest signals came from generator type and local motion structure.

Scalar fields contributed only weakly, with anisotropy providing a secondary effect. No sharp threshold law was identified.

**Operational consequence:**  
Established a leakage-free baseline for gateway prediction and identified typed motion as the primary predictive signal.

---

## OBS-036

**Date:** 2026-04-09  
**State:** Refining the symbolic gateway-state alphabet yields only marginal improvement; coarse state labels were not the main bottleneck

**Claim:**  
Refining the symbolic gateway-state alphabet added almost no predictive power, ruling out coarse symbolic resolution as the main bottleneck.

**Summary:**  
The symbolic state space used in gateway prediction was refined by incorporating local field structure into more detailed state labels.

Comparative modeling showed only negligible improvement over the coarse symbolic representation. This indicated that predictive limitations were not primarily due to insufficient state resolution.

The dominant predictive signals remained unchanged, confirming the robustness of earlier findings.

**Operational consequence:**  
Eliminated symbolic refinement as a primary direction for improving gateway prediction.

---

## OBS-037

**Date:** 2026-04-09  
**State:** One-step history does not strengthen the gateway law; short-memory context is not the missing predictor

**Claim:**  
Adding one-step prehistory did not improve gateway prediction, ruling out a short-memory Markov-style boundary law.

**Summary:**  
A history-aware predictor incorporating one-step prehistory was evaluated after removing leakage-prone features.

The resulting model showed no meaningful improvement over the launch-only predictor. This indicated that short-memory context does not significantly contribute to gateway prediction.

The dominant role of typed local motion persisted, while scalar and history features remained weak.

**Operational consequence:**  
Ruled out one-step memory as a key factor in gateway prediction, narrowing the search for missing structure.

---

## OBS-038

**Date:** 2026-04-09  
**State:** Pooled gateway model shown to hide real family-specific boundary laws, especially for the stable seam corridor

**Claim:**  
The pooled gateway model concealed a materially clearer local boundary law for the stable seam corridor while failing to capture reorganization-heavy under the same predictor class.

**Summary:**  
Gateway prediction models were re-estimated separately for each route family. This revealed that predictive structure differed significantly across families.

Stable seam corridor showed improved predictability and clearer local structure, while reorganization-heavy remained poorly captured by local predictors.

The pooled model had averaged over these differences, masking family-specific behavior.

**Operational consequence:**  
Established the need for family-specific modeling of gateway behavior.

---

## OBS-039

**Date:** 2026-04-09  
**State:** Reorganization-heavy resolved as a path-context law rather than a local gateway law

**Claim:**  
Reorganization-heavy crossing was shown to depend on broader path context rather than on local launch-state structure alone.

**Summary:**  
For the reorganization-heavy family, local predictors performed near chance, indicating lack of local boundary structure.

When broader path-context features were introduced, predictive performance improved substantially. These features included prior sector exposure and recent trajectory structure.

This demonstrated that reorganization-heavy is governed by a context-dependent law rather than a local rule.

**Operational consequence:**  
Introduced path-context modeling as necessary for explaining reorganization-heavy behavior.

---

## OBS-040

**Date:** 2026-04-09  
**State:** Temporal depth promoted to a first-class variable; seam families found to have distinct finite memory horizons

**Claim:**  
Seam families were shown to occupy distinct finite memory regimes: immediate, one-step, and two-step.

**Summary:**  
Predictive models were evaluated across increasing history horizons. Performance gains varied by family.

Branch-exit showed no improvement with added history, indicating immediate behavior. Stable seam corridor improved at one step and then saturated. Reorganization-heavy required deeper context.

This established temporal depth as a measurable and family-specific property.

**Operational consequence:**  
Enabled explicit modeling of memory horizon as a parameter in predictive analysis.

---

## OBS-041

**Date:** 2026-04-10  
**State:** Family-specific memory compression identified; reorganization-heavy shown to be a compressive long-memory regime

**Claim:**  
Seam families were shown to differ not only in memory depth but also in how strongly core and escape motifs compress older history.

**Summary:**  
Analysis of suffix sufficiency showed that certain states act as compression points where older history becomes redundant.

Reorganization-heavy exhibited strong compression through recurring core and escape motifs, while stable seam corridor showed rapid compression consistent with short memory.

Branch-exit showed minimal compression due to limited effective history.

**Operational consequence:**  
Introduced memory compression and forgetting nodes as structural features of seam dynamics.

---

## OBS-042

**Date:** 2026-04-10  
**State:** Canonical seam-family temporal regimes synthesized into one stabilized comparative framework

**Claim:**  
The seam-family arc was synthesized into three canonical temporal regimes: branch-exit, stable seam corridor, and reorganization-heavy.

**Summary:**  
Results from prior observations were consolidated into a single comparative framework describing seam families.

Each family was characterized by its routing behavior, temporal depth, and memory compression profile.

This synthesis provided a unified description of seam dynamics across multiple observational layers.

**Operational consequence:**  
Established a canonical taxonomy for seam families to guide future analysis and documentation.

---

## OBS-043

**Date:** 2026-04-15  
**State:** Response-eigenvector flow established as a seam-engaged dynamical layer on the manifold, with route-family structure resolved at first pass

**Claim:**  
The response eigenvector field induces a real seam-engaged flow picture, and that flow resolves into stable first-pass route families.

**Summary:**  
The dominant eigenvector of the response tensor defines a coherent local direction field in the embedded manifold. Integration of this field over the node graph produces structured trajectories that are not random but exhibit region-dependent directional organization.

Across all tested regimes, flow remains strongly seam-engaged. Relaxed integration reveals longer, outward-routing trajectories while preserving high seam-contact share. This shows that the seam functions not only as a local capture region but also as a launch region for broader routing.

The resulting trajectories admit a stable first-pass route-family decomposition into seam-hugging, release-directed, short-trapped, and mixed paths. Phase-crossing behavior is distributed across these families and is therefore best treated as a path attribute rather than a distinct class.

Seam-bundle mismatch scalars modulate routing without destroying seam engagement. In particular, neighborhood directional mismatch reduces cross-phase release behavior while preserving the presence of the underlying route families.

**Operational consequence:**  
Introduces a response-guided flow layer with route-family structure, enabling trajectory-level analysis of seam engagement, release behavior, and scalar-modulated routing.

**Recovery note:**  
Normalized from extended OBS-043 entry; content consolidated without loss of stated results or distinctions between flow regimes, route families, and scalar modulation.

---

## OBS-044

**Date:** 2026-04-16  
**State:** Continuous response-flow reconstruction established as a seam-preserving smoothing of the OBS-043 dynamical layer, with route-family comparison against the discrete baseline completed

**Claim:**  
Continuous reconstruction preserves the seam-engaged character and route-family structure of response-guided flow, while remaining more conservative than the relaxed discrete baseline in path extent and phase crossing.

**Summary:**  
A continuous response-flow reconstruction was introduced as a local smoothing of the OBS-043 discrete flow using interpolated response eigenvectors in embedded space. The resulting trajectories remain coherent and strongly seam-engaged, confirming that seam-centered flow is not an artifact of graph discretization.

The reconstruction produces smoother paths with stronger average seam adjacency, while reducing total path extent and phase-crossing frequency relative to the relaxed discrete regime. Seam-contact share is preserved exactly, indicating that seam engagement is a robust structural property of the flow.

The first-pass route-family structure—seam-hugging, release-directed, short-trapped, and mixed—survives continuous reconstruction. However, the distribution shifts modestly toward smoother release-directed behavior, and cross-phase motion is damped across all major families.

Termination behavior becomes more structured under the continuous solver. Extended seam-hugging and release-directed paths are primarily limited by support radius, while short-trapped paths are limited by local angular inconsistency, replacing the uniform forward-neighbor exhaustion seen in the discrete baseline.

**Operational consequence:**  
Establishes a seam-preserving continuous reconstruction of response-guided flow with stable route-family structure, enabling separation of robust dynamical features from solver-dependent effects.

**Recovery note:**  
Normalized from extended OBS-044 entry; detailed quantitative comparisons and termination diagnostics preserved in summarized form without altering reported values or conclusions.

---

## OBS-045

**Date:** 2026-04-16  
**State:** Controlled support expansion shown to increase continuous path extent without recovering additional cross-phase release

**Claim:**  
Support expansion increases continuous path extent while preserving seam engagement, but does not recover additional cross-phase release.

**Summary:**  
A controlled support-radius sweep was performed on the continuous response-flow reconstruction introduced in OBS-044. The test varied only the local support radius while keeping the interpolation model family, seed policy, seam-contact threshold, step-size scale, and local consistency logic fixed. This made the experiment a clean single-axis test of whether broader support alone could recover the missing release behavior.

Increasing support radius produced longer trajectories and more steps, confirming that the continuous solver is support-limited in extent. Seam-contact share remained fixed at the OBS-044 level across the entire sweep, while average seam distance increased only modestly. This shows that broader support allows somewhat wider continuation without destroying seam-guided flow.

However, neither overall phase-sign crossing nor release-directed cross-phase share improved under broader support. Route-family shares also remained unchanged. The missing broader release behavior therefore does not appear to be primarily caused by conservative support bounds.

The result narrows the remaining explanation space. Support expansion changes how far the current continuous solver travels, but not the deeper cross-phase release structure it expresses. The remaining deficit is therefore more likely tied to the local interpolation rule or the continuous local reconstruction model itself than to support radius alone.

**Operational consequence:**  
Support-envelope tuning can extend continuous trajectories without compromising seam engagement, but it should not be treated as the primary route for recovering the missing cross-phase release behavior of the continuous solver.

**Recovery note:**  
Normalized from the extended OBS-045 entry; quantitative sweep outcomes and the distinction between extent limitation and release limitation were preserved without adding new interpretation.

---

## OBS-046

**Date:** 2026-04-16  
**State:** Interpolation-model sensitivity established; minimally averaged and non-averaged local steering recover modestly more phase-crossing behavior than the broad averaged continuous baseline

**Claim:**  
Cross-phase release in continuous response-flow reconstruction is modestly sensitive to the interpolation model, with minimally averaged local blending recovering more release behavior than broad averaged interpolation while preserving seam engagement.

**Summary:**  
An interpolation-model sensitivity sweep was performed to test whether the under-recovery of cross-phase release in continuous flow is driven by the local interpolation rule rather than support limitations. The experiment compared broad kNN averaging, narrower averaging, non-averaged steering, and minimally averaged blending within a fixed solver configuration.

Reducing averaging width within the same kNN family increased path extent but did not improve phase-crossing or release behavior, ruling out averaging width alone as the primary cause. A qualitative shift in interpolation model class was required to produce any recovery signal.

Non-averaged steering and minimally averaged blending both increased overall phase-crossing relative to the OBS-044 baseline. The strongest improvement in release-directed cross-phase behavior was obtained with the top2_blend model, which outperformed both the broad averaged baseline and nearest-anchor steering while maintaining identical seam-contact share.

The results establish that interpolation model class directly affects the expression of release behavior. Broad averaged interpolation is now a dominated baseline, while minimal local blending provides the best current seam-preserving compromise, improving release without destabilizing route-family structure.

**Operational consequence:**  
Continuous-flow reconstruction should adopt a minimally averaged local interpolation rule as the new baseline, as support tuning and averaging width alone are insufficient to recover release behavior.

**Recovery note:**  
Normalized from the extended OBS-046 entry; comparative model results and the distinction between averaging width and model-class effects were preserved without modification.

---

## OBS-047

**Date:** 2026-04-16  
**State:** Minimal-blend continuous baseline stabilized; top2_blend selected as the best current seam-preserving continuous reconstruction baseline

**Claim:**  
Minimal local blending (top2_blend) stabilizes as the best current seam-preserving continuous baseline, improving phase crossing and release-directed behavior while maintaining seam engagement.

**Summary:**  
A stabilization sweep was conducted to determine whether the top2_blend interpolation model remains the best continuous reconstruction baseline under modest tuning of support radius and step size. The experiment compared three model families—broad averaged (knn_avg_k8), nearest-anchor, and top2_blend—across a controlled parameter grid.

The legacy broad-averaged baseline remained clearly dominated, with lower phase crossing and weaker release-directed behavior than the alternatives. nearest_anchor continued to perform well on overall phase crossing and trajectory extent but did not match top2_blend on release-directed cross-phase recovery.

The top2_blend model achieved the strongest joint performance, matching the best phase-crossing levels while delivering the highest release-directed cross-phase share, all with unchanged seam-contact share. The optimal configuration was the conservative setting (support_radius_scale = 3.5, step_size_scale = 0.15), indicating that improved performance arises from the interpolation rule rather than more aggressive solver expansion.

The selected baseline preserves an interpretable route-family structure and shifts the solver into a primarily support-limited regime, with reduced reliance on angular-consistency constraints. This marks a transition from exploratory model comparison to a stabilized continuous reconstruction reference.

**Operational consequence:**  
Establishes top2_blend (support_radius_scale = 3.5, step_size_scale = 0.15) as the canonical continuous baseline for response-flow reconstruction, enabling consistent downstream analysis and comparison.

**Recovery note:**  
Normalized from the extended OBS-047 entry; selection criteria, parameter values, and comparative outcomes preserved without modification.

---

## OBS-048

**Date:** 2026-04-16  
**State:** Route-family identity shown to be only weakly pointwise recoverable and better understood as a distributed recoverable object

**Claim:**  
Route-family identity is only weakly recoverable from local observables and is better understood as a distributed object supported across broader path context.

**Summary:**  
A recoverability ladder was constructed to test whether seam-family identity can be inferred from local event-level observables or requires broader distributed context. The experiment evaluated three feature tiers: pointwise local features, local plus neighborhood features, and local plus neighborhood plus short route context.

Pointwise local features yielded weak performance, indicating that family identity is not strongly encoded in local crossing state, anisotropy, relational structure, or distance measures. Adding neighborhood features produced only modest gains, showing that limited distributed local support is insufficient for strong recovery.

Short route-context features produced the largest improvement, confirming that family identity depends on path history. However, even the enriched feature set failed to achieve strong recovery, with accuracy and macro F1 remaining low. This demonstrates that family identity is only partially visible within the current local and short-context observables.

At the family level, stable seam corridor showed high local recoverability, consistent with its role as a local gateway regime. Branch-exit showed moderate improvement with context, while reorganization-heavy remained poorly recoverable across all feature sets. This aligns with earlier results identifying reorganization-heavy as the most path-context-dependent family.

**Operational consequence:**  
Family classification and analysis should not rely on pointwise or short-context features alone; richer distributed representations are required to capture seam-family identity.

**Recovery note:**  
Normalized from the extended OBS-048 entry; recoverability ladder structure and per-family behavior preserved without inference.

---

## OBS-049

**Date:** 2026-04-20  
**State:** First full corpus-Cp observatory closure achieved: complete trajectory coverage, complete pipeline execution, and full TUI inspection surface now operating on the finished Cp manifold

**Claim:**  
The observatory achieves its first full corpus-level closure, with complete trajectory coverage, successful end-to-end pipeline execution, and operational TUI inspection on the finished Cp manifold.

**Summary:**  
The corpus-Cp trajectory manifold has been completed at full resolution (750/750), removing prior gaps in source coverage. This establishes a fully realized manifold as the basis for all downstream observatory layers.

The full canonical pipeline, spanning geometry, phase, operator, and topology stacks, now executes successfully end-to-end on the completed dataset. This confirms that the observatory can transition from raw trajectory generation to fully derived artifact construction without failure.

The TUI inspection surface operates correctly across all available modes and overlays on the completed Cp outputs. This demonstrates that the observatory supports not only generation and derivation, but also coherent interactive inspection of the full manifold.

Together, these results establish the first complete operational closure of the observatory at corpus scale, marking the transition from partial experimental assembly to a fully functioning end-to-end system.

**Operational consequence:**  
Shifts repository focus from closure validation to post-closure work, including pipeline hardening, artifact canonicalization, and refinement of the TUI and manifold representation.

**Recovery note:**  
Normalized from the extended OBS-049 entry; emphasis on closure criteria (coverage, pipeline execution, inspection) preserved without modification.

---

## OBS-050

**Date:** 2026-04-27  
**State:** First predictive seam-coupling test completed; recovery-like roughness escalation is distinguished more by retained seam proximity than by inward seam-motion slope

**Claim:**  
Recovery-like roughness escalation is primarily characterized by retained seam coupling rather than inward seam-distance slope.

**Summary:**  
A first predictive test was conducted to determine whether instability regimes (roughness escalation windows) contain measurable structure distinguishing recovery-like from nonrecovery-like behavior. The initial hypothesis—that recovery corresponds to inward motion toward the seam—was tested and rejected, as recovering windows did not exhibit consistently negative seam-distance slope.

Instead, a stronger and more consistent signal emerged when analyzing seam-band occupancy. Recovery-like windows were far more likely to remain within seam-coupled regions (core and near bands), while nonrecovering windows overwhelmingly occurred in far, decoupled regions. This difference produced a large effect size, with recovery-like windows approximately an order of magnitude more likely to be seam-coupled.

The result shows that productive instability is not defined by local inward motion, but by remaining structurally coupled to the seam during escalation. Seam-distance slope alone is therefore insufficient as a predictive indicator, while seam-band persistence provides a much stronger discriminator.

This establishes the first predictive observatory result: instability posture can be partially classified based on geometric coupling to the seam, separating recovery-like from nonrecovery-like regimes.

**Operational consequence:**  
Shifts predictive analysis from slope-based indicators to seam-coupling persistence metrics, enabling classification of instability regimes based on seam-band occupancy.

**Recovery note:**  
Normalized from the extended OBS-050 entry; predictive comparison (slope vs coupling) and quantitative coupling signal preserved without modification.

---

## OBS-051

**Date:** 2026-04-27  
**State:** First local-divergence test completed; within seam-coupled escalation windows, recovery-like regimes are dynamically more bounded than nonrecovering regimes

**Claim:**  
Within seam-coupled roughness-escalation windows, recovery-like regimes exhibit significantly lower local divergence than nonrecovering regimes.

**Summary:**  
A Lyapunov-like local divergence instrument was introduced to distinguish bounded from explosive behavior within seam-coupled instability windows. Building directly on OBS-050, the analysis restricted attention to escalation windows that remained in seam-coupled bands, isolating the role of dynamic boundedness from structural coupling.

The local divergence proxy compared how nearby windows in state space separate over time, yielding a quantitative measure of bounded versus explosive evolution. Results showed a strong separation: recovery-like regimes had substantially lower mean local divergence than nonrecovering regimes, indicating more constrained and less explosive behavior within the same seam-coupled context.

This demonstrates that seam coupling alone is not sufficient for recovery-like behavior. Instead, productive instability requires both retained coupling and bounded divergence. Some seam-coupled windows still exhibit highly unstable dynamics, confirming that the seam can host both recoverable and nonrecoverable behavior.

Together with OBS-050, this establishes a two-stage predictive structure: recovery-like instability is characterized by both persistence within seam-coupled regions and lower local divergence during escalation.

**Operational consequence:**  
Introduces a two-stage instability discriminator combining seam-coupling persistence and local divergence, enabling separation of bounded versus explosive regimes within seam-coupled dynamics.

**Recovery note:**  
Normalized from the extended OBS-051 entry; divergence formulation, group separation, and combined interpretation with OBS-050 preserved without modification.

---

## OBS-052

**Date:** 2026-05-01  
**State:** Attractor basin mapping established from recurrence, boundedness, roughness, seam-drift, and recovery landing density

**Claim:**  
Recovery-like coupled windows preferentially land in seam-aligned, recurrent, bounded, low-drift node regions, forming a distinct alignment-sink basin class separate from a seam-far decoupled sink regime.

**Summary:**  
A first node-level attractor-basin analysis was constructed by integrating recurrence, local divergence, roughness, seam-drift, and recovery landing density into a composite attractor score. This extends the observatory from window-level instability analysis (OBS-050, OBS-051) to node-level landing structure, allowing recovery behavior to be studied as a geometric distribution over the manifold.

The analysis identifies two stable basin classes. The alignment-sink class is characterized by higher attractor scores, lower divergence, lower roughness, and higher recovery landing density, and is predominantly seam-aligned. The decoupled-sink class is seam-far, with higher roughness and lower recovery favorability, though still recurrent. This establishes that recovery-like dynamics do not terminate uniformly, but preferentially accumulate in specific structured regions.

Although numerical outputs vary across runs in exact node rankings and class sizes, the higher-level two-class basin structure remains stable. The result is therefore robust at the class level, even if individual node identities are not yet fixed. This supports a structured attractor interpretation rather than a single undifferentiated sink.

Overall, the manifold now exhibits a first attractor-basin layer in which recovery-like bounded dynamics are linked to seam-aligned landing regions, while a contrasting seam-far recurrent regime coexists with weaker recovery association.

**Operational consequence:**  
Introduces a node-level attractor-basin instrument linking recovery behavior to landing geometry, enabling classification of nodes into seam-aligned and seam-far sink regimes based on composite dynamical and geometric features.

**Recovery note:**  
First-pass basin construction; class structure is stable but node-level rankings and exact statistics remain provisional pending run provenance stabilization.

---

## OBS-053

**Date:** 2026-05-01  
**State:** Family-structured external witnessing established in the GPT-5.2 linked-response subset

**Claim:**  
Link-bearing responses in the GPT-5.2 corpus exhibit a non-random, family-structured mode of external witnessing rather than incidental link insertion.

**Summary:**  
A qualitative, response-level analysis was conducted on the linked-response subset across corpora `C`, `Cp`, and `Cp4`, all generated under GPT-5.2. Each response was annotated using a structured taxonomy capturing family mode, packet structure, motifs, and witnessing role. This established a controlled observational basis for evaluating whether links function as structured elements rather than incidental additions.

Three distinct family-level witnessing modes were identified. In `C`, links form geometric externalizations, organizing conceptual space through staged, often atlas-like constructions. In `Cp`, links appear as compact formal-structural packets, tightly aligned with the logical content of the prose. In `Cp4`, links operate as distributed-emergence packets, emphasizing coherence arising from interaction, network formation, and developmental structure.

These patterns demonstrate that links act as external witnesses whose structure reflects the underlying discourse regime. The linked-response layer therefore preserves family distinctions not only in language but in how external conceptual material is selected, organized, and deployed.

**Operational consequence:**  
Establishes a response-level taxonomy for linked outputs and promotes the linked-response layer to a valid observatory surface, enabling structured annotation and cross-family comparison of external witnessing behavior.

**Recovery note:**  
Scope restricted to GPT-5.2 corpus stage; qualitative and annotation-backed; taxonomy and specimen set refined with provenance correction (e.g., removal of Cp:27).

---

## OBS-054

**Date:** 2026-05-01  
**State:** Linked-response taxonomy instrument consolidated at repository level

**Claim:**  
The linked-response study has matured into a stable, repository-level observatory instrument with defined ontology, artifacts, and method.

**Summary:**  
Following the establishment of family-structured external witnessing, the linked-response layer was formalized into a repository-ready instrument. A response-level taxonomy artifact was constructed with one row per linked response, capturing corpus identity, link structure, family mode, subtype, packet architecture, motifs, and interpretive annotations. This converts the stage from qualitative narrative into a structured and reusable dataset.

A controlled vocabulary was stabilized across three family modes—geometric externalization, formal-structural packetization, and distributed-emergence packetization—along with a second layer of subtype labels. These vocabularies define the current ontology of external witnessing and enable consistent classification across the linked-response subset.

Repository-facing documentation was produced to preserve the stage, including taxonomy definitions, annotation protocol, and artifact indexing. Scope constraints were explicitly frozen: the study is limited to GPT-5.2, applies only to the linked-response subset, and remains qualitative and annotation-backed. Provenance discipline was incorporated through explicit correction procedures, including removal of misclassified specimens.

The result is a transition from conversational observation to a durable observatory layer with stable terminology, reproducible method, and preserved artifacts.

**Operational consequence:**  
Promotes the linked-response layer to a reusable research instrument, enabling structured analysis, comparison, and downstream quantitative or cross-layer integration without re-deriving the taxonomy.

**Recovery note:**  
Instrument is stable but provisional; lacks inter-rater validation, full statistical treatment, and cross-model generalization; scope explicitly limited to GPT-5.2 linked-response subset.

---

## OBS-055 — Downstream inherited-family survival control specification

**Date:** 2026-05-07
**State:** Downstream inherited-family survival controls specified

* Question: How should downstream inherited-family survival be tested under route-origin perturbation?
* Method: Defined the downstream validation framework for motif, generator, proto-groupoid, gateway, and canonical-family survival under controlled origin replacement.
* Artifacts: Control specification layer referenced by later OBS-056–OBS-067 studies.
* Result: Established the validation ladder separating provenance, origin sensitivity, decoy replacement, and downstream survival testing.
* Guardrail: Specification study only; no survival claims or downstream robustness results established.

---

## OBS-056 — Route-class provenance chain stabilization

**Date:** 2026-05-07
**State:** Route-class provenance stabilization established for C and Cp

* Question: Where does route_class originate, and how does it propagate downstream?
* Method: Traced provenance from OBS-022 scene-route metadata through OBS-030 route-class assignment into downstream family/gateway artifacts.
* Artifacts: Provenance-trace audit outputs over C and Cp artifact stores.
* Result: Established that downstream route-class structure is inherited from a small selected origin substrate (8 / 8 / 8 paths per class in C and Cp).
* Guardrail: Provenance audit only; did not establish route-class validity, robustness, or downstream survival.

---

## OBS-057 — Leave-one-out origin-substrate stability

**Date:** 2026-05-07
**State:** Route-class origin substrate leave-one-out stability established for C and Cp

* Question: Is the small OBS-022 / OBS-030 origin substrate dominated by single selected paths?
* Method: Removed one selected origin path at a time and recomputed OBS-030 transition-signature distributions.
* Artifacts: obs057_origin_path_profile.csv, obs057_leave_one_out_drift.csv, transition-signature and transition-distribution summaries.
* Result: Dominant transition signatures remained stable under leave-one-out deletion across all classes in C and Cp.
* Guardrail: Origin-level only; did not establish matched-decoy robustness, downstream survival, or cross-corpus equivalence.

---

## OBS-058 — Profile-exact matched-decoy survival

**Date:** 2026-05-07
**State:** Profile-exact matched-decoy route-origin survival established across C and Cp

* Question: Are selected origin paths uniquely necessary, or replaceable by profile-equivalent decoys?
* Method: Replaced selected paths with distinct non-selected profile-exact decoys matched on path-profile features.
* Artifacts: obs058_matched_decoy_pairs.csv, baseline/replacement transition signatures, drift summaries.
* Result: All dominant route-origin transition signatures survived full profile-exact replacement across C and Cp.
* Guardrail: Established profile-exact decoy survival only; not arbitrary-decoy or downstream survival.

---

## OBS-059 — Nearest non-exact decoy survival

**Date:** 2026-05-07
**State:** Nearest non-exact decoy dominant-signature survival established across C and Cp

* Question: Do dominant transition signatures survive after excluding profile-exact twins?
* Method: Replaced selected origin paths with nearest eligible non-exact decoys using deterministic nearest-neighbor matching.
* Artifacts: obs059_matched_nonexact_decoy_pairs.csv, replacement drift summaries, transition-distribution reports.
* Result: Dominant top-1 transition signatures survived across all route classes in C and Cp despite nonzero distribution drift.
* Guardrail: Demonstrated dominant-signature survival only; full transition-distribution invariance was not established.

---

## OBS-060 — Rank-k local robustness and ensemble non-exchangeability

**Date:** 2026-05-07
**State:** Non-exact decoy ensemble route-origin controls executed across C and Cp

* Question: Does robustness extend beyond nearest-neighbor replacement into broader non-exact ensembles?
* Method: Tested deterministic rank-k decoys (k=1,2,3,5,10) and broad random non-exact ensemble replacements.
* Artifacts: Rank-k replacement summaries and random ensemble survival-rate tables.
* Result: Deterministic local rank-k replacement preserved dominant transition signatures, while broad random ensembles revealed strong class- and corpus-dependent non-exchangeability.
* Guardrail: Origin-substrate ensemble study only; no downstream motif/proto/gateway survival claims.

---

## OBS-061 — Distance-banded robustness-radius mapping

**Date:** 2026-05-08
**State:** Distance-banded non-exact decoy ensembles executed across C and Cp

* Question: Over what decoy-distance scale do route-origin signatures remain stable?
* Method: Sampled random non-exact decoys from ranked candidate bands: 1–10, 11–50, 51–250, 251–1000, and all non-exact candidates.
* Artifacts: Distance-banded survival-rate summaries and drift metrics across 250 iterations per band.
* Result: All route classes survived through rank 11–50, but broader bands showed class- and corpus-specific robustness-radius decay.
* Guardrail: Origin-level robustness-radius map only; downstream symbolic layers not yet tested.

---

## OBS-062 — Downstream motif/generator survival controls

**Date:** 2026-05-08
**State:** Downstream motif-generator survival controls executed across C and Cp

* Question: Do downstream motif and generator layers survive route-origin decoy replacement?
* Method: Rebuilt motifs, completed generators, and generator compositions under controlled non-exact decoy replacement.
* Artifacts: Motif-class, completed-generator, and generator-composition survival summaries.
* Result: Motif classes were highly robust; completed generators were partially robust and corpus-sensitive; generator compositions were the most origin-sensitive layer.
* Guardrail: Did not test proto-groupoid, gateway, or canonical-family survival.

---

## OBS-063 — Generator compression sensitivity audit

**Date:** 2026-05-08
**State:** Generator compression sensitivity audit executed across C and Cp

* Question: Why are completed-generator and composition layers more fragile than motif classes?
* Method: Audited the symbolic compression chain: motif class → reduced word → generator → completed generator → composition.
* Artifacts: Margin audits, reduced-word change diagnostics, anchor-composition analyses.
* Result: Sensitivity arose from low-margin generator tie breaks, coarse motif compression, and composition-level algebraic anchors.
* Guardrail: Diagnostic audit only; did not revise generator rules or introduce soft assignments.

---

## OBS-064 — Proto-groupoid symbolic trace cache

**Date:** 2026-05-08
**State:** Proto-groupoid symbolic trace cache built for C and Cp

* Question: Can symbolic extraction be separated from decoy-control aggregation for efficient downstream studies?
* Method: Built reusable cached symbolic traces from scene routes through motifs, generators, compositions, proto-edges, and proto-relations.
* Artifacts: Cached per-path symbolic tables covering generator, composition, proto-edge, proto-relation, and sector-relation layers.
* Result: Reproduced OBS-062/063 symbolic baselines while exposing explicit proto-groupoid-ready structures for downstream controls.
* Guardrail: Infrastructure/cache study only; did not test proto-groupoid survival.

---

## OBS-065 — Proto-groupoid decoy survival controls

**Date:** 2026-05-09
**State:** Proto-groupoid decoy survival controls completed for C and Cp

* Question: Do proto-groupoid signatures survive route-origin decoy replacement?
* Method: Evaluated survival across generator, composition, proto-edge, proto-sector-edge, proto-relation, and proto-sector-relation layers under non-exact decoys.
* Artifacts: Proto survival summaries, cross-layer failure modes, proto-anchor candidate tables.
* Result: Proto-groupoid survival was layer-specific and finite; sector-level algebra was often more robust than reduced-state relations; fine proto-relations were broadly fragile but contained algebraic anchors.
* Guardrail: Did not establish gateway/canonical-family survival or revise generator rules.

---

## OBS-066 — Gateway and canonical-family survival controls

**Date:** 2026-05-09
**State:** Gateway and canonical-family decoy survival controls completed for C and Cp

* Question: Do gateway/canonical-family summaries survive after proto-groupoid recomputation under decoy replacement?
* Method: Evaluated gateway events, gateway relations, canonical-family relations, and anchor-family summaries under the OBS-065 replacement regimes.
* Artifacts: Gateway survival summaries, canonical-family relation summaries, anchor-family diagnostics, cross-layer consequence modes.
* Result: Seam-centered classes often preserved coarse gateway/canonical relations despite fine proto drift, while branch_exit remained relation-order sensitive; broad all-nonexact decoys still degraded many downstream layers.
* Guardrail: Gateway/canonical layers were treated as coarse projections, not independent predictive models or causal mechanisms.

---

## OBS-067 — Proto-to-gateway/canonical survival coupling meta-analysis

**Date:** 2026-05-12  
**State:** Proto-to-gateway/canonical survival coupling meta-analysis completed for C and Cp

- **Question:** How are proto-groupoid survival and downstream gateway/canonical-family survival coupled across replacement regimes?
- **Method:** Joined OBS-065 and OBS-066 regime-level summary artifacts into a coupling ledger over corpus × rank band × replacement class.
- **Artifacts:** `obs067_layer_coupling_table.csv`, `obs067_projection_absorption_modes.csv`, `obs067_proto_sufficiency_table.csv`, `obs067_anchor_transfer_table.csv`, `obs067_cross_corpus_contrast.csv`, and `obs067_proto_gateway_coupling_meta_report.md`, joined from OBS-065/066 summary inputs.
- **Result:** Proto survival was neither necessary nor sufficient for downstream survival; gateway/canonical projection sometimes absorbed fine proto drift, sometimes failed jointly, and anchor-family survival formed a partially independent coupling axis.
- **Guardrail:** Meta-analysis only; did not rerun decoy experiments, recompute symbolic traces, or establish causal gateway mechanisms.

---

## OBS-068 — Proto-groupoid returnability / partial-inverse specification

**Date:** 2026-05-16
**State:** Proto-groupoid returnability and partial-inverse design specified; execution deferred

* Question: Can proto-groupoid “invertibility” be operationalized as bounded-cost returnability to anchor/canonical families rather than strict algebraic reversal?
* Method: Specified a minimax bottleneck-cost returnability framework over reduced symbolic transition graphs, including return bottleneck cost, return path length, residual drift classes, and directional asymmetry diagnostics.
* Artifacts: Proposed outputs include obs068_proto_inverse_edges.csv, obs068_returnability_summary.csv, obs068_proto_inverse_class_summary.csv, and obs068_proto_inverse_report.md.
* Result: Defined a projection-aware proto-inverse framework in which symbolic moves are evaluated by bounded-cost returnability to canonical/anchor-family equivalence classes rather than exact state reversal.
* Guardrail: Specification-only study; no returnability execution performed. Returnability is scoped to observed reduced symbolic artifact graphs and declared edge-cost estimators, not the full latent transition space of the model.

---

## OBS-069 — Scale-space observable diffusion pilot

**Date:** 2026-05-16
**State:** Canonical-MDS scale-space observable diffusion pilot executed; Fisher/geodesic distance recovery and full multiscale geometry recomputation deferred

* Question: Which node-level observable salience structures persist under graph diffusion, and which reorganize as local texture?
* Method: Joined canonical FIM, MDS, curvature, and response-operator outputs into a 75-node observable table keyed by (r, alpha); built a self-tuning kNN graph from canonical MDS-coordinate distances; applied log/robust scaling to heavy-tailed fields; excluded unstable scalar curvature; and diffused observables over an 8-step geometric scale ladder.
* Artifacts: outputs/obs069_inputs/obs069_canonical_node_table_pilot.csv, obs069_canonical_node_table_pilot_log_robust_scaled.csv, obs069_mds_distance_pilot.csv, outputs/obs069_scale_space_canonical_mds_pilot_log_robust_no_curvature/obs069_scale_ladder.csv, obs069_diffused_observables_t*.csv, obs069_observable_drift_summary.csv, obs069_topk_persistence_summary.csv, obs069_top10_nodes_by_scale.csv, and obs069_scale_space_report.md.
* Result: Observable variance decreased monotonically across scale while a seam-centered salience core persisted. Base top nodes node_0058 and node_0057 at r=0.25, alpha≈0.133–0.141, distance_to_seam=0, and high Lazarus/response strength remained top-ranked at t=10; nearby seam nodes at r=0.20 were absorbed into the persistent core while less seam-local or low-Lazarus nodes dropped out.
* Quantitative summary: Final top-k Jaccard vs base was 0.428571 (top-5), 0.666667 (top-10), and 0.37931 (top-20). Mean observable variance decreased from 2.3144 at t=0.1 to 0.195112 at t=10.
* Guardrail: Pilot graph uses canonical MDS-coordinate distance rather than the original Fisher/geodesic dissimilarity matrix. Scalar curvature was excluded from the interpretable run due to instability/outlier dominance. OBS-069 does not yet recompute FIM geometry, seams, attractors, Lazarus fields, or symbolic route structures at each scale.

---

## OBS-070 — Cp2 full-grid freeze-inactivity validation

**Date:** 2026-05-17  
**State:** Cp2 `full_v2` campaign completed and validated; freeze macrostate inactive across full grid.

- **Question:** Was the Cp2 smoke-test NaN behavior a localized pipeline failure, or a full-campaign measurement result caused by absence of freeze-state variation?
- **Method:** Completed the Cp2 `full_v2` campaign over the full 5 × 15 × 10 grid, validated row/job/trajectory consistency, and summarized freeze/entropy diagnostics by `(r, alpha)`.
- **Artifacts:** `outputs/corpora/Cp2/campaigns/full_v2/index.csv`, `trajectories/*.npz`, manifest/progress logs, and `outputs/corpora/Cp2/campaigns/full_v2/cp2_full_v2_validation_summary.csv`.
- **Result:** The campaign completed all `750 / 750` planned jobs with `750` unique index rows, `750` trajectory files, and `0` failures. Across all 750 runs, `piF_mean = 0.0`; consequently `corr0`, `best_corr`, and `delta_r2_freeze` are undefined/NaN for all rows, while entropy-derived summaries remain finite.
- **Interpretation:** The original Cp2 NaNs were not residual pipeline errors after patching. They reflect that the current freeze macrostate observable is inactive for Cp2 across the tested grid. Cp2 therefore suppresses the freeze-coupling measurement channel under the current corpus/operator and parameterization.
- **Guardrail:** This does not imply Cp2 has no structure, no phase behavior, or no meaningful geometry. It means the specific freeze macrostate observable is inactive under the current definition. Cp2 must be compared to C/Cp using entropy geometry, transition alternatives, route structure, response fields, scale-space behavior, or revised macrostate diagnostics rather than freeze-coupling metrics alone.

---

## OBS-071 — C vs Cp2 scoped observatory-chain comparison

**Date:** 2026-05-27
**State:** C vs Cp2 comparison completed over registry-visible observatory-chain artifacts; OBS-050 replicated qualitatively, OBS-051 bounded-recovery direction preserved with corpus-dependent band structure

* Question: Does the scoped Cp2 entropy-geometry observatory chain reproduce the structural seam-coupling and local bounded-recovery patterns previously observed in the canonical C observatory chain?
* Method: Added experiments/studies/compare_corpus_observatory_chain.py as a file-first comparison script over already-produced artifacts, comparing C outputs/ against Cp2 outputs/corpora/Cp2/campaigns/full_v2/pipeline/ at scale 100000 without recomputing geometry, paths, families, coupling, or divergence.
* Artifacts: outputs/comparisons/C_vs_Cp2_observatory_chain/corpus_root_manifest.csv, obs050_coupling_comparison.csv, obs051_banded_comparison.csv, family_substrate_comparison.csv, obs028c_seam_bundle_comparison.csv, node_field_comparison.csv, and comparison_summary.md; artifact availability check found 38 checked artifacts and 0 missing.
* Result: OBS-050 qualitatively replicated: recovering roughness-escalation segments were more often seam-coupled than nonrecovering segments in both C and Cp2. The contrast was weaker in Cp2 because nonrecovering segments were also more often seam-coupled.
* Result: OBS-051 bounded-recovery direction was preserved but band-localized in Cp2. C showed a broad bounded-recovery signal across all/core/near coupled bands, while Cp2 showed a strong core-band signal and weak or nearly neutral near-band behavior.
* Guardrail: Corpus/root-specific artifact comparison only. OBS-050 is the stronger canonical replication result; OBS-051 remains provisional because its expression is seam-band dependent. Cp2 near-band boundedness should not be overclaimed, and the comparison does not imply a universal claim about all corpora or model modes.

---

## OBS-072 — Cp2 nonrecovering seam-drift diagnostic

**Date:** 2026-05-27
**State:** Cp2 nonrecovering seam-coupled baseline drift isolated as a localized false-recovery compression mode; C0_instant control confirms the effect is Cp2-specific relative to the current C-like controls

* Question: Why are Cp2 nonrecovering segments more often seam-coupled than C-like controls, and is the drift broad or localized by route family, seam band, posture, or grid locus?
* Method: Added experiments/studies/obs072_cp2_nonrecovering_seam_drift.py to compare registry-visible OBS-050/051 artifacts plus path-node diagnostics, family assignments, Lazarus scores, criticality surfaces, and response-operator outputs between C0_instant and Cp2 at scale 100000; v2 enriched OBS-050 segments using path_id + center_step joins into path_node_diagnostics.csv.
* Artifacts: outputs/obs072_C0_instant_vs_Cp2_nonrecovering_seam_drift/ including comparison summaries and enriched segment diagnostics; upstream inputs included structural_coupling_segments.csv, path_node_diagnostics.csv, path_family_assignments.csv, scene_nodes.csv, lazarus_scores.csv, criticality_surface.csv, and response_operator_nodes.csv.
* Result: C0_instant reproduced the canonical C-like OBS-050 baseline, while Cp2 preserved an elevated nonrecovering seam-coupled baseline (0.082165 vs 0.037701). The Cp2 excess was highly localized: approximately 90.5% of Cp2 nonrecovering coupled segments belonged to off_seam_reorganizing | near | compression near r=0.2, alpha≈0.1329.
* Result: Cp2 nonrecovering coupled segments were more seam-adjacent, more Lazarus-loaded, and more critical than decoupled counterparts, but not response-strength elevated. The profile matched a localized compressive near-seam “false-recovery” mode rather than broad bounded recovery.
* Guardrail: OBS-072 is a strong diagnostic result but the interpretation remains provisional. It does not determine whether the underlying cause is tokenizer behavior, embedding geometry, corpus entropy, or response-generation regime. The comparison is corpus/root-specific (C0_instant vs Cp2) and should be read alongside OBS-071 and future OBS-073 controls.

---

## OBS-073 — Continuous-field groupoid reduction

**Date:** 2026-06-01
**State:** Continuous-field reduction of the OBS-072 Cp2 false-recovery locus established under seam/grid blinding and label-shuffle null control; broader symbolic route-family recovery remains supported but shortcut-sensitive; cross-corpus recovery-channel transfer remains provisional

* Question: Can symbolic/proto-groupoid route classes be reduced to continuous field geometry, and does the OBS-072 Cp2 false-recovery compression locus remain separable after seam/grid blinding and shuffled-label null controls?
* Method: Built path-level continuous-field feature tables from existing observatory-chain artifacts (path_node_diagnostics, path_diagnostics, path_family_assignments, OBS-050 structural-coupling segments, OBS-051 divergence summaries) for C0_instant and Cp2 at scale 100000. Reduced node trajectories into summary/path-shape features and evaluated symbolic targets using random-forest probes, cross-corpus transfer tests, seam/grid shortcut-removal variants, and label-shuffle null controls.
* Artifacts: outputs/obs073_continuous_field_groupoid_reduction_v5_full/ including obs073_feature_table.csv, obs073_target_manifest.csv, obs073_feature_manifest.csv, obs073_model_scores.csv, obs073_feature_importance_gini.csv, obs073_feature_importance_permutation.csv, obs073_confusion_matrices.csv, obs073_label_shuffle_summary.csv, obs073_label_shuffle_runs.csv, and obs073_summary.md.
* Result: The OBS-072 Cp2 false-recovery compression locus remained highly separable from true bounded recovery under full seam/grid blinding: balanced_accuracy = 0.991727, macro_F1 = 0.9852, n_rows = 8851, with shuffled-label null controls collapsing to chance (shuffle BA mean ≈ 0.499, empirical p = 0.01).
* Result: The strongest blinded features were dominated by continuous dynamical quantities rather than direct seam/location proxies: criticality_last_minus_first, holonomy/obstruction statistics, angular path deformation, and field-flow displacement features. This supports a field-dynamical interpretation of the Cp2 false-recovery mode rather than a pure seam-distance or hotspot shortcut.
* Result: Broader symbolic reductions showed a layered profile. outcome_group remained strongly portable under blinding; coupling_class and path_family retained meaningful but weaker cross-corpus transfer; recovery_channel remained highly corpus-conditioned despite strong within-corpus separability.
* Guardrail: OBS-073 does not claim universal reducibility of all symbolic classes to continuous fields. Results are model-specific, corpus-specific, artifact-root-specific, and within-Cp2 for the strongest OBS-072 locus result. Cross-corpus recovery-channel transfer remained poor, so recovery-channel labels should not yet be treated as universal groupoid generators.

---

## OBS-074 — Lexical substrate / field-geometry bridge

**Date:** 2026-06-04
**State:** Corpus-level lexical-control bridge established for C / Cp / Cp2; continuous-field separability survives corpus lexical controls and seam/grid blinding; path-level lexical control remains unavailable because current corpus JSON artifacts do not resolve to path IDs; Cp3 reserved for incomplete-response / missing-step degeneracy analysis

* Question: Do corpus-level lexical fingerprints explain the continuous-field separability observed in OBS-050–OBS-073, and does field-geometric signal survive lexical controls, seam blinding, and grid-location blinding?
* Method: Compared three predictor families across C, Cp, and Cp2: (1) field-only observatory features, including seam/grid-blinded variants; (2) corpus-level lexical fingerprints computed from corpus JSON response text; and (3) combined lexical-plus-field models. Conducted a lexical join audit to determine whether path-level response text could be aligned to observatory path IDs.
* Artifacts: obs074_summary.md, obs074_model_scores.csv, obs074_lexical_vs_field_read.csv, obs074_lexical_join_audit.csv, obs074_feature_importance_permutation.csv, plus bridge outputs under outputs/comparisons/obs074_lexical_field_bridge/C_Cp_Cp2_v3_smoke/.
* Result: Corpus lexical fingerprints alone were consistently weaker than field geometry. Representative balanced accuracies (lexical_only vs field_no_direct_seam_no_grid) were: path_family (0.3068 vs 0.8257), coupling_class (0.5102 vs 0.8707), outcome_group (0.5462 vs 0.9246), coupled_outcome_group (0.6172 vs 0.9524), and recovery_channel_structural (0.7587 vs 0.9877). Adding lexical fingerprints did not eliminate the field signal.
* Result: Blinded field-plus-lexical models remained strong after removing direct seam/proximity features and absolute grid-location features. Permutation importance remained dominated by field observables such as signed phase, path-angle dynamics, sector-change measures, and criticality-flow statistics.
* Result: The lexical path join audit reported zero overlap (lexical_path_join_mode = none, lexical_path_overlap_rows = 0). Current corpus JSON artifacts cannot map individual response texts to the path IDs used by the field observatory chain, preventing meaningful path-level lexical controls.
* Interpretation: OBS-074 narrows, but does not eliminate, the lexical-confound critique. The strongest supported statement is that continuous-field observables retain label-relevant structure beyond corpus-level lexical fingerprints. The stronger path-level lexical-control claim remains untested.
* Relationship to OBS-073: OBS-073 established recoverability of symbolic/proto-groupoid labels from continuous field geometry. OBS-074 adds a corpus-level lexical-control layer and finds that field separability survives lexical controls, seam blinding, and grid-location blinding.
* Guardrail: This is a corpus-level lexical-control study only. Lexical fingerprints may partly act as corpus-regime proxies rather than mechanistic linguistic explanations. No tokenizer-level, embedding-level, or path-resolved lexical claims are made. Path-level lexical confounds remain open pending response-text provenance linked to observatory path IDs.
* Next Step: Build path-resolved response-text provenance and rerun lexical-only and lexical-plus-field controls at path level. Treat Cp3 separately as an incomplete-response / missing-step degeneracy study rather than folding it into OBS-074.

---

## OBS-075 — Cp3 directional asymmetry

**Date:** 2026-06-06
**State:** Directional-transfer asymmetry established for Cp3 against Cp2; broad-boundary/noisy-Cp3 interpretation weakened but not eliminated; endpoint/velocity ablation required and delegated to OBS-075b

* Question: Does Cp3 behave like a generic noisy or broad-boundary training corpus, or does it show corpus-pair-specific directional transfer asymmetry, especially against Cp2 and especially for coupled/recovery targets?
* Method: Compared existing OBS-073 transfer-score artifacts for Cp2 ↔ Cp3 and Cp ↔ Cp3 without recomputing geometry, labels, feature tables, or predictions. Directional asymmetry was defined as BA(Cp3 → B) - BA(B → Cp3), and Cp2-specificity was computed as the Cp3/Cp2 asymmetry minus the Cp3/Cp asymmetry.
* Artifacts: outputs/comparisons/obs073_Cp2_vs_Cp3_v5_smoke/obs073_model_scores.csv, outputs/comparisons/obs073_Cp_vs_Cp3_v5_smoke/obs073_model_scores.csv, with outputs/comparisons/obs073_C_vs_Cp3_v5_smoke/obs073_summary.md used as secondary audit context.
* Result: Cp3 transfer asymmetry was strongest against Cp2 for coupled/recovery targets. Examples included recovery_channel_no_grid with Cp2 specificity 0.6422, recovery_channel_structural with 0.6165, coupled_outcome_group with 0.5963, and coupled_outcome_group_no_grid with 0.5936.
* Result: The anti-shortcut no_direct_seam_no_grid slice preserved positive Cp2 specificity for coupled/recovery targets, but at reduced magnitude: recovery_channel_no_direct_seam_no_grid retained specificity 0.2741, and coupled_outcome_group_no_direct_seam_no_grid retained 0.2119.
* Result: The asymmetry was target-specific, not global. coupling_class was a counterexample, with negative Cp3→Cp2 asymmetry across full, no-grid, and no-direct-seam/no-grid variants.
* Interpretation: OBS-075 weakens the simplest noisy-Cp3 or broad-boundary explanation because Cp3 does not transfer equivalently well to Cp. The strongest current reading is that Cp3’s transfer advantage is corpus-pair-specific and concentrated where coupled unresolved dynamics meet Cp2-style recovery structure.
* Guardrail: OBS-075 is a comparison of existing OBS-073 artifacts, not a recomputation. It establishes directional transfer asymmetry, not causal mechanism. The conservative seam/grid-blinded slice still permits endpoint, velocity, path-length, tortuosity, turning, and last_minus_first proxies.
* Next Step: OBS-075b must rerun the transfer tests after removing endpoint/velocity-like proxies and should compare random forests against lower-complexity models such as logistic regression, shallow random forests, and shallow decision trees.

---

## OBS-076 — Observable scale-space

**Date:** 2026-06-10
**State:** Observable scale-space established as a multiscale structural-field layer; Cp2/Cp3 factorization is classifier-visible from dynamic fields alone

* Question: Do PAM observable structures persist, migrate, split, merge, or factorize when viewed across diffusion scale rather than as a single fixed manifold slice?
* Method: Implemented a four-stage scale-space branch: OBS-076a diffused observable fields over a graph substrate, OBS-076b rebuilt observable-space geometry at each scale, OBS-076c tracked named structural supports across scale, and OBS-076d tested scale-conditioned Cp2/Cp3 separability using dynamic fields under a matched shared-14 observable contract.
* Artifacts: OBS-076a outputs include obs076a_diffusion_bundle.npz, obs076a_scale_ladder.csv, obs076a_observable_drift_summary.csv, and obs076a_topk_persistence_summary.csv; OBS-076b outputs include obs076b_node_geometry_by_scale.csv and obs076b_scale_geometry_summary.csv; OBS-076c outputs include obs076c_object_membership_by_scale.csv, obs076c_object_persistence.csv, and obs076c_object_overlap_by_scale.csv; OBS-076d outputs include dynamic-field separability summaries for Cp2/Cp3.
* Result: C preserved and coalesced high-energy support across scale, while Cp3 showed stronger high-energy support migration. Under the shared-14 substrate, C retained higher final-scale persistence (top5 Jaccard = 0.666667, top10 Jaccard = 0.666667) than Cp3 (top5 Jaccard = 0, top10 Jaccard = 0.111111).
* Result: Rebuilt observable geometry showed C as more persistent than Cp3, while Cp3 reorganized faster without collapsing into noise. Cp3 retained moderate phase coherence despite faster pairwise, embedding, energy, density, and seam-proxy reorganization.
* Result: Structural-object tracking showed a three-way pattern: C coalesced energy ≈ response ≈ Frobenius ≈ Lazarus; Cp2 factorized with energy ≈ negative phase and response ≈ Lazarus; Cp3 factorized differently with energy ≈ positive phase, response/Frobenius/Lazarus co-located but separate from energy, and seam ≈ negative phase.
* Result: Cp2 and Cp3 were nearly perfectly separable using dynamic diffused fields alone. In OBS-076d, all_dynamic_only reached BA = 0.9867 at t = 0.100 and BA = 1.0000 from t ≥ 0.373, with final-scale axes dominated by response tensor, coupling, phase, and Lazarus-gradient features.
* Interpretation: OBS-076 establishes observable scale-space as a provisional observatory layer. PAM structures are not only located on a single manifold slice; they exhibit multiscale persistence, migration, coalescence, and factorization.
* Guardrail: OBS-076 does not establish physical skyrmions, topologically protected semantic tubes, winding numbers, formal singularities, path-label transfer asymmetry, linguistic syntax attribution, or hidden-state topology. Pinch-point geometry, intrinsic dimension, path-label projection, and path/text interpretation remain unimplemented.
* Next Step: OBS-077 should begin from the OBS-076 scale-space stack with pinch-point geometry, intrinsic-dimension diagnostics, path-label projection, and later text/provenance joins.

---

## OBS-077 — Scale-space transition geometry

**Date:** 2026-06-10
**State:** Scale-space transition geometry connected to path-label cohorts and coupled-window divergence structure

* Question: When scale-space structural supports persist, move, split, merge, or reassign support across diffusion scale, which path populations participate in those transitions, and what local coupled-window dynamics do they carry?
* Method: Built a three-stage interpretation layer on top of OBS-076: OBS-077a detected support-transition and pinch-point geometry, OBS-077b projected path-label populations onto supports and transition cohorts, and OBS-077c joined those cohorts to OBS-051 coupled-window divergence/boundedness diagnostics.
* Artifacts: OBS-077a outputs under obs077a_pinch_point_geometry_shared14_mds_pilot_v2; OBS-077b outputs under obs077b_path_label_projection_shared14_mds_pilot; OBS-077c outputs under obs077c_window_coupling_bridge_shared14_mds_pilot_v2; cross-corpus summaries include obs077b_C_Cp2_Cp3_path_projection_summary.md and obs077c_C_Cp2_Cp3_window_divergence_bridge_summary.md; project docs include docs/05_project/077b_path_label_projection.md and docs/05_project/077c_window_local_divergence_bridge.md.
* Result (OBS-077a): Transition geometry revealed distinct support-transition modes: C: coalesced ridge deformation; Cp2: factor relocation; Cp3: support reassignment. C exhibited late support-stable deformation, Cp2 showed early energy/negative-phase relocation and response/Lazarus reorganization, and Cp3 showed mid-late response/Frobenius transition plus late energy intrinsic-dimension transition.
* Result (OBS-077b): Path-level projection succeeded across all corpora (100,000 path-id overlap in each corpus). Transition cohorts recruited distinct path populations: C: recovery/stability recruitment; Cp2: response-ridge recovery sorting; Cp3: nonrecovering off-seam energy settlement.
* Result (OBS-077c): Joining support cohorts to OBS-051 windows produced a three-way divergence signature: C: bounded recovery recruitment (divergence_z_sum = -0.829 for entrants into density_core 5→6); Cp2: high-divergence recovery sorting (+0.480 for entrants into response_ridge 4→5); Cp3: earlier divergence with later nonrecovering settlement (energy_ridge 6→7 entrants -0.314, while strongest divergence appeared in earlier supports such as coupling_negative 4→5 at +1.437).
* Integrated Finding: The C/Cp2/Cp3 distinction is visible across three connected layers: scale-space transition geometry, path-label occupancy, and local coupled-window divergence/boundedness. The resulting signature is: C = bounded recovery recruitment, Cp2 = high-divergence recovery sorting, Cp3 = earlier divergence, later nonrecovering settlement.
* Relation to OBS-076: OBS-076 established multiscale factorization; OBS-077 explains how those factors transition, which path populations occupy them, and what local dynamical signatures they carry.
* Relation to OBS-075: OBS-077 does not directly explain transfer asymmetry, but provides a plausible substrate: Cp2 localizes recovery-compatible structure in high-divergence response-ridge entry events, while Cp3 localizes nonrecovering structure in late energy-ridge settlement after earlier divergence events.
* Guardrail: OBS-077 is model-specific, corpus-specific, matched-contract-specific, and subset-specific where OBS-051 windows are involved. It does not establish formal attractors, topological defects, generated-text semantics, causal transfer mechanisms, or direct coupled_outcome_group localization.
* Next Step: OBS-078 should test whether the C/Cp2/Cp3 distinction is recoverable directly from OBS-077 mechanistic features, including transition geometry, support overlap/shape metrics, path-label cohort enrichments, and window-local divergence/boundedness contrasts.

---

## OBS-078 — Mechanistic stability signature

**Date:** 2026-06-11
**State:** OBS-077 transition mechanism shown to be classifier-visible, compressible to a low-dimensional local stability signature, and localizable back onto the interpreted object/cohort structure

* Question: Is the OBS-077 transition interpretation merely descriptive, or does it define a compact mechanistic signature that recovers the C/Cp2/Cp3 distinction under strict anti-leakage controls?
* Method: Built a transition-signature feature table from OBS-077 pinch-point geometry, path-label projection, and window-divergence artifacts. OBS-078a tested strict mechanistic classification; OBS-078b performed minimal-feature ablation; OBS-078c localized the resulting minimal signature back onto cases, objects, cohorts, and transitions.
* Artifacts: obs078a_feature_table.csv (168 rows, 111+ columns; C=49, Cp2=60, Cp3=59), classifier outputs under obs078a_mechanistic_signature_classifier_v2, plus OBS-078b ablation outputs and OBS-078c localization artifacts derived from the same feature table.
* Result (OBS-078a): Strong strict-control separability remained after removing global path counts, rank identity, scale identity, object identity, cohort identity, dominant-family identity, and dominant-reason identity. Object-blind performance remained high: full_obs077_signature BA=0.919, paths_plus_windows BA=0.914, window_divergence_only BA=0.880, geometry_only BA=0.864, versus permutation baselines near chance (≈0.33–0.34).
* Result (OBS-078b): The C/Cp2/Cp3 distinction compressed to a minimal three-feature stability signature: mean_lambda_local_mean, mean_delta_d_mean, and bounded_share_mean. This window_means_only panel achieved BA=0.866 using only three features. Expanded six-feature and nine-feature panels reached BA=0.912 and BA=0.925, respectively.
* Result (OBS-078c): The minimal signature localized back onto the interpreted OBS-077 structure. Global case signatures separated C from Cp2/Cp3: C showed low instability and high bounded stability (instability_signature_z = -1.316, bounded_stability_signature_z = +2.655), while Cp2 (+0.477, -0.984) and Cp3 (+0.608, -1.204) occupied higher-instability, lower-boundedness regimes.
* Result: Object localization matched the OBS-077 interpretation. C exhibited bounded stability across energy_ridge, response_ridge, frobenius_ridge, lazarus_concentration, and coupling_positive; Cp2 showed elevated instability in response/Frobenius/seam/coupling structures; Cp3 showed strongest instability around energy_ridge, density_core, and seam_proxy.
* Result: Cohort localization reinforced the transition reading. Cp3 entered cohorts showed especially high instability (instability_signature_z ≈ +0.815, bounded_stability_z ≈ -1.530), supporting the OBS-077 interpretation of earlier entry-localized instability followed by later settlement.
* Integrated Finding: OBS-078 establishes a three-step validation chain: OBS-078a demonstrated strict mechanistic separability, OBS-078b found a low-dimensional stability core, and OBS-078c localized that core back onto the original transition/cohort/object structure.
* Interpretation: The dominant separation is a local stability regime: C = bounded stability, Cp2 = high-divergence recovery sorting, Cp3 = high-displacement instability and settlement. The richer OBS-077 interpretation compresses to divergence, displacement, and boundedness dynamics.
* Relation to OBS-077: OBS-077 supplied the mechanistic interpretation; OBS-078 shows that interpretation is classifier-visible, strict-control robust, compressible, and localizable rather than purely descriptive.
* Relation to OBS-075: OBS-078 does not directly explain transfer asymmetry, but provides a compact mechanistic substrate suggesting that corpus differences may be grounded in local divergence/boundedness regimes.
* Guardrail: Results are model-specific, corpus-specific, artifact-specific, matched-contract-specific, and provisional with respect to causal transfer claims. OBS-078 does not establish universality, direct localization of OBS-075 target labels, generated-text causality, attractor structure, or topological mechanisms.
* Next Step: OBS-079 should test robustness of the three-feature stability core under alternate normalizations, leave-object-out and leave-transition-out validation, resampling, held-out transition subsets, bootstrap confidence intervals, and alternate observable contracts.

---

## OBS-079 — Stability core robustness

**Date:** 2026-06-11
**State:** OBS-078 low-dimensional local stability core shown to be structurally robust, bootstrap-stable, and pairwise anatomized across C / Cp2 / Cp3

* Question: Does the OBS-078 three-feature local stability core survive structural perturbation, resampling, and pairwise decomposition?
* Method: Tested the OBS-078 stability core (mean_lambda_local_mean, mean_delta_d_mean, bounded_share_mean) using three robustness passes: OBS-079a leave-structure-out validation, OBS-079b bootstrap confidence intervals, and OBS-079c pairwise stability classifiers.
* Artifacts: Uses outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv, derived from OBS-077 pinch-point transition geometry, path-label projection, and coupled-window divergence/boundedness bridge artifacts.
* Result (OBS-079a): The three-feature stability core remained predictive under held-out structures. Primary-valid leave-object-out, leave-cohort-out, leave-transition-out, and combined leave-structure schemes stayed well above dummy baselines (≈0.333), with representative mean BA values from 0.828 to 0.938 depending on scheme/model.
* Result (OBS-079b): Bootstrap CIs confirmed stable case-level separation. C remained bounded and low-instability (instability_signature_z = -1.316, CI [-1.562, -1.088]; bounded_stability_signature_z = +2.655, CI [+2.197, +3.144]), while Cp2 and Cp3 remained higher-instability and lower-boundedness.
* Result (OBS-079b): Pairwise bootstrap contrasts robustly separated C from Cp2 and Cp3. C vs Cp2 had instability diff = -1.793 and bounded-stability diff = +3.639; C vs Cp3 had instability diff = -1.925 and bounded-stability diff = +3.859. Cp2 vs Cp3 was subtler but still separated by displacement, boundedness, instability, and bounded-stability axes.
* Result (OBS-079c): Pairwise classifiers anatomized different stability axes: C vs Cp2 = divergence / boundedness split; C vs Cp3 = boundedness-dominant split; Cp2 vs Cp3 = displacement / lambda-delta split.
* Integrated Finding: OBS-079 upgrades OBS-078 from “the stability core classifies the observed table” to “the stability core survives held-out structures, bootstrap resampling, and pairwise decomposition.”
* Interpretation: The robust case-level skeleton is: C = bounded stability, Cp2 = high-divergence / low-boundedness sorting, Cp3 = high-displacement / low-boundedness instability-settlement.
* Relation to OBS-078: OBS-078 established that the stability core is classifier-visible, compressible, and localizable. OBS-079 establishes that it is structurally robust, measurement-stable, and pairwise interpretable.
* Guardrail: OBS-079 remains model-specific, corpus-specific, artifact-specific, matched-contract-specific, and strict-control-specific. It does not prove OBS-075 transfer causality, generated-text semantic causality, formal attractor basins, topology, or robustness beyond the tested matched shared-14 MDS-pilot contract.
* Next Step: OBS-080 should test contract sensitivity: alternate normalizations, rank transforms, leave-scale-band-out validation, alternate shared observable subsets, matched downsampling, group-level bootstraps, and additional corpora where available.

---

## OBS-080 — Stability core contract-sensitivity

**Date:** 2026-06-13
**State:** OBS-078/079 local stability core shown to be contract-stable across numeric transforms, scale-band restrictions, feature-family projections, and structural resampling

* Question: Does the OBS-078/079 three-feature local stability core persist under alternate measurement contracts, or is it an artifact of the original OBS-078 feature contract?
* Method: Tested the core (mean_lambda_local_mean, mean_delta_d_mean, bounded_share_mean) across four contract families: OBS-080a numeric transforms, OBS-080b scale-band restrictions, OBS-080c feature-family projections, and OBS-080d structural-resampling contracts.
* Artifacts: Uses outputs/comparisons/obs078a_mechanistic_signature_classifier_v2/obs078a_feature_table.csv; outputs under outputs/comparisons/obs080a_stability_core_transform_sensitivity/, obs080b_stability_core_scale_band_sensitivity/, obs080c_feature_family_contract_sensitivity/, and obs080d_structural_resampling_contract_sensitivity/.
* Result (OBS-080a): The stability core was transform-stable across raw, standard_z, robust_median_iqr, rank_percentile, quantile_normal, minmax, and signed_log1p_abs. Three-way best BA remained near 0.899–0.916, and pairwise results stayed strongly above permutation baselines.
* Result (OBS-080b): The core was scale-band stable but scale-position sensitive. C-vs-Cp2 and C-vs-Cp3 stayed strong across meaningful non-empty bands, while Cp2-vs-Cp3 was strongest in middle / mid-to-coarse / all-but-early transition corridors.
* Result (OBS-080c): The three-feature core was sufficient but not exclusive. stability_core_3 reached three-way BA 0.916, while stability_plus_geometry reached 0.973; Cp2-vs-Cp3 sharpened under geometry-enriched and broader strict numeric contracts.
* Result (OBS-080d): Structural resampling confirmed recomposition stability. C-vs-Cp2 and C-vs-Cp3 remained near-ceiling across row, object, cohort, transition, object-cohort, and object-transition bootstraps. Cp2-vs-Cp3 remained the sensitive diagnostic pair but survived under the compact core and sharpened under geometry-enriched contracts.
* Integrated Finding: The OBS-078/079 local stability core is contract-stable, sufficient, and redundantly supported by broader transition geometry.
* Interpretation: The compact core is best read as a reusable local stability invariant: C = bounded stability, Cp2 = high-divergence / low-boundedness sorting, Cp3 = high-displacement / low-boundedness instability-settlement.
* Guardrail: OBS-080 is a contract-sensitivity study over the OBS-078a feature table. It does not establish external generalization, causal control, model-independent universality, new corpus-level transfer, or formal topological invariance.
* Next Step: OBS-081 should define a Reusable Invariance Registry, distinguishing stable reusable invariants from context-sensitive / geometry-supported invariants.

---

## OBS-081 — Reusable Invariance Registry

**Date:** 2026-06-13
**State:** Reusable Invariance Registry established as the first operational RIG layer over PAM transition relations

* Question: Can OBS-080 contract-sensitivity be converted into explicit reusable-invariant records with carrier roles, geometry-needed levels, failure localizations, and repair annotations?
* Method: Synthesized OBS-080a–d outputs into relation × carrier registry records using experiments/studies/obs081_rig_invariance_registry.py.
* Artifacts: Outputs under outputs/rig_registry/: rig_input_manifest.csv, rig_relation_registry.csv, rig_survival_matrix.csv, rig_failure_localization.csv, rig_geometry_needed_ladder.csv, rig_repair_recommendations.csv, and rig_registry_report.md.
* Result: OBS-081 establishes the first operational Reusable Invariance Geometry layer. Contract-stability is now indexed as navigable invariant records containing relation, carrier, carrier role, RIG status, geometry-needed level, localized repair pressure, and repair recommendation.
* Registry status counts: redundant_reusable_invariant = 16, weak_redundant_carrier = 4, stable_reusable_invariant = 2, context_sensitive_reusable_invariant = 2. No rows were classified as accidental or insufficient after v2 recalibration.
* Core records: C_vs_Cp2__stability_core_3 and C_vs_Cp3__stability_core_3 are stable reusable invariants under the compact local stability core. Cp2_vs_Cp3__stability_core_3 and three_way__stability_core_3 are context-sensitive reusable invariants.
* Geometry-needed result: All four primary tasks are Level 1: compact core sufficient. Enriched geometry is not required for survival, but sharpens Cp2-vs-Cp3 and three-way precision.
* Interpretation: The compact stability core is now registered as a reusable invariant carrier, while geometry, enriched-geometry, non-window, strict numeric, and path-share carriers are recorded as redundant or weak-redundant supports.
* Guardrail: OBS-081 is registry synthesis over OBS-080 artifacts. It does not establish external generalization, causal control, intervention success, model-independent universality, or formal topological invariance.
* Next Step: OBS-082 should build a RIG Navigator / invariance console for browsing stable invariants, context-sensitive invariants, weak redundant carriers, failure localizations, geometry-needed levels, and repair recommendations.

---

## OBS-082 — RIG intervention-readiness audit

**Date:** 2026-06-16
**State:** RIG intervention-readiness audit completed; registry records are scoreable and diagnostic, but not yet hypothesis-ready

* Question: Which OBS-081 relation × carrier records have sufficient invariant strength, failure localization, repair specificity, geometry sufficiency, carrier convergence, and negative-control contrast to support conservative, testable intervention hypotheses within the current PAM artifact lineage?
* Method: Audited all OBS-081 registry records using six readiness dimensions: D1 invariance strength, D2 failure localization, D3 repair specificity, D4 geometry sufficiency, D5 carrier convergence, and D6 negative-control contrast. Composite readiness scores used direct OBS-080d structural-resampling evidence (score_basis = obs080d_carrier_mean_ba), while rig_status remained descriptive metadata rather than primary evidence.
* Artifacts: Generated under outputs/rig_registry/obs082_intervention_readiness/: obs082_input_manifest.csv, obs082_relation_readiness_scores.csv, obs082_candidate_intervention_hypotheses.csv, obs082_negative_control_contrasts.csv, obs082_failure_mode_inventory.csv, obs082_blockers.csv, and obs082_report.md.
* Result: All 24 OBS-081 relation × carrier records were successfully scored. There were no missing artifacts and no blocked records (Class X = 0).
* Readiness classification:
    * Class A (hypothesis-ready): 0 / 24
    * Class B (candidate-ready): 0 / 24
    * Class C (diagnostic-only): 24 / 24
    * Class D (registry-only): 0 / 24
    * Class X (blocked / insufficient evidence): 0 / 24
* Main limiting factors: The registry exhibits strong invariant survival evidence but lacks the contrast and localization necessary for intervention design. The dominant limitations were:
    * weak negative-control contrast: 24 / 24
    * generic repair specificity: 24 / 24
    * diffuse failure localization: 23 / 24
* Interpretation: OBS-082 establishes a maturity distinction between registered reusable invariance and intervention-ready reusable invariance. The current registry reaches a diagnostic level: it reliably describes regime structure and invariant survival, but does not yet provide sufficiently localized failure modes, matched contrasts, or repair pathways to justify intervention hypotheses.
* Maturity ladder:
    * Level 1 — Registered invariant: relation survives perturbation under tested contracts.
    * Level 2 — Diagnostic invariant: relation provides reliable structural characterization of the observed regimes.
    * Level 3 — Actionable invariant: relation has sufficient negative controls, failure localization, and repair specificity to define a conservative intervention hypothesis.
    The current OBS-081 registry reaches Level 2, but no relation reaches Level 3.
* Guardrail: OBS-082 does not demonstrate controllability, causal sufficiency, successful interventions, external generalization, or universal invariance. It is restricted to the OBS-081 registry, OBS-080 contract family, OBS-078/079/080 stability-core lineage, the C/Cp2/Cp3 comparison family, and current repository-generated artifacts.
* Next Step: OBS-083 should focus on strengthening intervention evidence rather than attempting interventions. Priority directions are:
    * constructing stronger matched negative-control relations,
    * improving failure localization,
    * increasing repair/enrichment specificity,
    * separating current diagnostic records into candidate-ready versus purely descriptive invariants.

Canonical OBS-082 result:
The RIG registry is scoreable and diagnostically useful, but not yet intervention-ready. The next scientific step is not intervention execution, but improving the contrast, localization, and repair evidence required for legitimate intervention hypotheses.

---

