# Related-work positioning

The paper is closest to **Sala, De Sa, Gu & Ré, *Representation Tradeoffs for
Hyperbolic Embeddings*, ICML 2018** (arXiv:1804.03329). Getting the boundary
right between their contribution and this one is the single most important thing
in the write-up: a reviewer who knows that paper will stop reading at any claim
that overstates novelty.

## Already published by Sala et al. (2018) — cite, do not claim

| Claim | Status in Sala et al. |
|---|---|
| τ is Sarkar's scale factor with a derivable threshold | Published. τ = −log(tan(α/2)); generalized to τ = O((1/r)·log deg_max) in Prop. 3.1. |
| Larger τ → lower distortion | Published. τ ∝ (1+ε)/ε gives worst-case distortion ≤ 1+ε. |
| Larger τ → more precision required | Published, with matching lower bound (Lemma D.1). |
| float64 kills MAP; high precision recovers MAP ≈ 1.0 | Published. Their Table 9: 128 bits → 0.347; 256 → 0.986; 512+ → 1.0. |
| A learned scale improves MAP in an SGD embedder | Published. Their Table 5. |
| MAP is local/rank-only; distortion is global; algorithms implicitly target different metrics | Published, stated in their Background. |
| Construction beats optimization at low dimension | Published. MAP 0.989 at d=2 on WordNet vs. 0.87 at d=200. |

## What is new in this work

1. **Single-method decoupling.** Sala showed *different algorithms* target
   different metrics. This shows that *one algorithm, varying one
   hyperparameter*, moves reconstruction MAP 0.78 → 0.99 while distortion stays
   flat near 0.11 and out-of-training-set retrieval falls. That is a strictly
   stronger, more actionable statement: cross-paper MAP comparisons are
   confounded by a knob that is typically neither reported nor held fixed.

2. **A new, much larger domain.** ICD-10-CM, 46,817 nodes / 46,816 edges. Sala
   used WordNet, phylogenetic trees, and small graphs (Diseases: 516 nodes). The
   medical-embedding line (Beam et al. and successors) reports MAP without
   distortion and without scale.

3. **A construction reference point on this graph.** Sarkar in 2D reaches MAP
   1.0000 at distortion 0.0192, establishing that the ~0.11 distortion of the
   optimizer is the optimizer's cost, not the graph's.

4. **A reporting protocol.** Report distortion alongside MAP; report the
   scale/temperature; scale-match when comparing methods; state arithmetic
   precision. This is the part meant to be cited.

5. **Replication (labeled as such).** The precision phenomenon reproduces on a
   different dataset with a different construction (direct Sarkar rather than
   Sala's h-MDS).

## Other references in the draft

Nickel & Kiela (Poincaré, NeurIPS 2017; Lorentz, ICML 2018); Sarkar (GD 2011,
the construction); Gu et al. (mixed-curvature product spaces, ICLR 2019); Yu &
De Sa (numerically accurate hyperbolic embeddings, NeurIPS 2019/2021);
Ganea et al. (entailment cones, ICML 2018); Bonnabel (Riemannian SGD);
Kochurov et al. (geoopt); Beam et al. (cui2vec — note this is Euclidean
word2vec/GloVe, not hyperbolic; the draft currently miscites it). Full list in
`docs/paper/paper_draft_2026-08-06.md`.

## Citations that could not be verified

During the project two references in earlier notes — a SNOMED-CT hyperbolic
paper and a patient-trajectory paper — could not be confirmed to exist and were
removed. Verify every citation against the actual source before submission;
never reinstate an unconfirmed one.
