# Known issues and caveats — read before citing any number

This project audited its own results hard, and several numbers that circulated
during the work do **not** belong in a publication. This file lists every one so
that nobody (including future-you) rebuilds a claim on a retracted number. It is
the public-facing version of the internal error ledger.

## The contribution is narrower than the title of early drafts implied

Sala, De Sa, Gu & Ré (ICML 2018, *Representation Tradeoffs for Hyperbolic
Embeddings*) already published: τ as Sarkar's scale factor with a derivable
threshold; the precision-versus-τ tradeoff with matching upper and lower bounds;
high-precision recovery of MAP where float64 fails (their Table 9); a learned
scale improving MAP in an SGD embedder (their Table 5); and the local-MAP /
global-distortion distinction. **Do not write that the scale is "unreported" or
that this work discovered the precision effect.** What is new here is (1)
single-method decoupling — one algorithm, one hyperparameter, MAP moving 0.78 →
0.99 while distortion stays flat; (2) a new, much larger medical ontology; (3) a
construction reference point on that graph; (4) a reporting protocol. The
precision result is a **replication** and must be labeled one everywhere.
Full breakdown: [`related_work_positioning.md`](related_work_positioning.md).

## Numbers that must NOT be used

These are void. They appear in older notebook outputs and must never enter a
table or the paper.

| Number | Where it came from | Why it is dead |
|---|---|---|
| MAP 0.716 / mean rank 20 / α 1.85 | d=10 novel-method run, 6/13 | Trained with RiemannianAdam — the wrong optimizer for the Poincaré ball; its per-coordinate rescaling is not manifold-aware. Superseded. |
| MAP 0.98 / rank 3 | kernel field, 7/8 | Temperature artifact: κ collapsed to near-constant at init, so this was a ×6 scale effect, not curvature. |
| MAP 0.923 | deeper-MLP field, 7/8 | The field was κ = −1.000 everywhere; the "improvement" was a constant scale. |
| MAP 0.33 | learned field, 7/6 | Run at another method's α = −2.0; a straightforwardly unfair comparison. |
| Any Sarkar MAP 0.006–0.043 | first construction, float64 | Implementation was broken; later rebuilt in arbitrary precision (`src/hyperbolic_icd10/sarkar.py`). |
| "0.775 is the d=10 ceiling, confirmed four ways" | pre-7/4 | It was under-training. Long runs (1,500–4,500 epochs, lr 50) pass it. |

## Open methodological blockers (fix before publication)

These are real and still binding. None is fixed by this repository's code alone.

1. **Best-epoch selection on the reporting set.** Phase-3 numbers picked the
   peak-MAP epoch by watching the same eval set they reported. That inflates
   every novel-method number. The paper-experiments notebook (09) introduces a
   VAL/TEST split and shows the anti-correlation survives `final`- and
   `val`-rule selection — but the *absolute* Phase-3 MAP values elsewhere are
   still test-selected. Report only VAL-selected or final-epoch numbers.

2. **Methods evaluated on different sets.** Phase-2 baselines used all 46,816
   edges; Phase-3 novel methods used a 2,000-edge subset; the construction's
   high-precision column used **40 queries**. Forty is not a reportable sample.
   Run everything through one evaluator on one held-out set before tabulating a
   cross-method comparison. (The `metrics` module here gives that one evaluator.)

3. **Single-seed headlines on a noisy metric.** `mean_rank` swung 54–167 across
   runs at fixed settings. Every comparative claim needs mean ± std over ≥5
   seeds and a named significance test. Several archived cells are n=1 or n=3.

4. **The τ threshold number in the draft is from an unconverged solver.** The
   August draft quotes τ > 5.02 at d=2. That came from a 1,500-iteration
   repulsion solver; the exact value for max branching 34 is **4.8239**
   (`scripts/audit_threshold.py`). Both lie inside the empirically bracketed
   interval (τ=3 fails, τ=5 succeeds), so the qualitative claim holds; the
   number does not. Also: the derivation tying the construction's edge length to
   the loss temperature is *argued, not derived* — flag it.

5. **Distortion here is not comparable to Sala's.** Sala's D(f) is the mean over
   *all* pairs of |d_emb − d_graph| / d_graph, unscaled. The distortion in this
   repo fits one global scalar first and samples 2,000 pairs. The two must not
   be compared directly; state the definition every time.

6. **Sampled reconstruction MAP ≫ full-graph MAP.** The reported MAP (~0.77 at
   τ=1) is a per-node MAP over a 1,000-edge *sample*. Computed over all 46,817
   nodes it is far lower — `results/paper_experiments/A_fullgraph_map.json`:
   0.384 / 0.411 / 0.435 at τ = 1 / 3 / 8. The sampled number is what the
   literature reports and is internally consistent across this project, but the
   gap is large and the paper must state which MAP it means.

7. **Held-out link prediction is at chance for every method.** The results speak
   to reconstruction of *known* relations, not generalization to new concepts.
   The closure evaluation tests unseen relations between *known* nodes only.

## The "curvature helps" result is confounded with scale

The position-dependent method multiplies the conformal factor,
λ_θ(x) = base_λ(‖x‖)·exp(α·κ(x)), which *is* a scale change. The +0.04–0.05 MAP
gain at α ≈ ±0.2 over α = 0 is currently indistinguishable from a temperature
effect, and the clean symmetric inverted-U across both signs of α is exactly
what a pure scale artifact looks like. Until a scale-matched ablation has run at
≥5 seeds, this is not presentable as a curvature result. The structural audit
(notebook 07, "why position-dependent curvature cannot help on this tree")
argues the effect requires more extreme branching heterogeneity than ICD-10's
4.29× section-level spread provides.

## Text inconsistencies in the August draft (no code can catch these)

Caught in `notebooks/10_paper_audit.ipynb` §5; fix in the LaTeX source:
abstract vs. intro closure numbers disagree (0.134 vs 0.113); "four embedding
methods" but Table 2 has three rows; "falls 63%" vs. Table 1's 69%; Limitations
says both "one ontology" and "two ontologies"; Beam et al. described as
hyperbolic but cui2vec is Euclidean; `geoopt 0.5.1` used but not cited.
