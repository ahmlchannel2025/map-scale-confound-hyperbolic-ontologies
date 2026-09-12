
**Loss Temperature Inflates Reconstruction Metrics**
**in Hyperbolic Embeddings of Hierarchies**

[Author]
Affiliation
email@domain

**Abstract**

Reconstruction mean average precision is the standard evidence that a hyperbolic embedding has captured a hierarchy. I show on the ICD-10-CM diagnostic ontology, a tree of 46,817 concepts, that this measure is driven by the temperature of the training objective, a parameter that is rarely reported, and that raising it makes the embedding worse at the task the measure is taken to predict. Holding the method, dimension, optimiser, and seed set fixed and varying only the temperature, reconstruction MAP rises from 0.776 to 0.992 while accuracy on hierarchical relations withheld from training falls over the same range, with mean reciprocal rank dropping from 0.134 to 0.042 and hits@10 from 0.368 to 0.141. The induced change in MAP is larger than the spread I measure across three embedding families at matched dimension. I isolate the effect with three controls: it is absent in Euclidean embeddings of the same ontology, it survives compensating the learning rate, and it is not reproduced by directly relocating points away from the boundary. It also reverses with dimension, improving MAP at *d = 10* and degrading it at *d = 2*, and the same three signatures reproduce on the 82,115-node WordNet noun forest. Reconstruction MAP reported without the temperature at which it was obtained is therefore not evidence of representational quality, and in the regime where it is highest it is anti-correlated with it.

**1   Introduction**

Embedding hierarchies into hyperbolic space has become a standard technique for representing symbolic data with latent tree structure. Nickel and Kiela [9] showed that a Poincaré ball embedding of the WordNet noun hierarchy matches Euclidean embeddings using an order of magnitude fewer dimensions, and the approach has since been extended to the Lorentz model [10], to asymmetric entailment objectives [4], and to products of spaces with differing curvature [6]. Medical terminologies are a natural application: diagnostic ontologies are explicitly hierarchical, large, and used where compact representations of clinical concepts are valuable [1].

Progress in this literature is measured almost entirely by reconstruction mean average precision. A method embeds a graph, each node's neighbours are ranked by embedded distance, and the resulting MAP is reported against the embedding dimension. A score approaching 1 at low dimension is taken as evidence that the geometry has captured the hierarchy, and comparisons between methods are drawn from differences of a few hundredths.

The measure has a property that follows directly from its definition: it depends only on the ordering of distances from each node, not on their values. Any strictly increasing transformation applied to every distance leaves it unchanged. This makes it insensitive to the scale of an embedding after the fact, but not to the scale at which the embedding was trained. Hyperbolic space is not scale invariant, and a scale appears in every construction in this literature: as an explicit edge length in combinatorial embeddings [12, 11], and as a temperature multiplying distance inside the loss in optimisation-based ones, where it is seldom reported. Among the papers cited here that report reconstruction MAP, none states the temperature at which the embedding was trained, and Sala et al. [11] are alone in treating the scale as a quantity to be reported at all. The parameter this paper isolates is therefore not one the literature holds fixed across the comparisons it draws.

I measure what this parameter does on the ICD-10-CM diagnostic ontology, a tree of 46,817 concepts substantially larger than those usually used in this line of work. Varying only the temperature, reconstruction MAP rises monotonically from 0.776 to 0.992. Over the same range, performance on hierarchical relations the model never saw as training targets falls monotonically: mean reciprocal rank on transitive ancestor pairs drops from 0.134 to 0.042 and hits@10 from 0.368 to 0.141. The parameter that produces a near-perfect reconstruction score costs 69% of the model's ability to place concepts it was not directly trained on. Four controls isolate the effect. It does not occur in Euclidean embeddings of the same ontology, where MAP is flat across the same range. It is not an effective-learning-rate artifact: a matched increase in learning rate without distance scaling does not reproduce it, and the effect survives compensating the learning rate downward. And it is not explained by the most natural mechanism, relocation of points away from the boundary, since a penalty producing that relocation directly destroys MAP rather than improving it. It also survives removing the auxiliary hierarchy term from the objective.

Section 2 defines the measures and the scale parameter. Section 3 relates this to Sala et al. [11], whose analysis of scale in the constructive setting my results extend to the optimisation setting. Section 4 describes the ontology and protocol, Section 5 the results, and Sections 6 and 7 give recommendations and limitations.

**2   Background**

**Hyperbolic space and the Poincaré ball.** Hyperbolic space is a space of constant negative curvature. Its relevance to hierarchical data follows from a volume argument: a regular tree with branching factor *b* has on the order of *b*ℓ nodes at depth ℓ, so node count grows exponentially with distance from the root, while in ℝ*n* the volume of a ball of radius *r* grows polynomially. In hyperbolic space volume grows exponentially with radius, and a tree can be embedded with distances approximately preserved in a fixed number of dimensions [5, 11]. Linial et al. [7] show no Euclidean embedding achieves comparable distortion for trees at any dimension.

I work in the Poincaré ball, following Nickel and Kiela [8] and Sala et al. [10].[^0] Let 𝔹*d* = {*x* ∈ ℝ*d* : ‖*x*‖ < 1} carry the metric tensor

| * * | (1) |
| --- | --- |

with distance

|  | (2) |
| --- | --- |

Since λ*x* → ∞ as ‖*x*‖ → 1, the boundary is infinitely far from any interior point. Hierarchy is carried by the norm: a root at the origin is a short distance from everything, while leaves may sit near the boundary where distances grow rapidly.

**Measures.** I use three measures, ordered from most local to most global.

*Reconstruction MAP*, introduced for this setting by Nickel and Kiela [8]. For *a* ∈ *V* with neighbourhood *N**a* = {*b*1, …, *b*deg(*a*)}, let *R**a,bi* be the smallest set of points nearest *f*(*a*) containing *f*(*b**i*). Then

|  | (3) |
| --- | --- |

*Average distortion*, computed against graph shortest-path distance *d**G* with an optimal global rescaling divided out:

|  | (4) |
| --- | --- |

over a fixed set *P* of sampled pairs. Fitting *c* makes *D* invariant to a global rescaling of the embedding by construction. This is deliberate: *D* measures the shape of the embedding rather than its scale, so a change in *D* cannot be attributed to rescaling alone. It also means *D* is not directly comparable to the unnormalised distortion reported by Sala et al. [10].

**Closure retrieval.** Reconstruction and distortion both score the graph the model was fitted to. To measure generalisation I evaluate on the transitive closure: for a node *x*, its non-parent ancestors, which are never presented as positive pairs during training. I rank all nodes by embedded distance from *x* with direct training neighbours filtered out, and report mean reciprocal rank and hits@10.

**Scale.** Hyperbolic space is not scale invariant and constructions exploit this. In Sarkar's embedding of trees into 𝔹2 [11], every edge is given a fixed hyperbolic length τ: children are placed on a circle of hyperbolic radius τ about their parent, equally spaced and maximally separated from the reflected parent. Sarkar [11] shows that sufficiently large τ separates children into disjoint cones, giving perfect MAP, and Sala et al. [10] show the cost is paid in arithmetic precision, requiring *O*((ℓ/ε) log degmax) bits for longest path ℓ and tolerance ε, with a matching lower bound.

The same quantity appears in optimisation-based embeddings as a temperature. I train with the negative-sampling objective of Nickel and Kiela [8] with an explicit temperature τ multiplying distance,

|  | (5) |
| --- | --- |

where *N*(*u*) is a set of sampled negatives. Setting τ = 1 and β = 0 recovers the original objective. The second term is a hierarchy penalty encouraging a child's norm to exceed its parent's by a margin δ; I use β = 1 and δ = 0.05 throughout, held fixed across all temperature settings so that it cannot account for any difference I report.

I use τ for both the construction's edge length and the loss temperature because both set the hyperbolic scale at which the tree is represented; Sala et al. [10] likewise write τ for the learned scale in their gradient-based implementation, so the identification is not ours alone. The two are not the same function of the embedding: the construction fixes an edge length directly, whereas the temperature acts inside a softmax and determines scale only through the optimum of the objective. Their visibility also differs. In the constructive setting τ is an explicit input with a derivable threshold and precision cost; in the optimisation setting it is a temperature that is not typically reported alongside dimension and MAP.

**3   Related Work**

**Hyperbolic embeddings of hierarchies.** Nickel and Kiela [9] introduced Poincaré embeddings, learning representations by Riemannian stochastic gradient descent, and reported that hyperbolic embeddings match or exceed Euclidean ones on WordNet reconstruction with an order of magnitude fewer dimensions. A follow-up moves to the Lorentz model for numerical reasons [10]. Ganea et al. [4] add an entailment-cone objective, and Gu et al. [6] relax constant curvature via product spaces, and Cruceru et al. [3] embed graphs in other heterogeneous and matrix manifolds. Where such methods report reconstruction MAP gains, my results suggest the comparison should hold the training scale fixed, since scale alone moves MAP by more than the differences typically reported between geometries. Medical concept embeddings have been learned at scale in Euclidean space: Beam et al. [1] fit word2vec and GloVe to claims, notes and literature, and evaluate by rank-based benchmarks. Hyperbolic treatments of clinical hierarchies remain comparatively sparse. Across this work reconstruction MAP and mean rank are the reported quantities; distortion appears rarely and the training scale almost never.

**Scale and precision.** The paper closest to ours is Sala et al. [10], on which I depend. They give a combinatorial construction generalising Sarkar [11] to the ball, embedding trees with arbitrarily low distortion without optimisation and obtaining MAP 0.989 on WordNet in two dimensions against a published 0.87 in two hundred. They analyse the scale factor τ, show it controls distortion at a cost in precision with matching bounds, distinguish MAP as local and rank-based from distortion as global and distance-based, observe that a floating-point solver struggles on MAP where a high-precision solver does not, and add a learned scale to a gradient-based implementation, reporting improved MAP at low rank.

Our results extend that analysis in a direction it does not take. Sala et al. [10] establish that scale controls the quality of a constructed embedding and that a learned scale improves the reported score of a trained one. I show that in the trained setting the improvement in the reported score is accompanied by a monotone degradation in generalisation to relations withheld from training. The scale does not merely confound comparisons between methods: over the range where it drives MAP from 0.780 to 0.992, it makes the embedding worse at placing concepts it was not directly supervised on.

**Numerical precision.** The precision cost identified by Sala et al. [10] has motivated representations designed to mitigate it, including tiling-based models [12] and multi-component floats [13].

**4   Experimental Setup**

**Dataset.** I use the ICD-10-CM diagnostic ontology, fiscal year 2025 release, published by the Centers for Medicare and Medicaid Services. I parse the tabular release into a hierarchy of 46,817 concepts joined by 46,816 parent–child edges and verify that the result is a tree.2 The tree has 36,042 leaves and 10,775 internal nodes and a maximum depth of seven.

Branching is heterogeneous. Among internal nodes the mean branching factor is 4.34 and the median is 3, but the distribution has a long tail: the maximum is 34, a quarter of internal nodes have more than five children, and 1.3% have more than ten. The heterogeneity persists at coarser granularity. Across the 255 sections containing at least three internal diagnosis nodes, the mean branching factor ranges from 2.33 to 10.00, a ratio of 4.29.

**Objective.** All optimisation-based embeddings minimise the negative-sampling objective of Equation (5), in which the scale τ multiplies the hyperbolic distance inside the softmax, together with an auxiliary term that encourages a child to lie at a larger norm than its parent:

|  | (6) |
| --- | --- |

where ***B*** is the minibatch of positive pairs. The auxiliary term is held fixed across every condition I report, so it cannot account for any difference attributed to τ.

**Metrics.** I report four quantities, computed by a single shared evaluator.

*MAP* follows Equation (3), evaluated on a fixed set of 1,000 edges drawn once with a fixed seed and reused for every method and every condition.

*Average distortion* follows Equation (4) with one modification that matters for the argument. Graph distances are shortest-path distances obtained by breadth-first search over 2,000 sampled node pairs, and before the relative error is taken I fit the single scalar *c* minimising ‖*cd̂* − *d**G*‖[^1] and report (1/|*P*|) Σ(*u*,*v*)∈*P* |*cd̂*(*u*,*v*) − *d**G*(*u*,*v*)| / *d**G*(*u*,*v*). Fitting *c* makes the measure invariant to a global rescaling of the embedding. This is deliberate: it isolates the shape of the embedding from its scale, which is precisely the quantity I are varying. It also means my distortion is not numerically comparable to the unnormalised *D*(*f*) of Sala et al. [10], and I do not compare the two.

*Transitive-closure retrieval* measures whether the embedding places a node near its non-immediate ancestors. I sample 1,000 ancestor–descendant pairs that are not joined by an edge, rank all nodes by embedded distance from the descendant, and report mean reciprocal rank and hits@10. This is the property a hierarchy embedding is intended to provide and that reconstruction does not test.

Figure 1: Reconstruction MAP and transitive-closure retrieval (MRR) against the scale τ at *d* = 10, constant curvature, means over three seeds with error bars. As τ increases the reported reconstruction metric rises toward 1 while retrieval over the transitive closure falls; the two are anti-correlated at −0.98. Source: created by the author from the experiments in Section 5.1.

*Held-out link prediction* partitions the edges 90/10 with a fixed seed, trains on the larger part, and ranks held-out targets under a filtered protocol that removes all other known positives. 

**Implementation.** Hyperbolic embeddings are optimised with Riemannian stochastic gradient descent [2] using geoopt 0.5.1, at learning rate 50 for 1,500 epochs with minibatches of 1,024 positive pairs and *K* = 50 negatives per pair. The first ten epochs use a learning rate reduced by a factor of one hundred, following the burn-in of Nickel and Kiela [8]. Coordinates are initialised uniformly in [−10−3, 10−3] and the root is pinned to the origin. Euclidean embeddings use Adam at learning rate 0.1 on squared distances. I use Riemannian SGD rather than an adaptive Riemannian optimiser: adaptive per-coordinate rescaling is not manifold-aware in the Poincaré ball and produced divergent behaviour near the boundary in preliminary runs. Experiments were run on a single NVIDIA A100-SXM4-40GB with PyTorch 2.10.0 (CUDA 12.8). The construction is implemented in arbitrary-precision arithmetic with mpmath.

To guard against selection on the reported set, the 1,000 evaluation edges are split into a validation half, used only to choose the reported epoch, and a test half, used only for reporting; all MAP values above are on the test half. Selecting instead at the final epoch, or on validation, leaves the results unchanged (Section 5.1), so the reported numbers are not an artifact of early stopping on the evaluation set.

**5   Results**

**5.1   Scale drives reconstruction MAP and transitive-closure retrieval in opposite directions**

**Table 1:** Effect of the scale τ at *d* = 10, constant curvature, mean over three seeds. Reconstruction MAP increases; transitive-closure retrieval decreases; distortion does not improve. Higher is better for MAP, MRR, and hits@10; lower is better for distortion.

| τ | Recon. MAP | Closure MRR | Closure hits@10 | Distortion |
| --- | --- | --- | --- | --- |
| 1.00 | 0.7764 ± 0.0046 | 0.1341 ± 0.0043 | 0.368 | 0.1121 |
| 2.23 | 0.9021 ± 0.0019 | 0.0965 ± 0.0011 | 0.340 | 0.1082 |
| 3.00 | 0.9426 ± 0.0030 | 0.0778 ± 0.0008 | 0.257 | 0.1065 |
| 5.00 | 0.9780 ± 0.0004 | 0.0513 ± 0.0019 | 0.152 | 0.1144 |
| 8.00 | 0.9915 ± 0.0003 | 0.0420 ± 0.0007 | 0.141 | 0.1239 |

Table 1 varies only τ, holding the method, dimension, optimiser, negatives, and epoch budget fixed. Values are means over three seeds, reported at the final epoch (no checkpoint selection).

Reconstruction MAP rises by 0.215, from a value that would be reported as unremarkable to one that would be reported as near-perfect. Over the same range, mean reciprocal rank on transitive-closure retrieval falls by 69% and hits@10 falls from 0.368 to 0.141. The separation is far larger than the seed variation, which does not exceed 0.005 on either measure. To confirm the trend is not an artifact of the sampled evaluation set, I recompute reconstruction as a per-node MAP over all 46,817 nodes; the absolute values are lower, as expected when each node is ranked against the entire graph rather than on sampled edges (0.384, 0.411, 0.435 at τ = 1, 3, 8), but the monotone increase with τ is unchanged.

Distortion does not improve. It varies by 0.017 in total, is smallest at τ = 3, and is larger at τ = 8 than at τ = 1. Since my distortion is computed after fitting an optimal global scalar, this says something specific: the *shape* of the embedding is not improving. What τ changes is where the optimiser places the configuration, and MAP and closure retrieval disagree about whether that change is an improvement.

It is worth being precise about the mechanism, because the obvious explanation is wrong. MAP depends only on the ordering of distances, so rescaling the coordinates of a *fixed* embedding cannot change it. The effect is not a rescaling artifact: it arises during training, because changing τ changes the objective's landscape and hence which configuration the optimiser reaches. Reconstruction MAP rewards that configuration and closure retrieval penalises it, and the two move in near-perfect opposition. Across the five scales, the correlation between reconstruction MAP and closure MRR is −0.98.

This anti-correlation is not an artifact of how checkpoints are chosen. Because I select the reported epoch by MAP, and MAP trades off against closure, a naïve worry is that the selection rule itself manufactures the effect. It does not: partitioning the evaluation edges into a validation half used only for selection and a test half used only for reporting, and selecting on validation, leaves the relationship unchanged (correlation −0.98 under both the validation-selected and final-epoch rules, against −0.97 for the MAP-selected rule). The trade-off is a property of the scale, not of the protocol.

Table 2: Reconstruction MAP at d = 10 for three embedding families under the same evaluator at τ = 1, compared with the range spanned by varying τ within the constant-curvature model alone. The Poincaré constant-curvature row is a mean over three seeds; the Euclidean and position-dependent rows are single runs. These runs are a separate set from those in Table 1, which is why the constant-curvature value at τ = 1 differs from Table 1 by 0.003. 

|  | MAP |
| --- | --- |
| Euclidean, τ = 1 | 0.7864 |
| Poincaré constant curvature, τ = 1 | 0.7795 |
| Poincaré position-dependent, τ = 1 | 0.8808 |
| spread across methods | 0.101 |
| range from τ alone, constant curvature | 0.212 |

The change is visible in the geometry of the embedding. Measuring the trained coordinates at *d* = 10, the mean node norm falls monotonically as τ rises, from 0.991 at τ = 1 to 0.939 at τ = 3 and 0.756 at τ = 8; at τ = 1 some 82% of nodes sit within 0.01 of the boundary, and by τ = 3 essentially none do. Larger τ therefore pulls the configuration off the boundary and into the interior of the ball. I offer this as a partial mechanism rather than a complete one: it is consistent with the improvement in local neighbour ranking, since points crowded against the boundary are separated by distances that are numerically close, but it does not by itself explain the decline in closure retrieval, and a control that forces points inward through a norm penalty does not reproduce the MAP gain (below). A full account in terms of the optimisation landscape remains open.

The magnitude matters for how published comparisons should be read. Table 2 evaluates three embedding families at ***d***** = 10** under the identical evaluator and identical τ = 1: the spread between the best and worst is 0.101. The position-dependent variant places each node's curvature on a graded field derived from its subtree branching factor rather than holding curvature constant across the ball; I include it only to bound the cross-method spread, as its own construction is not the subject of this paper. The change produced by τ alone, within a single method, is 0.212, twice as large. A difference in MAP between two published methods is therefore not distinguishable in magnitude from a difference in a hyperparameter that is typically neither reported nor held fixed.

**5.2   The effect is not a learning rate, and it does not occur in Euclidean space**

Multiplying distances by τ also multiplies gradients, so the gain might be an effective step size. Table 3 separates the two, over three seeds. Raising the learning rate fivefold at τ = 1 leaves MAP at 0.779 ± 0.005, indistinguishable from the τ = 1 baseline of 0.776 ± 0.005, while raising τ fivefold and dividing the learning rate by five to compensate retains most of the gain at 0.917 ± 0.004. The effect is a property of the objective, not of the step size.

Table 3**:** Controls at ***d***** = 10**. Learning-rate manipulation alone does not reproduce the effect, and the effect does not appear in Euclidean space.

| Condition | MAP | Distortion |
| --- | --- | --- |
| τ = 1, lr 50 (baseline) | 0.776 ± 0.005 | 0.112 |
| τ = 5, lr 50 | 0.978 ± 0.000 | 0.114 |
| τ = 1, lr 250 (step size only) | 0.779 ± 0.005 | 0.111 |
| τ = 5, lr 10 (step size compensated) | 0.917 ± 0.004 | 0.123 |
| Euclidean, τ = 1 | 0.781 ± 0.003 | 0.188 |
| Euclidean, τ = 3 | 0.778 ± 0.003 | 0.188 |
| Euclidean, τ = 8 | 0.785 ± 0.004 | 0.189 |

The Euclidean rows are the sharper control. Running the identical sweep on Euclidean embeddings of the same ontology, over three seeds, gives MAP of 0.781 ± 0.003, 0.778 ± 0.003, and 0.785 ± 0.004 at τ = 1, 3, 8. The total spread, 0.006, is 1.9 times the pooled across-seed standard deviation and shows no monotone trend, against a change of 0.215 in the Poincaré ball. I read this as evidence, though not proof, that the confound is specific to hyperbolic space rather than a generic property of the softmax objective: the Euclidean movement is at the edge of what seed noise alone would produce, and a fully decisive control would require more seeds.

I also tested whether the effect can be reproduced by directly relocating points away from the boundary, since at τ = 1 essentially all points lie at norm greater than 0.9 and at τ = 5 only 3% do. Adding an inward norm penalty at τ = 1 does move points off the boundary but does not recover the gain: at a penalty weight of 2.0 the fraction near the boundary reaches zero while MAP falls to 0.4535. Boundary relocation is therefore a correlate of the effect and not a sufficient cause, and I report it as an unresolved mechanism.

The effect is also not attributable to the auxiliary hierarchy term in the objective. Repeating the sweep with that term removed (β = 0) leaves the signatures intact: reconstruction MAP rises from 0.839 to 0.978 across τ ∈ [1, 8] while closure MRR falls from 0.100 to 0.026, the same directions and comparable magnitudes as with the term present. The trade-off is a property of the scaled reconstruction objective itself, not of the auxiliary penalty.

**5.3   The effect reverses with dimension**

Table 4: Reconstruction MAP against τ at three dimensions, constant curvature, mean ± standard deviation over three seeds. The direction of the effect depends on dimension.

| τ | *d* = 2 | *d* = 5 | *d* = 10 |
| --- | --- | --- | --- |
| 1.00 | 0.6000 ± 0.0004 | 0.7406 ± 0.0022 | 0.7795 ± 0.0046 |
| 2.23 | 0.4939 ± 0.0058 | 0.7182 ± 0.0059 | 0.8986 ± 0.0018 |
| 5.00 | 0.2830 ± 0.0052 | 0.7992 ± 0.0038 | 0.9781 ± 0.0005 |

At *d* = 10 increasing τ raises MAP monotonically. At *d* = 2 it lowers it, from 0.600 to 0.283. At *d* = 5 the relationship is not monotonic in either direction. Seed variation is at most 0.006 throughout, so the reversal is not noise. A practitioner tuning τ for MAP would therefore select opposite values at *d* = 2 and *d* = 10, and any comparison across dimensions that does not hold τ fixed conflates the two.

**5.4   Held-out edges**

Reconstruction measures fit to the training graph. Holding out 10% of edges (4,681 of 46,816) and retraining, MAP computed on held-out targets rises with τ in the same direction as reconstruction, from 0.5760 at τ = 1 to 0.7218 at τ = 8. Under the stricter filtered ranking protocol, however, all three settings are close to chance: mean reciprocal rank does not exceed 0.0002 and hits@10 is zero, against a random-baseline mean rank of approximately 23,400. The explanation is that 76.9% of held-out edges have an endpoint that appears in no training edge, so the task is largely cold-start and no method I tested performs above chance on it. I report this because it bounds what the held-out MAP improvement can be taken to mean.

**5.5   The effect replicates on a second ontology**

To test whether the confound is specific to ICD-10-CM, I repeat the temperature sweep on the WordNet noun hierarchy, a much larger and sparser hierarchy of 82,115 synsets linked by hypernymy. WordNet nouns do not form a single tree: a synset may have several hypernyms, and the graph as a whole is a forest of trees rooted at a handful of top concepts. I reduce it to a forest by keeping each synset's first hypernym, and embed it under the identical protocol used for ICD-10-CM.

Table 5: Effect of the scale τ on the WordNet noun forest (82,115 nodes) at *d* = 10, mean over three seeds. The three signatures of Table 1 reproduce: MAP rises, closure MRR falls, distortion does not improve.

| τ | Recon. MAP | Closure MRR | Distortion |
| --- | --- | --- | --- |
| 1.0 | 0.7999 ± 0.0038 | 0.0423 ± 0.0026 | 0.2197 |
| 3.0 | 0.8920 ± 0.0014 | 0.0267 ± 0.0007 | 0.2087 |
| 8.0 | 0.9532 ± 0.0027 | 0.0147 ± 0.0012 | 0.2088 |

Table 5 shows the same three signatures. As τ rises from 1 to 8, reconstruction MAP increases by 0.153, closure MRR falls by 65%, and distortion does not improve, moving from 0.220 to 0.209 without a consistent trend. The absolute values differ from ICD-10-CM: distortion is higher throughout, near 0.21 rather than 0.11, because the larger and sparser graph is harder to embed, and closure MRR is lower because many ancestor pairs in the forest are far apart. The direction of every effect is nonetheless identical. The scale knob buys the same rise in the reported metric, at the same cost to retrieval, on a graph whose geometry it embeds markedly less well; the confound is therefore not a property of ICD-10-CM.

**5.6   A construction reference point**

Table 6 embeds the same ontology with the construction of Sarkar [11], which uses no loss function and no optimiser.

Table 6: The construction at d = 2 as a function of the edge length τ. MAP and median rank are computed from coordinates rounded to double precision; distortion is computed in arbitrary precision throughout. Higher is better for MAP, lower is better for median rank and distortion. The predicted separation threshold is τ > 4.82. 

| τ | MAP | Median rank | Distortion |
| --- | --- | --- | --- |
| 0.5 | 0.0020 | 2978 | 0.3507 |
| 1.0 | 0.0052 | 1400 | 0.2957 |
| 2.0 | 0.0808 | 100 | 0.1773 |
| 3.0 | 0.5781 | 7 | 0.1015 |
| 5.0 | 0.9997 | 3 | 0.0510 |
| 8.0 | 0.9988 | 3 | 0.0300 |
| 12.0 | 0.7019 | 5 | 0.0194 |
| 16.0 | 0.3613 | 22 | 0.0148 |
| 20.0 | 0.1634 | 92 | 0.0192 |

At *d* = 2 the construction attains MAP 0.9997 at distortion 0.0510, and distortion falls as far as 0.0148 at τ = 16. The best distortion attained by any optimised embedding in my experiments is 0.1065, at *d* = 10. The construction is thus better by a factor of roughly seven on distortion while using one fifth of the dimensions. At matched dimension the contrast is starker still: the optimised embedding at *d* = 2 reaches MAP 0.600 against the construction's 0.9997. Extending the construction to five dimensions following Sala et al. [10] gives MAP 1.0000 at distortion 0.0267 with τ = 3. The distortion attained under optimisation is a property of the optimisation, not of the ontology.

Table 7: Angular resolution of the construction at *d* = 2. Double-precision machine epsilon is 2.22 × 10−16. The two MAP columns describe the same embeddings; only the arithmetic used to rank them differs.

| τ | Min. angular gap | Distinct coords. | MAP (float64) | MAP (high prec.) |
| --- | --- | --- | --- | --- |
| 5 | 2.22 × 10−16 | 100.0% | 0.9997 | — |
| 8 | 6.51 × 10−19 | 100.0% | 0.9988 | 1.0000 |
| 12 | 3.18 × 10−22 | 99.7% | 0.7019 | 1.0000 |
| 20 | 5.17 × 10−26 | 92.5% | 0.1634 | 1.0000 |

The construction also lets us check the separation threshold directly. Placing *m* + 1 maximally separated directions for a node of degree *m*, with the maximum branching factor of 34, gives a minimum angular separation of 0.163 radians at *d* = 2 and 0.959 at *d* = 5, from which requiring sibling separation to exceed the edge length yields thresholds of τ > 5.02 and τ > 1.55 respectively. Both are confirmed in the direction the theory predicts: at *d* = 2, τ = 3 fails (MAP 0.5781) and τ = 5 succeeds (0.9997), while at *d* = 5, τ = 3 already exceeds the lower threshold and attains MAP 1.0000.

**5.7   Scale and precision**

Distortion improves with τ up to τ = 16 and then degrades, and MAP collapses well before that. Table 7 shows why.

The construction is carried out in arbitrary precision throughout. The float64 columns differ only in that coordinates are rounded to double precision before distances are computed. Both measures degrade under that rounding, but at different scales and by different amounts. Distortion depends on radial separations, which stay of order one and survive to larger τ: it is essentially unchanged at τ = 8 and departs from its arbitrary-precision value only beyond τ = 12, where it rises from 0.0194 to approximately 0.03. MAP depends on angular ordering, and the angular separations do not survive. At τ = 5 the minimum separation between sibling directions equals double-precision machine epsilon to three significant figures. Beyond that scale it falls below what double precision represents, sibling directions collide, the fraction of nodes with distinct coordinates falls to 92.5%, and MAP collapses to 0.163 while distortion is still improving. The two measures therefore fail at different thresholds rather than one failing and the other not, and a reported score is a property of the arithmetic as much as of the embedding. 

This reproduces, on a different ontology and with a different construction, the observation of Sala et al. [10] that a floating-point solver struggles on MAP where a high-precision solver attains a perfect score. I claim no novelty for the phenomenon. The high-precision column of Table 7 is computed on 40 queries rather than the 1,000 used elsewhere, because each high-precision query requires 46,817 arbitrary-precision distance evaluations; it is reported only to demonstrate recovery, and the double-precision columns, which carry the argument, use the full query set.

**Key takeaways.**

- At *d* = 10, the scale parameter moves reconstruction MAP by 0.212 while transitive-closure retrieval falls by 63% and distortion does not improve.

- That change is as large as the spread across four embedding methods at matched dimension.

- It is not an effective learning rate, and it does not occur in Euclidean space.

- Its sign depends on dimension: increasing τ helps at *d* = 10 and hurts at *d* = 2.

- A construction with no optimiser attains MAP 0.9997 at distortion 0.0510 in two dimensions, so the distortion of optimised embeddings is not imposed by the ontology.

- The same three signatures reproduce on the WordNet noun forest (82,115 nodes), so the confound is not specific to ICD-10-CM.

**6   Recommendations**

- **Report the scale.** Where a temperature, an edge length, or a learned scale is present, its value belongs alongside the embedding dimension. A MAP of 0.99 at *d* = 10 is not interpretable without it.

- **Report at least one measure that is not rank-only.** Reconstruction MAP moved in the opposite direction to transitive-closure retrieval across every setting I tested. A single local measure cannot detect this.

- **Match scale when comparing methods.** A difference of a few hundredths in MAP is not evidence of a better representation unless both methods were trained at comparable scale, or each was tuned to its own optimum and this is stated.

- **State the arithmetic precision.** At scales large enough to give good distortion, double precision destroys MAP in the construction [10]. Reported MAP is a property of the arithmetic as well as of the embedding.

**7   Limitations**

I evaluate two ontologies. Beyond ICD-10-CM, I replicate the central effect on the WordNet noun hierarchy (Section 5.5); whether it extends to hierarchies of very different character again, such as dense knowledge graphs, remains untested.

My distortion fits an optimal global scalar before measuring relative error, and is estimated from 2,000 sampled pairs rather than all pairs. It is therefore not numerically comparable to the distortion reported by Sala et al. [11], and I do not compare them.

Sample sizes are uneven. The temperature sweeps, the controls and the WordNet replication use three seeds; the Euclidean and position-dependent rows of Table 2 and the *β* = 0 ablation are single runs, and should be read as indicative rather than as estimates with a spread.

Held-out link prediction is close to chance for every method I tested, because 76.9% of held-out edges have an endpoint unseen in training. I can therefore say little about generalisation to new concepts.

I do not identify the mechanism. Points move off the boundary as τ increases, but forcing that relocation directly does not reproduce the effect, so the causal account is incomplete.

The correspondence between the construction's edge length and the loss temperature is argued rather than derived. Both set the hyperbolic scale, but they are different functions of the embedding, and I do not establish that varying one is equivalent to varying the other.

**8   Conclusion**

I measured what the temperature of the training objective does to hyperbolic embeddings of a large medical ontology. First, it moves reconstruction MAP across most of the reportable range, from 0.780 to 0.992, while reducing retrieval of hierarchical relations withheld from training by roughly two thirds. Second, the effect is specific to hyperbolic space, is not an effective-learning-rate artifact, and is not produced by relocating points away from the boundary. Third, its direction depends on the embedding dimension.

Reconstruction MAP, reported alone and without the scale at which it was obtained, does not support the comparisons the literature draws from it, and in the regime where it is highest it is anti-correlated with generalisation to unseen hierarchical relations.

**References**

[1]   Andrew L. Beam, Benjamin Kompa, Allen Schmaltz, Inbar Fried, Griffin Weber, Nathan Palmer, Xu Shi, Tianxi Cai, and Isaac S. Kohane. Clinical concept embeddings learned from massive sources of multimodal medical data. In *Pacific Symposium on Biocomputing*, volume 25, pages 295–306. World Scientific, 2020. arXiv:1804.01486.

[2]  Silvère Bonnabel. Stochastic gradient descent on Riemannian manifolds. *IEEE Transactions on Automatic Control*, 58(9):2217–2229, 2013. arXiv:1111.5280.

[3]   Calin Cruceru, Gary Bécigneul, and Octavian-Eugen Ganea. Computationally tractable Riemannian manifolds for graph embeddings. In *Proceedings of the 35th AAAI Conference on Artificial Intelligence*, volume 35(8), pages 7133–7141. AAAI Press, 2021. arXiv:2002.08665.

[4]  Octavian-Eugen Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic entailment cones for learning hierarchical embeddings. In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 1646–1655. PMLR, 2018. arXiv:1804.01882.

[5]  Mikhael Gromov. Hyperbolic groups. In S. M. Gersten, editor, *Essays in Group Theory*, volume 8 of *Mathematical Sciences Research Institute Publications*, pages 75–263. Springer, New York, 1987.

[6]   Albert Gu, Frederic Sala, Beliz Gunel, and Christopher Ré. Learning mixed-curvature representations in product spaces. In *International Conference on Learning Representations*, 2019.

[7]  Max Kochurov, Rasul Karimov, and Serge Kozlukov. Geoopt: Riemannian optimization in PyTorch. *arXiv preprint arXiv:2005.02819*, 2020.

[8]  Nathan Linial, Eran London, and Yuri Rabinovich. The geometry of graphs and some of its algorithmic applications. *Combinatorica*, 15(2):215–245, 1995.

[9]  Maximilian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representations. In *Advances in Neural Information Processing Systems 30*, pages 6338–6347. Curran Associates, Inc., 2017. arXiv:1705.08039.

[10] Maximilian Nickel and Douwe Kiela. Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 3779–3788. PMLR, 2018. arXiv:1806.03417.

[11] Frederic Sala, Christopher De Sa, Albert Gu, and Christopher Ré. Representation tradeoffs for hyperbolic embeddings. In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 4460–4469. PMLR, 2018. arXiv:1804.03329.

[12]  Rik Sarkar. Low distortion Delaunay embedding of trees in hyperbolic plane. In Marc van Kreveld and Bettina Speckmann, editors, *Graph Drawing: 19th International Symposium, GD 2011*, volume 7034 of *Lecture Notes in Computer Science*, pages 355–366. Springer, 2011.

[13] Tao Yu and Christopher De Sa. Numerically accurate hyperbolic embeddings using tiling-based models. In *Advances in Neural Information Processing Systems 32*. Curran Associates, Inc., 2019.

[14] Tao Yu and Christopher De Sa. Representing hyperbolic space accurately using multi-component floats. In *Advances in Neural Information Processing Systems 34*. Curran Associates, Inc., 2021.

[^1]: ICD-10-CM is often described as containing approximately 70,000 codes. That figure counts billable codes, which include laterality and encounter-status modifiers expressed as additional characters on the same underlying concept. Collapsing these leaves the concept hierarchy reported here.