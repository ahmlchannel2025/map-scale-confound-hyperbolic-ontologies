5/26/26 — Foundations Workshop: Phase 1 Completed (Real Analysis → Calculus → Training Mechanics)
Accomplished
Finished workshop Phase 1. Closed out Rudin Ch 2 (topology) and Ch 9 (multivariable calculus → Inverse Function Theorem), then pushed all the way through the mechanics of one training step. No proofs — intuition + project ties throughout.
Inverse Function Theorem. A C¹ map from R^n to R^n is locally invertible near any point where its Jacobian is invertible (det ≠ 0), and the inverse is smooth. Project tie: exp and log maps are local inverses of each other, and IFT is what guarantees that smooth local inverse exists — the whole reason Phase 1 ended here.
Derivative = best linear map. The derivative at a point is the matrix of the linear map that best approximates the function nearby. 1D = slope number; many-D = Jacobian matrix. Gradient = the Jacobian's special case when output is a single scalar (one row).
Determinant = volume-scaling factor. det ≠ 0 means no dimension squashed flat → map is reversible. det = 0 means a dimension collapsed → information destroyed → no inverse. This is exactly the IFT's invertibility condition.
exp / log maps. exp_p(v) = "start at p, walk along the geodesic in direction v, report where you land" (curved addition; takes optimizer steps). log_p(q) = "what vector connects p to q" (curved subtraction; its length IS the distance). On the Poincaré ball both have closed forms (tanh / artanh).
One training step, end to end. Six stages: (1) read points u,v,w; (2) distances via log map (closed-form artanh); (3) loss → one scalar; (4) backward() builds gradients by chaining Jacobians; (5) Riemannian rescale (×(1−‖x‖²)²/4) converts Euclidean grad → manifold-correct grad; (6) exp map moves the nodes. Positions change ONLY in stage 6.
Forward vs backward clarified. Forward = evaluate functions (f₁, f₂) producing values; Jacobians are NOT used, only recorded. Backward = multiply Jacobians (J₂·J₁) producing sensitivities (∂loss/∂param). Same steps, opposite directions, different objects flowing.
Gradient internals. Each gradient number = a per-coordinate sensitivity ("nudge this coord → loss moves this much"). Whole-model gradient is one long row (because loss is one scalar), sliceable into one D-vector force per node. exp map consumes it as negate-and-scale velocity → new position.
Scale of training pinned down. node = one code; edge = one parent-child link (46,816 of them = the training data); batch = 1024 edges processed together; epoch = all edges once (~46 batches); run = 300 epochs (~13,800 updates); grid = 12 runs. Each backward sums forces over all 1024 edges in the batch onto shared node rows.
Computation graph. PyTorch's recorded breadcrumb trail of every forward operation (+ the local info to differentiate each). backward() walks it in reverse applying each step's Jacobian; rebuilt fresh each forward, discarded after — why kernel reset / missing intermediates break backward().
Continuing / Open
Workshop now entering Phase 2 (Riemannian mechanics): tangent spaces, the metric tensor as the position-dependent ruler, geodesics, exp/log formalized. Done-criterion: compute by hand the tangent space + metric tensor at a generic point on the sphere and on the Poincaré ball.
3 phases after that: 3 variable curvature & geodesics, 4 PyTorch/geoopt + RSGD-vs-RAdam-vs-Adam, 5 synthesis back to results.
Chapter-classification interim eval still unrun (~half day). Three Phase 3 math derivations not started. Second-pass lit search overdue. Poincaré d=10 NN diagnostic drafted but unrun; master ref doc needs the 5/17 review folded in.
To Do Next
Immediate: Begin workshop Phase 2 (metric tensor as the central object — the thing the novel method makes learnable). Run the chapter-classification interim eval in parallel.
Short term (1–2 wks): Workshop Phases 2–3. Complete the three Phase 3 math derivations. Begin Phase 3 Layer 1 — parameterize λ_θ(x) = base_λ(‖x‖)·(1 + α·ρ_θ(x)); smoke-test α=0 against the Poincaré baseline.
Medium term: Workshop Phases 4–5. Implement Method 1 (tree-anchored curvature) at d=5/10/50, K=50; then Method 2 (alternating optimization) as stretch.
Standing: Second-pass lit search before Phase 3 code. Download HPO/Mondo/Orphanet to data/raw/ now. Start MIMIC-IV credentialing (1–3 wk lead). Update master ref + handoff each session.

