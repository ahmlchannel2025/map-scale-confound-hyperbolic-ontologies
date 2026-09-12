MATH ISSUES TO RESOLVE DURING PHASE 3 DERIVATION WORK

=====================================================

Recorded from Stanford Agentic Review v3 (May 11, 2026)

1. CONFORMAL LENGTH FORMULA DIRECTION

   Issue: For g = λ² g^E, the arc length is ds = λ ‖dx‖ (multiply by λ).

   For g = (1/λ²) g^E, the arc length is ds = ‖dx‖ / λ (divide by λ).

   v4 proposal mixed these conventions inconsistently.

   Fix: Pick one convention. Standard convention in Poincaré work is

        g_x = (4/(1−‖x‖²)²) g^E, so the factor is in the numerator

        and arc length multiplies. Use this convention throughout

        and rewrite the line-integral approximation as

        d̂(u,v) = ∫₀¹ ‖u−v‖ · λ(u + t(v−u)) dt where λ here

        denotes the conformal factor (not its inverse).

2. K = −λ⁻² Δ log λ IS 2D ONLY

   Issue: This formula gives Gaussian curvature in 2D. In higher

   dimensions, sectional curvature depends on direction and second

   derivatives of λ in directions perpendicular to the section plane.

   Fix: Two options to consider during derivation work:

   (a) Restrict the rigorous math derivation to 2D. Empirical work

       can use higher dimensions with directly parameterized λ_θ(x)

       and curvature verified numerically rather than analytically.

   (b) Use the higher-dimensional formula for sectional curvature

       of a conformally Euclidean metric, which involves both Δ log λ

       and ‖∇ log λ‖². The general result is more complex but tractable

       for the specific parametric forms we use.

3. CARTAN-HADAMARD NOT GUARANTEED BY κ(v) ≤ 0 AT NODES

   Issue: Cartan-Hadamard requires κ(x) ≤ 0 everywhere on the

   manifold, not just at the embedded node positions.

   Fix: Verify κ(x) ≤ 0 empirically by sampling x at many points

   in the manifold during training. If violations occur, constrain

   the parametric form of λ more aggressively. Cartan-Hadamard

   should be treated as an empirically-verified property, not an

   analytically-guaranteed one.

4. ERROR BOUND FOR LINE INTEGRAL APPROXIMATION

   Issue: |d − d̂| ≤ C ‖u−v‖² sup|∇ log λ| was stated without

   derivation, and conflates geodesic deviation error with

   numerical quadrature error.

   Fix: Derive two separate bounds during Phase 3:

   (a) Geodesic-vs-straight-line error: bounded by Christoffel

       symbol magnitudes and segment length squared.

   (b) Numerical quadrature error: standard Gauss-Legendre bound,

       essentially zero for our segment lengths.

   Verify both empirically against numerical ODE on a held-out

   sample of pairs.

5. KERNEL SMOOTHING SCALABILITY

   Issue: Full kernel smoothing is O(N) per query. For 46,500 nodes

   and millions of distance computations per epoch, this is slow.

   Fix: Use compact-support kernels with k-nearest-neighbor cutoff.

   Each query only sums over the k nearest embedded nodes (e.g., k=50).

   With spatial indexing (HNSW or similar), each query becomes O(log N).

6. DERIVED VS INVERTED CURVATURE PRESCRIPTION

   Issue: Going from κ(x) → λ(x) requires solving a PDE, which is

   expensive and possibly unstable.

   Fix: Parameterize λ_θ(x) directly as a function of position and

   local tree statistics. Don't invert from κ at all. Curvature

   κ(x) becomes a derived diagnostic property of λ_θ(x) rather

   than a prescribed target.

These will be resolved during Phase 3 (math derivation and novel

method implementation). For Phase 2 baselines, none of these

affect the work — baselines use standard constant-curvature methods.