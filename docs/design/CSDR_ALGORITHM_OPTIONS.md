# CSDR algorithm-level optimization options (survey record)

Date: 2026-09-09. Read-only survey at solver HEAD `4ab9b8e` (surveyor ran two
bounded Julia processes; no solves). This note records the ranked findings and
the explicit non-opportunities so they are not re-investigated. It is context,
not a qualification claim.

## Verified structural facts (frozen α3)

- Input SHA `2e7bac1d…c53f97d7`; `B` is 8400×42 with **298,146 nonzeros (84.51%
  dense)**; no zero equality columns/spectral rows or duplicate equality
  columns; all 8400 objective coefficients nonzero; equality RHS zero.
- Float64 SVD diagnostic: singular values ≈ 32,622.3 / 0.50074 (all 42
  directions resolved in that diagnostic; **not a rank certificate**).
- Triangular Gram: **7,585,200 multiply-add terms** vs ≈49,392 scalar flops for
  dense 42×42 LU (different units; establishes the work imbalance).
- Model construction (frozen-driver style, warmed): **1.3214 s median and
  15,850,197,600 cumulative allocated bytes per build**, outside the solve
  timer.
- Existing observational receipt: 105 iterations, valid certificate,
  41.31 s / 23.06 s at 1/4 threads, digest `3a7833…`.

## Ranked opportunities

1. **Amortize scan construction; bulk affine assembly.** `src/modeling/affine.jl`
   concatenates/copies/sorts/rebuilds on every affine addition. Retain
   immutable sampled data/equality panel/structure for repeated
   fixed-structure builds; build coefficient lists once. 100 identical
   structures could save ~131 s and ~1.57 TB cumulative allocations
   (extrapolation). **R2 gives no sparse-symbolic saving for α3**: it uses a
   dense `ProviderLPLUCache` (`fixed_trace_q3.jl:932-960`) and
   `product_cone_hsd.jl:546-560` returns before the session context reaches the
   general core. Changing `N_alpha/N_mu/N_a/spin cutoff` is not a c/b-only
   update — group reuse by exact structure only.
2. **Newton epoch reduction via targeted step/centrality control.** Historical
   101→93 iterations failed strict objective agreement; an extra corrector every
   iteration needs ~18 fewer iterations just to break even. Screening target
   5–10 fewer iterations, selective corrections only.
3. **Certified precision ladder.** Potentially large (lower-precision Gram/cone/
   vector work with target-precision verification and fail-closed promotion);
   high certificate risk; an x2 certificate does not satisfy the x4 contract.
4. **Factored equality operators / matrix-free Schur.** F3L storage
   26,127,360→2,023,056 bytes but slower forward/adjoint and only ~1.11×
   identity-metric Schur action (Float64). α3 is a poor Krylov target
   (~705,600 MACs per action → ~10.75 actions per Gram assembly). Better for
   larger α/smeared panels.
5. **Smeared formulation quadrature efficiency** (active direction). Halving Nμ
   roughly halves Q3 blocks; a different finite approximation, not equivalent to
   frozen α3; requires paired extrema at matched scientific convergence.

## Non-opportunities (do not re-investigate)

Small 42×42 LU; further Q3 local elimination/SoA/free variables (implemented);
predictor/corrector factor reuse (implemented); sparse/reordered α3 Gram
(84.51% dense, overlapping supports); rank dropping (no evidence); generic
presolve (not wired, cone-row/objective safety insufficient); crossing-orbit
aggregation (crossing enters through invariant kernels, not spectral orbits);
pure-ρ ansatz (different parameterization, loses disjoint-tail specialization);
FFT/Toeplitz (non-uniform mapped energies, no translation invariance); chordal
decomposition (blocks already independent 2×2); SDPB-style tighter tolerances
(do not themselves reduce iterations).

## Frozen-fingerprint disposition

`CSDR/CURRENT_WORK.md` demands the legacy 101-iteration objective/digest;
the scientific-core roadmap assigns that fingerprint to the old algorithm and
requires a separate standard-v1 baseline. Per supervisor decision: **the roadmap
governs this line of work; the legacy guard stays unchanged; 105 iterations /
`3a7833…` is an unqualified current-algorithm observational baseline**. A new
default baseline requires owner reconciliation plus matched
source/input/environment receipts and an independent original-coordinate
certificate.

## Recommended first experiment

Bulk affine construction (see opportunity 1). Accept gate: canonical
coefficients/constants/cone layout/reconstruction exactly equal; no rank/
arithmetic/tolerance/default change; ≥5% warmed construction time improvement;
no allocation increase; paired solves (when run) preserve the current-algorithm
terminal fields and original-coordinate certification.
