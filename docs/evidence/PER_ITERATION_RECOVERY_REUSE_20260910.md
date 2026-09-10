# Per-iteration terminal-recovery reuse (Float64)

## Problem

`src/hsd/product_cone_solve.jl` calls `_product_hsd_candidate_result!` at the end of every
accepted iteration of the product-HSD loop (`:733`, also `:522`, `:637`, `:706`, `:721`).
When the cheap direct verifier fails, the candidate falls through to
`_product_hsd_refined_optimal_result!`, whose only gate is
`recovered_residual <= sqrt(tol)` (1e-4 at tol=1e-8). That gate holds for most late
iterations, so a full terminal-grade recovery was paid repeatedly inside the solve:

1. a least-squares solve against the primal operator `Ad`,
2. a dense `(n+1) x m` reconstruction of the dual operator `[A'; b']`,
3. a dense wide least-squares solve against that operator,
4. a **second, unconditional** dense `(n+1) x m` reconstruction whose only consumer is the
   `cone === :exp` structural loop - dead work on a pure SOC/linear model.

`Ad`, `b` and the cone layout are fixed for the life of a solve, so items 2-4 are loop
invariant and items 1-3 refactorize an unchanged operator on every attempt.

This was the dominant unexplained cost on the finite full-unitarity dual family: 61% (C4)
and 69% (S256) of the core wall sat outside every populated phase timer, and per-iteration
core grew 5.94x for 1.91x more variables (cubic 7.01x, quadratic 3.66x).

## Change

`ProductHSDTerminalRecoveryCache{T}` (owned by `ProductConeHSDState`, never shared across
solves) stores the primal factorization, the dense dual operator, and its factorization,
keyed to the exact operator instance. The recovery path builds them once and reuses them;
the duplicate Exp-check reconstruction is materialized only when an Exp block actually
exists. Square operators keep the plain `\` dispatch and are never cached.

**No arithmetic changes.** For every non-square operator, `A \ rhs` is exactly
`qr(A[, ColumnNorm()]) \ rhs` - verified bit-identical in Julia 1.12.6 for dense wide, dense
tall, sparse wide and sparse tall operators; only square operators differ (they use LU) and
those are excluded. The cached rhs construction is character-for-character the one the
uncached helper used.

The cached path is Float64-only; every other arithmetic keeps its existing
provider-driven refinement unchanged.

## Evidence

Frozen arrays, same harness, threads=1, BLAS/OMP/MKL=1, warm reps:

| case | iters | solve-API median (baseline -> fixed) | core median | speedup | witnesses |
|---|---|---|---|---|---|
| FWD-C4 (140 vars, n=10) | 14 -> 14 | 0.110645 s -> 0.079537 s | 0.104617 s -> 0.072268 s | **1.39x** (core 1.45x) | bit-identical |
| S256 (268 vars, n=2) | 15 -> 15 | 0.650892 s -> 0.395847 s | 0.600038 s -> 0.357111 s | **1.64x** (core 1.68x) | bit-identical |
| H_J0_smoke_qp1 (28 vars) | 15 -> 15 | - | - | - | bit-identical |

Per-iteration core: 7.473 -> 5.162 ms (C4) and 40.003 -> 23.807 ms (S256).

Bit-identity covers the full witness, not only the objective: for both C4 and S256 the
primal `x`, the equality dual, all SOC duals and all tail duals are byte-for-byte equal
between the baseline and the fixed build; `cert_valid` stays `true` and the iterations are
unchanged. The 28-variable smoke case additionally reproduces the baseline
primal/dual objective exactly.

Regression: `test/certificate_layout_storage.jl` 48/48.

Artifacts: `timing/fix_check/` (fixed-run JSON, witnesses, and the applied diff),
baselines `timing/sdpx_c4.json` + `timing/scaled/scaled256_sdpx.json`.

## Not claimed

No claim for other problem families, other arithmetic (the MultiFloat/BigFloat refinement
path is deliberately untouched), threaded or larger runs, or the paper's exact numbers.
