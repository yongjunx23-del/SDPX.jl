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

## Why the win is 1.4-1.6x and not more: the wide-QR solve is overhead-bound

An instrumented attribution lane (separate worktree, baseline `e6a2acf`) measured the
baseline cost of the recovered items on C4 and S256 and found exactly **10 eligible
recoveries per solve, of which only 1 is accepted** (the other 9 pay the full least-squares
cost and then fail the post-refinement maxabs/cone gates), with the dual wide solve
accounting for 54-55% of core (58.8 ms of 109.8 ms on C4; 255.7 ms of 465.1 ms on S256).

A temporary counter probe on this build confirmed the cache is fully effective: every solve
reports exactly **1 build and 9 reuses**, primal and dual, with no partial misses. So the
limited win is not a cache failure - it is the cost structure of the operator itself:

| operator | `qr(ColumnNorm)` | cached `F \ b` | uncached `A \ b` | cached vs uncached |
|---|---|---|---|---|
| 141x772 (C4 dual) | 2.772 ms | 2.710 ms | 5.433 ms | **2.0x** |
| 772x140 (C4 primal) | 2.152 ms | 0.185 ms | 2.342 ms | 12.7x |
| 268x1284 (S256 dual) | 15.030 ms | 9.510 ms | 24.741 ms | **2.6x** |
| 1284x267 (S256 primal) | 9.848 ms | 0.546 ms | 10.449 ms | 19.1x |

For the **wide** dual operator the solve after caching is still 2.0-2.6x slower than needed
because `F \ rhs` must apply the full 772- (resp. 1284-) wide reflector block: the operation
is dominated by temporary allocation and LAPACK call overhead, not by flops. Predicting the
observed savings from these microbenchmarks reproduces the measured 32.3 ms (C4) closely.

Consequence for the next change: caching the wide-QR factor is exhausted as a lever. The
remaining recovery cost can only be removed by (a) not paying 9 of 10 recoveries, or
(b) replacing the wide least-squares by the `(n+1)x(n+1)` Gram/minimum-norm form that the
high-precision path already uses - which is mathematically identical but changes
conditioning and the last bits, so it is a numerics decision, not a mechanical reuse.

## Not claimed

No claim for other problem families, other arithmetic (the MultiFloat/BigFloat refinement
path is deliberately untouched), threaded or larger runs, or the paper's exact numbers.
