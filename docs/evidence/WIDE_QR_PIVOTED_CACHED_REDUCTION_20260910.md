# Guarded cached reduction for the wide pivoted-QR solve (Float64)

## Why

The Float64 terminal recovery solves the wide underdetermined system
`D dy = rhs` with `D = [A'; b']`, size `(n+1) x m` (141x772 on FWD-C4,
268x1284 on S256). Julia's `F \ rhs` for a wide `QRPivoted` factor takes the
`rnk < n` branch of `ldiv!(::QRPivoted, ::AbstractMatrix, rcond)`
(LinearAlgebra v1.12.6, `src/qr.jl:568-646`): it copies the whole `rnk x n`
factor block through `LAPACK.tzrzf!` and then applies `LAPACK.ormrz!` on every
call. That work is right-hand-side independent but was paid once per candidate.

Measured standalone: `F \ rhs` 2.710 ms on 141x772 and 9.433 ms on 268x1284
(allocating 1.04 / 2.97 MB), against `qr(D, ColumnNorm())` 2.834 / 14.916 ms.
Preallocating a caller buffer and calling `ldiv!` directly is bitwise
identical to `F \ rhs` but no faster, which located the cost inside `ldiv!`
rather than in caller allocation. A probe over one S256 solve showed ten
corrected candidates failing only the dual-cone gate and one further corrected
candidate passing and terminating the solve, so the repeated work cannot be
skipped by any pre-solve test: the cone verdict depends on the correction.

## What landed

`src/kernels/wide_qr_pivoted.jl` caches only the right-hand-side independent
part of the audited sequence - the rank estimate `rnk` and the `tzrzf!`
reduction `(C, tau)` - and replays the remaining `lmul!` / triangular /
`ormrz!` / permutation steps unchanged.

Guards, exactly as required by the review that authorized this change:

- Version gate on the audited Julia release `(1, 12)`, never overridden by a
  passing self-check. Provenance recorded per solve:
  `(julia = "1.12.6", linearalgebra = "1.12.0", blas = "lbt", adapter_revision = 1)`.
- Build-time self-check on each newly built operator: a deterministic,
  finite, non-zero probe is solved both by the untouched factor and by the
  cached reduction; agreement is required **bit for bit** as Float64 bit
  patterns (so signed zeros count), results must be finite, and the probe,
  `F.factors`, `F.tau` and `F.p` must be unmodified. Any mismatch, non-finite
  result or unsupported input disables the cached solve for that operator and
  the ordinary `F \ rhs` runs instead, with the refusal reason recorded.
- Restricted to dense, wide, real Float64 `QRPivoted` factors. Square, tall,
  non-Float64, empty and version-gated inputs return `nothing`.
- The adapter never mutates the factor or the caller right-hand side, and owns
  its own buffers; contaminated scratch is overwritten, not read.
- MIT notice and upstream provenance are retained in the adapter file.

The self-check demonstrates agreement for that operator, probe and runtime. It
is explicitly not a proof for every right-hand side, rank boundary or release,
which is why it is combined with the release gate and an unconditional
fallback rather than treated as authorization on its own.

## Tests

`test/wide_qr_pivoted_adapter.jl` (registered in `test/runtests.jl`), 110 tests:

- differential bit-identity against `F \ rhs` for four shapes and four random
  right-hand sides each, plus all-zero and signed-zero right-hand sides;
- full-row-rank, genuinely rank-deficient, duplicated-column, near-threshold,
  all-zero and zero-row operators, with repeated solves and deliberately
  contaminated scratch;
- non-mutation of factors, pivot vector, cached reduction and caller rhs;
- fallback for tall, square, Float32 and empty operators;
- a deliberately perturbed cached reduction must be refused by the self-check;
- recorded provenance and the release-gate expression;
- an end-to-end SDPX solve still certifies.

Full regression suite green (the only error observed before commit was the
suite's own dirty-source-tree guard).

## End-to-end result (frozen arrays, threads=1, warm, Float64)

| case | iters | solve-API baseline -> now | cumulative | core baseline -> now | cumulative | witnesses |
|---|---|---|---|---|---|---|
| FWD-C4 | 14 -> 14 | 0.110645 -> 0.052204 s | 2.12x | 0.104617 -> 0.046012 s | 2.27x | byte-identical |
| S256 | 15 -> 15 | 0.650892 -> 0.194002 s | 3.36x | 0.600038 -> 0.166638 s | 3.60x | byte-identical |

Cumulative includes the recovery-factor reuse, the certificate-replay reorder,
the staged Schur contraction and this adapter. Primal `x`, equality dual, all
SOC duals and all tail duals remain byte-identical to the pre-change baseline;
status, iterations and certificate validity are unchanged. Activation was
observed as `selfcheck_passed` with zero ordinary-solve fallbacks over the
timed solves.

## Not claimed

No claim for non-Float64 arithmetic (the high-precision path keeps its existing
provider-driven refinement), for other Julia releases, or for the Gram /
minimum-norm reformulation, which was considered and explicitly not authorized.
No matched-quality speed comparison against Clarabel is implied.
