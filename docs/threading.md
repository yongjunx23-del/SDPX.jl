# Threading guide

Start Julia with `-t N` (or `JULIA_NUM_THREADS=N`) to enable threaded block factorization, the Schur build, and the line search. Threading is automatic — no option to turn on; `solve!` inspects `Threads.nthreads()` and the block count `L` itself and falls back to serial code when threading wouldn't help (`nthreads() <= 1` or `L <= 1`).

## Scheduling

Blocks are assigned to threads via longest-processing-time (LPT) greedy partitioning, weighted by an estimate of each block's cost (`k[l]^3 + m·k[l]^2/2`), not by contiguous chunking. This matters for bootstrap-shaped problems, where block sizes `k[l]` are heterogeneous — contiguous chunking can leave one thread with all the large blocks and another with all the small ones.

## The Schur build specifically

Dense and sparse storage use different race-free accumulation strategies:

- Dense blocks accumulate into one private `m×m` buffer per LPT bin. Those
  buffers are reduced serially in fixed bin order.
- Sparse blocks transform only their active variables and write compact
  upper-pair values into a buffer owned by that PSD block. Exact arrow
  problems assemble directly into compact global, coupling, and local blocks
  instead of a dense `m x m` matrix.

Neither path uses locks or atomics. Results are deterministic at a fixed
thread count. Changing the thread count can change dense reduction order and
therefore the final rounding by a small amount.

Residual construction, predictor/corrector right-hand sides, direction
recovery, local arrow factorization, and local arrow solves also use the
cached LPT schedule. At 180/360/600/900 CSDR blocks, the measured eight-thread
speedups are `1.11x`, `2.68x`, `3.50x`, and `3.81x`. The smallest case is
still below the useful parallel granularity; larger cases increasingly
amortize the barriers.

## `BigFloat` does not thread

**This is the one thing worth reading carefully if you're solving at `BigFloat`.** All threaded code paths automatically detect `BigFloat` and fall back to fully serial execution, regardless of `-t`. This was not a design choice made in advance — it was found empirically during development: the exact same LPT-scheduled, partial-buffer-accumulating code, applied to the exact same problem, gave bit-identical results for `Float64` and silently *wrong* results (differing from the correct answer by orders of magnitude, no error thrown) for `BigFloat`. The cause is that MPFR (which `BigFloat` is built on) is not safe to call truly concurrently from multiple OS threads in Julia — a lower-level issue than object aliasing (which is handled everywhere else in this codebase via disjoint per-block/per-task buffers).

Practically: if you need parallelism at very high precision, `Float64x4`/`Float64x{8}` (via `MultiFloats.jl`) thread normally and are bitstypes with no such restriction — consider one of those instead of `BigFloat` if your precision needs (a few hundred bits) allow it.

## What remains serial

The reduced global arrow factorization remains serial, as does the generic
dense KKT factorization used for non-arrow problems and explicit equality
columns. Multi-node distributed factorization is not implemented. For a
cluster, use Julia threads within one node and distribute independent SDP
instances across processes or job-array tasks.
