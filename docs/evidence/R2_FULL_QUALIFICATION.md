# R2 Symbolic/Numeric Separation Evidence (narrow scope)

**Status: implementation + narrow tests only. R2 is NOT fully closed.**

This file records what has actually been executed and what remains open. It must
not be read as a claim of full R2 lifecycle/resource qualification.

## R2-A: session-local symbolic reuse (gate passed)

- `NativeExecutionContext` is threaded through the internal seam
  `_solve_prepared! -> _bridge_sdp_solve -> _optimize_impl ->
  _public_optimize_native_hsd -> _public_native_hsd_core ->
  _product_cone_hsd_state -> _prepare_product_hsd_symmetric_core ->
  prepare_symmetric_core_state -> _build_float64_core_cache`.
  The public `optimize!` signature is unchanged and lease-free.
- `SessionSymbolicSlot` is retained per `SolveState{Float64}`.
- Checkout happens **before** validation (approved lease protocol): a failed
  update (NaN objective/RHS, structural mismatch, non-optimal result) still
  revokes the retained entry and advances the attempt, so no stale factor
  survives a failed update.
- Ordinary provider selection runs first and is independent of any lease; a
  lease only authorizes reuse of a compatible CHOLMOD entry (a lease must not
  select the provider).
- **Gate** (`validation/scientific_core/test_r2a_symbolic_numeric_separation.jl`):
  Warm 100 updates on an already-warmed session = **0** new symbolic analyses;
  Cold 100 solves on a fresh session = **exactly 1**.

## R2-B: invalidation transactions (narrow tests)

- Structural mismatch (`PreparedStructureMismatch`) revokes/discards the leased
  entry; structure-cache clear/disable advances the generation and forces
  clean re-analysis on the next solve, then reuse resumes on the new
  generation.
- Failed updates (NaN) revoke the retained entry and advance the attempt
  counter (verified by the checkout-before-validation transaction).

## R2-C/D: narrow scope only (NOT full closure)

- R2-C currently exercises **sequential** independent sessions plus manual
  `busy` rejection. It does **not** establish multithread/task-level
  concurrency qualification.
- R2-D currently measures allocation variation across ten solves. It does
  **not** establish retained-live-object bounds, actual capacity, MPFR/GMP or
  thread scratch, GC overlap, or RSS behavior, and it is not a complete phase
  accounting of the peak.

Full R2 closure still requires: multithread/task-level concurrency
qualification, a complete retained-live-object/peak bound, and phase/RSS
accounting. Until then the memory-admission and full-lifecycle gates remain
open.
