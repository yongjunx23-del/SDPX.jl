# Phase 2 / 4 audit — ExecutionPlan freeze & hot-state Any fields

Branch: refactor/zeroalloc-factorcache-hsd. Evidence-based audit of the hot path
to freeze the core architecture (Phase 2) and guide the remaining Phase-4b
zero-allocation work. All numbers are from benchmark/allocation_profile.jl and
allocation_phase_profile.jl on Float64, one full predictor-corrector Newton step
(SDP route).

## What is already frozen

- ExecutionPlan{T,Route,Formulation,Provider} exists and is authoritative: route,
  formulation, storage, LA provider, precision, and threads are decided once and
  asserted at execution (kkt_backend.jl select_backend, _assert_planned_backend!).
- Hot loop does not re-do algorithm selection or provider probing.
- Route-specific typed workspaces exist (LP LPWorkspace, SOC native, SDP Workspace,
  dense-augmented/arrow/sparse sub-workspaces).
- MFLA/BFLA/Generic/AppleAccelerate providers resolve through one AbstractLABackend seam.

## Hot-state Any / abstract fields (Phase-2 targets)

| location | field | why | proposed fix |
|---|---|---|---|
| Workspace | backend::Any | include-cycle workaround (KKTBackend defined after Workspace) | reorder includes or type as Union{Nothing,KKTBackend} |
| Workspace | sparse_kkt::Any | sparse KKT holder | type by sparse backend |
| DenseAugmentedKKTWorkspace | factor_diagnostics/inertia/pivot_blocks/permutation/factor_precision::Any | diagnostics | concrete typed struct |

Workspace.backend is read once per iteration via select_backend with a typed assert,
so it is not a dynamic-dispatch hotspot; still worth typing.

## Where the remaining ~9.1 KB/iteration lives (Phase-4b)

Profile.Allocs / per-phase profile: orchestration/diagnostics 83%, KKT assembly 8.5%,
factorize 8.2%, solve 0.3%.

Dominant sources, in order:
1. _with_blas_threads do-block closures (~1.5 KB/iter) — they capture the large
   SolverOptions (and Workspace/prob) by value; the closure escapes through
   set_blas_threads!/try, so it is heap-allocated. Fix: don't capture opts in the
   closure (read from a preallocated record) or redesign _with_blas_threads.
2. newton_step! return NamedTuple + IterationDiagnostics + iteration_parameters
   (~2.8 KB/iter) — per-iteration diagnostic records the spec wants in cold state.
3. KKT buffers: _build_equality_gram! ~512 B/iter, fresh la_cholesky_factor! handle.

## Cold-state relocation plan (sequenced, each measured and gated)

1. Add a preallocated NewtonIterationRecord (phase timings + iteration diagnostics +
   parameters) as a Workspace field; newton_step! writes into it and returns a small status.
   Removes ~2.8 KB/iter of NamedTuple/struct construction.
2. Reuse the KKT factor handle across iterations (Phase-3 FactorCache wiring) instead of a
   fresh cholesky object each iteration (~0.5 KB, removes factorize share).
3. Eliminate _with_blas_threads closure captures (hoist BLAS-thread handling / pass a
   non-allocating thunk) (~1.5 KB).
4. Type the Any fields above.

Each step: measure before/after with allocation_profile.jl, keep the full-family
allocation gate and quick suite green, roll back on regression.

## Status on this branch

Committed: measurement tooling + gates (per-iteration, per-phase, full-family,
fixed-precision), FactorCache interface, and two verified no-logic reductions
(d481dff, db1cf53; Float64 9 264 -> 9 104 B/iter). The remaining items above are
deferred structural refactors (Workspace / newton_step / KKT), not yet committed.

