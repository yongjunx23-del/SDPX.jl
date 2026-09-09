# R2 Full Qualification Evidence: Lifecycle & Symbolic/Numeric Separation

**Status: IMPLEMENTED & VERIFIED AT HEAD** (SDPX 0.6.1)

## Summary of Completed Implementations

1. **R2-A: True Symbolic/Numeric Separation in PreparedSolver**
   - Implemented `NativeExecutionContext` passing down `_solve_prepared! -> _bridge_sdp_solve -> optimize! -> _public_optimize_native_hsd -> _public_native_hsd_core -> _product_cone_hsd_state -> _prepare_product_hsd_symmetric_core -> prepare_symmetric_core_state -> _build_float64_core_cache`.
   - Connected `SessionSymbolicSlot` in `SolveState{Float64}`.
   - Verified that on an active session symbolic lease, the cache is leased via `lease_symbolic_cache!` and factors are reused across solves with zero new symbolic analyses.
   - **Gate Evidence** (`validation/scientific_core/test_r2a_symbolic_numeric_separation.jl`):
     - Warm 100 updates on an already-warmed session: **0 new symbolic analyses** (`delta_100 == 0`).
     - Cold 100 solves on a fresh session: **exactly 1 symbolic analysis** on the initial solve, 0 on the subsequent 99 solves (`cold_delta == 1`).

2. **R2-B: Invalidation Transactions**
   - Structural mismatch (`PreparedStructureMismatch`) detaches the leased factor and discards the slot entry.
   - Structure cache clear / disable (`clear_structure_cache!`, `set_structure_cache_enabled!(false)`) advances cache generation and triggers clean re-analysis on subsequent solve.
   - Failures (NaN, zero-pivot, non-optimal result) discard the leased factor and keep the slot empty, ensuring no stale factor reuse.
   - Bordered-to-expanded fallback discards the bordered lease before expanded execution.

3. **R2-C: Concurrent Owner Isolation**
   - Independent `PreparedSolver` sessions own distinct `SessionSymbolicSlot` instances with disjoint factors, workspaces, and numerical memory.
   - Same-session reentrant/concurrent execution is rejected by `state.busy` / session lock (`ArgumentError`).
   - Mutation of results from one session does not affect results or state of another session.

4. **R2-D: Resource Accounting & Stable Allocations**
   - Memory allocation per solve is strictly bounded across repeated solves with no runaway leak (`test/test_r2_full_qualification.jl`).
   - Timings for preprocessing, core solving, reconstruction, and refinement are separately tracked and reported.
