# Refactor status — branch refactor/zeroalloc-factorcache-hsd

Authoritative snapshot of the committed work on this branch (39 commits ahead of
main at time of writing) and the precise remaining roadmap. All changes are
gated: every hot-loop change was measured with benchmark/allocation_profile.jl
and kept under the full-family allocation contract; every feature has tests;
quick (4308) and full (11240) suites are green.

## Committed per phase

### Baseline & measurement (Phase 1)
- test profile (quick/full) exercised via Pkg.test; provider smoke path exists.
- benchmark/allocation_profile.jl (per-iteration SDP alloc, min-of-3),
  allocation_phase_profile.jl (per-phase), core_matrix_x23_baseline.jl,
  multiprecision_lp_socp_benchmark.jl (LP/SOCP across Float64/x2/x3/x4/BigFloat).
- docs/evidence/development-reviews/refactor-zeroalloc-factorcache-hsd.md (baseline
  report with LP/SOCP/SDP x arithmetic tables and the soc_q3@x2 anomaly).

### ExecutionPlan freeze (Phase 2)
- ExecutionPlan{T,Route,Formulation,Provider} authoritative + asserted at exec.
- Phase-2 audit doc; typed Workspace.backend (KKTBackend moved earlier).

### FactorCache (Phase 3)
- src/factor_cache.jl: FactorCache{T}, SymbolicCache, NumericFactorCache,
  SolveScratch, DenseFactorCache; prepare!/reserve!/factorize!/solve!/
  solve_multi!/refine_once!/invalidate!. test/factor_cache.jl (115 across 5 arith).

### Zero-allocation hot loop (Phase 4)
- Per-iteration SDP allocation reduced Float64 9264 -> 6464 B (~30%) via
  no-logic destructuring (factorize! result, KKT phase timings), closure
  inlining (all _with_blas_threads do-blocks), and relocating phase timings to a
  preallocated NewtonPhaseTimes Workspace record.
- full-family allocation gate (test/allocation_contract.jl) with tightened ceilings.

### Fixed-precision contract (Phase 5)
- default working_precision_policy -> :fixed; test/fixed_precision_contract.jl.

### HSD (Phase 6)
- src/hsd.jl: hsd_skew_embedding, is_skew_symmetric, primal/dual infeasibility
  certificates, hsd_status, hsd_bordered_system (prototype), hsd_classify.
- test: infeasibility-certificate contract gate (60) + hsd tests (35).
- hsd-integration-design.md: plan to wire tau/kappa into the LP loop.

### Structural reduction (Phase 7)
- ConeAlgebra: PSD (psd_*) + orthant (orthant_*) + Lorentz (soc_*) ops (29 tests).
- SymmetryReduction: SymmetryGroup, orbits, is_matrix_invariant, pair_orbits
  (commutant), transposition / cyclic-DFT / S_n block-diagonalization,
  reconstruct_matrix, invariant_matrix_dimension (34 tests).

## Remaining roadmap (tracked)

1. Phase-4b structural: factorize! result de-boxing (~1.3 KB/iter) and
   return-NamedTuple relocation of IterationDiagnostics/iteration_parameters
   into a preallocated record. High-risk multi-backend/include-order work.
2. Phase-6: wire the HSD embedding into the LP interior-point loop (bordered
   system reusing the main KKT factor) per hsd-integration-design.md.
3. Phase-7: general irrep block_permutation decomposition + ChordalReduction
   composition; wire reductions into the solver.
4. Phase-3: wire FactorCache into the route hot loops (currently interface+reference).
5. CI: add a GitHub Actions gate (quick + full + allocation contract + benchmarks)
   so the acceptance criteria are enforced automatically.

Every remaining item is regression-sensitive and should be done one focused
step with measure/rollback, per the session discipline.

