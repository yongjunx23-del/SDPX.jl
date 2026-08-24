# Baseline — refactor/zeroalloc-factorcache-hsd

Branch: refactor/zeroalloc-factorcache-hsd (from main @ d185450).

Goal under test: steady-state zero-allocation Newton iterations across
Float64 / Float64x2 / Float64x3 / Float64x4 / BigFloat, unified
matrix / Workspace / FactorCache / symbolic-analysis reuse, original-coordinate
certificates, HSD infeasibility detection, and structural reduction.

Environment (this run):
- Julia 1.12.6, JULIA_DEPOT_PATH=…/SDPX/.julia-depot.
- Providers loaded: MultiFloatLinearAlgebra, BigFloatLinearAlgebra, GenericLinearAlgebra,
  MultiFloats (via a combined provider dev-environment that devs the local SDPX checkout).
- Host: apple-m4, 4 BLAS threads, single Julia thread for solves.

## 1. Baseline gates (all green on the base commit)

| Gate | Result | Time |
|------|--------|------|
| SDPX_TEST_PROFILE=quick (Pkg.test, 40 files) | PASS — 4046 pass, 1 broken, 0 fail | 4m35s |
| provider smoke (test/provider_smoke.jl, all target) | PASS — 180 pass | 2m36s |

## 2. Multi-precision LP / SOCP / SDP baseline

Reproducible driver: benchmark/core_matrix_x23_baseline.jl (Float64x2/x3) and
benchmark/runner.jl core_matrix (Float64 / Float64x4 / BigFloat256). Raw rows are
regenerated under benchmark/out/ (gitignored).

All rows below are single timed solve, warm-up excluded; allocB = @timed bytes of the
whole solve (not per-iteration); residuals/gap/certificate are original-coordinate values.

| problem | arithmetic | provider | status | iter | sec | allocB | p_res | d_res | gap | cert |
|---|---|---|---|---|---|---|---|---|---|---|
| lp_box | float64 | auto | Optimal | 7 | 0.0004 | 156016 | 0.0 | 7.5e-15 | 3.6e-9 | true |
| lp_box | float64x2 | auto | Optimal | 19 | 0.0076 | 1848104 | 0.0 | 6.3e-32 | 3.6e-21 | true |
| lp_box | float64x3 | auto | Optimal | 19 | 0.0057 | 1914312 | 0.0 | 2.3e-48 | 3.6e-21 | true |
| lp_box | float64x4 | multifloat | Optimal | 19 | 0.0070 | 2040088 | 0.0 | 9.4e-64 | 3.6e-21 | true |
| lp_box | BigFloat256 | bfla | Optimal | 19 | 0.0061 | 2066976 | 0.0 | 4.4e-58 | 3.6e-21 | true |
| soc_q3 | Float64 | auto | Optimal | 5 | 0.0004 | 76272 | 2.9e-9 | 5.1e-11 | 1.0e-9 | true |
| soc_q3 | Float64x2 | auto | Stalled | 20 | 0.0013 | 1373752 | 1.6e-50 | 1.7e-16 | 1.7e-16 | false |
| soc_q3 | Float64x3 | auto | Optimal | 11 | 0.0015 | 1332440 | 2.9e-21 | 5.1e-23 | 1.0e-21 | true |
| soc_q3 | Float64x4 | multifloat | Optimal | 11 | 0.0014 | 1360152 | 2.9e-21 | 5.1e-23 | 1.0e-21 | true |
| soc_q3 | BigFloat256 | bfla | Optimal | 11 | 0.0017 | 2023856 | 2.9e-21 | 5.1e-23 | 1.0e-21 | true |
| sdp_dense | Float64 | auto | Optimal | 8 | 0.0056 | 926224 | 0.0 | 7.1e-15 | 2.1e-9 | true |
| sdp_dense | float64x2 | auto | Optimal | 19 | 0.0155 | 3550696 | 0.0 | 1.0e-23 | 8.3e-22 | true |
| sdp_dense | float64x3 | auto | Optimal | 19 | 0.0160 | 3875368 | 0.0 | 5.4e-42 | 8.3e-22 | true |
| sdp_dense | float64x4 | multifloat | Optimal | 19 | 0.0158 | 4227432 | 0.0 | 4.1e-58 | 2.5e-21 | true |
| sdp_dense | BigFloat256 | bfla | Optimal | 20 | 0.0164 | 3389872 | 0.0 | 3.3e-53 | 9.7e-22 | true |

**Baseline finding:** soc_q3 at Float64x2 with the auto provider reaches only
Stalled (gap/d_res ~ 1.7e-16, certificate invalid). The same problem solves at every other
precision and at Float64x4 / BigFloat256. This is a numeric-behavior divergence worth a
dedicated regression gate before changing any hot-loop code (a candidate for Phase 5
fixed-precision and Phase 6 HSD work).

allocB is a whole-solve figure. **Per-iteration steady-state allocation is now measured** by
benchmark/allocation_profile.jl (one full predictor-corrector `newton_step!`, SDP route, min-of-3
after JIT warm-up). These are Julia heap bytes per iteration (BigFloat excludes MPFR-native heap):

| arithmetic | per-iteration Julia alloc (B) |
|---|---|
| Float64 | 7 360 |
| Float64x2 | 11 264 |
| Float64x3 | 12 720 |
| Float64x4 | 15 520 |
| BigFloat256 | 103 040 |

(Phase-4b destructuring reductions, measured per commit: `d481dff` (factorize! result) and
`db1cf53` (KKT phase timings) each saved ~80 B/iter across the family; cumulative Float64
9 264 -> 9 104. `predictor_diagnostics`/`iteration_parameters` destructuring gave zero measured
benefit and were reverted. The remaining ~9.1 KB is dominated by `_with_blas_threads` closures
(~1.5 KB, capturing large SolverOptions/Workspace) and the `newton_step!` return NamedTuple
+ `IterationDiagnostics` (~2.8 KB), which need the cold-state workspace refactor.)

So the hot loop is **not yet zero-allocation** (the Phase-4 target). `test/allocation_contract.jl`
registers a Float64 per-iteration regression ceiling (64 KB) plus an Optimal/certificate semantic gate.

`Profile.Allocs` on one Float64 `newton_step!` (9296 B total) attributes the hot-loop allocation to:

- NamedTuple construction + field access inside `newton_step!` (`step.jl` ~695/797/806/802/947,
  ≈2.8 KB/iter via `boot.jl:792 NamedTuple`) — the dominant source; these are per-iteration
  diagnostic records that the spec wants moved to cold state.
- `vec` / `reshape` in the block kernels (`schur.jl` `buildP_owned!`/`accumulate_v_owned!`).
- closure allocation in `_with_blas_threads` do-blocks (`step.jl:126`).
- `_kkt_factorization_quality` (`step.jl:346`).

These are the concrete Phase-4b targets for driving the iteration toward zero allocation.

`benchmark/allocation_phase_profile.jl` produces the **per-phase staged profile** (Phase-1b
deliverable) by attributing every `Profile.Allocs` byte to its producing phase (Float64):

| phase | bytes/iter | share |
|---|---|---|
| orchestration / diagnostics (newton_step NamedTuples, IterationDiagnostics, parameter records) | 7 920 | 83.2% |
| KKT assembly (schur / block kernels incl. `vec`/`reshape`) | 800 | 8.4% |
| factorize (fresh Cholesky factor handle per iteration) | 768 | 8.1% |
| KKT solve (predictor / corrector / refinement) | 32 | 0.3% |

So the numerical solve path is already essentially allocation-free; the Phase-4b win is to eliminate the
Newton orchestration/diagnostic NamedTuple construction (move it to cold state per the spec), and to reuse
the factor handle across iterations (Phase-3 FactorCache wiring).

## 3. Phase 3 increment — provider-neutral FactorCache (this branch)

Added a provider-neutral factor-cache interface and a reference dense implementation
(src/factor_cache.jl, wired into the module; test/factor_cache.jl):

- SymbolicCache (structure / ordering / permutation, reused across iterations).
- NumericFactorCache (current factor + matrix_epoch; -1 = invalid).
- SolveScratch (single-RHS, multi-RHS, refinement buffers).
- API: prepare!, reserve!, factorize!(cache,A,epoch), solve!, solve_multi!,
  refine_once!, invalidate!, isvalid, matrix_epoch, factor_kind.

The matrix_epoch contract guarantees one factorization per epoch with reuse for predictor,
corrector, and refinement; multi-RHS solves reuse one factor; invalidate! forces refactor.

Test: Phase 3 FactorCache — 115 pass across Float64, Float64x2/x3/x4, and BigFloat
(factor-once, epoch-refactor, single/multi-RHS reuse, refinement, invalidation).

The cache is deliberately additive and **not yet wired into the LP/SOC/SDP hot loops** (they
already carry route-specific typed factor storage). Wiring it in is the next step and must be
allocation-profiled against this baseline.

## 4. Remaining phases / blockers

- Phase 4 (zero-allocation contract): needs per-iteration and per-phase allocation profiles
  (assemble_kkt! / factorize_cached! / predictor / corrector / refinement / line search)
  across the five arithmetics; not yet measured. Must report Julia-alloc vs MPFR-native separately.
- Phase 2 (ExecutionPlan freeze): mostly present; audit Any/abstract fields in hot state.
- Phase 5 (fixed-precision contract): `test/fixed_precision_contract.jl` (added this branch)
  gates that `working_precision_policy=:fixed` uses one precision end-to-end (no ladder for
  fixed-width types, one rung at requested bits for BigFloat, valid certificate, no unauthorized
  fallback). The default is still `:auto`; flipping it to `:fixed` needs a ladder regression pass.
- Phase 6 (HSD): LP path already uses tau/kappa; no full bordered-system HSD embedding yet.
- Phase 7 (reduction): chordal.jl exists; no SymmetryReduction or ConeAlgebra layer yet.
- Known baseline anomaly to track: soc_q3 @ Float64x2 stalls.

