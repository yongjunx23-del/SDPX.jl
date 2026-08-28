# Legacy engine remaining-reference manifest (Phase 10 prep)

> **Status:** maintained by the test/CI migration wave (`agent/legacy-test-ci-migration`).
> **Purpose:** Phase 10 deletes the legacy engines.  This manifest records every
> remaining reference to the six legacy engine files so the deletion work can be
> planned file-by-file without a repository-wide archaeology pass.  It is a
> living document: each wave that removes a reference must update it.
>
> **Scope:** `src/lp_solver.jl`, `src/lp_sparse.jl`,
> `src/solver/interior_point.jl`, `src/step.jl` (+ `src/step_hot.jl`),
> `src/hsd/nonnegative_hsd.jl`, `src/soc_native.jl`.

## 1. Load order / include sites

`src/SDPX.jl` includes the legacy files in this order:

| Line | Include |
|---|---|
| 76  | `include("step_hot.jl")` |
| 83  | `include("hsd/nonnegative_hsd.jl")` |
| 102 | `include("cold_start.jl")` (shared, NOT legacy) |
| 107 | `include("soc_native.jl")` |
| 133 | `include("step.jl")` |
| 135 | `include("lp_sparse.jl")` |
| 136 | `include("lp_solver.jl")` |
| 137 | `include("solver/interior_point.jl")` |

## 2. `src/lp_solver.jl` — dedicated legacy LP solver (`solve_lp!`)

- Defines: `solve_lp!`, `LPScaling`, `LPDirectionGateRecord`, `LPWorkspace`
  internals, `_lp_*` kernels (`_lp_auto_parameter_resolution`, `_scale_lp!`,
  `_lp_phase2_cold_start!`, `_lp_initialization_record`).
- **src callers:**
  - `src/solver/interior_point.jl` — calls `solve_lp!` for LP blocks inside the
    legacy SDP path (also calls `_scale_lp!` before it).
  - `src/pipeline/diagnostics.jl` — comment references `_scale_lp!`.
  - `src/types/plans.jl` — comment references `_scale_lp!`.
- **Public-API wrappers still exposing legacy LP entry points:**
  - `src/frontend/high_level_solve.jl` — `solve_lp` / `solve_socp` compatibility
    wrappers.
  - `src/lp_api.jl` — `solve_lp` wrapper.
  - `bin/sdpx_solve.jl` — CLI bridge over `ingest`/`SolverOptions`/`solve!`.
- **Tests:** `test/lp_regressions.jl` (legacy LP solver regression suite, uses
  `solve_lp`, `_resolve_lp_backend!`, `_presolve_lp_rows`); `test/solver_regressions.jl`;
  `test/preprocessing_regressions.jl`; `test/options_interface.jl`
  (`SolverOptions`); `test/cli_bridge.jl` (legacy CLI); `test/pipeline.jl`.
- **Docs:** `docs/PLAN.md`, `docs/design/ROADMAP_1_0.md`, review/audit evidence
  documents (historical).

## 3. `src/lp_sparse.jl` — legacy sparse LP system

- Defines: `select_lp_formulation`, `LP_SPARSE_*`, `lp_sparse_candidate`,
  `lp_sparse_factor!`, `lp_sparse_solve!`.
- **src callers:** `src/lp_solver.jl` (lines 452/480/2282/2316).
- **Tests:** `test/lp_sparse.jl` (full file); `test/sparse_execution_round6.jl`
  (`lp_sparse_candidate`/`lp_sparse_factor!`); `test/direction_accuracy_lp.jl`
  (`_lp_sparse_assemble`, `_lp_sparse_k0_infinity_norm`,
  `_lp_sparse_regularized_action!`).
- **Docs:** `docs/PLAN.md`, review evidence (historical).

## 4. `src/solver/interior_point.jl` — legacy SDP interior-point engine

- Defines: `SDPProblem`, `SolverOptions`, `Workspace`, `solve!`, `ingest`,
  `newton_step!` orchestration inputs, `_kkt_cold_start_initialization`,
  `_solve_sdp_core!`, `block_norm_stats`.
- **src callers:** 61 files mention `SDPProblem`/`SolverOptions`/`Workspace`
  (types, pipeline, kernels, la_backend, frontend, public).  Representative:
  `src/pipeline/*.jl`, `src/kernels/*.jl`, `src/types/*.jl`,
  `src/public/optimize.jl`, `src/schur.jl`, `src/workspace.jl`.
- **Tests:** `test/result_certificate.jl`, `test/nonfinite_fail_closed.jl`,
  `test/performance_trace.jl`, `test/correctness.jl`, `test/sparse.jl`,
  `test/genericity.jl`, `test/extended_precision_blas.jl`, `test/mixed_cones.jl`,
  `test/error_handling.jl`, `test/examples.jl`, `test/moi_regressions.jl`,
  `test/executed_diagnostics.jl`, `test/termination_metadata.jl`,
  `test/infeasibility_diagnostics.jl`, `test/ingest_regressions.jl`,
  `test/threads.jl`, `test/mixed_precision_kkt_regressions.jl`, and others.
- **Docs:** `docs/PLAN.md`, `docs/design/ROADMAP_1_0.md`, review evidence.

## 5. `src/step.jl` / `src/step_hot.jl` — legacy Newton step orchestration

- `src/step.jl` defines `newton_step!`, `compute_residuals!`, KKT phase helpers
  for the legacy SDP engine.
- **src callers:** `src/solver/interior_point.jl`, `src/workspace.jl`,
  `src/schur.jl` (comment), `src/types/constraints.jl`.
- `src/step_hot.jl` defines `HotStepState` + `step!` (zero-alloc LP hot step;
  superseded by `ProductConeHSDState`).
- **Tests:** `test/hot_step_zeroalloc.jl` (whole file); `test/factorizations_gate.jl`
  (whole file — runs `newton_step!` / `_kkt_cold_start_initialization`).
- **Docs:** `docs/PLAN.md`, review evidence.

## 6. `src/hsd/nonnegative_hsd.jl` — legacy Nonnegative (LP-only) HSD engine

- Defines: `hsd_step!`, `hsd_cold_start!`, `hsd_solve!`, `_hsd_border_solve!`,
  `_hsd_form_schur!`, `_hsd_direction!`, `_hsd_line_search!`, infeasibility
  fallbacks.
- **src callers:** none after this migration wave — only docstring/comment
  cross-references remain in `src/hsd/hsd.jl`, `src/hsd/native_hsd_public.jl`,
  `src/ir/canonical.jl` (comment), `src/hsd/product_cone_solve.jl` (comment).
- **Tests (kernel-level, deliberately retained as unit gates):**
  - `test/hsd_border_failure.jl` — `_p0b_border_call_noreturn!` still calls
    `SDPX._hsd_border_solve!` (low-level fail-closed/zero-alloc kernel gate).
- **Docs:** `docs/design/HSD_FORMULATION.md`, `docs/PLAN.md`, review evidence.
- **Note:** `ProductConeHSDState` reuses `HSDState` (defined in
  `src/hsd/hsd.jl`, which is NOT legacy) and the shared `hsd_residual!`,
  `hsd_conic_iterate`, `verify_*!` certificate machinery.

## 7. `src/soc_native.jl` — NativeSOCPlan Lorentz execution

- Defines: `NativeSOCPlan`, `NativeSOCWorkspace`, `_native_soc_*` kernels
  (cold start, direction, assembly, solve).
- **src callers:** `src/pipeline/plan.jl`, `src/types/plans.jl` (payload types),
  `src/public/optimize.jl`, `src/frontend/high_level_solve.jl`,
  `src/soc_presolve.jl`, `src/validation.jl`, `src/ir/lower_soc.jl` (lowering),
  `src/kkt/specializations/fixed_trace_q3.jl`.
- **Tests:** `test/soc_native_solver.jl`, `test/soc_regressions.jl`,
  `test/moi_native_soc.jl`, `test/moi.jl`, `test/soc_metric_sparse.jl`,
  `test/soc_singleton_presolve.jl`, `test/public/result_optimize.jl`
  (`lower_soc_native`), `test/modeling/lower_soc.jl` (`lower_soc_native`).
- **Docs:** `docs/PLAN.md`, review evidence.

## 8. Migration status after this wave (test/CI only)

Migrated to native product HSD (`ProductConeHSDState` +
`product_hsd_cold_start!`/`product_hsd_step!`/`product_hsd_solve!`):

- `benchmark/hsd_allocation.jl`
- `test/hsd_zeroalloc.jl`
- `test/hsd_direction_lp.jl`
- `test/hsd_full_newton_oracle.jl`
- `test/hsd_border_failure.jl` (legacy `hsd_step!` testset → product caller)
- `test/hsd_nonnegative.jl`

`engine=:legacy` references removed from:

- `test/moi_native_hsd.jl`, `test/public/settings_outputs.jl`,
  `test/public/native_hsd_optin.jl`, `test/benchmark_runner.jl`.

Source-text assertions retargeted from legacy files to product files:

- `test/adaptive_parameter_policy.jl` (now asserts
  `src/hsd/product_cone_hsd.jl`, `src/hsd/initialize.jl`,
  `src/hsd/product_cone_solve.jl`).

Workflows:

- `.github/workflows/test.yml`, `.github/workflows/optimization-benchmark.yml`
  (step names/comments now name the native product-HSD allocation gate).

## 9. Known product-path translation risks (preserved assertions)

These migrated assertions keep their original tolerances; if the native
product-HSD route classifies a fixture differently from the legacy engine, the
assertion fails visibly (fail-closed, never loosened):

- `test/hsd_nonnegative.jl` — "badly-scaled LP fails closed in unresolved
  precision" expects `ProductHSDBreakdown` for Float64 and
  `ProductHSDOptimal` for Float64x2/3/4; primal/dual-infeasible statuses and
  rank-ambiguous/rank-incompatible classifications map to their
  `ProductHSD*` enums.
- `test/hsd_border_failure.jl` — the exact full-step denominator fault is
  classified `HSDStepSingularKKT` by the product caller (documented by the
  pre-existing "product caller classifies the singular full border" testset).
- `test/benchmark_runner.jl` — explicit `engine=:native_hsd` requests on the
  benchmark harness fail closed (`:error`/`:none`) until a native solve
  adapter is registered in the harness.

## 10. Remaining `:legacy` tokens NOT in engine-selector scope

The linear-algebra backend selector `:legacy` (→ `:sdpx_legacy_la`) and the
benchmark harness route label `:sdpx_legacy` are separate namespaces from the
`engine=:legacy` selector.  They are exercised by
`test/la_backend_regressions.jl`, `test/v05_core_invariants.jl`,
`test/mfla_backend.jl`, `test/dense_augmented_kkt.jl`,
`test/generic_la_backend.jl`, `test/benchmark_runner.jl` (via `:auto`),
`benchmark/runner_impl.jl`, `benchmark/fresh_process_campaign.jl`,
`benchmark/compare_impl.jl`, and `benchmark/README.md`.  These are out of scope
for the `engine=:legacy` deletion unless the LA-backend vocabulary is also
retired.
