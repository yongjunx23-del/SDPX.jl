# Legacy engine remaining-reference manifest (Phase 10 prep)

> **Status:** verified at the Phase 10 fifth wave (`agent/legacy-source-deletion`).
> **Purpose:** Phase 10 deletes the legacy engines.  This manifest records every
> remaining reference to the six legacy engine files so the deletion work can be
> planned file-by-file without a repository-wide archaeology pass.  It is a
> living document: each wave that removes a reference must update it.
>
> **Scope:** `src/lp_solver.jl`, `src/lp_sparse.jl`,
> `src/solver/interior_point.jl`, `src/step.jl` (+ `src/step_hot.jl`),
> `src/hsd/nonnegative_hsd.jl`, `src/soc_native.jl`.
>
> **Fifth-wave finding:** none of the six files was deletable at this wave.
> Every file still hosts symbols required by remaining production `src` code
> and/or by active tests in `test/runtests.jl` (quick and full profiles).
> Section 11 lists the exact blocking symbols per file.  The include sites,
> the `Printf` runtime dependency, and the legacy `solve!`/`solve_lp!`
> reachability were all verified and are unchanged.

## 1. Load order / include sites

`src/SDPX.jl` includes the legacy files in this order (line numbers verified at
the fifth wave; they shifted by +1 when `entrypoint_bridge.jl` was added):

| Line | Include |
|---|---|
| 77  | `include("step_hot.jl")` |
| 84  | `include("hsd/nonnegative_hsd.jl")` |
| 103 | `include("cold_start.jl")` (shared, NOT legacy) |
| 108 | `include("soc_native.jl")` |
| 134 | `include("step.jl")` |
| 136 | `include("lp_sparse.jl")` |
| 137 | `include("lp_solver.jl")` |
| 138 | `include("solver/interior_point.jl")` |

## 2. `src/lp_solver.jl` — dedicated legacy LP solver (`solve_lp!`)

- Defines: `solve_lp!`, `LPScaling`, `LPDirectionGateRecord`, `LPWorkspace`
  internals, `_lp_*` kernels (`_lp_auto_parameter_resolution`, `_scale_lp!`,
  `_lp_phase2_cold_start!`, `_lp_initialization_record`).
- **src callers:**
  - `src/solver/interior_point.jl` — calls `solve_lp!` for LP blocks inside the
    legacy SDP path (also calls `_scale_lp!` before it).
  - `src/pipeline/diagnostics.jl` — comment references only (`solve_lp!`,
    `_scale_lp!`).
  - `src/types/plans.jl` — comment reference only (`_scale_lp!`).
  - `src/public/optimize.jl` — comment reference only (`solve_lp!`).
- **Public-API wrappers (now native):** `src/frontend/high_level_solve.jl`
  (`solve_lp` / `solve_socp` compatibility wrappers call the native LP
  frontend), `src/lp_api.jl` (`solve_lp` wrapper), `bin/sdpx_solve.jl` (CLI
  bridge over the native entrypoint bridge).
- **Tests (legacy LP regression/kernel gates, retained; all in the active
  suite):** `test/lp_regressions.jl` (`LPWorkspace`, `_lp_assemble_hessian!`,
  `_resolve_lp_backend!`, `_lp_solve_factor!`, `_lp_populate_kkt!`,
  `_extract_lp_diagonal_nonnegative`, `_presolve_lp_rows`, `LPScaling`,
  `_lp_cold_start_failure_result`, `_lp_auto_parameter_resolution`,
  `_lp_regularization_floor`, `_extract_lp_rows`, `LPDiagonalMatrix`,
  `LPReducedFactor`, `LPStandardFormSystem`, `_lp_executed_backend`);
  `test/direction_accuracy_lp.jl` (`LPWorkspace`, `LPDiagonalMatrix`,
  `LPStandardFormSystem`, `_lp_direction_accuracy_gate!`,
  `_lp_direction_acceptance_tolerance`, `_lp_populate_kkt!`,
  `_lp_regularization_floor`, `_lp_dense_k0_infinity_norm`,
  `_lp_reduced_k0_infinity_norm`, `_lp_sparse_k0_infinity_norm`,
  `_lp_sparse_regularized_action!`, `_lp_sparse_assemble`, `LPSparseSystem`);
  `test/architecture_regressions.jl` (`LPWorkspace`, `_resolve_lp_backend!`,
  `_lp_executed_backend`); `test/lp_sparse.jl` (`LPWorkspace`,
  `_lp_workspace_bytes`, `_extract_lp_rows`, `_extract_lp_rows_sparse`);
  `test/bigfloat_ownership_regressions.jl` (`_extract_lp_rows`);
  `test/sparse_execution_round6.jl`; `test/kkt_sparse_backend.jl`;
  plus `test/solver_regressions.jl`, `test/preprocessing_regressions.jl`,
  `test/options_interface.jl`, `test/cli_bridge.jl`, `test/pipeline.jl`
  (legacy `solve!`/`SolverOptions` surface).
- **Docs:** `docs/PLAN.md`, `docs/design/ROADMAP_1_0.md`, review/audit evidence
  documents (historical).

## 3. `src/lp_sparse.jl` — legacy sparse LP system

- Defines: `select_lp_formulation`, `LP_SPARSE_*`, `lp_sparse_candidate`,
  `lp_sparse_factor!`, `lp_sparse_solve!`, `LPSparseSystem`,
  `_lp_sparse_assemble`, `formulation_backend`.
- **src callers:** `src/lp_solver.jl` (lines 452/480/2282/2316, unchanged).
  No remaining production `src` caller outside the legacy LP pair.
- **Tests (retained):** `test/lp_sparse.jl` (full file);
  `test/sparse_execution_round6.jl` (`lp_sparse_candidate`/
  `lp_sparse_factor!`/`select_lp_formulation`); `test/kkt_sparse_backend.jl`
  (`select_lp_formulation`); `test/direction_accuracy_lp.jl`
  (`_lp_sparse_assemble`, `_lp_sparse_k0_infinity_norm`,
  `_lp_sparse_regularized_action!`).
- **Docs:** `docs/PLAN.md`, review evidence (historical).

## 4. `src/solver/interior_point.jl` — legacy SDP interior-point engine

- Defines: `SDPProblem`, `SolverOptions`, `Workspace`, `solve!`, `ingest`,
  `newton_step!` orchestration inputs, `_kkt_cold_start_initialization`,
  `_solve_sdp_core!`, `block_norm_stats`, checkpoint API
  (`save_checkpoint`/`load_checkpoint`), precision-ladder machinery.
  (`ingest` itself is defined in `src/ingest.jl`, which is NOT legacy.)
- **Remaining production `src` callers of its symbols (real calls, not
  comments):**
  - `src/public/optimize.jl:1483` — calls `_scaled_identity` (defined
    `src/solver/interior_point.jl:453/461`).
  - `src/pipeline/diagnostics.jl:276` — calls `_ladder_retry_decision`
    (defined `src/solver/interior_point.jl:3137`).
  - `src/adaptive_parameters.jl:775` — calls `_replace_solver_options`
    (defined `src/solver/interior_point.jl:135`).
  - `src/preprocessing.jl:2118` and `src/validation.jl:151/373/964` — call
    `dual_objective(prob::SDPProblem, y, Y)` (defined
    `src/solver/interior_point.jl:7`).
  - `src/kernels/threaded.jl` calls `step.jl` symbols that `interior_point.jl`
    in turn drives (see §5).
- **Tests:** `test/result_certificate.jl`, `test/nonfinite_fail_closed.jl`,
  `test/performance_trace.jl`, `test/correctness.jl`, `test/sparse.jl`,
  `test/genericity.jl`, `test/extended_precision_blas.jl`, `test/mixed_cones.jl`,
  `test/error_handling.jl`, `test/examples.jl`, `test/moi_regressions.jl`,
  `test/executed_diagnostics.jl`, `test/termination_metadata.jl`,
  `test/infeasibility_diagnostics.jl`, `test/ingest_regressions.jl`,
  `test/threads.jl`, `test/mixed_precision_kkt_regressions.jl`,
  `test/solver_regressions.jl` (`_solve_sdp_core!`),
  `test/factorizations_gate.jl` and `test/allocation_contract.jl`
  (`_kkt_cold_start_initialization`), `test/precision_ladder_plan.jl`
  (`_build_precision_ladder_plan`, `adaptive_working_precision_bits`),
  `test/extensions_regressions.jl` (`save_checkpoint`/`load_checkpoint`),
  `test/dense_augmented_kkt.jl`, `test/kkt_regressions.jl`,
  `test/bigfloat_sparse_schur_regressions.jl`, `test/pipeline.jl`,
  `test/adaptive_parameter_policy.jl`, `test/v05_core_invariants.jl`,
  `test/fixed_precision_contract.jl`, `test/bfla_backend.jl`, and others
  (27 active test files in `test/runtests.jl` call `SDPX.solve!`; the external
  `test/provider_smoke.jl` harness is intentionally not part of `runtests.jl`).
- **Docs:** `docs/PLAN.md`, `docs/design/ROADMAP_1_0.md`, review evidence.

## 5. `src/step.jl` / `src/step_hot.jl` — legacy Newton step orchestration

> **Helper-migration wave status (this wave):** the residual/factor/RHS
> helpers with native consumers were moved to their owners:
> `compute_residuals!`, `factor_blocks!`, `_predictor_corrector_rhs!`, and
> `_has_owned_bigfloat_equality_arrow` now live in
> `src/kernels/threaded.jl`. `src/step.jl` is **legacy-only**: every symbol
> it still defines (`newton_step!` and its private helpers) is consumed only
> by `src/solver/interior_point.jl` and its active tests. It is gated for
> **atomic deletion together with `src/solver/interior_point.jl`** after the
> test/E2E migration; the `include("step.jl")` site is retained until then.

- `src/step.jl` defines `newton_step!` and its private helpers
  (`_block_primal_residual_norm`, `_with_blas_threads`, `_kkt_blas_threads`,
  `_skip_automatic_refinement`, `_cholesky_diagonal_quality`,
  `_block_factorization_margins`, `_kkt_factorization_quality`,
  `_relative_regularization_from_attempts`,
  `_same_normalized_complementarity`,
  `_predictor_complementarity_diagnostics!`,
  `_affine_predictor_diagnostics!`, `_legacy_predictor_diagnostics!`,
  `line_search!`, `fraction_to_boundary_search!`) for the legacy SDP engine.
- **Remaining production `src` callers (real calls, not comments):**
  - `src/solver/interior_point.jl` — calls `newton_step!` and the step
    helpers (legacy pair).
  - `src/kernels/threaded.jl` — consumes the migrated kernels
    (`fraction_to_boundary_search!` / `line_search!` remain in `step.jl`
    and are called from `threaded_line_search!`; `compute_residuals!`,
    `factor_blocks!`, `_predictor_corrector_rhs!`,
    `_has_owned_bigfloat_equality_arrow` are now defined there).
  - `src/workspace.jl`, `src/schur.jl`, `src/types/constraints.jl` — comment
    references only.
- `src/step_hot.jl` defines `HotStepState` + `step!` (zero-alloc LP hot step;
  superseded by `ProductConeHSDState`; NOT part of this wave's deletion list).
- **Tests:** `test/hot_step_zeroalloc.jl` (whole file);
  `test/factorizations_gate.jl` (whole file — runs `newton_step!` /
  `_kkt_cold_start_initialization`); `test/solver_regressions.jl`,
  `test/bigfloat_sparse_schur_regressions.jl` (`_affine_predictor_diagnostics!`,
  `_legacy_predictor_diagnostics!`, `_same_normalized_complementarity`,
  `_skip_automatic_refinement`, `compute_residuals!`, `factor_blocks!`);
  `test/extended_precision_blas.jl`, `test/schur_scheduler_regressions.jl`,
  `test/allocation_contract.jl`, `test/sparse.jl`, `test/mfla_backend.jl`,
  `test/pipeline.jl`, `test/kkt_regressions.jl`, `test/sparse_schur_round7.jl`,
  `test/coo_contraction_regression.jl` (`factor_blocks!`).
- **Docs:** `docs/PLAN.md`, review evidence.

## 6. `src/hsd/nonnegative_hsd.jl` — legacy Nonnegative (LP-only) HSD engine

- Defines: `hsd_step!`, `hsd_cold_start!`, `hsd_solve!`, `_hsd_border_solve!`,
  `_hsd_form_schur!`, `_hsd_direction!`, `_hsd_line_search!`, infeasibility
  fallbacks (`_farkas_*`, `_try_farkas!`, `_try_dual_ray!`), AND the shared
  `HSDState` kernels listed below.
- **Remaining production `src` callers (real calls, NOT docstring-only; this
  corrects the earlier "no src callers" claim):** the native product-cone HSD
  path calls seven kernels defined here:
  - `_hsd_matrix_finite` — `src/hsd/product_cone_hsd.jl:1206,2834`
  - `_hsd_direction_finite` — `src/hsd/product_cone_hsd.jl:1347,2389,2501,2656`;
    `src/hsd/predictor_corrector.jl:264,413`
  - `_hsd_maxinf` — `src/hsd/product_cone_solve.jl:313,314,319`;
    `src/hsd/linesearch.jl:82,83,111,112`
  - `_hsd_residual_homotopy_ok` — `src/hsd/product_cone_solve.jl:317`;
    `src/hsd/linesearch.jl:117`
  - `_hsd_scatter_dx!` — `src/hsd/product_cone_hsd.jl:1316,2380,2456`
  - `_hsd_update_record!` — `src/hsd/product_cone_hsd.jl:3009`
  - `_hsd_trial_residual!` — `src/hsd/product_cone_solve.jl:312`;
    `src/hsd/linesearch.jl:110`
  - Docstring/comment cross-references also remain in `src/hsd/hsd.jl`,
    `src/hsd/native_hsd_public.jl`, `src/ir/canonical.jl` (comment),
    `src/hsd/product_cone_solve.jl` (comment).
- **Tests (kernel-level, deliberately retained as unit gates):**
  - `test/hsd_border_failure.jl` (quick profile) — `_p0b_border_call_noreturn!`
    calls `SDPX._hsd_border_solve!` (lines 70/130/146/158/171).
  - `test/hsd_rank_reduction_precision.jl` (quick profile) — calls
    `SDPX.hsd_solve!` (lines 127/157/163/179).
- **Docs:** `docs/design/HSD_FORMULATION.md`, `docs/PLAN.md`, review evidence.
- **Note:** `ProductConeHSDState` reuses `HSDState` (defined in
  `src/hsd/hsd.jl`, which is NOT legacy) and the shared `hsd_residual!`,
  `hsd_conic_iterate`, `verify_*!` certificate machinery.

## 7. `src/soc_native.jl` — NativeSOCPlan Lorentz execution

- Defines: `NativeSOCPlan`, `NativeSOCWorkspace`, `NativeSOCDiagnostics`,
  `_native_soc_*` kernels (cold start, direction, assembly, solve),
  `GeneralLorentzExecution`, `FixedTraceQ3Execution`,
  `_build_native_soc_payload`, `_solve_native_soc_core`.
- **Reachability:** NativeSOC is still the production SOC engine for the
  `ConicProblem` path.  `src/public/optimize.jl` (`_public_solve_soc_core` →
  `_run_native_soc_frontend`) and `src/frontend/high_level_solve.jl`
  (`solve_socp`) both call `_solve_native_soc_core`
  (`src/frontend/high_level_solve.jl:165,202`); `src/pipeline/plan.jl` builds
  the payload through `_build_native_soc_payload` and carries
  `NativeSOCPlan`/`FixedTraceQ3Execution`; `src/soc_presolve.jl`,
  `src/validation.jl`, `src/public/optimize.jl` and
  `src/frontend/high_level_solve.jl` construct/dispatch on
  `NativeSOCDiagnostics`; `src/kkt/specializations/fixed_trace_q3.jl` mentions
  `GeneralLorentzExecution` in a docstring only.  (`lower_soc_native` lives in
  `src/ir/lower_soc.jl`, NOT here.)
- **Tests:** `test/soc_native_solver.jl`, `test/soc_regressions.jl`,
  `test/moi_native_soc.jl`, `test/moi.jl`, `test/soc_metric_sparse.jl`,
  `test/soc_singleton_presolve.jl`, `test/provider_smoke.jl`
  (`_solve_native_soc_core`), `test/public/result_optimize.jl`
  (`lower_soc_native`), `test/modeling/lower_soc.jl` (`lower_soc_native`).
- **Docs:** `docs/PLAN.md`, review evidence.

## 8. Migration status after the test/CI and entrypoint waves (historical)

Migrated to native product HSD (`ProductConeHSDState` +
`product_hsd_cold_start!`/`product_hsd_step!`/`product_hsd_solve!`):

- `benchmark/hsd_allocation.jl`
- `test/hsd_zeroalloc.jl`
- `test/hsd_direction_lp.jl`
- `test/hsd_full_newton_oracle.jl`
- `test/hsd_border_failure.jl` (legacy `hsd_step!` testset → product caller;
  low-level `_hsd_border_solve!` kernel gates retained, see §6)
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

Production entrypoints (prepared sessions, all-auto frontend, CLI bridge,
benchmark harness) were migrated to the native entrypoint bridge
(`src/entrypoint_bridge.jl`) and call the public `optimize!` seam with
`engine=:native_hsd`; no legacy `solve!`/`solve_lp!` route is reachable from
them.  The legacy `solve!`/`solve_lp!`/`_solve_sdp_core!` surfaces remain
package-internal and are exercised only by the retained kernel/regression
tests listed in §2–§5.

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

## 11. Fifth-wave source-deletion verdict (this wave)

Verified at the fifth wave (`agent/legacy-source-deletion`): **none of the six
files was deleted**, because each still hosts symbols required by remaining
production `src` code or by active tests.  Deleting any file now would break
`SDPX.jl` loading and/or the active `test/runtests.jl` suites (CI default is
`SDPX_TEST_PROFILE=quick`), which the Phase 10 contract forbids removing.

| File | Blocking production `src` references | Blocking active-test references |
|---|---|---|
| `src/lp_solver.jl` | none (comments only; `solve_lp!` called only from `solver/interior_point.jl`) | `LPWorkspace`, `LPScaling`, `LPReducedFactor`, `LPStandardFormSystem`, `LPDiagonalMatrix`, `_lp_*` kernels, `_resolve_lp_backend!`, `_presolve_lp_rows`, `_extract_lp_rows`, `_lp_workspace_bytes` in `test/lp_regressions.jl`, `test/direction_accuracy_lp.jl`, `test/architecture_regressions.jl`, `test/lp_sparse.jl`, `test/bigfloat_ownership_regressions.jl` |
| `src/lp_sparse.jl` | none outside the legacy LP pair | `select_lp_formulation`, `lp_sparse_candidate`, `lp_sparse_factor!`, `lp_sparse_solve!`, `LPSparseSystem`, `_lp_sparse_assemble` in `test/lp_sparse.jl`, `test/sparse_execution_round6.jl`, `test/kkt_sparse_backend.jl`, `test/direction_accuracy_lp.jl` |
| `src/solver/interior_point.jl` | `_scaled_identity` (`public/optimize.jl:1483`), `_ladder_retry_decision` (`pipeline/diagnostics.jl:276`), `_replace_solver_options` (`adaptive_parameters.jl:775`), `dual_objective(prob,y,Y)` (`preprocessing.jl:2118`, `validation.jl:151/373/964`) | `solve!` (27 active files), `_solve_sdp_core!`, `_kkt_cold_start_initialization`, `save_checkpoint`/`load_checkpoint`, `recommended_parameters`, `adaptive_working_precision_bits`, `block_norm_stats`, `_equality_factor_diagnostics`, `_build_precision_ladder_plan`, `_solve_pipeline!`, `_sdp_newton_termination_metadata`, `BestIterateWorkspace`, `_store_best_iterate!`, `_accepted_sdp_trial_residuals!` |
| `src/step.jl` | legacy-only after the helper-migration wave: `newton_step!` + private helpers consumed only by `src/solver/interior_point.jl`; `kernels/threaded.jl` still calls the retained `line_search!`/`fraction_to_boundary_search!` | `newton_step!`, `_affine_predictor_diagnostics!`, `_legacy_predictor_diagnostics!`, `_same_normalized_complementarity`, `_skip_automatic_refinement`, `factor_blocks!`, `compute_residuals!` in `test/factorizations_gate.jl`, `test/allocation_contract.jl`, `test/solver_regressions.jl`, `test/bigfloat_sparse_schur_regressions.jl`, `test/extended_precision_blas.jl`, `test/schur_scheduler_regressions.jl` and others |
| `src/hsd/nonnegative_hsd.jl` | shared `HSDState` kernels used by the production product-cone HSD path: `_hsd_matrix_finite`, `_hsd_direction_finite`, `_hsd_maxinf`, `_hsd_residual_homotopy_ok`, `_hsd_scatter_dx!`, `_hsd_update_record!`, `_hsd_trial_residual!` (see §6) | `hsd_solve!` (`test/hsd_rank_reduction_precision.jl`, quick), `_hsd_border_solve!` (`test/hsd_border_failure.jl`, quick) |
| `src/soc_native.jl` | production SOC engine for the `ConicProblem` path: `NativeSOCPlan`, `FixedTraceQ3Execution`, `_build_native_soc_payload` (`pipeline/plan.jl`), `_solve_native_soc_core` (`frontend/high_level_solve.jl:165,202`), `NativeSOCDiagnostics` (`soc_presolve.jl`, `validation.jl`, `public/optimize.jl`, `frontend/high_level_solve.jl`) | `test/soc_native_solver.jl`, `test/moi_native_soc.jl`, `test/soc_metric_sparse.jl`, `test/soc_singleton_presolve.jl`, `test/soc_regressions.jl`, `test/provider_smoke.jl` |

Ancillary checks performed this wave (no change needed):

- `engine=:legacy` selectors: already removed from the public surface
  (`src/public/settings.jl` `_validate_engine`, `src/moi_wrapper.jl`) and
  rejected with migration errors; the `:legacy` tokens that remain are the
  linear-algebra backend/provider vocabulary of §10.
- `Printf`: still used by `src/solver/interior_point.jl` (`print_iter`,
  lines 27/29), so the `Printf` runtime dependency is retained.
- Exports (`src/SDPX.jl`): the public export set is native-only and unchanged.
