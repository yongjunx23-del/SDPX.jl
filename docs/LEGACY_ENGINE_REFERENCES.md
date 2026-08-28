# Legacy engine remaining-reference manifest (Phase 10 prep)

> **Status:** `src/hsd/nonnegative_hsd.jl` was deleted in the subsequent
> nonnegative migration wave; generic helpers were migrated out of
> `src/solver/interior_point.jl` in the sixth wave. The other legacy files remain
> gated on removal of their production and active-test dependencies; see §4,
> §6, and §11.
> **Purpose:** Phase 10 deletes the legacy engines.  This manifest records every
> remaining reference to the six legacy engine files so the deletion work can be
> planned file-by-file without a repository-wide archaeology pass.  It is a
> living document: each wave that removes a reference must update it.
>
> **Scope:** `src/lp_solver.jl`, `src/lp_sparse.jl`,
> `src/solver/interior_point.jl`, `src/step.jl` (+ `src/step_hot.jl`),
> `src/hsd/nonnegative_hsd.jl`, `src/soc_native.jl`.
>
> **Fifth-wave baseline:** all six files were referenced at the time of that
> audit. Later waves deleted `src/hsd/nonnegative_hsd.jl` and extracted generic
> helpers from `interior_point.jl` and `step.jl`. Section 11 records the current
> blockers. The remaining legacy solve loops are not moved or renamed merely to
> claim deletion.

## 1. Load order / include sites

`src/SDPX.jl` includes the legacy files in this order (line numbers verified at
the fifth wave; they shifted by +1 when `entrypoint_bridge.jl` was added):

| Line | Include |
|---|---|
| 77  | `include("step_hot.jl")` |
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
  `_solve_sdp_core!`, `block_norm_stats`.
  (`ingest` itself is defined in `src/ingest.jl`, which is NOT legacy.)
- **Sixth-wave helper migration (`agent/commandcode-migration-interior`):**
  the generic helper groups below were moved out of the legacy engine to
  natural native owners, with byte-identical bodies and no behavior change.
  `interior_point.jl` remains included (line 139) and still hosts the full
  legacy solve loop, so all existing test references keep loading.
  - `dual_objective(prob::SDPProblem, y, Y)` →
    `src/certificates/certificates.jl` (objective/certificate seam).
  - `save_checkpoint` / `load_checkpoint` / `save_checkpoint_jld2` /
    `load_checkpoint_jld2` stubs → new `src/checkpoint.jl` (included right
    after `types/problems.jl`, where `Checkpoint` is defined).
  - `_replace_solver_options` / `_reround_solver_options` →
    `src/pipeline/options.jl` (`SolverOptions` value transforms; used by
    `src/adaptive_parameters.jl`, `src/step.jl`, and the legacy engine).
  - `_scaled_identity` → `src/pipeline/helpers.jl` (owned-scalar allocation
    utilities; used by `src/public/optimize.jl` and the legacy engine).
  - `_tolerance_precision_diagnostic`, `adaptive_working_precision_bits`,
    `_working_precision_success`, `_record_working_precision!`,
    `_build_precision_ladder_plan`, `_ladder_retry_decision`,
    `_patch_ladder_report!`, `_merge_ladder_result` →
    `src/pipeline/attempts.jl` (A0/A1 execution-attempt and precision-ladder
    authority; used by `src/pipeline/diagnostics.jl` and the legacy engine).
- **Remaining production `src` callers of its symbols (real calls, not
  comments):**
  - none outside the legacy engine itself. The four cross-file callers listed
    by earlier waves now resolve to the migrated homes above.
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

## 6. `src/hsd/nonnegative_hsd.jl` — legacy Nonnegative (LP-only) HSD engine (DELETED)

- **Deleted** by the `commandcode-migration-wave.nonnegative` wave.  The seven
  shared `HSDState` kernels that the production product-cone HSD path calls
  (`_hsd_matrix_finite`, `_hsd_direction_finite`, `_hsd_maxinf`,
  `_hsd_residual_homotopy_ok`, `_hsd_scatter_dx!`, `_hsd_update_record!`,
  `_hsd_trial_residual!`) moved verbatim to `src/hsd/common_runtime.jl`
  (included from `src/SDPX.jl` immediately after `hsd/hsd.jl`).
- The standalone legacy solver entry points `hsd_step!`, `hsd_cold_start!`,
  `hsd_solve!` and the Farkas infeasibility fallbacks (`_farkas_*`,
  `_try_farkas!`, `_try_dual_ray!`, `_hsd_border_solve!`, `_hsd_form_schur!`,
  `_hsd_direction!`, `_hsd_line_search!`) had no production `src` callers and
  were deleted without copying.  The production path uses
  `product_hsd_step!`/`product_hsd_solve!` (`src/hsd/product_cone_hsd.jl`,
  `src/hsd/product_cone_solve.jl`).
- **Remaining references (owned by the separate test/E2E migration wave):**
  - `test/hsd_rank_reduction_precision.jl` (quick profile) calls
    `SDPX.hsd_solve!` (lines 127/157/163/179) — will fail to compile until that
    file is translated or removed.
  - `test/hsd_border_failure.jl` (quick profile) calls
    `SDPX._hsd_border_solve!` (lines 70/130/146/158/171) — same.
  - `test/hsd_zeroalloc.jl` and `docs/design/HSD_FORMULATION.md`,
    `docs/PLAN.md` contain textual `nonnegative_hsd` references only.
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

The fifth-wave audit (`agent/legacy-source-deletion`) originally found all six
files referenced. The subsequent migration wave deleted
`src/hsd/nonnegative_hsd.jl` after extracting its seven product-HSD helpers;
the other five files still host symbols required by production `src` code or
active tests. Deleting those files now would break `SDPX.jl` loading and/or the
active `test/runtests.jl` suites.

| File | Blocking production `src` references | Blocking active-test references |
|---|---|---|
| `src/lp_solver.jl` | none (comments only; `solve_lp!` called only from `solver/interior_point.jl`) | `LPWorkspace`, `LPScaling`, `LPReducedFactor`, `LPStandardFormSystem`, `LPDiagonalMatrix`, `_lp_*` kernels, `_resolve_lp_backend!`, `_presolve_lp_rows`, `_extract_lp_rows`, `_lp_workspace_bytes` in `test/lp_regressions.jl`, `test/direction_accuracy_lp.jl`, `test/architecture_regressions.jl`, `test/lp_sparse.jl`, `test/bigfloat_ownership_regressions.jl` |
| `src/lp_sparse.jl` | none outside the legacy LP pair | `select_lp_formulation`, `lp_sparse_candidate`, `lp_sparse_factor!`, `lp_sparse_solve!`, `LPSparseSystem`, `_lp_sparse_assemble` in `test/lp_sparse.jl`, `test/sparse_execution_round6.jl`, `test/kkt_sparse_backend.jl`, `test/direction_accuracy_lp.jl` |
| `src/solver/interior_point.jl` | none remaining outside the legacy engine itself — cross-file consumers now resolve against certificates, checkpoint, and pipeline owner files after the sixth-wave helper migration | `solve!` (27 active files), `_solve_sdp_core!`, `_kkt_cold_start_initialization`, `recommended_parameters`, `block_norm_stats`, `_equality_factor_diagnostics`, `_solve_pipeline!`, `_sdp_newton_termination_metadata`, `BestIterateWorkspace`, `_store_best_iterate!`, `_accepted_sdp_trial_residuals!` and other loop-owned symbols |
| `src/step.jl` | after the helper-migration wave, `compute_residuals!`, `factor_blocks!`, `_predictor_corrector_rhs!`, and `_has_owned_bigfloat_equality_arrow` live in `src/kernels/threaded.jl`; `step.jl` retains `newton_step!` and private legacy helpers consumed by `src/solver/interior_point.jl`, plus retained line-search helpers called by the threaded legacy path | `newton_step!`, `_affine_predictor_diagnostics!`, `_legacy_predictor_diagnostics!`, `_same_normalized_complementarity`, `_skip_automatic_refinement`, `factor_blocks!`, `compute_residuals!` in active legacy regression tests |
| `src/hsd/nonnegative_hsd.jl` | **DELETED** — shared kernels moved to `src/hsd/common_runtime.jl`; no production `src` references remain | old `hsd_solve!` and `_hsd_border_solve!` tests are owned by the test/E2E migration wave |
| `src/soc_native.jl` | production SOC engine for the `ConicProblem` path: `NativeSOCPlan`, `FixedTraceQ3Execution`, `_build_native_soc_payload` (`pipeline/plan.jl`), `_solve_native_soc_core` (`frontend/high_level_solve.jl:165,202`), `NativeSOCDiagnostics` (`soc_presolve.jl`, `validation.jl`, `public/optimize.jl`, `frontend/high_level_solve.jl`) | `test/soc_native_solver.jl`, `test/moi_native_soc.jl`, `test/soc_metric_sparse.jl`, `test/soc_singleton_presolve.jl`, `test/soc_regressions.jl`, `test/provider_smoke.jl` |

Ancillary checks performed this wave (no change needed):

- `engine=:legacy` selectors: already removed from the public surface
  (`src/public/settings.jl` `_validate_engine`, `src/moi_wrapper.jl`) and
  rejected with migration errors; the `:legacy` tokens that remain are the
  linear-algebra backend/provider vocabulary of §10.
- `Printf`: still used by `src/solver/interior_point.jl` (`print_iter`,
  lines 27/29), so the `Printf` runtime dependency is retained.
- Exports (`src/SDPX.jl`): the public export set is native-only and unchanged.
