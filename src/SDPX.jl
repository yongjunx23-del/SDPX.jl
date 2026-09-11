module SDPX

using LinearAlgebra, Base.Threads, MathOptInterface, SHA, SparseArrays, Serialization
using LinearAlgebra: LowerTriangular, UpperTriangular, Symmetric, issuccess, mul!

include("cone_algebra.jl")
include("cones/symmetric/SymmetricCones.jl")
include("cones/nonsymmetric/dense3.jl")
include("cones/exponential.jl")
include("cones/exp_logarithmic.jl")
include("cones/power.jl")
# Internal Phase-3 nonsymmetric references. The line search is
# production-shaped and allocation-free for fixed-width arithmetic; the full
# Newton routine is deliberately a cold, independent sign/direction oracle.
include("cones/nonsymmetric/linesearch3.jl")
include("cones/nonsymmetric/types.jl")
include("cones/nonsymmetric/exact_spd3.jl")
include("cones/nonsymmetric/conjugate3.jl")
include("cones/nonsymmetric/scaling3.jl")
include("cones/nonsymmetric/corrector3.jl")
include("cones/nonsymmetric/initialization3.jl")
include("modeling/domains.jl")
include("modeling/refs.jl")
include("modeling/types.jl")
include("ir/types.jl")
include("program/transforms.jl")
include("ir/storage.jl")
include("ir/layout.jl")
include("cones/runtime/types.jl")
include("cones/runtime/product.jl")
include("cones/runtime/symmetric_api.jl")
include("cones/runtime/nonsymmetric_api.jl")
include("ir/canonical.jl")
include("program/equilibrate.jl")
# RSOC transform is defined after canonical IR types so its map can be used
# by the canonicalizer; `program/transforms.jl` is included once above.
include("program/transforms_rsoc.jl")
include("modeling/model.jl")
include("modeling/affine.jl")
include("modeling/constraints.jl")
include("modeling/starts.jl")
include("ir/reconstruction.jl")
include("modeling/compile.jl")
include("ir/route.jl")
include("types/core.jl")
include("types/backends.jl")
include("types/workspaces.jl")
include("types/constraints.jl")
include("types/problems.jl")
include("checkpoint.jl")
include("types/plans.jl")
include("types/results.jl")
include("memory_utils.jl")
include("frontend/solve_options.jl")
include("midend/resolve_options.jl")
include("public/settings.jl")
include("hsd/factor_pair_admission.jl")
include("public/outputs.jl")
include("kernels/api.jl")
include("kernels/generic.jl")
include("kernels/bigfloat.jl")
# Guarded cached reduction for the wide pivoted-QR solve; defines
# `ProductHSDWideQRReduction` before the product-cone state that caches it.
include("kernels/wide_qr_pivoted.jl")
include("la_backends/legacy.jl")
include("la_backend.jl")
include("kernels/extended_precision_blas/ExtendedPrecisionBLAS.jl")
include("blas_backend.jl")
include("factor_cache/state.jl")
include("factor_cache/requirements.jl")
include("factor_cache/api.jl")
include("factor_cache/symbolic_analysis_counter.jl")
include("factor_cache/routes.jl")
include("factor_cache/session_symbolic_lease.jl")
include("types/execution_context.jl")
include("kkt_route.jl")
include("kkt/system.jl")
include("kkt/scalar_closure.jl")
include("kkt/residual_workspace.jl")
include("kkt/block_incidence.jl")
include("kkt/factor_receipt.jl")
# Cross-solve structure cache + phase-timing accumulator (review slices 1-2):
# the symmetric-core KKT workspace and pattern constructor reference both.
include("hsd/phase_timings.jl")
include("factor_cache/structure_cache.jl")
include("kkt/symmetric_core.jl")
# Thin INTERNAL EXPERIMENTAL sparse-core wrapper (R3 bounded).  Included
# after `kkt/symmetric_core.jl` because the wrapper constructor consumes
# `SymmetricCorePattern`; it is not part of any public/native route.
include("factor_cache/routes/experimental_sparse_core.jl")
include("kkt/expanded_quasidefinite.jl")
include("kkt/psd_panels.jl")
include("kkt/reduced_schur.jl")
include("step_hot.jl")
# HSD state and the shared/product-cone HSD state machine. Included here
# (after the IR, factor-cache routes and the KKT route driver) so they can
# reference CanonicalConicProgram, ConeProductLayout, AbstractFactorCache and
# HotRouteCache.
include("hsd/equality_reduction_sparse.jl")
include("hsd/hsd.jl")
include("hsd/common_runtime.jl")
include("hsd/nonsymmetric_coupled.jl")
include("hsd/product_cone_hsd.jl")
include("hsd/nonsymmetric_schur3.jl")
include("hsd/product_cone_solve.jl")
include("certificates/certificates.jl")
include("hsd/equality_reduction.jl")
include("ingest.jl")
include("lp_api.jl")
include("soc.jl")
# Direct fixed-dimension KKT local contributions for product-HSD planning.
include("kkt/specializations/fixed_trace_q3.jl")
include("soc_lorentz_kernels.jl")
include("cold_start.jl")
include("hsd/initialize.jl")
include("midend/problem_features.jl")
include("midend/auto_planner.jl")
include("midend/formulation_planner.jl")
include("pipeline/helpers.jl")
include("pipeline/options.jl")
include("pipeline/classify.jl")
include("pipeline/resources.jl")
include("pipeline/route.jl")
include("pipeline/plan.jl")
include("pipeline/presolve.jl")
include("pipeline/workspace_estimate.jl")
include("pipeline/attempts.jl")
include("pipeline/diagnostics.jl")
include("pipeline/timing.jl")
include("stagnation.jl")
include("kernels/mixed_precision_kkt.jl")
include("kernels/sparse_coo.jl")
include("kernels/constraint_contractions.jl")
include("sparse_la.jl")
include("nullspace.jl")
include("chordal.jl")
include("preprocessing.jl")
include("prepared.jl")
include("validation.jl")
include("spectrum.jl")
include("frontend/high_level_solve.jl")
include("performance_trace.jl")
include("public/result.jl")
# R0-P4 opt-in half-Power factor-pair backend: reviewed arithmetic ported into
# an internal package namespace (design step 4).  These modules are internal
# implementation details; they are not exported and they do not change the
# default dense-metric route.  Independent research oracles stay in
# validation/scientific_core/.
include("hsd/factor_pair/factor_preserving_affine.jl")
include("hsd/factor_pair/native_factor_affine_certificate.jl")
include("hsd/factor_pair/half_power_native_corrector.jl")
include("hsd/factor_pair/factor_combined_epoch.jl")
include("hsd/factor_pair/native_half_pair.jl")
include("hsd/factor_pair/factor_pair_hsd.jl")
include("hsd/core_route_planner.jl")
include("hsd/native_hsd_public.jl")
include("public/optimize.jl")
include("accuracy_contract.jl")
include("entrypoint_bridge.jl")
include("moi_wrapper.jl")

# --- rebuild packet: integration cutover (@integration/core-cutover) ---------
#
# The 14 entry points added by the packet's S01-S06 tasks. Seven further files
# are deliberately NOT listed here, because a sibling above already includes
# them and listing them again would define their contents twice in two
# different namespaces:
#
#   certification/{status,direction}.jl   included by certification/original.jl
#                                         INSIDE `module SDPXCertification`
#   solver/{iterate,session,residuals,
#           globalization,recovery}.jl    included by solver/loop.jl
#
# Order is load-bearing, not stylistic:
#   * S03's four files resolve cross-file bindings at definition time;
#     `operator.jl` must precede the other three, and `session.jl` before
#     `strategy.jl` (whose methods are annotated on its types).
#   * S05's three files are included in the order the adapter contract declares.
#   * S04's `certification/original.jl` opens `module SDPXCertification` and
#     pulls its two siblings in itself.
#
# Verified before wiring: a copy with exactly these includes precompiles clean
# (`✓ SDPX`, zero warnings) and passes the full inherited suite (170 testsets,
# 9395 assertions, exit 0).
include("core/compiled_problem.jl")
include("core/transforms.jl")
include("kkt/operator.jl")
include("kkt/session.jl")
include("kkt/strategy.jl")
include("kkt/refinement_policy.jl")
include("la/protocol.jl")
include("la/admission.jl")
include("la/factor_lease.jl")
include("solver/loop.jl")
include("certification/original.jl")
include("planning/costs.jl")
include("planning/resources.jl")
include("planning/setup.jl")
# S07 session layer. ORDER IS MEASURED, NOT ASSUMED:
# `update.jl` MUST precede `replay.jl` (replay's struct fields name
# SessionTolerance/SessionScalarPayload/ProblemFingerprint; the driver measured
# update-then-replay compiling and replay-then-update failing).
include("session/update.jl")
include("session/replay.jl")
# `cancellation.jl` needs `solver_binding_is_complete` from solver/session.jl,
# which is reached through solver/loop.jl at line 194, so it must come after it.
include("session/cancellation.jl")

# v0.5 has one public modeling/solve interface.  Mature problem, workspace,
# provider, and legacy solve types remain package-internal implementation
# details and are intentionally not re-exported as parallel entry points.
export Model, variable!, constraint!, objective!
export Minimize, Maximize
export Settings, Limits, Tolerances, Outputs
export ZeroCone, Nonnegative, Reals, LorentzCone, RotatedLorentzCone
export PSDCone, ExponentialCone, PowerCone
export optimize!, execution_plan
export status, value, dual, dual_slack
export primal_objective, dual_objective
export objective_value, dual_objective_value
export primal_residual, dual_residual, relative_gap
export iterations, solve_time
export is_optimal, is_primal_infeasible, is_dual_infeasible
export primal_status, dual_status, termination_status
export num_variables, num_constraints
export variable_by_name, constraint_by_name, variable_names, constraint_names
export certificate, diagnostics, iteration_history, performance_trace
export Optimizer
export clear_structure_cache!, set_structure_cache_enabled!
export symbolic_analysis_count, symbolic_analysis_counts, symbolic_analysis_delta
export NonsymmetricBackendChoice, NativeNonsymmetricBackend
export ExperimentalHalfPowerFactorPairBackend, UnsupportedBackendError
export AccuracyContract, AccuracyClass, UnsupportedAccuracyContext
export AccuracyVerified, AccuracyUnsupported, AccuracyNumericalFailure
export AccuracyInfrastructureFailure
export accuracy_contract, accuracy_class

# Symmetric-cone algebra (Subagent I) lives in the nested module
# `SymmetricCones` (Nonnegative / SOC / PSDTriangle kernels). It is not part of
# the frozen public export set; callers reach it as `SDPX.SymmetricCones`.
import .SymmetricCones

end
