module SDPX

using LinearAlgebra, Base.Threads, MathOptInterface, Printf, SHA, SparseArrays, Serialization
using LinearAlgebra: LowerTriangular, UpperTriangular, Symmetric, issuccess, mul!

include("cone_algebra.jl")
include("cones/symmetric/SymmetricCones.jl")
include("cones/nonsymmetric/dense3.jl")
include("cones/exponential.jl")
include("cones/power.jl")
# Internal Phase-3 nonsymmetric references. The line search is
# production-shaped and allocation-free for fixed-width arithmetic; the full
# Newton routine is deliberately a cold, independent sign/direction oracle.
include("cones/nonsymmetric/linesearch3.jl")
include("cones/nonsymmetric/types.jl")
include("cones/nonsymmetric/conjugate3.jl")
include("cones/nonsymmetric/scaling3.jl")
include("cones/nonsymmetric/corrector3.jl")
include("cones/nonsymmetric/initialization3.jl")
include("cones/nonsymmetric/full_newton_reference.jl")
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
include("program/presolve.jl")
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
include("types/plans.jl")
include("types/results.jl")
include("memory_utils.jl")
include("frontend/solve_options.jl")
include("midend/resolve_options.jl")
include("public/settings.jl")
include("public/outputs.jl")
include("kernels/api.jl")
include("kernels/generic.jl")
include("kernels/bigfloat.jl")
include("la_backends/legacy.jl")
include("la_backend.jl")
include("la/sparse_capabilities.jl")
include("kernels/extended_precision_blas/ExtendedPrecisionBLAS.jl")
include("blas_backend.jl")
include("factor_cache/state.jl")
include("factor_cache/requirements.jl")
include("factor_cache/api.jl")
include("factor_cache/routes.jl")
include("kkt_route.jl")
include("kkt/system.jl")
include("kkt/block_incidence.jl")
include("kkt/factor_receipt.jl")
include("kkt/expanded_quasidefinite.jl")
include("kkt/reduced_schur.jl")
include("kkt/psd_panels.jl")
include("step_hot.jl")
# HSD state, Nonnegative (LP) HSD predictor/corrector, and certificate
# verification. Included here (after the IR, factor-cache routes and the KKT
# route driver) so they can reference CanonicalConicProgram, ConeProductLayout,
# AbstractFactorCache and HotRouteCache.
include("hsd/equality_reduction_sparse.jl")
include("hsd/hsd.jl")
include("hsd/nonnegative_hsd.jl")
include("hsd/nonsymmetric_coupled.jl")
include("hsd/product_cone_hsd.jl")
include("hsd/nonsymmetric_schur3.jl")
include("hsd/product_cone_solve.jl")
include("certificates/certificates.jl")
include("hsd/equality_reduction.jl")
include("ingest.jl")
include("lp_api.jl")
include("ir/lower_lp.jl")
include("ir/lower_sdp.jl")
include("ir/lift_psd.jl")
include("soc.jl")
# P4 direct fixed-dimension KKT local contributions.  Included after
# soc.jl (ConicProblem/SOCConstraint) and before soc_native.jl, which
# delegates its fixed-trace Q3 execution to this specialization.
include("kkt/specializations/fixed_trace_q3.jl")
include("ir/lower_soc.jl")
include("soc_lorentz_kernels.jl")
include("cold_start.jl")
include("hsd/initialize.jl")
include("midend/problem_features.jl")
include("midend/auto_planner.jl")
include("midend/formulation_planner.jl")
include("soc_native.jl")
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
include("adaptive_parameters.jl")
include("stagnation.jl")
include("kernels/mixed_precision_kkt.jl")
include("workspace.jl")
include("schur.jl")
include("kernels/sparse_coo.jl")
include("kkt.jl")
include("kkt_formulations/dense_augmented.jl")
include("kkt_backend.jl")
include("sparse_la.jl")
include("nullspace.jl")
include("chordal.jl")
include("reduction_plan.jl")
include("preprocessing.jl")
include("step.jl")
include("kernels/threaded.jl")
include("lp_sparse.jl")
include("lp_solver.jl")
include("solver/interior_point.jl")
include("prepared.jl")
include("validation.jl")
include("soc_presolve.jl")
include("spectrum.jl")
include("frontend/high_level_solve.jl")
include("performance_trace.jl")
include("public/result.jl")
include("hsd/native_hsd_public.jl")
include("public/optimize.jl")
include("moi_wrapper.jl")

# v0.5 has one public modeling/solve interface.  Mature problem, workspace,
# provider, and legacy solve types remain package-internal implementation
# details and are intentionally not re-exported as parallel entry points.
export Model
export variable!, constraint!, objective!
export set_start!, set_dual_start!, set_dual_slack_start!
export Reals, Nonnegative, Nonpositive, ZeroCone
export LorentzCone, RotatedLorentzCone, PSDCone, ExponentialCone, PowerCone
export Minimize, Maximize
export Settings, Tolerances, Limits, Outputs
export optimize!, execution_plan
export status, value, dual, dual_slack
export primal_objective, dual_objective
export certificate, diagnostics, iteration_history, performance_trace
export Optimizer

# Symmetric-cone algebra (Subagent I) lives in the nested module
# `SymmetricCones` (Nonnegative / SOC / PSDTriangle kernels). It is not part of
# the frozen public export set; callers reach it as `SDPX.SymmetricCones`.
import .SymmetricCones

end
