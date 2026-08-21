using SDPX
using Test

const QUICK_TESTS = (
    "la_backend_regressions.jl",
    "generic_la_backend.jl",
    "dense_augmented_kkt.jl",
    "soc_native_algebra.jl",
    "soc_metric_sparse.jl",
    "soc_singleton_presolve.jl",
    "soc_native_solver.jl",
    "moi_vector_cones.jl",
    "moi_native_soc.jl",
    "bfla_backend.jl",
    "mfla_backend.jl",
    "auto_planner.jl",
    "frontend_auto_options.jl",
    "architecture_regressions.jl",
    "modeling/foundation_types.jl",
    "modeling/route_classifier.jl",
    "modeling/exp_route_phase_a.jl",
    "modeling/lower_lp.jl",
    "modeling/lower_soc.jl",
    "modeling/lower_sdp.jl",
    "public/settings_outputs.jl",
    "public/result_optimize.jl",
    "public_api.jl",
    "performance_trace.jl",
    "termination_metadata.jl",
    "sdpa_benchmark_loader.jl",
    "benchmark_pathological.jl",
    "benchmark_registry.jl",
    "benchmark_core_matrix.jl",
    "benchmark_core_matrix_fresh_process.jl",
    "netlib_benchmark_loader.jl",
    "cbf_benchmark_loader.jl",
    "benchmark_compare.jl",
    "benchmark_fresh_process.jl",
    "prepared_structure.jl",
    "v05_core_invariants.jl",
    "sparse_execution_round6.jl",
    "sparse_schur_round7.jl",
    "coo_contraction_regression.jl",
    "precision_ladder_plan.jl",
)

# Keep this list in the historical full-suite order. Some older tests share
# helpers through that order (notably correctness.jl -> sparse.jl).
const FULL_TESTS = (
    "la_backend_regressions.jl",
    "generic_la_backend.jl",
    "dense_augmented_kkt.jl",
    "bfla_backend.jl",
    "mfla_backend.jl",
    "correctness.jl",
    "genericity.jl",
    "extended_precision_blas.jl",
    "sparse.jl",
    "coo_contraction_regression.jl",
    "moi_vector_cones.jl",
    "moi_native_soc.jl",
    "moi.jl",
    "threads.jl",
    "pipeline.jl",
    "adaptive_parameter_policy.jl",
    "preprocessing_regressions.jl",
    "lp_regressions.jl",
    "direction_accuracy_lp.jl",
    "soc_regressions.jl",
    "soc_native_algebra.jl",
    "soc_metric_sparse.jl",
    "soc_singleton_presolve.jl",
    "soc_native_solver.jl",
    "auto_planner.jl",
    "fixed_trace_benchmark_regressions.jl",
    "soc_q3_kernel_regressions.jl",
    "solver_regressions.jl",
    "kkt_regressions.jl",
    "kkt_sparse_backend.jl",
    "sparse_sdp_kkt.jl",
    "mixed_precision_kkt_regressions.jl",
    "result_certificate.jl",
    "infeasibility_diagnostics.jl",
    "options_interface.jl",
    "frontend_auto_options.jl",
    "architecture_regressions.jl",
    "modeling/foundation_types.jl",
    "modeling/route_classifier.jl",
    "modeling/exp_route_phase_a.jl",
    "modeling/lower_lp.jl",
    "modeling/lower_soc.jl",
    "modeling/lower_sdp.jl",
    "public/settings_outputs.jl",
    "public/result_optimize.jl",
    "public_api.jl",
    "performance_trace.jl",
    "termination_metadata.jl",
    "sdpa_benchmark_loader.jl",
    "benchmark_pathological.jl",
    "benchmark_registry.jl",
    "benchmark_core_matrix.jl",
    "benchmark_core_matrix_fresh_process.jl",
    "netlib_benchmark_loader.jl",
    "cbf_benchmark_loader.jl",
    "benchmark_compare.jl",
    "benchmark_fresh_process.jl",
    "prepared_structure.jl",
    "moi_regressions.jl",
    "extended_blas_regressions.jl",
    "bigfloat_kernel_regressions.jl",
    "bigfloat_ownership_regressions.jl",
    "bigfloat_sparse_schur_regressions.jl",
    "schur_scheduler_regressions.jl",
    "ingest_regressions.jl",
    "spectrum_regressions.jl",
    "extensions_regressions.jl",
    "lp_sparse.jl",
    "examples.jl",
    "nullspace_reduction.jl",
    "error_handling.jl",
    "executed_diagnostics.jl",
    "cli_bridge.jl",
    "aqua.jl",
    "shadowing_guard.jl",
    "gates.jl",
    "v05_core_invariants.jl",
    "precision_ladder_plan.jl",
    "sparse_execution_round6.jl",
    "sparse_schur_round7.jl",
)

function _test_profile()
    profile = Symbol(lowercase(strip(get(ENV, "SDPX_TEST_PROFILE", "quick"))))
    profile in (:quick, :full) || throw(ArgumentError(
        "SDPX_TEST_PROFILE must be quick or full, got $(repr(profile))",
    ))
    return profile
end

const TEST_PROFILE = _test_profile()

# These packages are only needed by full-suite extension tests. Avoiding their
# load cost is part of making the normal edit-test loop small.
if TEST_PROFILE === :full
    using JLD2
end

const SELECTED_TESTS = TEST_PROFILE === :quick ? QUICK_TESTS : FULL_TESTS

@info "SDPX test profile" profile=TEST_PROFILE files=length(SELECTED_TESTS)

@testset "cold-start math helpers" begin
    include(joinpath(@__DIR__, "cold_start.jl"))
end

@testset "SDPX.jl ($(TEST_PROFILE))" begin
    for file in SELECTED_TESTS
        include(joinpath(@__DIR__, file))
    end
end
