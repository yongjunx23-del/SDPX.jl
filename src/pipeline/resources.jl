function _available_memory_bytes()
    return ExtendedPrecisionBLAS._system_free_memory_bytes()
end
function _lp_extended_crossover(
    ::Type{T},
    classification::ProblemClassification,
    opts::SolverOptions{T},
    thread_count::Int,
    memory_budget_bytes::Int,
    available_memory_bytes::Int,
) where {T}
    features = ExtendedPrecisionBLAS.CrossoverFeatures(
        rows=classification.cone_rows,
        columns=classification.variables,
        matrix_dimension=1,
        average_nnz=Float64(classification.variables),
        active_density=classification.coefficient_density,
        expected_schur_density=classification.expected_schur_density,
        thread_count=thread_count,
        memory_budget_bytes=memory_budget_bytes,
        sparse_input=false,
    )
    return ExtendedPrecisionBLAS.choose_crossover(
        T,
        features;
        mode=opts.extended_precision_blas,
        available_memory_bytes=available_memory_bytes,
    )
end

"""
    _lp_bigfloat_thread_limit(classification, algorithm) -> Int

The dedicated standard-form LP path owns disjoint BigFloat panel rows and
Schur tiles, so it can safely use Julia threads.  Other BigFloat paths still
use the conservative serial default because their mutable MPFR storage is not
partitioned at this planning seam.  Keep the crossover based on the actual
panel work rather than enabling threads for small LPs where task barriers
dominate.
"""
@inline function _lp_bigfloat_thread_limit(
    classification::ProblemClassification,
    algorithm::Symbol,
)
    algorithm === :lp_primal_dual || return 1
    classification.arithmetic === :bigfloat || return 1
    classification.equalities > 0 || return 1
    work = Int128(classification.variables) *
           Int128(max(classification.equalities, 1))
    # The panel and Schur tile loops are safely threadable, but the remaining
    # BigFloat predictor/residual reductions become synchronization-bound on
    # this LP family.  These conservative bands are based on the cluster
    # crossover sweep: 8 workers is best for 250k--1M scalar panel entries;
    # only a substantially larger panel is allowed to use 16.  No default
    # path is opened at 32+ workers, where MPFR task overhead dominates.
    work < 250_000 && return 1
    work < 1_000_000 && return 8
    work < 4_000_000 && return 16
    return 32
end


"""
    automatic_scaling_policy(algorithm)

Select the scaling stage without probing numerical values: LP routes use
geometric scaling and every SDP route uses automatic Ruiz scaling. Explicit
`scaling=:none` or `:equilibrate` choices bypass this policy in
[`build_execution_plan`](@ref).
"""
@inline function automatic_scaling_policy(algorithm::Symbol)
    algorithm === :lp_primal_dual && return :lp_geometric
    return :sdp_ruiz
end
