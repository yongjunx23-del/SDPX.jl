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
    physical_core_count() -> Int

Physical cores available, distinct from `Sys.CPU_THREADS`.

Plan §18.4 requires that requested workers, effective workers and actual
physical cores be reported separately, and that oversubscribed workers are not
described as core scaling. That distinction cannot be made from
`Threads.nthreads()` alone: on an SMT machine half the "cores" share execution
units, and on a heterogeneous machine (performance plus efficiency cores) a
block-parallel region runs at the speed of its slowest worker.

This is measured, not assumed, and falls back to the logical count when the
platform does not report it.
"""
function physical_core_count()
    if Sys.isapple()
        try
            return parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
        catch exception
            _recoverable(exception) || rethrow()
        end
    elseif Sys.islinux()
        try
            cores = Set{Tuple{String,String}}()
            for block in split(read("/proc/cpuinfo", String), "\n\n")
                package = match(r"physical id\s*:\s*(\d+)", block)
                core = match(r"core id\s*:\s*(\d+)", block)
                (package === nothing || core === nothing) && continue
                push!(cores, (package.captures[1], core.captures[1]))
            end
            isempty(cores) || return length(cores)
        catch exception
            _recoverable(exception) || rethrow()
        end
    end
    return Sys.CPU_THREADS
end

"""
    schur_bin_report(::Type{T}, m, L, threads) -> NamedTuple

Whether the per-worker Schur accumulators were capped below the requested
worker count, which automatic memory fraction was selected, and what that
cost.

The accumulators are full `m x m` matrices, one per bin, so their total scales
as `threads * m^2`. `_schur_parallel_bins` caps them at an automatically
selected fraction of free memory, which trades parallelism for memory.
Section 18.4 asks that a change in algorithm selection between thread counts
be reported rather than inferred from disappointing scaling, and section 19.3
asks for an informative estimate rather than a silent degradation.

Eligible multi-threaded dense Float64 workspaces own disjoint
lower-triangle output columns and allocate no partials, so `assembly_mode` is
`:column_owned`, `selected_bins == requested_bins`, `capped === false`, and
`total_bytes` is zero. Single-thread and single-block workspaces also store no
partials, but report `:serial` because they do not execute the owner kernel.
`would_have_been_bytes` keeps the historical per-bin cost for the diagnostic's
avoided-memory display. Pass `dense_owner=true` only for a known eligible
`DenseCons` route; the dimension-only default remains the legacy partial
report because sparse Float64 dense-Schur assembly still uses partials.
`_schur_parallel_bins` itself remains the legacy cap for sparse and
extended-arithmetic routes.
"""
function schur_bin_report(::Type{T}, m::Integer, L::Integer,
                          threads::Integer;
                          free_memory_bytes::Union{Nothing,Integer}=nothing,
                          dense_owner::Bool=false) where {T}
    requested = max(1, min(Int(threads), Int(L)))
    bytes_each = saturating_bytes(max(sizeof(T), 8), Int(m), Int(m))
    if requested == 1
        return (
            requested_bins=requested,
            selected_bins=requested,
            capped=false,
            memory_fraction=0.0,
            memory_budget_bytes=0,
            bytes_per_bin=bytes_each,
            total_bytes=0,
            would_have_been_bytes=saturating_bytes(requested, bytes_each),
            assembly_mode=:serial,
            owner_tasks=1,
        )
    end
    if dense_owner
        T === Float64 || throw(ArgumentError(
            "dense column ownership is available only for Float64",
        ))
        return (
            requested_bins=requested,
            selected_bins=requested,
            capped=false,
            memory_fraction=0.0,
            memory_budget_bytes=0,
            bytes_per_bin=bytes_each,
            total_bytes=0,
            would_have_been_bytes=saturating_bytes(requested, bytes_each),
            assembly_mode=:column_owned,
            owner_tasks=min(max(Int(threads), 1), max(Int(m), 1)),
        )
    end
    available = free_memory_bytes === nothing ?
        ExtendedPrecisionBLAS._system_free_memory_bytes() :
        Int(free_memory_bytes)
    selected = _schur_parallel_bins(T, Int(m), Int(L), Int(threads);
        free_memory_bytes=available)
    memory_fraction = _schur_accumulator_memory_fraction(
        T,
        Int(m),
        Int(L),
        Int(threads),
        available,
    )
    memory_budget_bytes =
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            available,
            memory_fraction,
        )
    return (
        requested_bins=requested,
        selected_bins=selected,
        capped=selected < requested,
        memory_fraction=memory_fraction,
        memory_budget_bytes=memory_budget_bytes,
        bytes_per_bin=bytes_each,
        total_bytes=saturating_bytes(selected, bytes_each),
        would_have_been_bytes=saturating_bytes(requested, bytes_each),
        assembly_mode=:partial_accumulators,
        owner_tasks=0,
    )
end

"""
    worker_report(requested, selected) -> NamedTuple

The three counts §18.4 asks to be kept apart, plus whether the request exceeds
the hardware. Reporting `oversubscribed` explicitly is the point: a speedup
measured with more workers than cores is not core scaling, and labelling it as
such is how misleading scaling numbers get published.
"""
function worker_report(requested::Integer, selected::Integer)
    physical = physical_core_count()
    # `Sys.CPU_THREADS` is Julia's view, which can be narrowed by affinity,
    # `JULIA_CPU_THREADS`, or a container limit — on this development machine it
    # reports 4 against 10 physical cores. Reporting that as "logical cores"
    # would be actively wrong, so it is labelled for what it is and the OS is
    # asked separately for the hardware count.
    logical = physical
    if Sys.isapple()
        try
            logical = parse(Int, strip(read(`sysctl -n hw.logicalcpu`, String)))
        catch exception
            _recoverable(exception) || rethrow()
        end
    elseif Sys.islinux()
        logical = max(Sys.CPU_THREADS, physical)
    end
    return (
        requested_workers=Int(requested),
        effective_workers=Int(selected),
        physical_cores=physical,
        logical_cores=logical,
        julia_visible_cores=Sys.CPU_THREADS,
        oversubscribed=selected > physical,
    )
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
