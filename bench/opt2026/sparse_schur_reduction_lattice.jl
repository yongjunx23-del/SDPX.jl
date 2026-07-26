#!/usr/bin/env julia

"""Compare equal-width and pair-balanced Task_Low08 Schur reduction."""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

let path = joinpath(
        @__DIR__,
        "..",
        "lattice_bootstrap",
        "benchmark_sdpx_float64_solve.jl",
    )
    source = read(path, String)
    include_string(
        @__MODULE__,
        replace(source, r"\nmain\(ARGS\)\s*$" => "\n"),
        path,
    )
end

function pair_work(cons::SDPX.SparseCons, variables::Int)
    work = zeros(Int, variables)
    @inbounds for ids in cons.schur_order
        count = length(ids)
        for position in eachindex(ids)
            work[ids[position]] += count - position + 1
        end
    end
    return work
end

function partition_loads(work, boundaries)
    loads = zeros(Int, length(boundaries) - 1)
    @inbounds for task in eachindex(loads)
        for column in
            boundaries[task]:(boundaries[task + 1] - 1)
            loads[task] += work[column]
        end
    end
    return loads
end

function best_reduction!(
    workspace,
    constraints,
    boundaries;
    repetitions::Int=5,
)
    copyto!(workspace.schur_column_boundaries, boundaries)
    SDPX.reduce_sparse_schur!(workspace, constraints)
    best = Inf
    allocated = 0
    for _ in 1:repetitions
        GC.gc()
        stats = @timed SDPX.reduce_sparse_schur!(
            workspace,
            constraints,
        )
        if stats.time < best
            best = stats.time
            allocated = stats.bytes
        end
    end
    return best, allocated
end

function main()
    length(ARGS) >= 1 ||
        error("usage: sparse_schur_reduction_lattice.jl INPUT")
    data = read_problem(abspath(ARGS[1]))
    equality_ids, _, _, _ = equality_basis(data.B, data.b)
    problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B[:, equality_ids],
        data.b[equality_ids];
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    constraints = problem.cons::SDPX.SparseCons{Float64}
    workspace = SDPX.Workspace(
        problem;
        thread_count=Threads.nthreads(),
    )
    workspace.arrow === nothing ||
        error("Task_Low08 unexpectedly selected the arrow backend")
    workspace.dense_sparse_assembly &&
        error("benchmark expects the packed sparse Schur path")
    workspace.schur_lower_only ||
        error("benchmark expects lower-triangle Schur storage")

    X = [
        Matrix{Float64}(100.0I, dimension, dimension)
        for dimension in problem.dims.k
    ]
    Y = [
        Matrix{Float64}(0.001I, dimension, dimension)
        for dimension in problem.dims.k
    ]
    SDPX.factor_blocks!(workspace, X, Y) ||
        error("initial block factorization failed")
    @sync for bin in workspace.schur_bins
        isempty(bin) && continue
        Threads.@spawn begin
            for block in bin
                SDPX.sparse_schur_block!(
                    workspace.blk[block],
                    constraints,
                    block,
                    X[block],
                    Y[block],
                )
            end
        end
    end

    tasks = length(workspace.schur_column_boundaries) - 1
    variables = problem.dims.m
    equal_boundaries = [
        min((task - 1) * cld(variables, tasks) + 1, variables + 1)
        for task in 1:tasks
    ]
    push!(equal_boundaries, variables + 1)
    balanced_boundaries = copy(workspace.schur_column_boundaries)
    work = pair_work(constraints, variables)
    equal_loads = partition_loads(work, equal_boundaries)
    balanced_loads = partition_loads(work, balanced_boundaries)

    equal_time, equal_allocated = best_reduction!(
        workspace,
        constraints,
        equal_boundaries,
    )
    equal_schur = similar(workspace.S)
    SDPX.materialize_schur!(equal_schur, workspace)
    balanced_time, balanced_allocated = best_reduction!(
        workspace,
        constraints,
        balanced_boundaries,
    )
    balanced_schur = similar(workspace.S)
    SDPX.materialize_schur!(balanced_schur, workspace)
    relative_error =
        norm(equal_schur - balanced_schur) /
        max(norm(equal_schur), 1.0)

    @printf("threads=%d tasks=%d pairs=%d\n",
        Threads.nthreads(), tasks, sum(work))
    println("equal_boundaries=", equal_boundaries)
    println("balanced_boundaries=", balanced_boundaries)
    println("equal_loads=", equal_loads)
    println("balanced_loads=", balanced_loads)
    @printf(
        "equal_width_seconds=%.6f balanced_seconds=%.6f speedup=%.3f\n",
        equal_time,
        balanced_time,
        equal_time / balanced_time,
    )
    println("equal_width_allocated_bytes=", equal_allocated)
    println("balanced_allocated_bytes=", balanced_allocated)
    @printf("relative_schur_error=%.6e\n", relative_error)
    relative_error <= 1e-15 ||
        error("balanced reduction changed the Schur matrix")
end

main()
