#!/usr/bin/env julia

"""
Measure exact-arrow sparse Schur assembly scaling and compact per-worker
reduction memory on deterministic Float64 and Float64x4 problems.

Usage:

    JULIA_NUM_THREADS=8 julia --project=. \
        docs/evidence/bench/threading/benchmark_sparse_schur_scheduler.jl output.csv
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using SparseArrays

BLAS.set_num_threads(1)

function arrow_problem(::Type{T}, blocks::Int, shared::Int) where {T}
    variables = shared + blocks
    coefficients = [
        Vector{SparseMatrixCSC{T,Int}}(undef, variables)
        for _ in 1:blocks
    ]
    for block in 1:blocks, variable in 1:variables
        if variable <= shared || variable == shared + block
            diagonal_one = T(mod(17variable + 11block, 31) + 3) / T(19)
            off_diagonal = T(mod(7variable + 13block, 17) - 8) / T(29)
            diagonal_two = T(mod(5variable + 23block, 37) + 5) / T(23)
            coefficients[block][variable] = sparse(
                T[
                    diagonal_one off_diagonal
                    off_diagonal diagonal_two
                ],
            )
        else
            coefficients[block][variable] = spzeros(T, 2, 2)
        end
    end
    return SDPX.ingest(
        ones(T, variables),
        coefficients,
        [zeros(T, 2, 2) for _ in 1:blocks],
        zeros(T, variables, 0),
        T[];
        sparse=true,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
end

function median_measurement(operation, repetitions::Int)
    times = Float64[]
    allocations = Int[]
    for _ in 1:repetitions
        GC.gc()
        measurement = @timed operation()
        push!(times, measurement.time)
        push!(allocations, measurement.bytes)
    end
    sort!(times)
    sort!(allocations)
    return (
        seconds=times[cld(length(times), 2)],
        minimum_seconds=first(times),
        allocated_bytes=allocations[cld(length(allocations), 2)],
        minimum_allocated_bytes=first(allocations),
    )
end

function synchronization_probe(bins)
    @sync for bin in bins
        isempty(bin) && continue
        Threads.@spawn nothing
    end
    return nothing
end

function relative_error(reference, candidate)
    scale = max(maximum(abs, reference), one(eltype(reference)))
    return Float64(maximum(abs, candidate .- reference) / scale)
end

function run_case(
    ::Type{T},
    arithmetic::String,
    case_name::String,
    blocks::Int,
    shared::Int,
    repetitions::Int,
) where {T}
    problem = arrow_problem(T, blocks, shared)
    identities = [Matrix{T}(I, 2, 2) for _ in 1:blocks]
    serial = SDPX.Workspace(problem; thread_count=1)
    serial.arrow === nothing &&
        error("synthetic sparse problem did not select the arrow backend")
    SDPX.factor_blocks!(serial, identities, identities) ||
        error("identity factorization failed")
    SDPX.schur_build!(
        serial,
        problem,
        problem.cons,
        identities,
        identities,
    )
    reference = zeros(T, problem.dims.m, problem.dims.m)
    SDPX.materialize_schur!(reference, serial)

    rows = NamedTuple[]
    serial_seconds = 0.0
    for requested_threads in (1, 2, 4, 8)
        requested_threads > Threads.nthreads() && continue
        workspace = SDPX.Workspace(
            problem;
            thread_count=requested_threads,
        )
        SDPX.factor_blocks!(workspace, identities, identities) ||
            error("identity factorization failed")
        operation = () -> SDPX.threaded_schur_build!(
            workspace,
            problem,
            problem.cons,
            identities,
            identities,
        )
        operation()
        operation()
        measurement = median_measurement(operation, repetitions)
        requested_threads == 1 &&
            (serial_seconds = measurement.seconds)

        materialized = similar(reference)
        SDPX.materialize_schur!(materialized, workspace)
        arrow = workspace.arrow
        reduction_bytes = Base.summarysize(arrow.Sredpartial)
        synchronization = median_measurement(
            () -> synchronization_probe(workspace.schur_bins),
            max(repetitions, 9),
        )
        push!(
            rows,
            (
                arithmetic=arithmetic,
                case=case_name,
                julia_threads=Threads.nthreads(),
                requested_threads=requested_threads,
                selected_threads=workspace.thread_count,
                blocks=blocks,
                shared_variables=shared,
                total_variables=problem.dims.m,
                schur_bins=length(workspace.schur_bins),
                build_seconds=measurement.seconds,
                build_minimum_seconds=measurement.minimum_seconds,
                speedup=serial_seconds / max(measurement.seconds, eps()),
                build_allocated_bytes=measurement.allocated_bytes,
                compact_reduction_megabytes=reduction_bytes / 1.0e6,
                per_worker_reduction_megabytes=
                    reduction_bytes /
                    max(length(arrow.Sredpartial), 1) /
                    1.0e6,
                synchronization_seconds=synchronization.seconds,
                synchronization_allocated_bytes=
                    synchronization.allocated_bytes,
                relative_schur_error=
                    relative_error(reference, materialized),
                peak_rss_megabytes=Sys.maxrss() / 1.0e6,
            ),
        )
    end
    return rows
end

function write_rows(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do output
        names = propertynames(first(rows))
        println(output, join(names, ','))
        for row in rows
            println(
                output,
                join((getproperty(row, name) for name in names), ','),
            )
        end
    end
end

function main(arguments)
    output = isempty(arguments) ?
             joinpath(@__DIR__, "results", "sparse-schur-scheduler.csv") :
             abspath(first(arguments))
    rows = NamedTuple[]
    append!(
        rows,
        run_case(Float64, "Float64", "small", 64, 8, 9),
    )
    append!(
        rows,
        run_case(Float64, "Float64", "medium", 512, 24, 7),
    )
    append!(
        rows,
        run_case(Float64x4, "Float64x4", "small", 64, 8, 5),
    )
    append!(
        rows,
        run_case(Float64x4, "Float64x4", "medium", 256, 16, 3),
    )
    write_rows(output, rows)
    for row in rows
        @printf(
            "%-10s %-6s threads=%d build=%9.6f s speedup=%5.2fx alloc=%7d compact=%7.3f MB error=%.3e\n",
            row.arithmetic,
            row.case,
            row.requested_threads,
            row.build_seconds,
            row.speedup,
            row.build_allocated_bytes,
            row.compact_reduction_megabytes,
            row.relative_schur_error,
        )
    end
    println("Wrote ", output)
end

main(ARGS)
