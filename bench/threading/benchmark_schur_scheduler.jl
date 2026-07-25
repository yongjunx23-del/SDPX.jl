#!/usr/bin/env julia

"""
Measure Schur assembly scheduling overhead, task-local accumulator memory, and
thread scaling on deterministic dense Float64 and Float64x4 problems.

Usage:

    JULIA_NUM_THREADS=8 julia --project=. \
        bench/threading/benchmark_schur_scheduler.jl output.csv

The benchmark intentionally keeps BLAS single-threaded. Parallelism therefore
comes only from SDPX's block scheduler, avoiding Julia-task × BLAS
oversubscription.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX

BLAS.set_num_threads(1)

function dense_problem(::Type{T}, blocks::Int, variables::Int, dimension::Int) where {T}
    coefficients = Vector{Array{T,3}}(undef, blocks)
    for block in 1:blocks
        panel = zeros(T, variables, dimension, dimension)
        @inbounds for variable in 1:variables
            for column in 1:dimension
                for row in 1:column
                    numerator =
                        mod(17variable + 31block + 13row + 7column, 41) - 20
                    value = T(numerator) / T(23)
                    panel[variable, row, column] = value
                    panel[variable, column, row] = value
                end
            end
        end
        coefficients[block] = panel
    end
    constants = [zeros(T, dimension, dimension) for _ in 1:blocks]
    return SDPX.ingest(
        ones(T, variables),
        coefficients,
        constants,
        zeros(T, variables, 0),
        T[];
        sparse=false,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
end

function median_measurement(operation, repetitions::Int)
    times = Vector{Float64}(undef, repetitions)
    bytes = Vector{Int}(undef, repetitions)
    for repetition in 1:repetitions
        GC.gc()
        measurement = @timed operation()
        times[repetition] = measurement.time
        bytes[repetition] = measurement.bytes
    end
    sort!(times)
    sort!(bytes)
    return (
        seconds=times[cld(repetitions, 2)],
        minimum_seconds=first(times),
        allocated_bytes=bytes[cld(repetitions, 2)],
        minimum_allocated_bytes=first(bytes),
    )
end

function synchronization_probe(bins)
    @sync for bin in bins
        isempty(bin) && continue
        Threads.@spawn nothing
    end
    return nothing
end

function reference_reduce_schur_partials!(workspace)
    SDPX._zero_schur_accumulator!(workspace.S, workspace)
    partials = workspace.Spartial
    isempty(partials) && return workspace.S
    rows = size(workspace.S, 1)
    task_count = min(workspace.thread_count, rows)
    if task_count <= 1
        for partial in partials
            SDPX.kaxpby!(
                one(eltype(workspace.S)),
                partial,
                one(eltype(workspace.S)),
                workspace.S,
            )
        end
        return workspace.S
    end
    chunk = cld(rows, task_count)
    @sync for task in 1:task_count
        first_row = (task - 1) * chunk + 1
        first_row > rows && continue
        last_row = min(task * chunk, rows)
        Threads.@spawn begin
            @inbounds for partial in partials
                for column in axes(workspace.S, 2)
                    for row in first_row:last_row
                        workspace.S[row, column] += partial[row, column]
                    end
                end
            end
        end
    end
    return workspace.S
end

function reference_threaded_dense_schur_build!(
    workspace,
    problem,
    X,
    Y,
)
    block_count, variables, _, dimensions = problem.dims
    if workspace.thread_count <= 1 ||
       block_count <= 1 ||
       length(workspace.schur_bins) <= 1
        return SDPX.schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
    end
    for partial in workspace.Spartial
        SDPX._zero_schur_accumulator!(partial, workspace)
    end
    @sync for (bin_index, bin) in enumerate(workspace.schur_bins)
        isempty(bin) && continue
        Threads.@spawn begin
            partial = workspace.Spartial[bin_index]
            for block in bin
                dimension = dimensions[block]
                dimension == 0 && continue
                block_workspace = workspace.blk[block]
                source = reshape(
                    problem.cons.Av[block],
                    dimension,
                    dimension * variables,
                )
                copyto!(block_workspace.Ppanel, source)
                SDPX.ktrsm!(
                    block_workspace.LX,
                    block_workspace.Ppanel,
                )
                for variable in 1:variables
                    columns = (
                        (variable - 1) * dimension + 1
                    ):(variable * dimension)
                    SDPX.ktrmm!(
                        view(block_workspace.Ppanel, :, columns),
                        block_workspace.MY,
                    )
                end
                transformed = reshape(
                    block_workspace.Ppanel,
                    dimension * dimension,
                    variables,
                )
                SDPX._dense_gram_add!(partial, transformed)
            end
        end
    end
    reference_reduce_schur_partials!(workspace)
    SDPX._dense_gram_lower_only(eltype(workspace.S)) &&
        SDPX._mirror_schur_lower!(workspace.S)
    return workspace.S
end

function maximum_relative_error(reference, candidate)
    scale = max(maximum(abs, reference), one(eltype(reference)))
    return Float64(maximum(abs, candidate .- reference) / scale)
end

function append_rows(path::String, rows)
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

function run_case(
    ::Type{T},
    arithmetic::String,
    case_name::String,
    blocks::Int,
    variables::Int,
    dimension::Int,
    repetitions::Int,
) where {T}
    problem = dense_problem(T, blocks, variables, dimension)
    identity_blocks = [
        Matrix{T}(I, dimension, dimension)
        for _ in 1:blocks
    ]

    serial_workspace = SDPX.Workspace(problem; thread_count=1)
    SDPX.factor_blocks!(
        serial_workspace,
        identity_blocks,
        identity_blocks,
    ) || error("identity factorization failed")
    SDPX.schur_build!(
        serial_workspace,
        problem,
        problem.cons,
        identity_blocks,
        identity_blocks,
    )
    reference = copy(serial_workspace.S)

    rows = NamedTuple[]
    serial_build_seconds = 0.0
    for requested_threads in (1, 2, 4, 8)
        requested_threads > Threads.nthreads() && continue
        workspace = SDPX.Workspace(
            problem;
            thread_count=requested_threads,
        )
        SDPX.factor_blocks!(workspace, identity_blocks, identity_blocks) ||
            error("identity factorization failed")
        operation = () -> SDPX.threaded_schur_build!(
            workspace,
            problem,
            problem.cons,
            identity_blocks,
            identity_blocks,
        )
        reference_operation = () ->
            reference_threaded_dense_schur_build!(
                workspace,
                problem,
                identity_blocks,
                identity_blocks,
            )
        reference_operation()
        operation()
        reference_build =
            median_measurement(reference_operation, repetitions)
        build = median_measurement(operation, repetitions)
        requested_threads == 1 &&
            (serial_build_seconds = build.seconds)
        error = maximum_relative_error(reference, workspace.S)

        reduction = if isempty(workspace.Spartial)
            (
                seconds=0.0,
                minimum_seconds=0.0,
                allocated_bytes=0,
                minimum_allocated_bytes=0,
            )
        else
            reduction_lower_only =
                !workspace.extended_precision.lower_only &&
                SDPX._dense_gram_lower_only(T)
            SDPX._reduce_schur_partials!(
                workspace,
                reduction_lower_only,
            )
            optimized_reduction = median_measurement(
                () -> SDPX._reduce_schur_partials!(
                    workspace,
                    reduction_lower_only,
                ),
                repetitions,
            )
            reference_reduction = median_measurement(
                () -> reference_reduce_schur_partials!(workspace),
                repetitions,
            )
            merge(
                optimized_reduction,
                (
                    reference_seconds=reference_reduction.seconds,
                    reference_allocated_bytes=
                        reference_reduction.allocated_bytes,
                ),
            )
        end

        synchronization_probe(workspace.schur_bins)
        synchronization = median_measurement(
            () -> synchronization_probe(workspace.schur_bins),
            max(repetitions, 9),
        )

        partial_bytes = Base.summarysize(workspace.Spartial)
        partial_count = length(workspace.Spartial)
        push!(
            rows,
            (
                arithmetic=arithmetic,
                case=case_name,
                julia_threads=Threads.nthreads(),
                requested_threads=requested_threads,
                selected_threads=workspace.thread_count,
                blocks=blocks,
                variables=variables,
                block_dimension=dimension,
                schur_bins=length(workspace.schur_bins),
                partial_accumulators=partial_count,
                partial_megabytes=partial_bytes / 1.0e6,
                per_worker_partial_megabytes=
                    partial_count == 0 ? 0.0 :
                    partial_bytes / partial_count / 1.0e6,
                build_seconds=build.seconds,
                reference_build_seconds=reference_build.seconds,
                implementation_speedup=
                    reference_build.seconds / max(build.seconds, eps()),
                thread_speedup=
                    serial_build_seconds / max(build.seconds, eps()),
                build_minimum_seconds=build.minimum_seconds,
                build_allocated_bytes=build.allocated_bytes,
                reference_build_allocated_bytes=
                    reference_build.allocated_bytes,
                build_minimum_allocated_bytes=
                    build.minimum_allocated_bytes,
                reduction_seconds=reduction.seconds,
                reference_reduction_seconds=
                    hasproperty(reduction, :reference_seconds) ?
                    reduction.reference_seconds : 0.0,
                reduction_speedup=
                    hasproperty(reduction, :reference_seconds) ?
                    reduction.reference_seconds /
                    max(reduction.seconds, eps()) : 1.0,
                reduction_allocated_bytes=reduction.allocated_bytes,
                reference_reduction_allocated_bytes=
                    hasproperty(reduction, :reference_allocated_bytes) ?
                    reduction.reference_allocated_bytes : 0,
                synchronization_seconds=synchronization.seconds,
                synchronization_allocated_bytes=
                    synchronization.allocated_bytes,
                relative_schur_error=error,
                peak_rss_megabytes=Sys.maxrss() / 1.0e6,
            ),
        )
    end
    return rows
end

function main(arguments)
    output = isempty(arguments) ?
             joinpath(@__DIR__, "results", "schur-scheduler.csv") :
             abspath(first(arguments))
    rows = NamedTuple[]
    append!(
        rows,
        run_case(Float64, "Float64", "small", 8, 96, 6, 9),
    )
    append!(
        rows,
        run_case(Float64, "Float64", "medium", 8, 512, 10, 7),
    )
    append!(
        rows,
        run_case(Float64x4, "Float64x4", "small", 8, 48, 4, 5),
    )
    append!(
        rows,
        run_case(Float64x4, "Float64x4", "medium", 8, 128, 8, 3),
    )
    append_rows(output, rows)
    for row in rows
        @printf(
            "%-10s %-6s threads=%d build=%9.6f s speedup=%5.2fx reduce=%8.6f s alloc=%7d sync=%8.6f s partial=%8.2f MB error=%.3e\n",
            row.arithmetic,
            row.case,
            row.requested_threads,
            row.build_seconds,
            row.thread_speedup,
            row.reduction_seconds,
            row.build_allocated_bytes,
            row.synchronization_seconds,
            row.partial_megabytes,
            row.relative_schur_error,
        )
    end
    println("Wrote ", output)
end

main(ARGS)
