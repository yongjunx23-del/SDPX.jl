#!/usr/bin/env julia

"""Benchmark SDPX automatic dense/sparse dispatch across arithmetic types."""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using SparseArrays

BLAS.set_num_threads(1)

function lattice_like_coefficients(
    ::Type{T};
    blocks::Int,
    variables::Int,
    dimension::Int,
    active_percent::Int=65,
    entries_per_matrix::Int=3,
) where {T}
    coefficients = Vector{Vector{SparseMatrixCSC{T,Int}}}(undef, blocks)
    upper = [
        (row, column)
        for column in 1:dimension
        for row in 1:column
    ]
    for block in 1:blocks
        coefficient_block = Vector{SparseMatrixCSC{T,Int}}(undef, variables)
        for variable in 1:variables
            active = mod(variable * 17 + block * 13, 100) < active_percent
            if !active
                coefficient_block[variable] = spzeros(T, dimension, dimension)
                continue
            end
            rows = Int[]
            columns = Int[]
            values = T[]
            for entry in 1:entries_per_matrix
                coordinate = mod1(
                    variable * 37 + block * 53 + entry * 29,
                    length(upper),
                )
                row, column = upper[coordinate]
                value = T(0.25 + 0.01 * mod(variable + 3block + entry, 19))
                push!(rows, row)
                push!(columns, column)
                push!(values, value)
                if row != column
                    push!(rows, column)
                    push!(columns, row)
                    push!(values, value)
                end
            end
            coefficient_block[variable] = sparse(
                rows,
                columns,
                values,
                dimension,
                dimension,
            )
        end
        coefficients[block] = coefficient_block
    end
    return coefficients
end

function run_case(
    ::Type{T};
    blocks::Int,
    variables::Int,
    dimension::Int,
    precision_bits::Int=256,
) where {T}
    return setprecision(BigFloat, precision_bits) do
        coefficients = lattice_like_coefficients(
            T;
            blocks,
            variables,
            dimension,
        )
        c = ones(T, variables)
        C = [zeros(T, dimension, dimension) for _ in 1:blocks]
        B = zeros(T, variables, 1)
        b = zeros(T, 1)

        ingest_stats = @timed SDPX.ingest(
            c,
            coefficients,
            C,
            B,
            b;
            sparse=:auto,
            verbosity=0,
        )
        problem = ingest_stats.value
        workspace_stats = @timed SDPX.Workspace(problem)
        workspace = workspace_stats.value
        X = [Matrix{T}(I, dimension, dimension) for _ in 1:blocks]
        Y = [Matrix{T}(I, dimension, dimension) for _ in 1:blocks]
        SDPX.factor_blocks!(workspace, X, Y) ||
            error("identity block factorization failed")

        # Warm the specialized kernels before measuring.
        SDPX.threaded_schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        repetitions = T === Float64 ? 3 : 1
        schur_seconds = minimum(1:repetitions) do _
            @elapsed SDPX.threaded_schur_build!(
                workspace,
                problem,
                problem.cons,
                X,
                Y,
            )
        end
        adaptive_schur = copy(workspace.S)

        dense_problem = SDPX.ingest(
            c,
            coefficients,
            C,
            B,
            b;
            sparse=false,
            verbosity=0,
        )
        dense_workspace = SDPX.Workspace(dense_problem)
        SDPX.factor_blocks!(dense_workspace, X, Y) ||
            error("dense reference block factorization failed")
        SDPX.threaded_schur_build!(
            dense_workspace,
            dense_problem,
            dense_problem.cons,
            X,
            Y,
        )
        dense_schur_seconds = minimum(1:repetitions) do _
            @elapsed SDPX.threaded_schur_build!(
                dense_workspace,
                dense_problem,
                dense_problem.cons,
                X,
                Y,
            )
        end
        relative_error = Float64(
            norm(adaptive_schur - dense_workspace.S) /
            max(norm(dense_workspace.S), eps(T)),
        )

        analysis = problem.structure
        return (
            arithmetic=string(T),
            julia_threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(),
            blocks,
            variables,
            dimension,
            profile=string(analysis.profile),
            selected_storage=string(analysis.selected_storage),
            coefficient_density=analysis.coefficient_density,
            block_pattern_density=analysis.block_pattern_density,
            schur_density=analysis.schur_density,
            schur_backend=string(analysis.schur_backend),
            dense_sparse_assembly=workspace.dense_sparse_assembly,
            ingest_seconds=ingest_stats.time,
            workspace_seconds=workspace_stats.time,
            schur_seconds,
            dense_schur_seconds,
            schur_speedup=dense_schur_seconds / schur_seconds,
            workspace_megabytes=Base.summarysize(workspace) / 1.0e6,
            relative_schur_error=relative_error,
        )
    end
end

function write_results(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do output
        println(
            output,
            "arithmetic,julia_threads,blas_threads,blocks,variables,dimension," *
            "profile,selected_storage," *
            "coefficient_density,block_pattern_density,schur_density," *
            "schur_backend,dense_sparse_assembly,ingest_seconds," *
            "workspace_seconds,schur_seconds,dense_schur_seconds," *
            "schur_speedup,workspace_megabytes," *
            "relative_schur_error",
        )
        for row in rows
            @printf(
                output,
                "%s,%d,%d,%d,%d,%d,%s,%s,%.9g,%.9g,%.9g,%s,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                row.arithmetic,
                row.julia_threads,
                row.blas_threads,
                row.blocks,
                row.variables,
                row.dimension,
                row.profile,
                row.selected_storage,
                row.coefficient_density,
                row.block_pattern_density,
                row.schur_density,
                row.schur_backend,
                row.dense_sparse_assembly,
                row.ingest_seconds,
                row.workspace_seconds,
                row.schur_seconds,
                row.dense_schur_seconds,
                row.schur_speedup,
                row.workspace_megabytes,
                row.relative_schur_error,
            )
        end
    end
end

function main(arguments)
    output = isempty(arguments) ?
             joinpath(@__DIR__, "adaptive-results.csv") :
             abspath(first(arguments))
    rows = Any[
        run_case(Float64; blocks=16, variables=700, dimension=28),
        run_case(Float64x4; blocks=8, variables=180, dimension=18),
        run_case(BigFloat; blocks=5, variables=80, dimension=12),
    ]
    write_results(output, rows)
    for row in rows
        println(row)
    end
end

main(ARGS)
