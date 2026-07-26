#!/usr/bin/env julia

"""
Measure SDPX's adaptive Float64 path on the exported Task_Low08 instance.

The benchmark intentionally times one Schur-complement build, rather than a
full solve, so it isolates the structural optimization introduced for this
large, equality-heavy SDP.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using SparseArrays

const MAGIC = collect(codeunits("LATSDP01"))

function read_vector(input::IO, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    read!(input, values)
    return values
end

function read_problem(path::String, ::Type{T}) where {T}
    open(path, "r") do input
        read_vector(input, UInt8, length(MAGIC)) == MAGIC ||
            error("Unexpected lattice benchmark file format")
        variables = Int(read(input, Int64))
        equalities = Int(read(input, Int64))
        block_count = Int(read(input, Int64))
        tolerance = T(read(input, Float64))

        coefficients =
            Vector{Vector{SparseMatrixCSC{T,Int}}}(undef, block_count)
        constants = Vector{Matrix{T}}(undef, block_count)
        dimensions = Vector{Int}(undef, block_count)
        for block_index in 1:block_count
            dimension = Int(read(input, Int32))
            dimensions[block_index] = dimension
            entry_count = Int(read(input, Int64))
            rows_by_slot = Dict{Int,Vector{Int}}()
            columns_by_slot = Dict{Int,Vector{Int}}()
            values_by_slot = Dict{Int,Vector{T}}()
            constant = zeros(T, dimension, dimension)
            for _ in 1:entry_count
                row = Int(read(input, Int32)) + 1
                column = Int(read(input, Int32)) + 1
                slot = Int(read(input, Int32))
                value = T(read(input, Float64))
                if slot == 0
                    constant[row, column] += value
                    row != column && (constant[column, row] += value)
                else
                    rows = get!(rows_by_slot, slot, Int[])
                    columns = get!(columns_by_slot, slot, Int[])
                    values = get!(values_by_slot, slot, T[])
                    push!(rows, row)
                    push!(columns, column)
                    push!(values, value)
                    if row != column
                        push!(rows, column)
                        push!(columns, row)
                        push!(values, value)
                    end
                end
            end
            block = [
                haskey(rows_by_slot, variable) ?
                sparse(
                    rows_by_slot[variable],
                    columns_by_slot[variable],
                    values_by_slot[variable],
                    dimension,
                    dimension,
                ) :
                spzeros(T, dimension, dimension)
                for variable in 1:variables
            ]
            coefficients[block_index] = block
            # SDPX represents sum_i x_i A_i - C >= 0.
            constants[block_index] = -constant
        end

        equality_nnz = Int(read(input, Int64))
        equality_rows = Vector{Int}(undef, equality_nnz)
        equality_columns = Vector{Int}(undef, equality_nnz)
        equality_values = Vector{T}(undef, equality_nnz)
        for entry_index in 1:equality_nnz
            equality_rows[entry_index] = Int(read(input, Int32)) + 1
            equality_columns[entry_index] = Int(read(input, Int32)) + 1
            equality_values[entry_index] = T(read(input, Float64))
        end
        equality_constants = T.(read_vector(input, Float64, equalities))
        eof(input) || error("Unexpected trailing data")
        equality_matrix = sparse(
            equality_rows,
            equality_columns,
            equality_values,
            equalities,
            variables,
        )
        objective = zeros(T, variables)
        objective[1] = one(T)
        return (
            c=objective,
            A=coefficients,
            C=constants,
            B=Matrix(transpose(equality_matrix)),
            b=-equality_constants,
            tolerance,
            dimensions,
        )
    end
end

function json_string(value)
    value isa Bool && return value ? "true" : "false"
    value isa Number && return isfinite(value) ? repr(value) : "\"$value\""
    value isa Symbol && return "\"$value\""
    value isa AbstractString &&
        return "\"" * replace(value, '\\' => "\\\\", '"' => "\\\"") * "\""
    error("Unsupported JSON value $(typeof(value))")
end

function write_json(path::String, values)
    mkpath(dirname(path))
    open(path, "w") do output
        println(output, "{")
        pairs = collect(values)
        for (index, (key, value)) in enumerate(pairs)
            comma = index == length(pairs) ? "" : ","
            println(output, "  \"", key, "\": ", json_string(value), comma)
        end
        println(output, "}")
    end
end

function validate_identity_schur(problem::SDPX.SDPProblem{T}, workspace; samples::Int=256) where {T}
    cons = problem.cons::SDPX.SparseCons{T}
    variables = problem.dims.m
    state = UInt64(0x9e3779b97f4a7c15)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for _ in 1:samples
        state =
            state * UInt64(6364136223846793005) +
            UInt64(1442695040888963407)
        row = Int(rem(state, UInt64(variables))) + 1
        state =
            state * UInt64(6364136223846793005) +
            UInt64(1442695040888963407)
        column = Int(rem(state, UInt64(variables))) + 1
        expected = zero(T)
        for block in cons.Asp
            expected += dot(block[row], block[column])
        end
        actual = workspace.S[row, column]
        absolute_error = Float64(abs(actual - expected))
        relative_error =
            Float64(abs(actual - expected) / max(abs(expected), one(T)))
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
    end
    return maximum_absolute_error, maximum_relative_error
end

function main(arguments)
    length(arguments) >= 2 || error(
        "usage: benchmark_sdpx_adaptive_float64.jl INPUT OUTPUT " *
        "[BLAS_THREADS] [Float64|Float64x4|BigFloat] [PRECISION_BITS]",
    )
    input_path = abspath(arguments[1])
    output_path = abspath(arguments[2])
    blas_threads = length(arguments) >= 3 ? parse(Int, arguments[3]) : 1
    arithmetic = length(arguments) >= 4 ? arguments[4] : "Float64"
    T = arithmetic == "Float64" ? Float64 :
        arithmetic == "Float64x4" ? Float64x4 :
        arithmetic == "BigFloat" ? BigFloat :
        error("Unsupported arithmetic $arithmetic")
    precision_bits = length(arguments) >= 5 ? parse(Int, arguments[5]) : 256
    T === BigFloat && setprecision(BigFloat, precision_bits)
    SDPX.set_blas_threads!(blas_threads)

    load_timing = @timed read_problem(input_path, T)
    data = load_timing.value
    ingest_timing = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    problem = ingest_timing.value
    workspace_timing = @timed SDPX.Workspace(problem)
    workspace = workspace_timing.value
    X = [Matrix{T}(I, k, k) for k in data.dimensions]
    Y = [Matrix{T}(I, k, k) for k in data.dimensions]
    SDPX.factor_blocks!(workspace, X, Y) ||
        error("Identity block factorization failed")

    # Ask Julia to compile the concrete hot-call signature before timing.
    precompile(
        SDPX.threaded_schur_build!,
        (
            typeof(workspace),
            typeof(problem),
            typeof(problem.cons),
            typeof(X),
            typeof(Y),
        ),
    )
    GC.gc()
    schur_seconds = @elapsed SDPX.threaded_schur_build!(
        workspace,
        problem,
        problem.cons,
        X,
        Y,
    )
    maximum_absolute_error, maximum_relative_error =
        validate_identity_schur(problem, workspace)
    factor_seconds = NaN
    factor_ok = false
    equality_factor_pivoted = false
    regularization_attempts = 0
    if T === Float64
        options = SDPX.SolverOptions{T}(verbosity=0)
        precompile(
            SDPX.factor_kkt!,
            (typeof(workspace), typeof(problem), typeof(options)),
        )
        factor_result = nothing
        factor_seconds = @elapsed factor_result =
            SDPX.factor_kkt!(workspace, problem, options)
        factor_ok = factor_result.ok
        equality_factor_pivoted = factor_result.q_pivoted
        regularization_attempts = factor_result.reg_attempts
    end
    analysis = problem.structure
    result = [
        "benchmark" => "Task_Low08 SDPX adaptive Schur assembly",
        "arithmetic" => string(T),
        "precision_bits" => T === BigFloat ? precision_bits : precision(T),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => SDPX.blas_threads(),
        "variables" => problem.dims.m,
        "equalities" => problem.dims.n,
        "psd_blocks" => problem.dims.L,
        "coefficient_density" => analysis.coefficient_density,
        "active_density" => analysis.active_density,
        "block_pattern_density" => analysis.block_pattern_density,
        "schur_density" => analysis.schur_density,
        "schur_exact" => analysis.schur_exact,
        "profile" => analysis.profile,
        "selected_storage" => analysis.selected_storage,
        "psd_kernel" => analysis.psd_kernel,
        "schur_backend" => analysis.schur_backend,
        "dense_sparse_assembly" => workspace.dense_sparse_assembly,
        "input_build_seconds" => load_timing.time,
        "ingest_seconds" => ingest_timing.time,
        "workspace_seconds" => workspace_timing.time,
        "workspace_megabytes" => Base.summarysize(workspace) / 1.0e6,
        "schur_build_seconds" => schur_seconds,
        "kkt_factor_seconds" => factor_seconds,
        "kkt_factor_ok" => factor_ok,
        "equality_factor_pivoted" => equality_factor_pivoted,
        "regularization_attempts" => regularization_attempts,
        "schur_frobenius_norm" =>
            T === Float64 ? norm(workspace.S) : "not evaluated",
        "sampled_schur_max_absolute_error" => maximum_absolute_error,
        "sampled_schur_max_relative_error" => maximum_relative_error,
    ]
    write_json(output_path, result)
    for (key, value) in result
        @printf("%-28s %s\n", key, value)
    end
end

main(ARGS)
