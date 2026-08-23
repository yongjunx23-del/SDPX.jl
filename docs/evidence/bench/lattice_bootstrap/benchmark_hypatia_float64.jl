#!/usr/bin/env julia

"""
Run the lattice-bootstrap Task_Low08 SDP with Hypatia in Float64.

The input is the compact binary produced by export_low08_binary.py. The model
uses sparse affine coefficient data and Hypatia's dense PSD barrier because the
aggregate PSD matrix patterns are effectively dense.
"""

using Hypatia
using LinearAlgebra
using Printf

const MOI = Hypatia.MOI
const MAGIC = collect(codeunits("LATSDP01"))

struct PSDBlockData
    dimension::Int
    rows::Vector{Int32}
    columns::Vector{Int32}
    slots::Vector{Int32}
    values::Vector{Float64}
end

struct LatticeData
    variable_count::Int
    equality_count::Int
    tolerance::Float64
    blocks::Vector{PSDBlockData}
    equality_rows::Vector{Int32}
    equality_columns::Vector{Int32}
    equality_values::Vector{Float64}
    equality_constants::Vector{Float64}
end

function read_vector(input::IO, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    read!(input, values)
    return values
end

function read_lattice_data(path::String)
    open(path, "r") do input
        magic = read_vector(input, UInt8, length(MAGIC))
        magic == MAGIC || error("Unexpected lattice benchmark file format")
        variable_count = Int(read(input, Int64))
        equality_count = Int(read(input, Int64))
        block_count = Int(read(input, Int64))
        tolerance = read(input, Float64)
        blocks = Vector{PSDBlockData}(undef, block_count)
        for block_index in 1:block_count
            dimension = Int(read(input, Int32))
            coefficient_count = Int(read(input, Int64))
            rows = Vector{Int32}(undef, coefficient_count)
            columns = similar(rows)
            slots = similar(rows)
            values = Vector{Float64}(undef, coefficient_count)
            for entry_index in 1:coefficient_count
                rows[entry_index] = read(input, Int32)
                columns[entry_index] = read(input, Int32)
                slots[entry_index] = read(input, Int32)
                values[entry_index] = read(input, Float64)
            end
            blocks[block_index] = PSDBlockData(
                dimension,
                rows,
                columns,
                slots,
                values,
            )
        end
        equality_nnz = Int(read(input, Int64))
        equality_rows = Vector{Int32}(undef, equality_nnz)
        equality_columns = similar(equality_rows)
        equality_values = Vector{Float64}(undef, equality_nnz)
        for entry_index in 1:equality_nnz
            equality_rows[entry_index] = read(input, Int32)
            equality_columns[entry_index] = read(input, Int32)
            equality_values[entry_index] = read(input, Float64)
        end
        equality_constants = read_vector(input, Float64, equality_count)
        eof(input) || error("Unexpected trailing data in lattice benchmark file")
        return LatticeData(
            variable_count,
            equality_count,
            tolerance,
            blocks,
            equality_rows,
            equality_columns,
            equality_values,
            equality_constants,
        )
    end
end

function triangle_index(row::Int, column::Int)
    @assert 1 <= row <= column
    return div(column * (column - 1), 2) + row
end

function build_model(data::LatticeData)
    model = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(model, data.variable_count)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.VariableIndex}(),
        variables[1],
    )

    root_two = sqrt(2.0)
    for block in data.blocks
        output_dimension = div(block.dimension * (block.dimension + 1), 2)
        constants = zeros(Float64, output_dimension)
        terms = MOI.VectorAffineTerm{Float64}[]
        sizehint!(terms, count(!iszero, block.slots))
        for entry_index in eachindex(block.values)
            row = Int(block.rows[entry_index]) + 1
            column = Int(block.columns[entry_index]) + 1
            slot = Int(block.slots[entry_index])
            scale = (row == column ? 1.0 : root_two)
            value = scale * block.values[entry_index]
            output_index = triangle_index(row, column)
            if slot == 0
                constants[output_index] += value
            else
                push!(
                    terms,
                    MOI.VectorAffineTerm(
                        output_index,
                        MOI.ScalarAffineTerm(value, variables[slot]),
                    ),
                )
            end
        end
        function_value = MOI.VectorAffineFunction(terms, constants)
        cone = MOI.Scaled(
            MOI.PositiveSemidefiniteConeTriangle(block.dimension),
        )
        MOI.add_constraint(model, function_value, cone)
    end

    equality_terms = MOI.VectorAffineTerm{Float64}[]
    sizehint!(equality_terms, length(data.equality_values))
    for entry_index in eachindex(data.equality_values)
        push!(
            equality_terms,
            MOI.VectorAffineTerm(
                Int(data.equality_rows[entry_index]) + 1,
                MOI.ScalarAffineTerm(
                    data.equality_values[entry_index],
                    variables[Int(data.equality_columns[entry_index]) + 1],
                ),
            ),
        )
    end
    equality_function = MOI.VectorAffineFunction(
        equality_terms,
        copy(data.equality_constants),
    )
    MOI.add_constraint(
        model,
        equality_function,
        MOI.Zeros(data.equality_count),
    )
    return model, variables
end

function validate_solution(data::LatticeData, variable_values::Vector{Float64})
    equality_residual = copy(data.equality_constants)
    for entry_index in eachindex(data.equality_values)
        equality_residual[Int(data.equality_rows[entry_index]) + 1] +=
            data.equality_values[entry_index] *
            variable_values[Int(data.equality_columns[entry_index]) + 1]
    end

    minimum_eigenvalues = Float64[]
    for block in data.blocks
        matrix = zeros(Float64, block.dimension, block.dimension)
        for entry_index in eachindex(block.values)
            row = Int(block.rows[entry_index]) + 1
            column = Int(block.columns[entry_index]) + 1
            slot = Int(block.slots[entry_index])
            multiplier = slot == 0 ? 1.0 : variable_values[slot]
            value = block.values[entry_index] * multiplier
            matrix[row, column] += value
            if row != column
                matrix[column, row] += value
            end
        end
        push!(minimum_eigenvalues, eigmin(Symmetric(matrix)))
    end
    return Dict{String, Any}(
        "max_absolute_linear_residual" => maximum(abs, equality_residual),
        "linear_residual_l2_norm" => norm(equality_residual),
        "minimum_psd_eigenvalue" => minimum(minimum_eigenvalues),
        "psd_block_minimum_eigenvalues" => minimum_eigenvalues,
    )
end

function json_escape(value::AbstractString)
    escaped = replace(
        value,
        '\\' => "\\\\",
        '"' => "\\\"",
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
    )
    return "\"$(escaped)\""
end

function json_value(value)
    if value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa AbstractString
        return json_escape(value)
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? repr(value) : json_escape(string(value))
    elseif value isa AbstractVector
        return "[" * join((json_value(item) for item in value), ",") * "]"
    elseif value isa AbstractDict
        items = sort!(collect(value); by = item -> string(first(item)))
        body = join(
            (
                json_escape(string(key)) * ":" * json_value(item)
                for (key, item) in items
            ),
            ",",
        )
        return "{" * body * "}"
    end
    error("Cannot encode $(typeof(value)) as JSON")
end

function write_json(path::String, result::AbstractDict)
    open(path, "w") do output
        println(output, json_value(result))
    end
end

function parse_arguments(arguments::Vector{String})
    length(arguments) >= 2 || error(
        "usage: benchmark_hypatia_float64.jl INPUT OUTPUT " *
        "[TIME_LIMIT_SECONDS] [ITERATION_LIMIT] [BLAS_THREADS]",
    )
    return (
        input = abspath(arguments[1]),
        output = abspath(arguments[2]),
        time_limit = length(arguments) >= 3 ? parse(Float64, arguments[3]) : 600.0,
        iteration_limit = length(arguments) >= 4 ? parse(Int, arguments[4]) : 500,
        blas_threads = length(arguments) >= 5 ? parse(Int, arguments[5]) : 4,
    )
end

function main(arguments::Vector{String})
    options = parse_arguments(arguments)
    mkpath(dirname(options.output))
    BLAS.set_num_threads(options.blas_threads)
    started = time()
    result = Dict{String, Any}(
        "arithmetic" => "Float64",
        "solver" => "Hypatia",
        "hypatia_version" => string(pkgversion(Hypatia)),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "time_limit_seconds" => options.time_limit,
        "iteration_limit" => options.iteration_limit,
    )
    try
        data = read_lattice_data(options.input)
        read_finished = time()
        model, variables = build_model(data)
        model_finished = time()
        optimizer = Hypatia.Optimizer{Float64}(
            verbose = true,
            iter_limit = options.iteration_limit,
            time_limit = options.time_limit,
            tol_rel_opt = data.tolerance,
            tol_abs_opt = data.tolerance,
            tol_feas = data.tolerance,
            preprocess = true,
            reduce = true,
            rescale = true,
            syssolver = Hypatia.Solvers.QRCholDenseSystemSolver{Float64}(),
            use_dense_model = false,
        )
        index_map = MOI.copy_to(optimizer, model)
        copy_finished = time()
        @printf(
            "Loaded %d variables, %d equalities, and %d PSD blocks; starting Hypatia with %d BLAS threads.\n",
            data.variable_count,
            data.equality_count,
            length(data.blocks),
            BLAS.get_num_threads(),
        )
        flush(stdout)
        MOI.optimize!(optimizer)
        solve_finished = time()

        termination_status = MOI.get(optimizer, MOI.TerminationStatus())
        result_count = MOI.get(optimizer, MOI.ResultCount())
        result["status"] = string(termination_status)
        result["raw_status"] = MOI.get(optimizer, MOI.RawStatusString())
        result["result_count"] = result_count
        result["barrier_iterations"] = MOI.get(
            optimizer,
            MOI.BarrierIterations(),
        )
        result["solver_time_seconds"] = MOI.get(
            optimizer,
            MOI.SolveTimeSec(),
        )
        if result_count > 0
            variable_values = [
                MOI.get(
                    optimizer,
                    MOI.VariablePrimal(),
                    index_map[variable],
                ) for variable in variables
            ]
            result["objective"] = MOI.get(optimizer, MOI.ObjectiveValue())
            result["diagnostics"] = validate_solution(data, variable_values)
        end
        validation_finished = time()
        result["timing_seconds"] = Dict{String, Any}(
            "input_read" => read_finished - started,
            "model_construction" => model_finished - read_finished,
            "optimizer_copy" => copy_finished - model_finished,
            "solve" => solve_finished - copy_finished,
            "validation" => validation_finished - solve_finished,
            "total" => validation_finished - started,
        )
    catch exception
        result["status"] = "ERROR"
        result["error"] = sprint(showerror, exception, catch_backtrace())
        result["timing_seconds"] = Dict("total" => time() - started)
        write_json(options.output, result)
        rethrow()
    end
    write_json(options.output, result)
    println(json_value(result))
end

main(ARGS)
