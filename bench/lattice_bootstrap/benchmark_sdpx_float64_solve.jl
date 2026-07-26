#!/usr/bin/env julia

"""Run and validate a complete Float64 SDPX solve of Task_Low08."""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

const MAGIC = collect(codeunits("LATSDP01"))

function read_vector(input::IO, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    read!(input, values)
    return values
end

function read_problem(path::String)
    open(path, "r") do input
        read_vector(input, UInt8, length(MAGIC)) == MAGIC ||
            error("Unexpected lattice benchmark file format")
        variables = Int(read(input, Int64))
        equalities = Int(read(input, Int64))
        block_count = Int(read(input, Int64))
        tolerance = read(input, Float64)

        coefficients =
            Vector{Vector{SparseMatrixCSC{Float64,Int}}}(undef, block_count)
        constants = Vector{Matrix{Float64}}(undef, block_count)
        dimensions = Vector{Int}(undef, block_count)
        for block_index in 1:block_count
            dimension = Int(read(input, Int32))
            dimensions[block_index] = dimension
            entry_count = Int(read(input, Int64))
            rows_by_slot = Dict{Int,Vector{Int}}()
            columns_by_slot = Dict{Int,Vector{Int}}()
            values_by_slot = Dict{Int,Vector{Float64}}()
            constant = zeros(Float64, dimension, dimension)
            for _ in 1:entry_count
                row = Int(read(input, Int32)) + 1
                column = Int(read(input, Int32)) + 1
                slot = Int(read(input, Int32))
                value = read(input, Float64)
                if slot == 0
                    constant[row, column] += value
                    row != column && (constant[column, row] += value)
                else
                    rows = get!(rows_by_slot, slot, Int[])
                    columns = get!(columns_by_slot, slot, Int[])
                    values = get!(values_by_slot, slot, Float64[])
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
            coefficients[block_index] = [
                haskey(rows_by_slot, variable) ?
                sparse(
                    rows_by_slot[variable],
                    columns_by_slot[variable],
                    values_by_slot[variable],
                    dimension,
                    dimension,
                ) :
                spzeros(Float64, dimension, dimension)
                for variable in 1:variables
            ]
            # SDPX uses sum_i x_i A_i - C >= 0.
            constants[block_index] = -constant
        end

        equality_nnz = Int(read(input, Int64))
        equality_rows = Vector{Int}(undef, equality_nnz)
        equality_columns = Vector{Int}(undef, equality_nnz)
        equality_values = Vector{Float64}(undef, equality_nnz)
        for entry_index in 1:equality_nnz
            equality_rows[entry_index] = Int(read(input, Int32)) + 1
            equality_columns[entry_index] = Int(read(input, Int32)) + 1
            equality_values[entry_index] = read(input, Float64)
        end
        equality_constants = read_vector(input, Float64, equalities)
        eof(input) || error("Unexpected trailing data")
        equality_matrix = sparse(
            equality_rows,
            equality_columns,
            equality_values,
            equalities,
            variables,
        )
        objective = zeros(Float64, variables)
        objective[1] = 1.0
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

function equality_basis(B, b)
    factorization = qr(B, ColumnNorm())
    diagonal = abs.(diag(factorization.R))
    tolerance =
        maximum(size(B)) * eps(Float64) * maximum(diagonal; init=0.0)
    rank = count(>(tolerance), diagonal)
    permutation = factorization.p
    independent = permutation[1:rank]
    dependent = permutation[(rank + 1):end]
    dependency_residual = 0.0
    if !isempty(dependent)
        R = factorization.R
        coefficients =
            UpperTriangular(R[1:rank, 1:rank]) \ R[1:rank, (rank + 1):end]
        predicted = transpose(coefficients) * b[independent]
        dependency_residual = maximum(abs, predicted - b[dependent])
    end
    return independent, rank, tolerance, dependency_residual
end

function validate_solution(problem, result, full_B, full_b)
    equality_residual = transpose(full_B) * result.x - full_b
    minimum_eigenvalue = Inf
    minimum_dual_eigenvalue = Inf
    block_minimum_eigenvalues = Float64[]
    dual_block_minimum_eigenvalues = Float64[]
    cons = problem.cons::SDPX.SparseCons{Float64}
    for block_index in 1:problem.dims.L
        matrix = -copy(problem.C[block_index])
        for variable in cons.active[block_index]
            coefficient = cons.Asp[block_index][variable]
            rows = rowvals(coefficient)
            values = nonzeros(coefficient)
            multiplier = result.x[variable]
            for column in axes(coefficient, 2)
                for index in nzrange(coefficient, column)
                    matrix[rows[index], column] += multiplier * values[index]
                end
            end
        end
        block_minimum = eigmin(Symmetric(matrix))
        push!(block_minimum_eigenvalues, block_minimum)
        minimum_eigenvalue = min(minimum_eigenvalue, block_minimum)
        dual_block_minimum = eigmin(Symmetric(result.Y[block_index]))
        push!(
            dual_block_minimum_eigenvalues,
            dual_block_minimum,
        )
        minimum_dual_eigenvalue =
            min(minimum_dual_eigenvalue, dual_block_minimum)
    end
    return (
        max_absolute_linear_residual=maximum(abs, equality_residual),
        linear_residual_l2_norm=norm(equality_residual),
        minimum_psd_eigenvalue=minimum_eigenvalue,
        psd_block_minimum_eigenvalues=block_minimum_eigenvalues,
        minimum_dual_psd_eigenvalue=minimum_dual_eigenvalue,
        dual_psd_block_minimum_eigenvalues=
            dual_block_minimum_eigenvalues,
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
    value === nothing && return "null"
    value isa Bool && return value ? "true" : "false"
    value isa AbstractString && return json_escape(value)
    value isa Symbol && return json_escape(string(value))
    value isa Enum && return json_escape(string(value))
    value isa Integer && return string(value)
    value isa AbstractFloat &&
        return isfinite(value) ? repr(value) : json_escape(string(value))
    value isa AbstractVector &&
        return "[" * join((json_value(item) for item in value), ",") * "]"
    if value isa NamedTuple
        return json_value(Dict(string(key) => item for (key, item) in pairs(value)))
    end
    if value isa AbstractDict
        items = sort!(collect(value); by=item -> string(first(item)))
        return "{" * join(
            (
                json_escape(string(key)) * ":" * json_value(item)
                for (key, item) in items
            ),
            ",",
        ) * "}"
    end
    error("Cannot encode $(typeof(value)) as JSON")
end

function write_json(path::String, result)
    mkpath(dirname(path))
    open(path, "w") do output
        println(output, json_value(result))
    end
end

function main(arguments)
    length(arguments) >= 2 || error(
        "usage: benchmark_sdpx_float64_solve.jl INPUT OUTPUT " *
        "[BLAS_THREADS] [MAX_TIME_SECONDS] [BETA] [GAMMA] " *
        "[OMEGA_P] [OMEGA_D] [classic|sdpb] [MAX_ITER] [TOLERANCE]",
    )
    input_path = abspath(arguments[1])
    output_path = abspath(arguments[2])
    blas_threads = length(arguments) >= 3 ? parse(Int, arguments[3]) : 4
    max_time = length(arguments) >= 4 ? parse(Float64, arguments[4]) : 600.0
    beta = length(arguments) >= 5 ? parse(Float64, arguments[5]) : 0.1
    gamma = length(arguments) >= 6 ? parse(Float64, arguments[6]) : 0.9
    omega_p = length(arguments) >= 7 ? parse(Float64, arguments[7]) : 1.0
    omega_d = length(arguments) >= 8 ? parse(Float64, arguments[8]) : 1.0
    predictor =
        length(arguments) >= 9 ? Symbol(arguments[9]) : :classic
    max_iter = length(arguments) >= 10 ? parse(Int, arguments[10]) : 100
    SDPX.set_blas_threads!(blas_threads)

    started = time()
    read_timing = @timed read_problem(input_path)
    data = read_timing.value
    solve_tolerance =
        length(arguments) >= 11 ? parse(Float64, arguments[11]) : data.tolerance
    presolve_timing = @timed equality_basis(data.B, data.b)
    equality_ids, equality_rank, rank_tolerance, dependency_residual =
        presolve_timing.value
    @printf(
        "Equality presolve: %d -> %d constraints (dependency residual %.3e)\n",
        length(data.b),
        equality_rank,
        dependency_residual,
    )
    solve_B = data.B[:, equality_ids]
    solve_b = data.b[equality_ids]
    ingest_timing = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        solve_B,
        solve_b;
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    problem = ingest_timing.value
    options = SDPX.SolverOptions{Float64}(
        β=beta,
        γ=gamma,
        Ωp=omega_p,
        Ωd=omega_d,
        ϵ_gap=solve_tolerance,
        ϵ_primal=solve_tolerance,
        ϵ_dual=solve_tolerance,
        iter_max=max_iter,
        max_time=max_time,
        verbosity=1,
        timing=true,
        restart=true,
        max_restarts=5,
        sparse=:auto,
        parameter_policy=:fixed,
        predictor=predictor,
        step_rule=:backtrack,
        refine_steps=1,
    )
    solve_started = time()
    result = SDPX.solve!(problem, options)
    solve_finished = time()
    diagnostics = validate_solution(problem, result, data.B, data.b)
    certificate = result.diagnostics === nothing ?
                  nothing :
                  result.diagnostics.selected_algorithms.certificate
    finished = time()

    output = Dict{String,Any}(
        "task_name" => "DreamTest",
        "arithmetic" => "Float64",
        "solver" => "SDPX",
        "sdpx_version" => string(pkgversion(SDPX)),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => SDPX.blas_threads(),
        "status" => string(result.status),
        "message" => result.message,
        "objective" => result.pObj,
        "dual_objective" => result.dObj,
        "relative_gap" => result.gap_rel,
        "reported_primal_residual" => result.p_res,
        "reported_dual_residual" => result.d_res,
        "iterations" => result.iterations,
        "restarts" => result.restarts,
        "regularizations" => result.regularizations,
        "variables" => problem.dims.m,
        "equalities" => length(data.b),
        "presolved_equalities" => problem.dims.n,
        "equality_rank" => equality_rank,
        "equality_rank_tolerance" => rank_tolerance,
        "equality_dependency_residual" => dependency_residual,
        "psd_blocks" => problem.dims.L,
        "structure" => SDPX.structure_summary(problem),
        "parameters" => Dict(
            "beta" => options.β,
            "gamma" => options.γ,
            "omega_p" => options.Ωp,
            "omega_d" => options.Ωd,
            "tolerance_gap" => options.ϵ_gap,
            "tolerance_primal" => options.ϵ_primal,
            "tolerance_dual" => options.ϵ_dual,
            "predictor" => options.predictor,
            "step_rule" => options.step_rule,
            "refine_steps" => options.refine_steps,
        ),
        "timing_seconds" => Dict(
            "input_build" => read_timing.time,
            "equality_presolve" => presolve_timing.time,
            "ingest" => ingest_timing.time,
            "solve" => solve_finished - solve_started,
            "validation" => finished - solve_finished,
            "total" => finished - started,
            "solver_phases" => result.timings,
        ),
        "diagnostics" => diagnostics,
        "certificate" => certificate,
        "parameter_history" => result.parameter_history,
        "variable_values" => result.x,
    )
    write_json(output_path, output)
    @printf(
        "SDPX status=%s objective=%.12g solve=%.3fs total=%.3fs eq_res=%.3e min_primal_eig=%.3e min_dual_eig=%.3e\n",
        result.status,
        result.pObj,
        solve_finished - solve_started,
        finished - started,
        diagnostics.max_absolute_linear_residual,
        diagnostics.minimum_psd_eigenvalue,
        diagnostics.minimum_dual_psd_eigenvalue,
    )
end

main(ARGS)
