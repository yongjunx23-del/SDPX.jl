#!/usr/bin/env julia

"""Matched native-SDPX and Convex.jl frontend benchmark.

Run one frontend in each process so peak RSS and allocation measurements are
not contaminated by the other frontend. Problem data are generated before the
timed region and are identical for a given case, size, and arithmetic type.
"""

import Convex
using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats: Float64x4
using Random
using SDPX
using SparseArrays
import MathOptInterface as MOI

function parse_arguments(arguments)
    values = Dict{String,String}()
    for argument in arguments
        startswith(argument, "--") || error("expected --name=value, got '$argument'")
        pieces = split(argument[3:end], "="; limit=2)
        length(pieces) == 2 || error("expected --name=value, got '$argument'")
        values[pieces[1]] = pieces[2]
    end
    return (
        case=Symbol(get(values, "case", "lp")),
        frontend=Symbol(get(values, "frontend", "native")),
        arithmetic=get(values, "arithmetic", "Float64"),
        size=parse(Int, get(values, "size", "0")),
        repetitions=parse(Int, get(values, "repetitions", "5")),
        threads=parse(Int, get(values, "threads", "1")),
        output=get(values, "output", ""),
    )
end

function arithmetic_type(name)
    name == "Float64" && return Float64
    name == "Float64x4" && return Float64x4
    name == "BigFloat256" && return BigFloat
    error("arithmetic must be Float64, Float64x4, or BigFloat256")
end

default_size(case::Symbol) = case === :lp ? 512 : case === :socp ? 24 : 12

function benchmark_tolerance(::Type{T}) where {T}
    T === Float64 && return T(1e-8)
    T === BigFloat && return parse(BigFloat, "1e-24")
    return T(1e-18)
end

function solver_options(::Type{T}, tolerance::T, threads::Int) where {T}
    return SDPX.SolverOptions{T}(
        sparse=:auto,
        parameter_policy=:auto,
        parameter_strategy=:adaptive,
        threads=threads,
        iter_max=400,
        max_time=1800.0,
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        verbosity=0,
        diagnostics=true,
        timing=true,
        precision_bits=T === BigFloat ? 256 : 997,
        working_precision_policy=:fixed,
    )
end

function convex_optimizer(::Type{T}, tolerance::T, threads::Int) where {T}
    attributes = Any[
        MOI.RawOptimizerAttribute("tolerance") => tolerance,
        MOI.RawOptimizerAttribute("max_iterations") => 400,
        MOI.RawOptimizerAttribute("time_limit") => 1800.0,
        MOI.RawOptimizerAttribute("sparse") => :auto,
        MOI.RawOptimizerAttribute("parameter_policy") => :auto,
        MOI.RawOptimizerAttribute("parameter_strategy") => :adaptive,
        MOI.RawOptimizerAttribute("diagnostics") => true,
        MOI.RawOptimizerAttribute("timing") => true,
        MOI.NumberOfThreads() => threads,
        MOI.Silent() => true,
    ]
    if T === BigFloat
        push!(
            attributes,
            MOI.RawOptimizerAttribute("precision") => precision(BigFloat),
            MOI.RawOptimizerAttribute("working_precision_policy") => :fixed,
        )
    end
    return MOI.OptimizerWithAttributes(SDPX.Optimizer{T}, attributes...)
end

function lp_data(::Type{T}, variables::Int) where {T}
    variables >= 2 || error("LP size must be at least 2")
    denominator = T(variables - 1)
    objective = [one(T) + T(i - 1) / denominator for i in 1:variables]
    return (objective=objective, variables=variables)
end

function sdp_data(::Type{T}, side::Int) where {T}
    side >= 2 || error("SDP size must be at least 2")
    rng = MersenneTwister(0x53445058 + side)
    adjacency = zeros(T, side, side)
    for column in 2:side, row in 1:(column - 1)
        value = T(0.25 + rand(rng))
        adjacency[row, column] = value
        adjacency[column, row] = value
    end
    laplacian = Matrix(Diagonal(vec(sum(adjacency; dims=2))) - adjacency)
    return (weight=laplacian / T(4), side=side)
end

function case_data(::Type{T}, case::Symbol, size::Int) where {T}
    case === :lp && return lp_data(T, size)
    case === :socp && return (dimension=size,)
    case === :sdp && return sdp_data(T, size)
    error("case must be lp, socp, or sdp")
end

function build_native_lp(::Type{T}, data) where {T}
    variables = data.variables
    coefficients = [
        SDPX.CompactScalarCoefficientVector(T, variables, variable, one(T))
        for variable in 1:variables
    ]
    constants = [zeros(T, 1, 1) for _ in 1:variables]
    problem = SDPX.ingest(
        data.objective,
        coefficients,
        constants,
        fill(one(T), variables, 1),
        T[one(T)];
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, data=data)
end

function build_native_socp(::Type{T}, data) where {T}
    dimension = data.dimension
    variables = dimension + 1
    side = dimension + 1
    coefficients = Vector{SparseMatrixCSC{T,Int}}(undef, variables)
    coefficients[1] = sparse(Matrix{T}(I, side, side))
    for index in 1:dimension
        coefficients[index + 1] = sparse(
            [1, index + 1],
            [index + 1, 1],
            T[one(T), one(T)],
            side,
            side,
        )
    end
    equality = zeros(T, variables, 1)
    equality[2:end, 1] .= one(T)
    objective = zeros(T, variables)
    objective[1] = one(T)
    problem = SDPX.ingest(
        objective,
        [coefficients],
        [zeros(T, side, side)],
        equality,
        T[one(T)];
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, data=data)
end

function build_native_sdp(::Type{T}, data) where {T}
    side = data.side
    variables = side * (side + 1) ÷ 2
    coefficients = Vector{SparseMatrixCSC{T,Int}}(undef, variables)
    objective = Vector{T}(undef, variables)
    equality = zeros(T, variables, side)
    variable = 0
    for column in 1:side, row in 1:column
        variable += 1
        if row == column
            coefficients[variable] = sparse(
                [row], [column], T[one(T)], side, side,
            )
            objective[variable] = -data.weight[row, column]
            equality[variable, row] = one(T)
        else
            coefficients[variable] = sparse(
                [row, column],
                [column, row],
                T[one(T), one(T)],
                side,
                side,
            )
            objective[variable] = -T(2) * data.weight[row, column]
        end
    end
    problem = SDPX.ingest(
        objective,
        [coefficients],
        [zeros(T, side, side)],
        equality,
        fill(one(T), side);
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, data=data)
end

function build_native(::Type{T}, case::Symbol, data) where {T}
    case === :lp && return build_native_lp(T, data)
    case === :socp && return build_native_socp(T, data)
    return build_native_sdp(T, data)
end

function build_convex_lp(::Type{T}, data) where {T}
    x = Convex.Variable(data.variables)
    problem = Convex.minimize(
        sum(data.objective .* x),
        [x >= zero(T), sum(x) == one(T)];
        numeric_type=T,
    )
    return (problem=problem, variable=x, data=data)
end

function build_convex_socp(::Type{T}, data) where {T}
    x = Convex.Variable(data.dimension)
    problem = Convex.minimize(
        Convex.norm2(x),
        [sum(x) == one(T)];
        numeric_type=T,
    )
    return (problem=problem, variable=x, data=data)
end

function build_convex_sdp(::Type{T}, data) where {T}
    X = Convex.Semidefinite(data.side)
    problem = Convex.maximize(
        Convex.tr(data.weight * X),
        [LinearAlgebra.diag(X) == fill(one(T), data.side)];
        numeric_type=T,
    )
    return (problem=problem, variable=X, data=data)
end

function build_convex(::Type{T}, case::Symbol, data) where {T}
    case === :lp && return build_convex_lp(T, data)
    case === :socp && return build_convex_socp(T, data)
    return build_convex_sdp(T, data)
end

minimum_eigenvalue(matrix) = eigmin(Symmetric(matrix))

function native_validation(case::Symbol, payload, result, options, tolerance)
    problem = payload.problem
    data = payload.data
    if case === :lp
        objective = dot(data.objective, result.x)
        equality = abs(sum(result.x) - one(eltype(result.x)))
        margin = minimum(result.x)
    elseif case === :socp
        vector = view(result.x, 2:length(result.x))
        objective = result.x[1]
        equality = abs(sum(vector) - one(eltype(result.x)))
        margin = minimum_eigenvalue(result.X[1])
    else
        matrix = result.X[1]
        objective = sum(data.weight .* matrix)
        equality = maximum(
            abs(matrix[index, index] - one(eltype(matrix)))
            for index in 1:data.side
        )
        margin = minimum_eigenvalue(matrix)
    end
    certificate = SDPX.result_certificate(problem, result, options)
    return (
        status=string(result.status),
        iterations=result.iterations,
        objective=objective,
        equality_violation=equality,
        cone_margin=margin,
        certificate_valid=certificate.valid,
        validation_gate=tolerance,
        source_model_variables=problem.dims.m,
        core_model_variables=problem.dims.m,
        core_model_equalities=problem.dims.n,
        core_model_psd_blocks=problem.dims.L,
    )
end

function convex_validation(case::Symbol, payload, tolerance)
    problem = payload.problem
    data = payload.data
    value = Convex.evaluate(payload.variable)
    if case === :lp
        vector = vec(value)
        objective = dot(data.objective, vector)
        equality = abs(sum(vector) - one(eltype(vector)))
        margin = minimum(vector)
    elseif case === :socp
        vector = vec(value)
        objective = norm(vector)
        equality = abs(sum(vector) - one(eltype(vector)))
        margin = problem.optval - norm(vector)
    else
        matrix = Matrix(value)
        objective = sum(data.weight .* matrix)
        equality = maximum(
            abs(matrix[index, index] - one(eltype(matrix)))
            for index in 1:data.side
        )
        margin = minimum_eigenvalue(matrix)
    end
    T = typeof(tolerance)
    floor = T === Float64 ? T(1e-6) : T(1e-12)
    gate = max(floor, T(100) * tolerance)
    valid = Convex.termination_status(problem) == MOI.OPTIMAL &&
            equality <= gate && margin >= -gate
    raw = MOI.get(problem.model, MOI.RawSolver())
    return (
        status=string(Convex.termination_status(problem)),
        iterations=MOI.get(problem.model, MOI.BarrierIterations()),
        objective=objective,
        equality_violation=equality,
        cone_margin=margin,
        certificate_valid=valid,
        validation_gate=gate,
        source_model_variables=MOI.get(problem.model, MOI.NumberOfVariables()),
        core_model_variables=length(raw.x),
        core_model_equalities=length(raw.y),
        core_model_psd_blocks=length(raw.X),
    )
end

function run_sample(::Type{T}, case, frontend, data, options, tolerance, threads) where {T}
    build_stats = frontend === :native ?
                  @timed(build_native(T, case, data)) :
                  @timed(build_convex(T, case, data))
    payload = build_stats.value
    solve_stats = if frontend === :native
        @timed SDPX.solve!(payload.problem, options)
    else
        @timed Convex.solve!(
            payload.problem,
            convex_optimizer(T, tolerance, threads);
            silent=true,
        )
    end
    validation_stats = frontend === :native ?
                       @timed(native_validation(
                           case,
                           payload,
                           solve_stats.value,
                           options,
                           tolerance,
                       )) :
                       @timed(convex_validation(case, payload, tolerance))
    core_seconds = frontend === :native ? solve_stats.time :
                   Float64(MOI.get(payload.problem.model, MOI.SolveTimeSec()))
    return (
        build_seconds=build_stats.time,
        frontend_solve_seconds=solve_stats.time,
        core_solver_seconds=core_seconds,
        frontend_overhead_seconds=max(0.0, solve_stats.time - core_seconds),
        validation_seconds=validation_stats.time,
        end_to_end_seconds=build_stats.time + solve_stats.time + validation_stats.time,
        build_allocated_bytes=build_stats.bytes,
        solve_allocated_bytes=solve_stats.bytes,
        validation_allocated_bytes=validation_stats.bytes,
        validation=validation_stats.value,
    )
end

function median_value(values)
    sorted = sort!(collect(values))
    count = length(sorted)
    return isodd(count) ? sorted[(count + 1) ÷ 2] :
           (sorted[count ÷ 2] + sorted[count ÷ 2 + 1]) / 2
end

function csv_value(value)
    text = string(value)
    return occursin(',', text) || occursin('"', text) ?
           "\"" * replace(text, '"' => "\"\"") * "\"" : text
end

function write_record(record, output)
    columns = collect(keys(record))
    lines = [
        join(string.(columns), ","),
        join((csv_value(getproperty(record, column)) for column in columns), ","),
    ]
    if isempty(output)
        println(join(lines, "\n"))
    else
        mkpath(dirname(abspath(output)))
        open(output, "w") do io
            println(io, join(lines, "\n"))
        end
        println("wrote ", abspath(output))
    end
end

function benchmark(::Type{T}, config) where {T}
    config.case in (:lp, :socp, :sdp) || error("case must be lp, socp, or sdp")
    config.frontend in (:native, :convex) || error("frontend must be native or convex")
    config.threads <= Threads.nthreads() || error(
        "requested $(config.threads) solver threads, but Julia has $(Threads.nthreads())",
    )
    size = config.size == 0 ? default_size(config.case) : config.size
    tolerance = benchmark_tolerance(T)
    data = case_data(T, config.case, size)
    options = solver_options(T, tolerance, config.threads)

    compile_seconds = @elapsed run_sample(
        T, config.case, config.frontend, data, options, tolerance, config.threads,
    )
    samples = NamedTuple[]
    for _ in 1:config.repetitions
        GC.gc()
        push!(
            samples,
            run_sample(
                T,
                config.case,
                config.frontend,
                data,
                options,
                tolerance,
                config.threads,
            ),
        )
    end
    validation = last(samples).validation
    total_allocations = [
        sample.build_allocated_bytes + sample.solve_allocated_bytes +
        sample.validation_allocated_bytes for sample in samples
    ]
    record = (
        case=config.case,
        frontend=config.frontend,
        arithmetic=config.arithmetic,
        precision_bits=precision(T),
        size=size,
        repetitions=config.repetitions,
        julia_threads=Threads.nthreads(),
        solver_threads=config.threads,
        blas_threads=SDPX.blas_threads(),
        blas_backend=SDPX.blas_backend(),
        hostname=gethostname(),
        compile_seconds=compile_seconds,
        build_seconds_median=median_value(getfield.(samples, :build_seconds)),
        frontend_overhead_seconds_median=median_value(
            getfield.(samples, :frontend_overhead_seconds),
        ),
        core_solver_seconds_median=median_value(
            getfield.(samples, :core_solver_seconds),
        ),
        frontend_solve_seconds_median=median_value(
            getfield.(samples, :frontend_solve_seconds),
        ),
        validation_seconds_median=median_value(
            getfield.(samples, :validation_seconds),
        ),
        end_to_end_seconds_median=median_value(
            getfield.(samples, :end_to_end_seconds),
        ),
        build_allocated_bytes_median=round(
            Int,
            median_value(getfield.(samples, :build_allocated_bytes)),
        ),
        solve_allocated_bytes_median=round(
            Int,
            median_value(getfield.(samples, :solve_allocated_bytes)),
        ),
        total_allocated_bytes_median=round(Int, median_value(total_allocations)),
        process_peak_rss_raw=Sys.maxrss(),
        status=validation.status,
        iterations=validation.iterations,
        objective=validation.objective,
        equality_violation=validation.equality_violation,
        cone_margin=validation.cone_margin,
        validation_gate=validation.validation_gate,
        certificate_valid=validation.certificate_valid,
        source_model_variables=validation.source_model_variables,
        core_model_variables=validation.core_model_variables,
        core_model_equalities=validation.core_model_equalities,
        core_model_psd_blocks=validation.core_model_psd_blocks,
    )
    write_record(record, config.output)
    validation.certificate_valid || error("numerical validation failed")
    return record
end

config = parse_arguments(ARGS)
T = arithmetic_type(config.arithmetic)
SDPX.set_blas_threads!(parse(Int, get(ENV, "SDPX_BLAS_THREADS", "1")))
if T === BigFloat
    setprecision(BigFloat, 256) do
        benchmark(BigFloat, config)
    end
else
    benchmark(T, config)
end
