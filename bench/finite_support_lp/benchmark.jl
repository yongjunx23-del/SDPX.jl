#!/usr/bin/env julia

"""Cluster benchmark for a finite-support nonnegative LP.

The input format is the existing finite-support regression format:

1. `number_of_variables number_of_equalities`
2. objective coefficients
3. equality right-hand side
4. one equality column per remaining line

The source files contain binary64 data printed with round-trip decimal
precision.  Every arithmetic mode therefore parses through `Float64` first and
then converts exactly to the requested type.  This keeps the mathematical data
identical while changing only solver arithmetic.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX

function arithmetic_type(name::AbstractString)
    name == "Float64" && return Float64
    name == "Float64x4" && return Float64x4
    name == "BigFloat256" && return BigFloat
    error("unknown arithmetic '$name'; use Float64, Float64x4, or BigFloat256")
end

convert_source(::Type{T}, token::AbstractString) where {T} = T(parse(Float64, token))

function parse_row(::Type{T}, line::AbstractString, expected::Int) where {T}
    tokens = split(strip(line))
    length(tokens) == expected || error(
        "expected $expected values, found $(length(tokens))",
    )
    return T[convert_source(T, token) for token in tokens]
end

function read_model(::Type{T}, path::AbstractString) where {T}
    lines = readlines(path)
    dimensions = parse.(Int, split(strip(first(lines))))
    length(dimensions) == 2 || error("first line must contain two dimensions")
    variables, equalities = dimensions
    length(lines) == 3 + equalities || error("model line count is inconsistent")
    objective = parse_row(T, lines[2], variables)
    rhs = parse_row(T, lines[3], equalities)
    B = Matrix{T}(undef, variables, equalities)
    for equality in 1:equalities
        B[:, equality] .= parse_row(T, lines[3 + equality], variables)
    end
    return objective, B, rhs
end

function build_problem(::Type{T}, path::AbstractString, direction::Symbol) where {T}
    objective, B, rhs = read_model(T, path)
    direction === :max && (objective .*= -one(T))
    variables = length(objective)
    coefficients = [
        SDPX.CompactScalarCoefficientVector(
            T,
            variables,
            variable,
            one(T),
        )
        for variable in 1:variables
    ]
    constants = [zeros(T, 1, 1) for _ in 1:variables]
    problem = SDPX.ingest(
        objective,
        coefficients,
        constants,
        B,
        rhs;
        sparse=true,
        verbosity=0,
    )
    return problem, objective, B, rhs
end

function independent_validation(result, objective, B, rhs)
    T = eltype(result.x)
    equality_residual = transpose(B) * result.x - rhs
    maximum_normalized = zero(T)
    maximum_absolute = zero(T)
    @inbounds for equality in eachindex(rhs)
        magnitude = abs(rhs[equality])
        for variable in eachindex(result.x)
            magnitude += abs(B[variable, equality]) * abs(result.x[variable])
        end
        residual = abs(equality_residual[equality])
        maximum_absolute = max(maximum_absolute, residual)
        maximum_normalized = max(
            maximum_normalized,
            residual / max(one(T), magnitude),
        )
    end
    nonnegative_violation = maximum(
        value -> max(zero(T), -value),
        result.x;
        init=zero(T),
    )
    objective_recomputed = dot(objective, result.x)
    minimum_scalar_psd = minimum(result.x; init=zero(T))
    return (
        equality_absolute=maximum_absolute,
        equality_normalized=maximum_normalized,
        nonnegative_violation=nonnegative_violation,
        objective_recomputed=objective_recomputed,
        minimum_scalar_psd=minimum_scalar_psd,
    )
end

function print_value(name, value)
    println(name, "=", value)
end

function run_benchmark(::Type{T}, model_path, direction, requested_threads) where {T}
    build_started = time_ns()
    problem, objective, B, rhs = build_problem(T, model_path, direction)
    build_seconds = (time_ns() - build_started) / 1.0e9

    tolerance = if T === Float64
        T(parse(Float64, get(ENV, "SDPX_TOLERANCE", "1e-9")))
    elseif T === BigFloat
        parse(BigFloat, get(ENV, "SDPX_TOLERANCE", "1e-40"))
    else
        T(parse(Float64, get(ENV, "SDPX_TOLERANCE", "1e-30")))
    end
    maximum_iterations = parse(Int, get(ENV, "SDPX_MAX_ITERATIONS", "600"))
    time_limit = parse(Float64, get(ENV, "SDPX_TIME_LIMIT", "1800"))
    options = SDPX.SolverOptions{T}(
        algorithm=:lp,
        parameter_policy=:auto,
        parameter_strategy=:adaptive,
        presolve=true,
        scaling=:equilibrate,
        sparse=:auto,
        threads=requested_threads,
        iter_max=maximum_iterations,
        max_time=time_limit,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        ϵ_gap=tolerance,
        verbosity=0,
        diagnostics=true,
        timing=true,
        precision_bits=T === BigFloat ? 256 : 997,
        working_precision_policy=:fixed,
        convert_inputs=T === BigFloat,
    )

    warmup = get(ENV, "SDPX_WARMUP", "1") == "1"
    if warmup
        SDPX.solve!(problem, options)
    end
    GC.gc()
    result = nothing
    solve_started = time_ns()
    allocated = @allocated result = SDPX.solve!(problem, options)
    solve_seconds = (time_ns() - solve_started) / 1.0e9
    validation_started = time_ns()
    validation = independent_validation(result, objective, B, rhs)
    certificate = SDPX.result_certificate(problem, result, options)
    validation_seconds = (time_ns() - validation_started) / 1.0e9
    diagnostics = hasproperty(result, :diagnostics) ? result.diagnostics : nothing
    selected = diagnostics !== nothing &&
               hasproperty(diagnostics, :selected_algorithms) ?
               diagnostics.selected_algorithms : NamedTuple()
    termination = hasproperty(result, :termination) ? result.termination : NamedTuple()
    executed = hasproperty(termination, :executed) ?
               termination.executed : selected

    print_value("arithmetic", T === BigFloat ? "BigFloat256" : string(T))
    print_value("precision_bits", T === BigFloat ? precision(BigFloat) : 8 * sizeof(T))
    print_value("variables", problem.dims.m)
    print_value("equalities", problem.dims.n)
    print_value("direction", direction)
    print_value("julia_threads_requested", requested_threads)
    print_value("julia_threads_available", Threads.nthreads())
    print_value("blas_backend", SDPX.blas_backend())
    print_value("blas_threads", SDPX.blas_threads())
    print_value("build_seconds", build_seconds)
    print_value("solve_seconds", solve_seconds)
    print_value("validation_seconds", validation_seconds)
    print_value("allocated_bytes", allocated)
    print_value("peak_rss_bytes", Sys.maxrss())
    print_value(
        "workspace_bytes",
        diagnostics !== nothing && hasproperty(diagnostics, :memory) ?
        diagnostics.memory.workspace_bytes : 0,
    )
    print_value(
        "removed_dependent_equalities",
        result.diagnostics.presolve.removed_dependent_equalities,
    )
    print_value("status", result.status)
    print_value("message", replace(result.message, '\n' => ' '))
    print_value("iterations", result.iterations)
    print_value("primal_objective", result.pObj)
    print_value("dual_objective", result.dObj)
    print_value("relative_gap", result.gap_rel)
    print_value("reported_primal_residual", result.p_res)
    print_value("reported_dual_residual", result.d_res)
    print_value("equality_absolute", validation.equality_absolute)
    print_value("equality_normalized", validation.equality_normalized)
    print_value("nonnegative_violation", validation.nonnegative_violation)
    print_value("minimum_scalar_psd", validation.minimum_scalar_psd)
    print_value("objective_recomputed", validation.objective_recomputed)
    print_value("certificate_valid", certificate.valid)
    # `selected_algorithms` is the pre-solve plan.  Dedicated LP dispatch is
    # finalized after presolve/scaling, so the plan may deliberately retain a
    # conservative generic label even when the reduced diagonal route ran.
    # Report both labels; benchmark consumers must use the executed fields for
    # kernel comparisons.
    print_value("planned_kkt", get(selected, :kkt, :unknown))
    print_value("planned_gram", get(selected, :gram, :unknown))
    print_value("planned_effective_threads", get(selected, :effective_threads, nothing))
    print_value("planned_schur_threads", get(selected, :schur_threads, nothing))
    print_value("planned_lp_pack_threads", get(selected, :lp_pack_threads, nothing))
    print_value("planned_factor_threads", get(selected, :factor_threads, nothing))
    for (label, field) in (
        ("executed_kkt", :kkt),
        ("executed_gram", :gram),
        ("effective_threads", :effective_threads),
        ("schur_threads", :schur_threads),
        ("lp_pack_threads", :lp_pack_threads),
        ("factor_threads", :factor_threads),
    )
        hasproperty(executed, field) &&
            print_value(label, getproperty(executed, field))
    end
    print_value(
        "warnings",
        diagnostics !== nothing && hasproperty(diagnostics, :warnings) ?
        join(diagnostics.warnings, " | ") : "",
    )
    for name in (
        :total,
        :lp_core,
        :residual,
        :gram_assembly,
        :kkt_factorization,
        :predictor_corrector,
        :update,
    )
        hasproperty(result.timings, name) &&
            print_value("timing_$(name)", getproperty(result.timings, name))
    end
    return result
end

length(ARGS) == 4 || error(
    "usage: benchmark.jl MODEL Float64|Float64x4|BigFloat256 min|max THREADS",
)
model_path, arithmetic_name, direction_name, thread_text = ARGS
T = arithmetic_type(arithmetic_name)
direction = Symbol(direction_name)
direction in (:min, :max) || error("direction must be min or max")
requested_threads = parse(Int, thread_text)
requested_threads <= Threads.nthreads() || error(
    "requested $requested_threads threads but Julia has $(Threads.nthreads())",
)
blas_threads = parse(Int, get(ENV, "SDPX_BLAS_THREADS", "1"))
SDPX.set_blas_threads!(blas_threads)

if T === BigFloat
    setprecision(BigFloat, 256) do
        run_benchmark(BigFloat, model_path, direction, requested_threads)
    end
else
    run_benchmark(T, model_path, direction, requested_threads)
end
