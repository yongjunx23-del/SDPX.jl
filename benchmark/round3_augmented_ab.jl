#!/usr/bin/env julia

# Lightweight Mac-first Round 3 formulation comparison.  The script reuses
# two Round 2 registry controls and adds six deterministic dense SDP stresses.
# It never downloads data and it refuses route/provider/fallback drift.

using SDPX
using LinearAlgebra
using Printf
using TOML
import BigFloatLinearAlgebra
import MultiFloatLinearAlgebra
import MultiFloats

include(joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry

const CASES = (
    (id="registry/sdp_dense", kind=:registry, parameter=nothing),
    (id="registry/sdp_equality_heavy", kind=:registry, parameter=nothing),
    (id="equality_condition_1e2", kind=:near_equality, parameter="1e-2"),
    (id="equality_condition_1e4", kind=:near_equality, parameter="1e-4"),
    (id="equality_condition_1e8", kind=:near_equality, parameter="1e-8"),
    (id="equality_condition_1e12", kind=:near_equality, parameter="1e-12"),
    (id="bad_scale_1e8", kind=:equality_scale, parameter="1e8"),
    (id="schur_scale_1e8", kind=:schur_scale, parameter="1e8"),
)

_typed(::Type{BigFloat}, text::AbstractString) = parse(BigFloat, text)
_typed(::Type{T}, text::AbstractString) where {T} = T(parse(BigFloat, text))

function _owned_zeros(::Type{T}, dimensions...) where {T}
    return SDPX.alloc_zeros(T, dimensions...)
end

function _dense_equality_problem(
    ::Type{T};
    mode::Symbol,
    parameter::AbstractString,
    variables::Int=6,
) where {T}
    coefficients = _owned_zeros(T, variables, variables, variables)
    offset = _owned_zeros(T, variables, variables)
    objective = _owned_zeros(T, variables)
    base = _owned_zeros(T, variables, variables - 1)
    rhs_base = _owned_zeros(T, variables - 1)
    transformation = _owned_zeros(T, variables - 1, variables - 1)
    scales = _owned_zeros(T, variables)

    @inbounds for index in 1:variables
        scales[index] = one(T)
    end
    if mode === :schur_scale
        dynamic_range = _typed(T, parameter)
        @inbounds for index in 1:variables
            exponent = T(index - 1) / T(variables - 1)
            scales[index] = dynamic_range^exponent
        end
    end

    @inbounds for index in 1:variables
        coefficients[index, index, index] = scales[index]
        offset[index, index] = one(T)
        objective[index] = scales[index]
    end
    @inbounds for equality in 1:(variables - 1)
        base[equality, equality] = scales[equality]
        base[equality + 1, equality] = scales[equality + 1]
        rhs_base[equality] = T(2)
        transformation[equality, equality] = one(T)
    end

    if mode === :near_equality
        delta = _typed(T, parameter)
        transformation[1, end] = one(T)
        transformation[end, end] = delta
    elseif mode === :equality_scale
        dynamic_range = _typed(T, parameter)
        @inbounds for equality in 1:(variables - 1)
            exponent = T(equality - 1) / T(variables - 2)
            transformation[equality, equality] = dynamic_range^exponent
        end
    elseif mode !== :schur_scale
        throw(ArgumentError("unknown Round 3 stress mode $mode"))
    end

    equality_matrix = _owned_zeros(T, variables, variables - 1)
    equality_rhs = _owned_zeros(T, variables - 1)
    @inbounds for column in axes(equality_matrix, 2)
        for row in axes(equality_matrix, 1)
            value = zero(T)
            for index in axes(base, 2)
                value += base[row, index] * transformation[index, column]
            end
            equality_matrix[row, column] = value
        end
        value = zero(T)
        for index in eachindex(rhs_base)
            value += transformation[index, column] * rhs_base[index]
        end
        equality_rhs[column] = value
    end

    return SDPX.ingest(
        objective,
        [coefficients],
        [offset],
        equality_matrix,
        equality_rhs;
        sparse=false,
        verbosity=0,
    )
end

function _problem(case, ::Type{T}) where {T}
    if case.id == "registry/sdp_dense"
        return build_problem(
            benchmark_spec("synthetic/sdp_dense"),
            T,
        ).problem
    elseif case.id == "registry/sdp_equality_heavy"
        return build_problem(
            benchmark_spec("synthetic/sdp_equality_heavy"),
            T,
        ).problem
    end
    return _dense_equality_problem(
        T;
        mode=case.kind,
        parameter=case.parameter,
    )
end

function _options(
    ::Type{T},
    formulation::Symbol,
    provider::Symbol,
) where {T}
    tolerance = T === BigFloat ? parse(BigFloat, "1e-30") : T(1e-10)
    return SDPX.SolverOptions{T}(
        algorithm=:sdp,
        presolve=false,
        scaling=:none,
        sparse=false,
        formulation=formulation,
        equality_solver=:normal_equations,
        linear_algebra_backend=provider,
        extended_precision_blas=:off,
        mixed_precision_kkt=:off,
        parameter_strategy=:adaptive,
        threads=1,
        verbosity=0,
        diagnostics=true,
        certification=true,
        timing=true,
        iter_max=150,
        working_precision_policy=:fixed,
        precision_bits=T === BigFloat ? 256 : SDPX.sig_bits(T),
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
    )
end

function _string_metric(value)
    value === nothing && return ""
    return string(value)
end

function _run_one(case, ::Type{T}, provider::Symbol, formulation::Symbol) where {T}
    problem = _problem(case, T)
    options = _options(T, formulation, provider)
    plan = SDPX.build_execution_plan(problem, options)
    expected_formulation = formulation === :augmented ?
                           :dense_augmented_kkt : :dense_normal_equations
    plan.kkt_formulation === expected_formulation || error(
        "$(case.id): planned $(plan.kkt_formulation), expected $expected_formulation",
    )
    plan.la_config.selected === provider || error(
        "$(case.id): planned provider $(plan.la_config.selected), expected $provider",
    )
    isempty(plan.la_config.fallback_chain) || error(
        "$(case.id): formulation A/B must have an empty fallback chain",
    )

    # Keep compilation/provider initialization out of the recorded Mac
    # comparison. Both variants still construct a fresh numerical Workspace
    # for the measured solve; only method compilation is warmed.
    warmup = SDPX.solve!(problem, options)
    warmup.status === SDPX.Optimal || error(
        "$(case.id) $formulation warmup stopped with $(warmup.status)",
    )
    elapsed = @elapsed result = SDPX.solve!(problem, options)
    selected = result.diagnostics.selected_algorithms
    executed_formulation = selected.executed_kkt_formulation
    executed_provider = selected.la_backend
    expected_factor = formulation === :augmented ?
                      :pivoted_symmetric_ldlt : :cholesky
    executed_formulation === expected_formulation || error(
        "$(case.id): executed formulation drifted to $executed_formulation",
    )
    executed_provider === provider || error(
        "$(case.id): executed provider drifted to $executed_provider",
    )
    selected.executed_factorization === expected_factor || error(
        "$(case.id): executed factorization drifted to $(selected.executed_factorization)",
    )
    selected.la_fallback_reason === :none || error(
        "$(case.id): unexpected LA fallback $(selected.la_fallback_reason)",
    )

    certificate = selected.certificate
    certificate_valid = get(certificate, :valid, false)
    result.status === SDPX.Optimal || error(
        "$(case.id) $formulation stopped with $(result.status)",
    )
    certificate_valid || error(
        "$(case.id) $formulation failed original-coordinate certification",
    )

    augmented = get(result.termination, :augmented_kkt, (available=false,))
    trace = SDPX.performance_trace(result)
    iteration_trace = trace.iteration
    return (
        case_id=case.id,
        arithmetic=string(T),
        provider=String(provider),
        formulation=String(formulation),
        status=string(result.status),
        objective=_string_metric(result.pObj),
        primal_residual=_string_metric(result.p_res),
        dual_residual=_string_metric(result.d_res),
        relative_gap=_string_metric(result.gap_rel),
        certificate_valid,
        iterations=result.iterations,
        regularizations=result.regularizations,
        elapsed_seconds=elapsed,
        recorded_total_seconds=_string_metric(trace.final.total_seconds),
        schur_assembly_seconds=
            _string_metric(iteration_trace.schur_assembly_seconds),
        kkt_factorization_seconds=
            _string_metric(iteration_trace.kkt_factorization_seconds),
        kkt_numeric_factorization_seconds=
            _string_metric(iteration_trace.kkt_numeric_factorization_seconds),
        predictor_solve_seconds=
            _string_metric(iteration_trace.predictor_solve_seconds),
        corrector_solve_seconds=
            _string_metric(iteration_trace.corrector_solve_seconds),
        refinement_seconds=
            _string_metric(iteration_trace.refinement_seconds),
        planned_formulation=String(plan.kkt_formulation),
        executed_formulation=String(executed_formulation),
        executed_factorization=String(selected.executed_factorization),
        executed_provider=String(selected.la_executed_provider),
        fallback_reason=String(selected.la_fallback_reason),
        augmented_dimension=get(augmented, :dimension, ""),
        inertia=_string_metric(get(augmented, :inertia, nothing)),
        factor_diagnostics=_string_metric(
            get(augmented, :factor_diagnostics, nothing),
        ),
    )
end

function _print_rows(rows)
    println(join((
        "case_id", "arithmetic", "provider", "formulation", "status",
        "objective", "primal_residual", "dual_residual", "relative_gap",
        "certificate_valid", "iterations", "regularizations",
        "elapsed_seconds", "recorded_total_seconds",
        "schur_assembly_seconds", "kkt_factorization_seconds",
        "kkt_numeric_factorization_seconds", "predictor_solve_seconds",
        "corrector_solve_seconds", "refinement_seconds",
        "executed_factorization", "inertia",
    ), '\t'))
    for row in rows
        println(join((
            row.case_id,
            row.arithmetic,
            row.provider,
            row.formulation,
            row.status,
            row.objective,
            row.primal_residual,
            row.dual_residual,
            row.relative_gap,
            row.certificate_valid,
            row.iterations,
            row.regularizations,
            @sprintf("%.6f", row.elapsed_seconds),
            row.recorded_total_seconds,
            row.schur_assembly_seconds,
            row.kkt_factorization_seconds,
            row.kkt_numeric_factorization_seconds,
            row.predictor_solve_seconds,
            row.corrector_solve_seconds,
            row.refinement_seconds,
            row.executed_factorization,
            row.inertia,
        ), '\t'))
    end
end

function _write_toml(path::AbstractString, rows)
    payload = Dict(
        "schema_version" => 1,
        "source_commit" => try
            readchomp(`git -C $(normpath(joinpath(@__DIR__, ".."))) rev-parse HEAD`)
        catch
            "unknown"
        end,
        "source_dirty" => try
            !isempty(readchomp(`git -C $(normpath(joinpath(@__DIR__, ".."))) status --short`))
        catch
            true
        end,
        "rows" => [Dict(string(key) => value for (key, value) in pairs(row))
                   for row in rows],
    )
    open(path, "w") do io
        TOML.print(io, payload)
    end
end

function main(args=ARGS)
    extended = "--extended" in args
    rows = NamedTuple[]
    for case in CASES, formulation in (:auto, :augmented)
        push!(rows, _run_one(case, MultiFloats.Float64x3, :multifloat, formulation))
    end
    if extended
        for case in CASES[1:2], formulation in (:auto, :augmented)
            push!(rows, _run_one(case, MultiFloats.Float64x4, :multifloat, formulation))
        end
        setprecision(BigFloat, 256) do
            for case in CASES[1:2], formulation in (:auto, :augmented)
                push!(rows, _run_one(case, BigFloat, :bfla, formulation))
            end
        end
    end
    _print_rows(rows)
    output = get(ENV, "SDPX_ROUND3_OUT", "")
    isempty(output) || _write_toml(output, rows)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
