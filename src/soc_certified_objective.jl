"""Immutable execution decision for the FixedTraceQ3 reduced dual."""
struct ReducedDualExecutionPlan
    algorithm::Symbol
    specialization::Symbol
    la_config::LABackendConfiguration
    threads::Int
    fallback_chain::Tuple
end

struct ReducedDualReconstructionToken{T}
    problem_fingerprint::NTuple{32,UInt8}
    reduction_fingerprint::NTuple{32,UInt8}
    y::Vector{T}
    tau::T
    smoothing_schedule::Vector{T}
    arithmetic::Symbol
    precision_bits::Int
    provider::Symbol
    requested_backend::Symbol
    threads::Int
    ownership::Symbol
end

"""
Answer-only result for a certified FixedTraceQ3 objective.

`lower` and `upper` are a numerical-certificate interval in the requested
arithmetic and tolerance.  `rigorous_interval=false` deliberately avoids
claiming outward-rounded real-number bounds.
"""
struct CertifiedObjective{T}
    status::SolveStatus
    objective::T
    lower::Union{Nothing,T}
    upper::Union{Nothing,T}
    interval_kind::Symbol
    rigorous_interval::Bool
    certificate::NamedTuple
    relative_gap::T
    y::Vector{T}
    arithmetic::Symbol
    precision_bits::Int
    algorithm::Symbol
    specialization::Symbol
    provider::Symbol
    smoothing_history::Vector{ReducedDualLBFGSStage{T}}
    reconstruction_token::Union{Nothing,ReducedDualReconstructionToken{T}}
    timings::NamedTuple
    counters::NamedTuple
    termination::NamedTuple
    polish::Symbol
end

function _convert_conic_problem(::Type{T}, problem::ConicProblem) where {T}
    T <: AbstractFloat || throw(ArgumentError(
        "reduced-dual arithmetic must be an AbstractFloat type",
    ))
    if eltype(problem) === T && T !== BigFloat
        return problem
    end
    cones = SOCConstraint[
        SOCConstraint(T.(cone.A), T.(cone.b); T) for cone in problem.cones
    ]
    return second_order_program(
        T.(problem.c),
        cones;
        Aeq=T.(problem.Aeq),
        beq=T.(problem.beq),
        T,
    )
end

@inline function _reduced_dual_number(::Type{T}, value) where {T}
    value isa AbstractString && return parse(T, value)
    return T(value)
end

function _reduced_dual_schedule(::Type{T}, smoothing, tolerance::T) where {T}
    schedule = if smoothing === nothing || smoothing === :auto
        target = max(tolerance, T(16) * eps(T))
        values = T[]
        value = one(T)
        relative_guard = one(T) + sqrt(eps(T))
        while value > target * relative_guard
            push!(values, value)
            value /= T(10)
        end
        (isempty(values) || values[end] != target) && push!(values, target)
        values
    elseif smoothing isa AbstractVector || smoothing isa Tuple
        T[_reduced_dual_number(T, value) for value in smoothing]
    else
        T[_reduced_dual_number(T, smoothing)]
    end
    !isempty(schedule) || throw(ArgumentError("smoothing schedule is empty"))
    all(value -> isfinite(value) && value > zero(T), schedule) ||
        throw(ArgumentError("smoothing values must be finite and positive"))
    all(schedule[index] <= schedule[index - 1] for index in 2:length(schedule)) ||
        throw(ArgumentError("smoothing schedule must be nonincreasing"))
    return schedule
end

function _plan_reduced_dual(
    ::Type{T}, requested::Symbol, threads::Int,
) where {T}
    threads > 0 || throw(ArgumentError("threads must be positive"))
    effective = requested
    if is_multifloat_arithmetic(T)
        # The reduced-dual contract requires MFLA for MultiFloat.  Do not use
        # plan_la_backend's compatibility fall-through to Standard.
        requested in (:auto, :multifloat) || throw(ArgumentError(
            "MultiFloat reduced-dual execution requires the MFLA provider",
        ))
        effective = :multifloat
    end
    config = plan_la_backend(
        T;
        requested=effective,
        route=:dense_gemv,
        threads,
        equality_solver=:normal_equations,
    )
    isempty(config.fallback_chain) || throw(ArgumentError(
        "reduced-dual dense GEMV plan must not contain a fallback",
    ))
    config.fallback_reason === :none || throw(ArgumentError(
        "reduced-dual provider planning failed: $(config.fallback_reason)",
    ))
    return ReducedDualExecutionPlan(
        :reduced_dual_lbfgs,
        :fixed_trace_q3,
        config,
        threads,
        (),
    )
end

function _reduced_dual_row_scaling(layout::FixedTraceQ3DualLayout{T}) where {T}
    scaling = alloc_zeros(T, size(layout.equality_panel, 1))
    @inbounds for row in axes(layout.equality_panel, 1)
        scale = zero(T)
        for column in axes(layout.equality_panel, 2)
            scale = max(scale, abs(layout.equality_panel[row, column]))
        end
        if iszero(scale)
            scaling[row] = one(T)
            continue
        end
        sum_squares = zero(T)
        for column in axes(layout.equality_panel, 2)
            value = layout.equality_panel[row, column] / scale
            sum_squares += value * value
        end
        scaled_norm = sqrt(sum_squares)
        inverse_norm = (one(T) / scale) / scaled_norm
        scaling[row] = isfinite(inverse_norm) ? min(one(T), inverse_norm) : one(T)
        scaling[row] > zero(T) || throw(ArgumentError(
            "equality row scaling underflowed at row $row",
        ))
    end
    return scaling
end

function _temporary_reduced_dual_result(
    problem::ConicProblem{T}, reconstructed, iterations::Int,
) where {T}
    primal = zero(T)
    @inbounds for index in eachindex(problem.c, reconstructed.x)
        primal += problem.c[index] * reconstructed.x[index]
    end
    dual_objective = zero(T)
    @inbounds for index in eachindex(problem.beq, reconstructed.equality_dual)
        dual_objective += problem.beq[index] * reconstructed.equality_dual[index]
    end
    @inbounds for block in eachindex(problem.cones)
        for coordinate in eachindex(problem.cones[block].b)
            dual_objective -= problem.cones[block].b[coordinate] *
                              reconstructed.dual[block][coordinate]
        end
    end
    scale = max(one(T), (abs(primal) + abs(dual_objective)) / T(2))
    return ConicResult{T}(
        NotStarted,
        "Reduced-dual candidate awaiting original-coordinate certification.",
        reconstructed.x,
        reconstructed.slack,
        reconstructed.dual,
        reconstructed.equality_dual,
        primal,
        dual_objective,
        abs(primal - dual_objective) / scale,
        T(Inf),
        T(Inf),
        iterations,
        nothing,
        nothing,
    )
end

function _reduced_dual_failure_status(reason::Symbol)
    reason === :iteration_limit && return IterLimit
    reason === :time_limit && return TimeLimit
    reason in (:line_search_failed, :non_descent_direction) && return Stalled
    return NumericalBreakdown
end

@inline function _numerical_certificate_interval(certificate)
    certificate.valid || return (nothing, nothing, :unavailable)
    primal = certificate.primal_objective
    dual = certificate.dual_objective
    # This is a finite-arithmetic numerical envelope, not a rigorous primal /
    # dual bound.  A tiny reversed gap is therefore represented honestly by
    # ordering the two certified endpoint values rather than exposing an
    # invalid lower > upper interval.
    orientation = dual <= primal ? :dual_primal : :roundoff_reversed
    return (min(primal, dual), max(primal, dual), orientation)
end

function _certified_from_ipm(
    result::ConicResult{T},
    problem::ConicProblem{T},
    options::SolverOptions{T},
    elapsed::Float64,
) where {T}
    certificate = result_certificate(problem, result, options)
    valid = result.status === Optimal && certificate.valid
    lower, upper, interval_orientation =
        _numerical_certificate_interval(certificate)
    wrapped_status = valid ? Optimal :
        result.status === Optimal ? NumericalFailure : result.status
    wrapped_termination = if result.status === Optimal && !certificate.valid
        merge(
            result.termination,
            (
                reason=:final_certificate_failed,
                previous=get(result.termination, :reason, :unavailable),
                certificate_failures=certificate.failures,
                interval_orientation,
            ),
        )
    else
        merge(result.termination, (; interval_orientation))
    end
    provider = result.diagnostics isa NativeSOCDiagnostics ?
        result.diagnostics.selected_algorithms.la_executed_provider : :unknown
    return CertifiedObjective{T}(
        wrapped_status,
        certificate.primal_objective,
        valid ? lower : nothing,
        valid ? upper : nothing,
        :numerical_certificate,
        false,
        certificate,
        certificate.gap_relative,
        _owned_array_copy(T, result.equality_dual),
        _la_arithmetic_symbol(T),
        T === BigFloat ? Base.precision(BigFloat) : sig_bits(T),
        :primal_dual_ipm,
        :fixed_trace_q3,
        provider,
        ReducedDualLBFGSStage{T}[],
        nothing,
        (total=elapsed, optional_ipm_polish=elapsed,
         strict_spectrum_reconstruction=nothing),
        (ipm_polish_iterations=result.iterations, certificate_calls=1),
        wrapped_termination,
        :native_soc_ipm,
    )
end

function _solve_value_typed(
    problem::ConicProblem{T};
    soc_algorithm::Symbol,
    tolerance::T,
    smoothing,
    polish::Symbol,
    linear_algebra_backend::Symbol,
    threads::Int,
    history_size::Int,
    maximum_iterations::Int,
    max_time::Float64,
    warm_start,
    timing::Bool,
    diagnostics::Bool,
) where {T<:AbstractFloat}
    soc_algorithm in (:primal_dual_ipm, :reduced_dual_lbfgs) ||
        throw(ArgumentError("unknown soc_algorithm $(repr(soc_algorithm))"))
    polish in (:none, :native_soc_ipm) || throw(ArgumentError(
        "polish must be :none or :native_soc_ipm",
    ))
    tolerance > zero(T) && isfinite(tolerance) || throw(ArgumentError(
        "tolerance must be finite and positive",
    ))
    ipm_options = SolverOptions(
        T;
        tolerance,
        maximum_iterations,
        time_limit=max_time,
        linear_algebra_backend,
        threads,
        verbosity=0,
        timing,
        diagnostics,
        certification=true,
        working_precision_policy=:fixed,
    )
    if soc_algorithm === :primal_dual_ipm
        started = time()
        result = solve_socp(
            problem;
            specialization=:fixed_trace,
            tolerance,
            maximum_iterations,
            max_time,
            verbosity=0,
            timing,
            diagnostics,
            linear_algebra_backend,
            threads,
            working_precision_policy=:fixed,
        )
        return _certified_from_ipm(result, problem, ipm_options, time() - started)
    end

    total_started = time()
    compile_timing = Ref((
        fixed_trace_compile=0.0,
        equality_panel_conversion=0.0,
        reduced_dual_setup=0.0,
    ))
    layout = _compile_fixed_trace_q3_dual(problem; timing=compile_timing)
    plan = _plan_reduced_dual(T, linear_algebra_backend, threads)
    backend = instantiate_la_backend(plan.la_config, T, threads)
    la_backend_reason(backend) === :none || throw(ArgumentError(
        "reduced-dual provider instantiated with a fallback reason",
    ))
    workspace = _fixed_trace_dual_workspace(layout)
    schedule = _reduced_dual_schedule(T, smoothing, tolerance)
    initial_y = warm_start === nothing ? alloc_zeros(T, length(problem.beq)) :
                _owned_array_copy(T, warm_start)
    length(initial_y) == length(problem.beq) || throw(DimensionMismatch(
        "warm_start has the wrong reduced dimension",
    ))
    dual_scaling = _reduced_dual_row_scaling(layout)
    initial = alloc_zeros(T, length(initial_y))
    physical_y = alloc_zeros(T, length(initial_y))
    @inbounds for index in eachindex(initial, initial_y, dual_scaling)
        initial[index] = initial_y[index] / dual_scaling[index]
    end
    eval_stats = FixedTraceDualEvalStats()
    fg! = function (gradient, point, tau)
        @inbounds for index in eachindex(point, physical_y, dual_scaling)
            physical_y[index] = point[index] * dual_scaling[index]
        end
        value = _fixed_trace_dual_evaluate!(
            layout,
            backend,
            physical_y,
            tau,
            workspace.u,
            workspace.x,
            gradient,
            workspace.w,
            workspace.rho,
            workspace.wnorm;
            stats=eval_stats,
        )
        @inbounds for index in eachindex(gradient, dual_scaling)
            gradient[index] *= dual_scaling[index]
        end
        return value
    end
    lbfgs_options = ReducedDualLBFGSOptions{T}(
        history_size=history_size,
        gradient_tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        max_time=max_time,
        minimum_step=max(eps(T), tolerance * tolerance),
    )
    optimized = _solve_reduced_dual_lbfgs(
        fg!, initial, schedule, lbfgs_options,
    )
    actual_tau = isempty(optimized.stages) ? schedule[1] :
                 optimized.stages[end].tau
    final_y = alloc_zeros(T, length(optimized.point))
    @inbounds for index in eachindex(final_y, optimized.point, dual_scaling)
        final_y[index] = optimized.point[index] * dual_scaling[index]
    end

    reconstruction_started = time_ns()
    _fixed_trace_dual_evaluate!(
        layout,
        backend,
        final_y,
        actual_tau,
        workspace.u,
        workspace.x,
        workspace.gradient,
        workspace.w,
        workspace.rho,
        workspace.wnorm;
        stats=eval_stats,
    )
    reconstructed = _fixed_trace_dual_reconstruct(
        layout, final_y, workspace.x, workspace.w, workspace.wnorm,
    )
    reconstruction_seconds = (time_ns() - reconstruction_started) / 1.0e9
    candidate = _temporary_reduced_dual_result(
        problem, reconstructed, optimized.iterations,
    )
    certificate_started = time_ns()
    certificate = result_certificate(problem, candidate, ipm_options)
    certificate_seconds = (time_ns() - certificate_started) / 1.0e9

    if polish === :native_soc_ipm
        polish_started = time()
        elapsed_before_polish = polish_started - total_started
        remaining_time = isfinite(max_time) ?
            max(0.0, max_time - elapsed_before_polish) : Inf
        polished = solve_socp(
            problem;
            specialization=:fixed_trace,
            tolerance,
            maximum_iterations,
            max_time=remaining_time,
            verbosity=0,
            timing,
            diagnostics,
            linear_algebra_backend,
            threads,
            working_precision_policy=:fixed,
        )
        polish_seconds = time() - polish_started
        wrapped = _certified_from_ipm(
            polished, problem, ipm_options, time() - polish_started,
        )
        pre_timings = merge(
            compile_timing[],
            optimized.timings,
            (
                equality_transpose_gemv=eval_stats.transpose_gemv_seconds,
                equality_forward_gemv=eval_stats.forward_gemv_seconds,
                block_support=eval_stats.block_support_seconds,
                primal_reconstruction=reconstruction_seconds,
                reduced_dual_certificate=certificate_seconds,
                optional_ipm_polish=polish_seconds,
                strict_spectrum_reconstruction=nothing,
                total=time() - total_started,
                certificate=get(polished.timings, :certification, 0.0),
            ),
        )
        pre_counters = merge(
            optimized.counters,
            (
                number_of_blocks=size(layout.active_ids, 2),
                reduced_dimension=length(layout.equality_rhs),
                panel_rows=size(layout.equality_panel, 1),
                panel_columns=size(layout.equality_panel, 2),
                panel_nnz=count(!iszero, layout.equality_panel),
                transpose_gemv_calls=eval_stats.transpose_gemv_calls,
                forward_gemv_calls=eval_stats.forward_gemv_calls,
                primal_reconstructions=1,
                certificate_calls=2,
                ipm_polish_iterations=polished.iterations,
            ),
        )
        return CertifiedObjective{T}(
            wrapped.status,
            wrapped.objective,
            wrapped.lower,
            wrapped.upper,
            wrapped.interval_kind,
            wrapped.rigorous_interval,
            wrapped.certificate,
            wrapped.relative_gap,
            _owned_array_copy(T, wrapped.y),
            layout.arithmetic,
            layout.precision_bits,
            :reduced_dual_lbfgs,
            :fixed_trace_q3,
            wrapped.provider,
            optimized.stages,
            nothing,
            pre_timings,
            pre_counters,
            (
                reason=wrapped.termination.reason,
                reduced_dual_reason=optimized.reason,
                polish=:native_soc_ipm,
                interval_orientation=get(
                    wrapped.termination, :interval_orientation, :unavailable,
                ),
                no_psd_lift=true,
                fallback_chain=(),
            ),
            :native_soc_ipm,
        )
    end

    certificate_valid = certificate.valid
    certified = optimized.status === :converged && certificate_valid
    status = certified ? Optimal : _reduced_dual_failure_status(optimized.reason)
    if !certificate_valid && optimized.status === :converged
        status = AlmostOptimal
    end
    reason = certified ? :converged_original_certificate :
             optimized.status === :converged ? :reduced_dual_not_certified :
             optimized.reason
    lower, upper, interval_orientation =
        _numerical_certificate_interval(certificate)
    token = ReducedDualReconstructionToken{T}(
        layout.problem_fingerprint,
        layout.reduction_fingerprint,
        _owned_array_copy(T, final_y),
        actual_tau,
        _owned_array_copy(T, schedule),
        layout.arithmetic,
        layout.precision_bits,
        la_backend_provider(backend),
        linear_algebra_backend,
        threads,
        layout.ownership,
    )
    total_seconds = time() - total_started
    timings = merge(
        compile_timing[],
        optimized.timings,
        (
            equality_transpose_gemv=eval_stats.transpose_gemv_seconds,
            equality_forward_gemv=eval_stats.forward_gemv_seconds,
            block_support=eval_stats.block_support_seconds,
            primal_reconstruction=reconstruction_seconds,
            certificate=certificate_seconds,
            optional_ipm_polish=nothing,
            strict_spectrum_reconstruction=nothing,
            total=total_seconds,
        ),
    )
    counters = merge(
        optimized.counters,
        (
            number_of_blocks=size(layout.active_ids, 2),
            reduced_dimension=length(layout.equality_rhs),
            panel_rows=size(layout.equality_panel, 1),
            panel_columns=size(layout.equality_panel, 2),
            panel_nnz=count(!iszero, layout.equality_panel),
            panel_density=isempty(layout.equality_panel) ? 0.0 :
                count(!iszero, layout.equality_panel) / length(layout.equality_panel),
            transpose_gemv_calls=eval_stats.transpose_gemv_calls,
            forward_gemv_calls=eval_stats.forward_gemv_calls,
            mfla_gemv_calls=la_backend_provider(backend) ===
                :multifloat_linear_algebra ?
                eval_stats.transpose_gemv_calls + eval_stats.forward_gemv_calls : 0,
            primal_reconstructions=1,
            certificate_calls=1,
            ipm_polish_iterations=0,
        ),
    )
    return CertifiedObjective{T}(
        status,
        certificate.primal_objective,
        certificate_valid ? lower : nothing,
        certificate_valid ? upper : nothing,
        :numerical_certificate,
        false,
        certificate,
        certificate.gap_relative,
        _owned_array_copy(T, final_y),
        layout.arithmetic,
        layout.precision_bits,
        :reduced_dual_lbfgs,
        :fixed_trace_q3,
        la_backend_provider(backend),
        optimized.stages,
        token,
        timings,
        counters,
        (
            reason,
            optimizer_reason=optimized.reason,
            smoothed=true,
            tau_final=actual_tau,
            certificate_failures=certificate.failures,
            interval_orientation,
            no_psd_lift=true,
            fallback_chain=plan.fallback_chain,
        ),
        polish,
    )
end

function _validate_reconstruction_bigfloat_precision(
    problem::ConicProblem{BigFloat}, bits::Int,
)
    operands = Any[problem.c, problem.beq]
    push!(operands, problem.Aeq isa SparseArrays.SparseMatrixCSC ?
        problem.Aeq.nzval : problem.Aeq)
    for cone in problem.cones
        push!(operands, cone.A isa SparseArrays.SparseMatrixCSC ?
            cone.A.nzval : cone.A)
        push!(operands, cone.b)
    end
    for operand in operands
        isempty(operand) && continue
        _bigfloat_uniform_precision_bits(operand) == bits || throw(ArgumentError(
            "reconstruction problem precision does not match the token",
        ))
    end
    return nothing
end

function _reconstruct_fixed_trace_solution_typed(
    problem::ConicProblem{T}, token::ReducedDualReconstructionToken{T},
) where {T}
    token.arithmetic == _la_arithmetic_symbol(T) || throw(ArgumentError(
        "reconstruction arithmetic does not match the token",
    ))
    token.ownership === :owned || throw(ArgumentError(
        "reconstruction token does not satisfy the owned-data contract",
    ))
    expected_bits = T === BigFloat ? token.precision_bits : sig_bits(T)
    token.precision_bits == expected_bits || throw(ArgumentError(
        "reconstruction precision does not match the arithmetic",
    ))
    T === BigFloat && _validate_reconstruction_bigfloat_precision(
        problem, token.precision_bits,
    )
    layout = _compile_fixed_trace_q3_dual(problem)
    layout.arithmetic == token.arithmetic || throw(ArgumentError(
        "compiled reconstruction arithmetic does not match the token",
    ))
    layout.precision_bits == token.precision_bits || throw(ArgumentError(
        "compiled reconstruction precision does not match the token",
    ))
    layout.problem_fingerprint == token.problem_fingerprint || throw(ArgumentError(
        "problem fingerprint does not match the reconstruction token",
    ))
    layout.reduction_fingerprint == token.reduction_fingerprint ||
        throw(ArgumentError("reduction fingerprint mismatch"))
    plan = _plan_reduced_dual(T, token.requested_backend, token.threads)
    backend = instantiate_la_backend(plan.la_config, T, token.threads)
    la_backend_reason(backend) === :none || throw(ArgumentError(
        "reconstruction provider instantiated with a fallback reason",
    ))
    la_backend_provider(backend) == token.provider || throw(ArgumentError(
        "reconstruction provider does not match the token",
    ))
    workspace = _fixed_trace_dual_workspace(layout)
    _fixed_trace_dual_evaluate!(
        layout, backend, token.y, token.tau, workspace.u, workspace.x,
        workspace.gradient, workspace.w, workspace.rho, workspace.wnorm,
    )
    return _fixed_trace_dual_reconstruct(
        layout, token.y, workspace.x, workspace.w, workspace.wnorm,
    )
end

"""
    solve_value(problem; ...)

Solve a verified FixedTraceQ3 model for an answer-only objective.  The default
algorithm is the explicit reduced dual with typed L-BFGS.  No PSD lift,
formulation switch, provider retry, or implicit IPM fallback is permitted.
"""
function solve_value(
    problem::ConicProblem;
    arithmetic::Type=eltype(problem),
    precision_bits::Union{Nothing,Integer}=nothing,
    soc_algorithm::Symbol=:reduced_dual_lbfgs,
    tolerance=nothing,
    smoothing=:auto,
    polish::Symbol=:none,
    linear_algebra_backend::Symbol=:auto,
    threads::Int=1,
    history_size::Int=10,
    maximum_iterations::Int=200,
    max_time::Real=Inf,
    warm_start=nothing,
    timing::Bool=true,
    diagnostics::Bool=true,
)
    arithmetic <: AbstractFloat || throw(ArgumentError(
        "arithmetic must be an AbstractFloat type",
    ))
    if arithmetic === BigFloat
        precision_bits === nothing && throw(ArgumentError(
            "BigFloat reduced-dual solves require explicit precision_bits",
        ))
        bits = Int(precision_bits)
        bits > 0 || throw(ArgumentError("precision_bits must be positive"))
        return setprecision(BigFloat, bits) do
            converted = _convert_conic_problem(BigFloat, problem)
            tol = tolerance === nothing ? auto_tolerance(BigFloat, bits) :
                  _reduced_dual_number(BigFloat, tolerance)
            _solve_value_typed(
                converted;
                soc_algorithm,
                tolerance=tol,
                smoothing,
                polish,
                linear_algebra_backend,
                threads,
                history_size,
                maximum_iterations,
                max_time=Float64(max_time),
                warm_start,
                timing,
                diagnostics,
            )
        end
    end
    precision_bits === nothing || throw(ArgumentError(
        "precision_bits is only valid with arithmetic=BigFloat",
    ))
    converted = _convert_conic_problem(arithmetic, problem)
    tol = tolerance === nothing ? auto_tolerance(arithmetic, sig_bits(arithmetic)) :
          _reduced_dual_number(arithmetic, tolerance)
    return _solve_value_typed(
        converted;
        soc_algorithm,
        tolerance=tol,
        smoothing,
        polish,
        linear_algebra_backend,
        threads,
        history_size,
        maximum_iterations,
        max_time=Float64(max_time),
        warm_start,
        timing,
        diagnostics,
    )
end

"""Cold reconstruction of generic conic primal/dual coordinates from a token."""
function reconstruct_fixed_trace_solution(
    problem::ConicProblem,
    certified::CertifiedObjective{T},
) where {T}
    token = certified.reconstruction_token
    token === nothing && throw(ArgumentError("result has no reconstruction token"))
    if T === BigFloat
        return setprecision(BigFloat, token.precision_bits) do
            converted = _convert_conic_problem(BigFloat, problem)
            _reconstruct_fixed_trace_solution_typed(converted, token)
        end
    end
    converted = _convert_conic_problem(T, problem)
    return _reconstruct_fixed_trace_solution_typed(converted, token)
end
