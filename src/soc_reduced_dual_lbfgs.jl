"""Configuration for SDPX's small typed L-BFGS implementation."""
Base.@kwdef struct ReducedDualLBFGSOptions{T}
    history_size::Int = 10
    gradient_tolerance::T = sqrt(eps(T))
    maximum_iterations::Int = 200
    max_time::Float64 = Inf
    armijo::T = T(1) / T(10_000)
    backtrack::T = T(1) / T(2)
    minimum_step::T = eps(T)
    initial_step::T = one(T)
end

struct ReducedDualLBFGSStage{T}
    tau::T
    iterations::Int
    objective::T
    gradient_norm::T
    reason::Symbol
end

struct ReducedDualLBFGSResult{T}
    point::Vector{T}
    objective::T
    gradient::Vector{T}
    gradient_norm::T
    status::Symbol
    reason::Symbol
    iterations::Int
    stages::Vector{ReducedDualLBFGSStage{T}}
    timings::NamedTuple
    counters::NamedTuple
end

@inline function _lbfgs_dot(left, right)
    value = zero(promote_type(eltype(left), eltype(right)))
    @inbounds for index in eachindex(left, right)
        value += left[index] * right[index]
    end
    return value
end

@inline function _lbfgs_norminf(vector)
    isempty(vector) && return zero(eltype(vector))
    value = zero(eltype(vector))
    @inbounds for entry in vector
        value = max(value, abs(entry))
    end
    return value
end

function _lbfgs_copy_column!(destination, source, column)
    @inbounds for row in axes(destination, 1)
        destination[row, column] = _ingest_owned_scalar(
            eltype(destination), source[row],
        )
    end
    return destination
end

@inline function _lbfgs_dot_column(matrix, column, vector)
    value = zero(promote_type(eltype(matrix), eltype(vector)))
    @inbounds for row in axes(matrix, 1)
        value += matrix[row, column] * vector[row]
    end
    return value
end

@inline function _lbfgs_dot_columns(left, left_column, right, right_column)
    value = zero(promote_type(eltype(left), eltype(right)))
    @inbounds for row in axes(left, 1)
        value += left[row, left_column] * right[row, right_column]
    end
    return value
end

function _lbfgs_direction!(
    direction::AbstractVector{T},
    gradient::AbstractVector{T},
    s_history::AbstractMatrix{T},
    y_history::AbstractMatrix{T},
    inverse_curvature::AbstractVector{T},
    alpha::AbstractVector{T},
    head::Int,
    count::Int,
) where {T}
    copy_owned!(direction, gradient)
    count == 0 && begin
        @inbounds for index in eachindex(direction)
            direction[index] = -direction[index]
        end
        return direction
    end
    capacity = size(s_history, 2)
    oldest = mod1(head - count + 1, capacity)
    @inbounds for position in count:-1:1
        column = mod1(oldest + position - 1, capacity)
        coefficient = inverse_curvature[column] *
                      _lbfgs_dot_column(s_history, column, direction)
        alpha[position] = coefficient
        for row in eachindex(direction)
            direction[row] -= coefficient * y_history[row, column]
        end
    end
    newest = head
    sy = _lbfgs_dot_columns(s_history, newest, y_history, newest)
    yy = _lbfgs_dot_columns(y_history, newest, y_history, newest)
    scaling = yy > zero(T) ? sy / yy : one(T)
    @inbounds for row in eachindex(direction)
        direction[row] *= scaling
    end
    @inbounds for position in 1:count
        column = mod1(oldest + position - 1, capacity)
        beta = inverse_curvature[column] *
               _lbfgs_dot_column(y_history, column, direction)
        coefficient = alpha[position] - beta
        for row in eachindex(direction)
            direction[row] += coefficient * s_history[row, column]
        end
    end
    @inbounds for row in eachindex(direction)
        direction[row] = -direction[row]
    end
    return direction
end

"""
    _solve_reduced_dual_lbfgs(fg!, initial, schedule, options)

Minimize a sequence of smooth objectives. `fg!(gradient, point, tau)` must
overwrite `gradient` and return the objective in the same arithmetic `T`.
No Hessian, Hessian-vector product, Gram matrix, or CG state is formed.
"""
function _solve_reduced_dual_lbfgs(
    fg!,
    initial::AbstractVector{T},
    schedule::AbstractVector{T},
    options::ReducedDualLBFGSOptions{T}=ReducedDualLBFGSOptions{T}(),
) where {T<:AbstractFloat}
    !isempty(schedule) || throw(ArgumentError("smoothing schedule is empty"))
    all(tau -> isfinite(tau) && tau > zero(T), schedule) ||
        throw(ArgumentError("every smoothing value must be finite and positive"))
    options.history_size > 0 || throw(ArgumentError("history_size must be positive"))
    options.maximum_iterations > 0 || throw(ArgumentError(
        "maximum_iterations must be positive",
    ))
    zero(T) < options.armijo < one(T) || throw(ArgumentError(
        "Armijo constant must lie in (0,1)",
    ))
    zero(T) < options.backtrack < one(T) || throw(ArgumentError(
        "backtrack factor must lie in (0,1)",
    ))

    dimension = length(initial)
    point = _owned_array_copy(T, initial)
    gradient = alloc_zeros(T, dimension)
    trial_point = alloc_zeros(T, dimension)
    trial_gradient = alloc_zeros(T, dimension)
    direction = alloc_zeros(T, dimension)
    step_delta = alloc_zeros(T, dimension)
    gradient_delta = alloc_zeros(T, dimension)
    s_history = alloc_zeros(T, dimension, options.history_size)
    y_history = alloc_zeros(T, dimension, options.history_size)
    inverse_curvature = alloc_zeros(T, options.history_size)
    alpha = alloc_zeros(T, options.history_size)

    stages = ReducedDualLBFGSStage{T}[]
    total_iterations = 0
    objective_evaluations = 0
    gradient_evaluations = 0
    line_search_trials = 0
    accepted_steps = 0
    rejected_steps = 0
    history_resets = 0
    direction_seconds = 0.0
    line_search_seconds = 0.0
    history_seconds = 0.0
    continuation_started = time_ns()
    wall_started = time()
    objective = T(Inf)
    gradient_norm = T(Inf)
    final_status = :not_started
    final_reason = :none

    for (stage_index, tau) in pairs(schedule)
        time() - wall_started >= options.max_time && begin
            final_status = :time_limit
            final_reason = :time_limit
            break
        end
        head = 0
        count = 0
        zero_owned!(s_history)
        zero_owned!(y_history)
        zero_owned!(inverse_curvature)
        objective = fg!(gradient, point, tau)
        objective_evaluations += 1
        gradient_evaluations += 1
        if !isfinite(objective)
            final_status = :failed
            final_reason = :nonfinite_objective
            push!(stages, ReducedDualLBFGSStage(
                tau, 0, objective, T(Inf), final_reason,
            ))
            break
        elseif !all(isfinite, gradient)
            final_status = :failed
            final_reason = :nonfinite_gradient
            push!(stages, ReducedDualLBFGSStage(
                tau, 0, objective, T(Inf), final_reason,
            ))
            break
        end
        gradient_norm = _lbfgs_norminf(gradient)
        # Intermediate continuation stages only need to enter the basin of
        # the next, less-smoothed objective. Requiring the final certificate
        # tolerance at every tau can spend the whole iteration budget solving
        # an approximation that will immediately be replaced.
        stage_tolerance = stage_index == length(schedule) ?
            options.gradient_tolerance :
            max(options.gradient_tolerance, sqrt(tau))
        stage_iterations = 0
        stage_reason = :none
        while gradient_norm > stage_tolerance
            total_iterations >= options.maximum_iterations && begin
                stage_reason = :iteration_limit
                final_status = :iteration_limit
                final_reason = stage_reason
                break
            end
            time() - wall_started >= options.max_time && begin
                stage_reason = :time_limit
                final_status = :time_limit
                final_reason = stage_reason
                break
            end
            direction_started = time_ns()
            _lbfgs_direction!(
                direction, gradient, s_history, y_history,
                inverse_curvature, alpha, head, count,
            )
            slope = _lbfgs_dot(gradient, direction)
            if !(isfinite(slope) && slope < zero(T))
                count = 0
                history_resets += 1
                @inbounds for index in eachindex(direction, gradient)
                    direction[index] = -gradient[index]
                end
                slope = -_lbfgs_dot(gradient, gradient)
            end
            direction_seconds += (time_ns() - direction_started) / 1.0e9
            if !(isfinite(slope) && slope < zero(T))
                stage_reason = :non_descent_direction
                final_status = :failed
                final_reason = stage_reason
                break
            end

            search_started = time_ns()
            step = options.initial_step
            accepted = false
            trial_objective = objective
            while step >= options.minimum_step
                @inbounds for index in eachindex(point)
                    trial_point[index] = point[index] + step * direction[index]
                end
                trial_objective = fg!(trial_gradient, trial_point, tau)
                objective_evaluations += 1
                gradient_evaluations += 1
                line_search_trials += 1
                if isfinite(trial_objective) && all(isfinite, trial_gradient) &&
                   trial_objective <= objective + options.armijo * step * slope
                    accepted = true
                    break
                end
                step *= options.backtrack
                rejected_steps += 1
            end
            line_search_seconds += (time_ns() - search_started) / 1.0e9
            if !accepted
                stage_reason = :line_search_failed
                final_status = :failed
                final_reason = stage_reason
                break
            end

            update_started = time_ns()
            @inbounds for index in eachindex(point)
                step_delta[index] = trial_point[index] - point[index]
                gradient_delta[index] = trial_gradient[index] - gradient[index]
            end
            sy = _lbfgs_dot(step_delta, gradient_delta)
            ss = _lbfgs_dot(step_delta, step_delta)
            yy = _lbfgs_dot(gradient_delta, gradient_delta)
            threshold = sqrt(eps(T)) * sqrt(max(ss * yy, zero(T)))
            if isfinite(sy) && sy > threshold
                head = mod1(head + 1, options.history_size)
                _lbfgs_copy_column!(s_history, step_delta, head)
                _lbfgs_copy_column!(y_history, gradient_delta, head)
                inverse_curvature[head] = one(T) / sy
                count = min(count + 1, options.history_size)
            else
                count = 0
                history_resets += 1
            end
            copy_owned!(point, trial_point)
            copy_owned!(gradient, trial_gradient)
            objective = trial_objective
            gradient_norm = _lbfgs_norminf(gradient)
            accepted_steps += 1
            total_iterations += 1
            stage_iterations += 1
            history_seconds += (time_ns() - update_started) / 1.0e9
        end
        if stage_reason === :none
            stage_reason = gradient_norm <= stage_tolerance ?
                           :converged : :smoothing_limit_reached
        end
        push!(stages, ReducedDualLBFGSStage(
            tau, stage_iterations, objective, gradient_norm, stage_reason,
        ))
        stage_reason in (:iteration_limit, :time_limit, :line_search_failed,
                         :non_descent_direction) && break
        final_status = :converged
        final_reason = :converged
    end

    continuation_seconds = (time_ns() - continuation_started) / 1.0e9
    final_status === :not_started && begin
        final_status = :failed
        final_reason = :not_started
    end
    return ReducedDualLBFGSResult{T}(
        _owned_array_copy(T, point),
        objective,
        _owned_array_copy(T, gradient),
        gradient_norm,
        final_status,
        final_reason,
        total_iterations,
        stages,
        (
            continuation=continuation_seconds,
            lbfgs_direction=direction_seconds,
            line_search=line_search_seconds,
            lbfgs_history_update=history_seconds,
        ),
        (
            lbfgs_iterations=total_iterations,
            objective_evaluations,
            gradient_evaluations,
            line_search_trials,
            accepted_steps,
            rejected_steps,
            history_resets,
            continuation_stages=length(stages),
        ),
    )
end
