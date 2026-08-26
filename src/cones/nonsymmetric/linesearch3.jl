# Three-dimensional nonsymmetric-cone fraction-to-boundary reference.
#
# For a current strict-interior point `p` and direction `d`, the routine searches
# the ray p + alpha*d.  It first derives the largest admissible alpha from the
# coordinates whose positivity is required by the selected cone side.  If that
# upper endpoint is not strictly feasible, a deterministic halving pass produces
# a feasible/infeasible bracket and bisection tightens it to the resolution of the
# scalar type.  The returned step is `safety * alpha_feasible` and is checked once
# more before it is accepted.  Invalid data, an invalid cone parameter, failure to
# bracket, and a bisection limit all return alpha == 0 (fail closed).
#
# The positional API is the allocation-free production-shaped reference:
#
#   nonsymmetric_fraction_to_boundary(tag, p, d, safety, alpha_limit,
#                                     max_backtracks, max_bisections)
#
# where `p` and `d` are `NTuple{3,T}`.  Dispatch on the isbits tag selects Exp
# primal/dual or Power primal/dual membership without a captured closure.

abstract type NonsymmetricStepTag end

struct ExpPrimalStepTag <: NonsymmetricStepTag end
struct ExpDualStepTag <: NonsymmetricStepTag end

struct PowerPrimalStepTag{T} <: NonsymmetricStepTag
    alpha::T
end

struct PowerDualStepTag{T} <: NonsymmetricStepTag
    alpha::T
end

@enum NonsymmetricStepStatus::UInt8 begin
    NS_STEP_ACCEPTED = 0x00
    NS_STEP_FULL_LIMIT = 0x01
    NS_STEP_INVALID_PARAMETER = 0x02
    NS_STEP_NONFINITE_INPUT = 0x03
    NS_STEP_NOT_INTERIOR = 0x04
    NS_STEP_NO_BRACKET = 0x05
    NS_STEP_BISECTION_LIMIT = 0x06
end

struct NonsymmetricStepResult{T}
    status::NonsymmetricStepStatus
    alpha::T
    alpha_feasible::T
    alpha_upper::T
    iterations::Int
end

@inline _ns_step_valid_tag(::ExpPrimalStepTag) = true
@inline _ns_step_valid_tag(::ExpDualStepTag) = true
@inline _ns_step_valid_tag(tag::PowerPrimalStepTag) =
    isfinite(tag.alpha) && zero(tag.alpha) < tag.alpha < one(tag.alpha)
@inline _ns_step_valid_tag(tag::PowerDualStepTag) =
    isfinite(tag.alpha) && zero(tag.alpha) < tag.alpha < one(tag.alpha)

@inline _ns_step_interior(::ExpPrimalStepTag, p::NTuple{3}) =
    exp_primal_interior(p[1], p[2], p[3])
@inline _ns_step_interior(::ExpDualStepTag, p::NTuple{3}) =
    exp_dual_interior(p[1], p[2], p[3])
@inline _ns_step_interior(tag::PowerPrimalStepTag, p::NTuple{3}) =
    power_primal_interior(p[1], p[2], p[3], tag.alpha)
@inline _ns_step_interior(tag::PowerDualStepTag, p::NTuple{3}) =
    power_dual_interior(p[1], p[2], p[3], tag.alpha)

@inline function _ns_step_finite(p::NTuple{3})
    return isfinite(p[1]) && isfinite(p[2]) && isfinite(p[3])
end

@inline function _ns_step_trial_interior(
    tag::NonsymmetricStepTag,
    p::NTuple{3,T},
    d::NTuple{3,T},
    alpha::T,
) where {T}
    trial = (
        p[1] + alpha * d[1],
        p[2] + alpha * d[2],
        p[3] + alpha * d[3],
    )
    return _ns_step_finite(trial) && _ns_step_interior(tag, trial)
end

@inline function _ns_positive_upper(value::T, direction::T, upper::T) where {T}
    if direction < zero(T)
        candidate = -value / direction
        return candidate < upper ? candidate : upper
    end
    return upper
end

@inline function _ns_step_positive_upper(
    ::ExpPrimalStepTag,
    p::NTuple{3,T},
    d::NTuple{3,T},
    upper::T,
) where {T}
    upper = _ns_positive_upper(p[2], d[2], upper)
    return _ns_positive_upper(p[3], d[3], upper)
end


@inline function _ns_step_positive_upper(
    ::ExpDualStepTag,
    p::NTuple{3,T},
    d::NTuple{3,T},
    upper::T,
) where {T}
    upper = _ns_positive_upper(-p[1], -d[1], upper)
    return _ns_positive_upper(p[3], d[3], upper)
end


@inline function _ns_step_positive_upper(
    ::PowerPrimalStepTag,
    p::NTuple{3,T},
    d::NTuple{3,T},
    upper::T,
) where {T}
    upper = _ns_positive_upper(p[1], d[1], upper)
    return _ns_positive_upper(p[2], d[2], upper)
end


@inline function _ns_step_positive_upper(
    ::PowerDualStepTag,
    p::NTuple{3,T},
    d::NTuple{3,T},
    upper::T,
) where {T}
    upper = _ns_positive_upper(p[1], d[1], upper)
    return _ns_positive_upper(p[2], d[2], upper)
end

@inline function _ns_step_failure(
    ::Type{T},
    status::NonsymmetricStepStatus,
    upper::T,
    iterations::Int,
) where {T}
    return NonsymmetricStepResult{T}(status, zero(T), zero(T), upper, iterations)
end

@inline function _ns_step_tolerance(lo::T, hi::T) where {T}
    scale = max(one(T), abs(lo), abs(hi))
    return (one(T) + one(T))^3 * eps(one(T)) * scale
end

function nonsymmetric_fraction_to_boundary(
    tag::NonsymmetricStepTag,
    point::NTuple{3,T},
    direction::NTuple{3,T},
    safety::T,
    alpha_limit::T,
    max_backtracks::Int,
    max_bisections::Int,
) where {T<:AbstractFloat}
    zero_t = zero(T)
    if !_ns_step_valid_tag(tag)
        return _ns_step_failure(T, NS_STEP_INVALID_PARAMETER, zero_t, 0)
    end
    if !_ns_step_finite(point) || !_ns_step_finite(direction) ||
       !isfinite(safety) || !isfinite(alpha_limit)
        return _ns_step_failure(T, NS_STEP_NONFINITE_INPUT, zero_t, 0)
    end
    if !(zero_t < safety < one(T)) || alpha_limit <= zero_t ||
       max_backtracks <= 0 || max_bisections <= 0
        return _ns_step_failure(T, NS_STEP_INVALID_PARAMETER, zero_t, 0)
    end
    if !_ns_step_interior(tag, point)
        return _ns_step_failure(T, NS_STEP_NOT_INTERIOR, zero_t, 0)
    end

    upper = _ns_step_positive_upper(tag, point, direction, alpha_limit)
    if !isfinite(upper) || upper <= zero_t
        return _ns_step_failure(T, NS_STEP_NO_BRACKET, upper, 0)
    end

    if _ns_step_trial_interior(tag, point, direction, upper)
        accepted = safety * upper
        if !_ns_step_trial_interior(tag, point, direction, accepted)
            return _ns_step_failure(T, NS_STEP_NO_BRACKET, upper, 0)
        end
        return NonsymmetricStepResult{T}(
            NS_STEP_FULL_LIMIT,
            accepted,
            upper,
            upper,
            0,
        )
    end

    infeasible = upper
    feasible = zero_t
    candidate = upper / (one(T) + one(T))
    bracket_iterations = 0
    bracketed = false
    while bracket_iterations < max_backtracks
        bracket_iterations += 1
        if candidate <= zero_t || candidate == infeasible
            break
        end
        if _ns_step_trial_interior(tag, point, direction, candidate)
            feasible = candidate
            bracketed = true
            break
        end
        infeasible = candidate
        candidate /= one(T) + one(T)
    end
    if !bracketed
        return _ns_step_failure(T, NS_STEP_NO_BRACKET, upper, bracket_iterations)
    end

    bisection_iterations = 0
    converged = false
    while bisection_iterations < max_bisections
        if infeasible - feasible <= _ns_step_tolerance(feasible, infeasible)
            converged = true
            break
        end
        midpoint = (feasible + infeasible) / (one(T) + one(T))
        if midpoint == feasible || midpoint == infeasible
            converged = true
            break
        end
        bisection_iterations += 1
        if _ns_step_trial_interior(tag, point, direction, midpoint)
            feasible = midpoint
        else
            infeasible = midpoint
        end
    end
    total_iterations = bracket_iterations + bisection_iterations
    if !converged
        return _ns_step_failure(T, NS_STEP_BISECTION_LIMIT, upper, total_iterations)
    end

    accepted = safety * feasible
    if !isfinite(accepted) || accepted <= zero_t ||
       !_ns_step_trial_interior(tag, point, direction, accepted)
        return _ns_step_failure(T, NS_STEP_NO_BRACKET, upper, total_iterations)
    end
    return NonsymmetricStepResult{T}(
        NS_STEP_ACCEPTED,
        accepted,
        feasible,
        upper,
        total_iterations,
    )
end

function nonsymmetric_fraction_to_boundary(
    tag::NonsymmetricStepTag,
    point::NTuple{3,T},
    direction::NTuple{3,T};
    safety::T = T(0.995),
    alpha_limit::T = one(T),
    max_backtracks::Int = 256,
    max_bisections::Int = 1024,
) where {T<:AbstractFloat}
    return nonsymmetric_fraction_to_boundary(
        tag,
        point,
        direction,
        safety,
        alpha_limit,
        max_backtracks,
        max_bisections,
    )
end
