#=====================================================================
    Automatic stagnation detection

    Replaces a fixed per-iteration `stall_tolerance` bar with a rolling
    window over all four convergence metrics, normalised by the
    tolerances actually requested, and a stop rule derived from the
    measured convergence rate rather than a hand-set constant.
=====================================================================#

"""Metrics below this multiple of `eps(T)` are at the precision floor: no
arithmetic at that width can push them lower, so a plateau there is exhaustion
rather than stagnation."""
const PRECISION_FLOOR_ULPS = 16

"""A window whose measured rate projects more than this multiple of the
remaining iteration budget is treated as not going to converge.

Deliberately well above 1. At 1 the rule stops whenever the projection merely
exceeds the budget, which is a *marginal* call made from a noisy local rate —
and it misfires badly: on the CSDR sparse model it ended the solve at iteration
34 with a projection of ~373 against 366 remaining, at gap 7.1e-3, when letting
it run reaches 8.4e-6. A large factor keeps the rule to what it is actually good
for — recognising a hopeless rate — and leaves `iter_max`/`max_time` to enforce
the budget, which is their job."""
const PROJECTION_SLACK = 10.0

"""
    StagnationDetector{T}

Rolling-window stagnation detector.

Tracks the four convergence metrics — primal residual, dual residual, relative
duality gap, and complementarity — each divided by the tolerance requested for
it, so that the combined `merit` is dimensionless and `merit <= 1` means
converged. Progress is then measured in nats of `log(merit)` per iteration over
a window, and the run is declared stalled when that rate cannot reach the
target.

Why not a fixed per-iteration bar: the previous rule required a
`stall_tolerance` relative improvement on *every* iteration and stopped after
`window` consecutive misses. That is a false positive on any slowly-but-steadily
converging solve — the CSDR sparse model improves ~0.05% per iteration near the
end, real progress that needs ~70 iterations in total (Clarabel takes exactly
that on the same problem), but every individual step missed a 0.1% bar, so the
solve was killed at iteration 27 with the gap still at 9e-4.

Scaling is automatic in both senses asked of it:

* **with the requested tolerance** — each metric is normalised by its own `ϵ`,
  so loosening a tolerance genuinely shortens the run instead of only changing
  the final check, and a metric already inside its tolerance stops dominating.
* **with the working precision** — a metric within `PRECISION_FLOOR_ULPS·eps(T)`
  cannot improve further at that width, so a plateau there is reported as
  `:precision_floor` (actionable: widen `T`) rather than `:no_progress`.
"""
mutable struct StagnationDetector{T}
    window::Int
    tolerance::T
    merits::Vector{T}
    at_floor::Vector{Bool}
    reason::Symbol
    rate::T
    projected::T
end

"""
    StagnationDetector{T}(window, tolerance)

`tolerance` is the minimum *cumulative* relative improvement required across the
whole window — the reinterpreted `SolverOptions.stall_tolerance`. Its old
meaning (required on every single iteration) is what made it a false positive on
slow-but-steady convergence; as a per-window figure the same number is a
sensible knob and existing settings keep working.
"""
function StagnationDetector{T}(window::Int, tolerance::Real=0) where {T}
    return StagnationDetector{T}(max(window, 0), T(max(tolerance, 0)), T[], Bool[],
        :none, zero(T), T(Inf))
end

"""
    stagnation_merit(detector, opts, p_res, d_res, gap_rel, complementarity, scale_p, scale_d)

Dimensionless progress measure: the worst of the four metrics relative to the
tolerance requested for it. `<= 1` means every metric is inside its tolerance.

Complementarity has no tolerance of its own — it is what the duality gap is
built from — so it is held to `ϵ_gap` on the same scale as the objective.
"""
function stagnation_merit(::StagnationDetector{T}, opts::SolverOptions{T},
    p_res::T, d_res::T, gap_rel::T, complementarity::T,
    scale_p::T, scale_d::T, objective_scale::T) where {T}
    safe(ϵ) = ϵ > zero(T) ? ϵ : eps(T)
    merit = (p_res / scale_p) / safe(opts.ϵ_primal)
    merit = max(merit, (d_res / scale_d) / safe(opts.ϵ_dual))
    merit = max(merit, gap_rel / safe(opts.ϵ_gap))
    if isfinite(complementarity) && complementarity > zero(T)
        merit = max(merit, complementarity / (safe(opts.ϵ_gap) * max(one(T), objective_scale)))
    end
    return merit
end

"""
    at_precision_floor(p_res, d_res, gap_rel, scale_p, scale_d, ::Type{T})

Whether every metric already sits within `PRECISION_FLOOR_ULPS` ulps of the
working precision, where no further iteration can help.
"""
function at_precision_floor(p_res::T, d_res::T, gap_rel::T,
    scale_p::T, scale_d::T) where {T}
    floor_value = T(PRECISION_FLOOR_ULPS) * eps(T)
    return (p_res / scale_p) <= floor_value &&
           (d_res / scale_d) <= floor_value &&
           gap_rel <= floor_value
end

"""
    observe!(detector, merit, floor_reached, iterations_remaining) -> Bool

Record one iteration and return whether the solve should stop as stalled.

The window must fill before any judgement is made. Once full, the merit is
compared with the merit `window` iterations ago:

* no improvement at all → `:no_progress` (or `:precision_floor` when the metrics
  cannot go lower at this width);
* improving, but at a rate that projects past the remaining iteration budget →
  `:too_slow`. This is the case a fixed bar cannot express, because whether a
  given rate is good enough depends on how far there is left to go.
"""
function observe!(detector::StagnationDetector{T}, merit::T,
    floor_reached::Bool, iterations_remaining::Int) where {T}
    detector.window > 0 || return false
    push!(detector.merits, merit)
    push!(detector.at_floor, floor_reached)
    if length(detector.merits) > detector.window + 1
        popfirst!(detector.merits)
        popfirst!(detector.at_floor)
    end
    length(detector.merits) > detector.window || return false

    reference = first(detector.merits)
    (isfinite(reference) && isfinite(merit) && reference > zero(T) && merit > zero(T)) ||
        return false

    # Nats of progress per iteration, measured across the whole window rather
    # than step by step.
    progress = log(reference) - log(merit)
    detector.rate = progress / T(detector.window)

    # `log(reference/merit) > tolerance` is the cumulative-improvement test:
    # for small improvements `log(1/(1-x)) ≈ x`, so this reads directly as
    # "the merit fell by more than `tolerance` across the window".
    if !(progress > detector.tolerance)
        detector.reason = all(detector.at_floor) ? :precision_floor : :no_progress
        detector.projected = T(Inf)
        return true
    end

    # Already converged on this measure: nothing to project.
    merit <= one(T) && return false

    remaining = log(merit)                       # nats still needed to reach 1
    detector.projected = remaining / detector.rate
    if detector.projected > T(PROJECTION_SLACK) * T(max(iterations_remaining, 1))
        detector.reason = all(detector.at_floor) ? :precision_floor : :too_slow
        return true
    end
    return false
end

"""
    stagnation_message(detector, tolerance) -> String

Human-readable termination reason, including the measured rate and projection
so the choice can be checked rather than taken on trust.
"""
function stagnation_message(detector::StagnationDetector{T}, tolerance) where {T}
    rate = round(Float64(detector.rate), sigdigits=3)
    if detector.reason === :precision_floor
        return "All convergence metrics reached the $(T) precision floor " *
               "(within $(PRECISION_FLOOR_ULPS) ulps) and stopped improving over " *
               "$(detector.window) iterations; the requested tolerance " *
               "$(Float64(tolerance)) is below what $(T) can deliver on this " *
               "problem. Use a wider arithmetic type. Returning the best iterate."
    elseif detector.reason === :too_slow
        projected = round(Float64(detector.projected), sigdigits=3)
        return "Convergence stagnated: over the last $(detector.window) iterations " *
               "the scaled merit improved at $(rate) nats/iteration, which projects " *
               "$(projected) further iterations to reach the requested tolerance — " *
               "more than the remaining budget. Returning the best iterate."
    else
        return "Convergence stagnated: the scaled residual/gap/complementarity " *
               "merit made no progress over $(detector.window) iterations at $(T) " *
               "precision. Returning the best iterate."
    end
end
