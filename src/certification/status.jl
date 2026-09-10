#=====================================================================#
#  S04 — terminal status vocabulary and telemetry.
#
#  ADR-003 §1/§3.  Four facts that must NEVER be merged:
#
#      1. numeric termination   — the iteration stopped, and why
#      2. numeric verified      — a certificate in ORIGINAL coordinates passed
#      3. strict error bound    — a *derived* forward bound, not an estimate
#      4. resource / support    — the run ran out of budget or capability
#
#  A status is composed from those four axes by [`compose_terminal_status`].
#  The composition performs **no promotional upgrade**: `Optimal` is reachable
#  only from (converged numeric exit) AND (verified optimal L3) AND (no
#  unsupported capability).  A `maxiter` / time / memory exit therefore keeps
#  its last valid state and is reported as `IterLimit` / `TimeLimit` /
#  `NumericalBreakdown`, never as `Optimal`.
#
#  Telemetry ([`CertificationTelemetry`]) is measurement only.  It is
#  deliberately a separate object from every status axis and from
#  [`acceptance_token`](@ref); nothing in the acceptance path reads it.
#=====================================================================#

"""
    Unmeasured

Sentinel for "this was not measured".  ADR-003 §3: unmeasured is `null` /
`not_run`, never `0`.  Use [`measured`](@ref) to convert a `Real` into a
`Measured`.
"""
struct Unmeasured end

const UNMEASURED = Unmeasured()

"""A measurement is either a real number or explicitly [`UNMEASURED`](@ref)."""
const Measured{T<:Real} = Union{T,Unmeasured}

"""`measured(3.0) === 3.0`; `measured(nothing) === UNMEASURED`."""
measured(x::Real) = x
measured(::Nothing) = UNMEASURED

"""Display helper: `"3.0"` or `"not_run"`, never `"0"` for a missing value."""
function measurement_label(x::Unmeasured)
    return "not_run"
end
measurement_label(x::Real) = string(x)

"""
    TerminalStatus

The composed public status.  `status` is a `Symbol` from the frozen
[`SolveStatus`](@ref) vocabulary; the four axes that produced it are retained
so a reader can see which fact produced which word.
"""
struct TerminalStatus
    status::Symbol
    termination::TerminationClass
    verification::VerificationClass
    error_bound::ErrorBoundClass
    capability::CapabilityClass
end

"""
    compose_terminal_status(termination, verification, capability, error_bound)
        -> TerminalStatus

Compose the four orthogonal facts into one public word.

The only path to `:Optimal` is `TERMINATION_CONVERGED` plus
`VERIFICATION_VERIFIED_OPTIMAL` plus `CAPABILITY_SUPPORTED`.  An unsupported
capability is `:InsufficientPrecision`-style explicit refusal, never an
acceptance; every resource exit maps to its own non-optimal word.
"""
function compose_terminal_status(termination::TerminationClass,
                                 verification::VerificationClass,
                                 capability::CapabilityClass,
                                 error_bound::ErrorBoundClass;
                                 certificate_attempted::Bool=true)
    if capability === CAPABILITY_UNSUPPORTED
        return TerminalStatus(:unsupported, termination, verification,
                              error_bound, capability)
    end
    if termination === TERMINATION_PROMOTABLE
        if verification === VERIFICATION_VERIFIED_OPTIMAL
            return TerminalStatus(:Optimal, termination, verification,
                                  error_bound, capability)
        elseif verification === VERIFICATION_VERIFIED_PRIMAL_INFEASIBLE
            return TerminalStatus(:PrimalInfeasible, termination, verification,
                                  error_bound, capability)
        elseif verification === VERIFICATION_VERIFIED_DUAL_INFEASIBLE
            return TerminalStatus(:DualInfeasible, termination, verification,
                                  error_bound, capability)
        end
        # `AlmostOptimal` means "the numeric iteration converged and a
        # certificate was attempted but did not reach Optimal".  A point that
        # was refused by a GUARD (non-finite input, inconsistent dual map,
        # dimension mismatch) never had its numeric quality assessed at all,
        # so reporting it as `AlmostOptimal` would overstate what was
        # measured.  That case is `:Rejected`.
        return TerminalStatus(certificate_attempted ? :AlmostOptimal : :Rejected,
                              termination, verification, error_bound, capability)
    end
    termination === TERMINATION_MAXITER &&
        return TerminalStatus(:IterLimit, termination, verification,
                              error_bound, capability)
    termination === TERMINATION_TIME_LIMIT &&
        return TerminalStatus(:TimeLimit, termination, verification,
                              error_bound, capability)
    termination === TERMINATION_MEMORY_LIMIT &&
        return TerminalStatus(:ResourceLimit, termination, verification,
                              error_bound, capability)
    return TerminalStatus(:NumericalBreakdown, termination, verification,
                          error_bound, capability)
end

"""Whether `status` may be reported as a certificate of optimality."""
status_is_optimal(s::TerminalStatus) = s.status === :Optimal

"""Explicit refusal: never a substitute for a required capability (ADR-003 §3)."""
is_unsupported(s::TerminalStatus) = s.status === :unsupported

# ---------------------------------------------------------------------
#  Strict error bound vs. estimate
# ---------------------------------------------------------------------

"""
    classify_error_bound(mode::Symbol) -> ErrorBoundClass

`mode` is what was actually done, not what was hoped for:

* `:strict`    — a forward bound was *derived* (interval / exact-arithmetic
  witness) and is valid for the reported point;
* `:estimated` — only a heuristic/estimated magnitude exists (a residual
  norm scaled by a condition estimate);
* `:not_run`   — nothing was computed.
"""
function classify_error_bound(mode::Symbol)
    mode === :strict && return ERROR_BOUND_STRICT
    mode === :estimated && return ERROR_BOUND_ESTIMATED
    return ERROR_BOUND_UNMEASURED
end

error_bound_label(c::ErrorBoundClass) =
    c === ERROR_BOUND_STRICT ? "strict" :
    c === ERROR_BOUND_ESTIMATED ? "estimated" : "not_run"

"""
    strict_error_bound(problem, point; tol) -> (class, bound)

Derive a forward error bound for the reported ORIGINAL point, or say honestly
that none was derived.

The bound returned here is a *residual-based* bound: it is strict for the
stated point only when the stationarity residual and the cone margin are
computed in the same arithmetic and the problem data are exact in it.  That
is not the general case, so the default classification is
[`ERROR_BOUND_ESTIMATED`](@ref) and a caller must pass
`witness=:exact_arithmetic` to claim strictness.  `NaN` is never converted
into a bound; an unavailable bound is [`UNMEASURED`](@ref).
"""
function strict_error_bound(problem::OriginalProblem{T}, point::OriginalPoint{T};
                            tol::Real=default_tol(T),
                            witness::Symbol=:none) where {T}
    valid_tolerance(tol) || return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    length(point.x) == variable_dimension(problem) ||
        return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    length(point.s) == row_dimension(problem) ||
        return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    all_finite(point.x) && all_finite(point.s) || 
        return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    stationarity = stationarity_residual(problem, point.x, point.y, point.s,
                                        point.tau, point.kappa)
    residual = max(primal_residual(stationarity.rP),
                   dual_residual(stationarity.rD), abs(stationarity.rG))
    isfinite(residual) || return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    margin = min_cone_margin(problem, point.s; scale=:primal)
    isfinite(margin) || return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    bound = residual / data_scale(problem)
    isfinite(bound) || return (ERROR_BOUND_UNMEASURED, UNMEASURED)
    if witness === :exact_arithmetic
        return (ERROR_BOUND_STRICT, T(bound))
    end
    return (ERROR_BOUND_ESTIMATED, T(bound))
end

# ---------------------------------------------------------------------
#  Telemetry — measurement that must not feed back into acceptance
# ---------------------------------------------------------------------

"""
    CertificationTelemetry

Detailed quality metrics for a certification run.  Per card implementation
step 3 (`输出详细质量指标但不让telemetry反向影响接受`) and ADR-003 §2, this
object is written by the gate and read by nobody in the acceptance path:
[`acceptance_token`](@ref) does not take it, and neither does
[`compose_terminal_status`](@ref).  A/B runs with telemetry on and off must
therefore agree bitwise on decisions and on every numeric field.

Timings are `UNMEASURED` when diagnostics are disabled — not `0.0`.
"""
struct CertificationTelemetry{T<:AbstractFloat}
    diagnostics_enabled::Bool
    primal_residual::Measured{T}
    dual_residual::Measured{T}
    stationarity::Measured{T}
    gap::Measured{T}
    complementarity::Measured{T}
    primal_cone_margin::Measured{T}
    dual_cone_margin::Measured{T}
    data_scale::Measured{T}
    map_adjoint_residual::Measured{T}
    certification_seconds::Measured{Float64}
    allocations_bytes::Measured{Int}
    iterations::Measured{Int}
    checks_run::Measured{Int}
end

"""An explicitly empty telemetry record: every field `not_run`."""
function unmeasured_telemetry(::Type{T}) where {T}
    return CertificationTelemetry{T}(
        false, UNMEASURED, UNMEASURED, UNMEASURED, UNMEASURED, UNMEASURED,
        UNMEASURED, UNMEASURED, UNMEASURED, UNMEASURED, UNMEASURED,
        UNMEASURED, UNMEASURED, UNMEASURED,
    )
end

"""
    certification_telemetry(metrics, provenance; seconds=nothing,
                            allocations=nothing, iterations=nothing,
                            checks_run=nothing) -> CertificationTelemetry

Snapshot telemetry from the measured quantities.  A quantity that was not
measured stays [`UNMEASURED`](@ref); it is never defaulted to zero.
"""
function certification_telemetry(metrics::CertificateMetrics{T},
                                 provenance::CertificateProvenance{T};
                                 seconds::Union{Nothing,Real}=nothing,
                                 allocations::Union{Nothing,Integer}=nothing,
                                 iterations::Union{Nothing,Integer}=nothing,
                                 checks_run::Union{Nothing,Integer}=nothing,
                                 map_adjoint_residual::Union{Nothing,Real}=nothing,
                                 ) where {T}
    return CertificationTelemetry{T}(
        provenance.telemetry_enabled,
        metrics.primal_residual,
        metrics.dual_residual,
        metrics.stationarity,
        metrics.gap,
        metrics.complementarity,
        metrics.primal_cone_margin,
        metrics.dual_cone_margin,
        metrics.data_scale,
        measured(map_adjoint_residual === nothing ? nothing : T(map_adjoint_residual)),
        measured(seconds === nothing ? nothing : Float64(seconds)),
        measured(allocations === nothing ? nothing : Int(allocations)),
        measured(iterations === nothing ? nothing : Int(iterations)),
        measured(checks_run === nothing ? nothing : Int(checks_run)),
    )
end

"""
    telemetry_agrees(a, b) -> Bool

Compare two telemetry records for the acceptance-relevant fields only
(metrics, checks, decisions).  Timing and allocation fields are compared
separately by the A/B test because a disabled diagnostics run reports them
as [`UNMEASURED`](@ref) rather than reproducing a wall-clock number.
"""
function telemetry_agrees(a::CertificationTelemetry{T},
                          b::CertificationTelemetry{T}) where {T}
    return a.primal_residual === b.primal_residual &&
           a.dual_residual === b.dual_residual &&
           a.stationarity === b.stationarity &&
           a.gap === b.gap &&
           a.complementarity === b.complementarity &&
           a.primal_cone_margin === b.primal_cone_margin &&
           a.dual_cone_margin === b.dual_cone_margin &&
           a.data_scale === b.data_scale &&
           a.map_adjoint_residual === b.map_adjoint_residual &&
           a.checks_run === b.checks_run
end

"""Whether both records report their timings as `not_run` (diagnostics off)."""
telemetry_timings_unmeasured(t::CertificationTelemetry) =
    t.certification_seconds isa Unmeasured && t.allocations_bytes isa Unmeasured
