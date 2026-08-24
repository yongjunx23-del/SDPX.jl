# ---------------------------------------------------------------------------
# Internal result protocol.
#
# LP/SDP and NativeSOC retain family-specific iterate layouts, but they share
# one typed lifecycle boundary. Public adapters consume `AbstractCoreResult`
# implementations and are the only layer that constructs the exported
# `Result`; numerical cores never return that public wrapper directly.
# ---------------------------------------------------------------------------

abstract type AbstractCoreResult{T} end
abstract type AbstractCoreDiagnostics end

core_status(result::AbstractCoreResult) = result.status
core_message(result::AbstractCoreResult) = result.message
core_iterations(result::AbstractCoreResult) = result.iterations
core_diagnostics(result::AbstractCoreResult) = result.diagnostics

# ---------------------------------------------------------------------------
# A0 — first-class execution-attempt records.
#
# One solve is one attempt: a single immutable snapshot of the planned route,
# the route that actually executed, and every divergence the plan authorized.
# These records are constructed exactly once per solve, inside
# diagnostics-enabled `_attach_diagnostics`; they never carry per-iteration
# state and never feed back into solver math.
# ---------------------------------------------------------------------------

"""Precision facts for one execution route: the arithmetic tag plus the
explicit significand width when execution reached that route. BigFloat facts
are captured inside the exact per-attempt `setprecision` scope; a route that
never executed records `nothing` instead of re-reading ambient precision.
`mode` records whether the route used the mixed-precision implementation of
its formulation."""
struct AttemptPrecisionFacts
    arithmetic::Symbol
    mode::Symbol
    explicit_bits::Union{Nothing,Int}
end

"""One immutable route snapshot: solver family, formulation, storage, LA
provider, precision, and thread count. `planned` comes from the
`ExecutionPlan`; `executed` mirrors the same dimensions from the actual
termination record, using `:not_executed` where the route never ran."""
struct ExecutionRouteFacts
    family::Symbol
    formulation::Symbol
    storage::Symbol
    provider::Symbol
    precision::AttemptPrecisionFacts
    threads::Int
end

"""One ordered fallback event. `authorized` is derived fail-closed from the
plan's fallback chains: a divergence that is not one of the plan's explicitly
authorized fallback targets is labeled `authorized=false`, never silently
accepted. `source` is `:plan` for planned provenance or `:runtime_executed`
for facts read from the termination record."""
struct FallbackEvent
    kind::Symbol
    reason::Symbol
    authorized::Bool
    source::Symbol
    detail::NamedTuple
end

"""Regularization facts. Regularization is a numerical retry inside the same
formulation, never a route switch, so it is recorded here and must not appear
among `fallback_events`. `count` is the solve-level counter already exposed
by `SDPResult.regularizations`; `events` carries only counts that were
explicitly recorded by an initialization/factor report, each with its own
source."""
struct RegularizationFacts
    count::Int
    events::Tuple{Vararg{NamedTuple}}
    executed_value::Union{Nothing,Real}
end

"""Certificate summary for the attempt. A certification downgrade (status
demoted because the original-coordinate gate failed) is a certificate fact,
not a fallback event."""
struct CertificateFacts
    available::Bool
    method::Symbol
    reason::Symbol
    valid::Union{Nothing,Bool}
    downgrade::Bool
    minimal_gate_valid::Union{Nothing,Bool}
end

"""Prepared-structure reuse facts. A6 owns truthful attempt-level reuse
reporting; until then the record states explicitly that reuse facts are
unavailable and that no reuse was reported."""
struct PreparedReuseFacts
    available::Bool
    reused::Bool
    source::Symbol
end

"""A0 first-class execution attempt: one immutable record of one solve.
`attempt_id`/`plan_id` are attempt-local stable indices for now; `planned`
and `executed` describe the same route dimensions so a divergence is visible
and the plan's authorization of every fallback event is checkable."""
struct ExecutionAttemptRecord
    attempt_id::Int
    plan_id::Int
    planned::ExecutionRouteFacts
    executed::ExecutionRouteFacts
    fallback_events::Tuple{Vararg{FallbackEvent}}
    regularization::RegularizationFacts
    certificate::CertificateFacts
    reuse::PreparedReuseFacts
    status::SolveStatus
    termination_reason::Symbol
end

# ---------------------------------------------------------------------------
# A1 — first-class BigFloat precision ladder.
#
# Pre-execution authority (`PrecisionAttemptSpec` / `PrecisionLadderPlan`) is
# split from post-execution diagnostics (`PrecisionAttemptReport` /
# `PrecisionLadderReport`). The ladder plan is built unconditionally before
# the first BigFloat rung; the reports are diagnostics-only and never carry a
# result, workspace, factor, or mutable BigFloat array.
# ---------------------------------------------------------------------------

"""One pre-execution rung of a BigFloat precision ladder: the exact
significand bits, and whether those bits are the caller's requested
`precision_bits` or an adaptive selector's lower choice. A rung never
contains an `ExecutionPlan`; each rung's child plan is built exactly once
inside that rung, when it executes."""
struct PrecisionAttemptSpec
    rung::Int
    bits::Int
    role::Symbol
end

"""The immutable pre-execution ladder authority for one BigFloat solve.
`policy` is `:fixed` or `:auto`; `requested_bits` is the caller's hard upper
bound; `selected_bits` is the adaptive selector's first-rung choice (the
requested bits for `:fixed` and resume); `floor_bits` is the selector floor;
`resume_bypass` records that resume bypasses the selector; `selection_reason`
is `:resume`, `:fixed`, `:adaptive_lower`, or `:adaptive_requested`. `rungs`
is the ordered rung list (exactly one rung for fixed / resume or when the
adaptive selector lands on the requested bits, at most two rungs otherwise);
`retry_statuses` is the fixed retry-eligibility set that drives the staging
loop (a rung retries only when its status is in this set and shared budget
remains); `time_budget` is the shared wall-clock policy. The plan holds no
results and no child `ExecutionPlan`s (those are built per rung)."""
struct PrecisionLadderPlan
    policy::Symbol
    requested_bits::Int
    selected_bits::Int
    floor_bits::Int
    resume_bypass::Bool
    selection_reason::Symbol
    rungs::Tuple{Vararg{PrecisionAttemptSpec}}
    retry_statuses::Tuple{Vararg{SolveStatus}}
    time_budget::Symbol

    function PrecisionLadderPlan(
        policy::Symbol,
        requested_bits::Int,
        selected_bits::Int,
        floor_bits::Int,
        resume_bypass::Bool,
        selection_reason::Symbol,
        rungs::Tuple{Vararg{PrecisionAttemptSpec}},
        retry_statuses::Tuple{Vararg{SolveStatus}},
        time_budget::Symbol=:shared_wall_clock,
    )
        1 <= length(rungs) <= 2 ||
            throw(ArgumentError(
                "a precision ladder has exactly one or two rungs",
            ))
        for (index, rung) in enumerate(rungs)
            rung.rung == index ||
                throw(ArgumentError(
                    "precision ladder rungs must be ordered 1, 2, ...",
                ))
            rung.bits > 0 ||
                throw(ArgumentError(
                    "precision ladder rung bits must be positive",
                ))
        end
        rungs[1].bits == selected_bits ||
            throw(ArgumentError(
                "precision ladder selected bits must match the first rung",
            ))
        floor_bits > 0 ||
            throw(ArgumentError(
                "precision ladder floor bits must be positive",
            ))
        floor_bits <= selected_bits <= requested_bits ||
            throw(ArgumentError(
                "precision ladder bits must satisfy " *
                "floor <= selected <= requested",
            ))
        return new(
            policy,
            requested_bits,
            selected_bits,
            floor_bits,
            resume_bypass,
            selection_reason,
            rungs,
            retry_statuses,
            time_budget,
        )
    end

    # Compatibility constructor for callers that only provide the ordered
    # rungs: the selected bits are the first rung's bits and the floor is
    # taken as the selected bits.
    function PrecisionLadderPlan(
        policy::Symbol,
        requested_bits::Int,
        rungs::Tuple{Vararg{PrecisionAttemptSpec}},
        retry_statuses::Tuple{Vararg{SolveStatus}},
        time_budget::Symbol=:shared_wall_clock,
    )
        selected = length(rungs) == 2 ? rungs[1].bits : requested_bits
        return PrecisionLadderPlan(
            policy,
            requested_bits,
            selected,
            selected,
            false,
            :compatibility,
            rungs,
            retry_statuses,
            time_budget,
        )
    end
end

"""Scalar execution facts for one ladder rung. `retry_decision` is the
plan-driven decision for the rung (`:success`, `:retry`, `:ineligible_status`,
`:no_time`, or `:terminal`); `remaining_budget_seconds` is the shared
wall-clock budget remaining when the rung completed. Diagnostics-only: no
result, workspace, factor, or mutable BigFloat arrays are retained."""
struct PrecisionAttemptScalarFacts
    status::SolveStatus
    termination_reason::Symbol
    elapsed_seconds::Float64
    success::Bool
    retry_decision::Symbol
    remaining_budget_seconds::Float64
end

"""Post-execution report for one ladder rung. Retains the rung's
`PrecisionAttemptSpec`, its child `ExecutionPlan` (built inside that rung),
its A0 `ExecutionAttemptRecord`, and scalar facts only."""
struct PrecisionAttemptReport
    spec::PrecisionAttemptSpec
    child_plan::ExecutionPlan
    record::ExecutionAttemptRecord
    facts::PrecisionAttemptScalarFacts
end

"""Post-execution ladder diagnostics: the immutable pre-execution plan plus
the ordered per-rung reports. Diagnostics-only; attempts remain accessible
flattened in rung order through `SolveDiagnostics.attempts`."""
struct PrecisionLadderReport
    plan::PrecisionLadderPlan
    attempts::Tuple{Vararg{PrecisionAttemptReport}}
end

"""
    SolveDiagnostics

Structured metadata accompanying a solve. `timings` contains phase-level
seconds, `memory` contains estimated solver workspace and process peak RSS,
and `warnings` records non-fatal fallbacks or numerical caveats.
"""
struct SolveDiagnostics <: AbstractCoreDiagnostics
    classification::ProblemClassification
    plan::ExecutionPlan
    presolve::PresolveReport
    timings::NamedTuple
    memory::NamedTuple
    selected_algorithms::NamedTuple
    parameter_history::Vector{NamedTuple}
    warnings::Vector{String}
    # Why the solve stopped, beyond the coarse `status`. For a `Stalled` run
    # this carries the stagnation detector's verdict (`:no_progress`,
    # `:too_slow`, `:precision_floor`) plus the measured convergence rate and
    # projected iterations, so the decision can be checked rather than trusted.
    termination::NamedTuple
    # A0 execution-attempt records. Constructed exactly once per solve inside
    # diagnostics-enabled `_attach_diagnostics`; empty when a diagnostics
    # object was built by a compatibility path that never ran a solve.
    attempts::Tuple{Vararg{ExecutionAttemptRecord}}
    # A1 precision-ladder diagnostics. `nothing` for solves that never ran a
    # BigFloat ladder; otherwise the full ladder plan plus per-rung reports.
    precision_ladder::Union{Nothing,PrecisionLadderReport}
end

"""Solve-local ladder bookkeeping threaded from the BigFloat staging loop into
`_attach_diagnostics`. Mutable because it accumulates the small per-rung
diagnostics reports (`PrecisionAttemptReport`s — never results, workspaces,
factors, or per-rung `SolveDiagnostics` snapshots) as the ladder advances; it
is never part of any public record. Diagnostics-disabled runs allocate the
empty report vector once and never fill it."""
mutable struct PrecisionLadderContext
    plan::PrecisionLadderPlan
    rung::Int
    attempt_id::Int
    explicit_bits::Int
    rung_started_ns::UInt64
    remaining_budget_seconds::Float64
    reports::Vector{PrecisionAttemptReport}
end

# Source compatibility for the pre-`termination` positional form.
SolveDiagnostics(classification, plan, presolve, timings, memory,
    selected_algorithms, parameter_history, warnings) =
    SolveDiagnostics(classification, plan, presolve, timings, memory,
        selected_algorithms, parameter_history, warnings, (reason=:none,),
        (), nothing)

# Source compatibility for the pre-`attempts` positional form.
SolveDiagnostics(classification, plan, presolve, timings, memory,
    selected_algorithms, parameter_history, warnings, termination) =
    SolveDiagnostics(classification, plan, presolve, timings, memory,
        selected_algorithms, parameter_history, warnings, termination, (),
        nothing)

# Source compatibility for the pre-`precision_ladder` positional form.
SolveDiagnostics(classification, plan, presolve, timings, memory,
    selected_algorithms, parameter_history, warnings, termination, attempts) =
    SolveDiagnostics(classification, plan, presolve, timings, memory,
        selected_algorithms, parameter_history, warnings, termination,
        attempts, nothing)

"""
    SDPResult{T}

Typed replacement for the old `Dict{String,Any}` return value (A1).
`result["x"]`, `result["status"]`, etc. keep working through the compatibility
`Base.getindex` methods below, so existing callers are unaffected;
new code should prefer the typed fields.
"""
struct SDPResult{T} <: AbstractCoreResult{T}
    status::SolveStatus
    message::String
    x::Vector{T}
    X::Vector{Matrix{T}}
    y::Vector{T}
    Y::Vector{Matrix{T}}
    pObj::T
    dObj::T
    gap_rel::T
    p_res::T
    d_res::T
    iterations::Int
    restarts::Int
    regularizations::Int
    timings::Union{Nothing,NamedTuple}
    parameter_history::Vector{NamedTuple}
    diagnostics::Union{Nothing,SolveDiagnostics}
    # Structured termination reason, carried from the solve loop so the
    # pipeline can copy it into `diagnostics.termination`. `(reason=:none,)`
    # when the run ended for an ordinary reason covered by `status`.
    termination::NamedTuple
end

# Source compatibility for the pre-`termination` positional form.
SDPResult{T}(status, message, x, X, y, Y, pObj, dObj, gap_rel, p_res, d_res,
    iterations, restarts, regularizations, timings, parameter_history,
    diagnostics) where {T} =
    SDPResult{T}(status, message, x, X, y, Y, pObj, dObj, gap_rel, p_res, d_res,
        iterations, restarts, regularizations, timings, parameter_history,
        diagnostics, (reason=:none,))

# Source compatibility for callers that constructed the pre-pipeline result
# positionally. New code should obtain results from `solve`/`solve!`.
SDPResult{T}(
    status,
    message,
    x,
    X,
    y,
    Y,
    pObj,
    dObj,
    gap_rel,
    p_res,
    d_res,
    iterations,
    restarts,
    regularizations,
    timings,
) where {T} = SDPResult{T}(
    status,
    message,
    x,
    X,
    y,
    Y,
    pObj,
    dObj,
    gap_rel,
    p_res,
    d_res,
    iterations,
    restarts,
    regularizations,
    timings,
    NamedTuple[],
    nothing,
)

function Base.getindex(r::SDPResult, k::AbstractString)
    k == "x" && return r.x
    k == "X" && return r.X
    k == "y" && return r.y
    k == "Y" && return r.Y
    k == "pObj" && return r.pObj
    k == "dObj" && return r.dObj
    k == "status" && return r.message
    k == "diagnostics" && return r.diagnostics
    k == "parameter_history" && return r.parameter_history
    throw(KeyError(k))
end
Base.haskey(::SDPResult, k::AbstractString) =
    k in (
        "x",
        "X",
        "y",
        "Y",
        "pObj",
        "dObj",
        "status",
        "diagnostics",
        "parameter_history",
    )

# --- Precision traits (Phase 4.1) ---

"""
    has_dynamic_precision(::Type{T})

`true` for `BigFloat`, where `setprecision` changes the working
precision at runtime; `false` for fixed-width bitstypes
(`Float64`, `MultiFloat`s, …), for which precision is
baked into the type and `precision_bits` is a no-op.
"""
has_dynamic_precision(::Type{BigFloat}) = true
has_dynamic_precision(::Type) = false

"""
    sig_bits(::Type{T})

Significand width in bits. Works for `BigFloat` (current global
`precision`), `Float64`, and any type implementing
`Base.precision` (MultiFloats).
"""
sig_bits(::Type{T}) where {T} = precision(T)

"""
    dynamic_range_limited(::Type{T})

`true` for types whose exponent range is bounded and which don't have
a dedicated `Inf` representation distinct from `NaN` (§4.2: e.g.
`MultiFloat`s inherit `Float64`'s ~10±308 range and collapse `±Inf` to
`NaN`). `solve!` runs an extra non-finite-iterate guard for these
types (raw high-degree-polynomial bootstrap data can exceed 10³⁰⁸),
converting an overflow into a reported `NumericalBreakdown` instead of
letting `NaN` silently propagate for the rest of the run. `false` by
default (including for `BigFloat`, whose exponent range is enormous).
"""
dynamic_range_limited(::Type) = false
