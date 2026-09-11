# ---------------------------------------------------------------------------
# Internal result protocol.
#
# Compatibility result adapters retain family-specific layouts, but they share
# one typed lifecycle boundary. Public adapters consume `AbstractCoreResult`
# implementations and are the only layer that constructs the exported
# `Result`; numerical cores never return that public wrapper directly.
# ---------------------------------------------------------------------------

abstract type AbstractCoreResult{T} end
abstract type AbstractCoreDiagnostics end



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
end


# Source compatibility for the pre-`termination` positional form.
SolveDiagnostics(classification, plan, presolve, timings, memory,
    selected_algorithms, parameter_history, warnings) =
    SolveDiagnostics(classification, plan, presolve, timings, memory,
        selected_algorithms, parameter_history, warnings, (reason=:none,))

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
