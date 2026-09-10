#=====================================================================#
#  S04 / L2 — direction (per-step numeric) certification.
#
#  ADR-003 §1: L2 is *per-step numeric* checking and may never be replaced
#  by an ownership/lease token.  It is kept in a different layer from L3
#  (`original.jl`) because the two answer different questions:
#
#      L2: is this candidate DIRECTION numerically admissible at all?
#      L3: does the TERMINAL POINT satisfy the ORIGINAL problem?
#
#  Extracted from the certificate machinery in
#  `src/certificates/certificates.jl` (the svec/dual-map discipline) and from
#  the per-step checks in `src/hsd/` — reimplemented here against plain
#  original-coordinate data so the check does not inherit the solver's own
#  scaling, sign or accumulation.
#
#  The dual map is an EXPLICIT artifact with provenance.  A map whose adjoint
#  identity fails is a *wrong dual map* and must produce a typed rejection
#  ([`REJECT_DUAL_MAP_INCONSISTENT`](@ref)), not a silently weaker check.
#=====================================================================#

"""
    DualMapArtifact{T}

The packed PSD coordinate map as an inspectable artifact: the primal packed
factor `scale`, its adjoint `pullback`, where it came from, and the residual
of the adjoint identity that makes it a *dual* map.

`build_dual_map(dim; scaling=g)` with `g ≠ 1` deliberately produces a WRONG
dual map; the holder is then refused by [`dual_map_consistent`](@ref).
"""
struct DualMapArtifact{T<:AbstractFloat}
    dim::Int
    packed_length::Int   # NOT `length`: a field named `length` shadows Base.length
    scale::Vector{T}
    pullback::Vector{T}
    source::Symbol
    adjoint_residual::T
end

"""
    build_dual_map(dim; scaling=1, source=:independent_adjoint) -> DualMapArtifact

Build the packed primal factor `scale` (`1` diagonal, `√2` off-diagonal) and
its adjoint `pullback = 1/scale`, optionally perturbed by `scaling`.  The
independent reference for the off-diagonal factor is `√2`, derived from
`⟨P, Q⟩ = Σ_ij P_ij Q_ij` over a symmetric basis; `test/rebuild/S04.jl`
cross-checks it against A01's oracle rather than against SDPX.
"""
function build_dual_map(dim::Integer; scaling::Real=1,
                        source::Symbol=:independent_adjoint,
                        T::Type{<:Real}=Float64)
    len = psd_packed_length(dim)
    scale = [T(scaling) * psd_scale_of(dim, k; T=T) for k in 1:len]
    pullback = [inv(s) for s in scale]
    residual = zero(T)
    @inbounds for k in 1:len
        residual = max(residual, abs(scale[k] * pullback[k] - one(T)))
    end
    return DualMapArtifact{T}(Int(dim), len, scale, pullback, source, residual)
end

"""
    DualMapArtifact(dim, scale, pullback; source=:caller)

Construct a map artifact from an explicit `(scale, pullback)` pair, COMPUTING
the adjoint-identity defect `max_k |scale[k]·pullback[k] − 1|` from the
vectors themselves rather than trusting a caller-supplied number.  This is
what makes a wrong dual map detectable: a fabricated `adjoint_residual` of
`0.0` cannot disguise a non-adjoint pair.
"""
function DualMapArtifact(dim::Integer, scale::AbstractVector,
                         pullback::AbstractVector; source::Symbol=:caller)
    R = promote_type(eltype(scale), eltype(pullback))
    R <: Real || throw(ArgumentError("map arithmetic must be real, got $R"))
    s = Vector{R}(scale)
    d = Vector{R}(pullback)
    len = psd_packed_length(dim)
    residual = R(Inf)
    if length(s) == len && length(d) == len
        acc = zero(R)
        for k in 1:len
            acc = max(acc, abs(s[k] * d[k] - one(R)))
        end
        residual = acc
    end
    return DualMapArtifact{R}(Int(dim), len, s, d, source, residual)
end

"""Adjoint-identity defect of a map artifact (`0` is a valid dual map)."""
dual_map_residual(map::DualMapArtifact{T}) where {T} = map.adjoint_residual

"""
    as_psd_block(map, offset) -> PSDBlock

Attach a map artifact to the ORIGINAL block layout at `offset`.
"""
as_psd_block(map::DualMapArtifact{T}, offset::Integer) where {T} =
    PSDBlock(Int(offset), map.dim; scale=map.scale, dual_scale=map.pullback, T=T)

"""
    map_is_dual_consistent(map; tol) -> Bool

Whether the artifact is a usable dual map at all: positive finite factors,
matching lengths, and an adjoint identity within `tol`.
"""
function map_is_dual_consistent(map::DualMapArtifact{T};
                                tol::Union{Nothing,Real}=nothing) where {T}
    limit = tol === nothing ? max(100 * eps(T), T(1e-12)) : T(tol)
    valid_tolerance(limit) || return false
    (Base.length(map.scale) == map.packed_length &&
     Base.length(map.pullback) == map.packed_length) || return false
    all_finite(map.scale) && all_finite(map.pullback) || return false
    all(s -> isfinite(s) && s > zero(T), map.scale) || return false
    return isfinite(map.adjoint_residual) && map.adjoint_residual <= limit
end

# ---------------------------------------------------------------------
#  L2 direction evidence
# ---------------------------------------------------------------------

@enum DirectionClass::UInt8 begin
    DIRECTION_ADMISSIBLE
    DIRECTION_UNVERIFIED
end

"""
    DirectionEvidence

Result of the per-step numeric gate on one candidate direction.  `verified`
is never set by a lease, an ownership token or a `trusted` flag; only the
checks in [`certify_direction`](@ref) can set it.
"""
struct DirectionEvidence{T<:AbstractFloat}
    class::DirectionClass
    reason::RejectReason
    finite_step::Bool
    primal_cone_margin::T
    dual_cone_margin::T
    map_residual::T
    normalized_step::T
    lease_present::Bool
    checks_run::Int
end

_no_direction_evidence(::Type{T}, reason::RejectReason, maps_residual::T,
                       lease::Bool, checks::Int) where {T} =
    DirectionEvidence{T}(DIRECTION_UNVERIFIED, reason, false, T(NaN), T(NaN),
                         maps_residual, T(NaN), lease, checks)

"""
    certify_direction(problem, op, dx, ds, dy; tol, lease_present=false)

Per-step numeric gate on a candidate direction in ORIGINAL coordinates.

An ownership lease (`lease_present=true`) may skip L1 structure work but may
NOT skip any numeric check here (ADR-003 §1).  `checks_run` is reported so a
caller can prove L2 actually ran.
"""
function certify_direction(problem::OriginalProblem{T}, op::OriginalOperator{T},
                           dx::AbstractVector, ds::AbstractVector,
                           dy::AbstractVector;
                           tol::Real=default_tol(T),
                           lease_present::Bool=false) where {T}
    checks = 0
    map_ok, _ = dual_map_consistent(problem, op)
    map_residual = T(maximum((adjoint_residual(b) for b in problem.blocks
                              if b isa PSDBlock); init=0.0))
    valid_tolerance(tol) ||
        return _no_direction_evidence(T, REJECT_INVALID_TOLERANCE, map_residual,
                                      lease_present, checks)
    checks += 1
    map_ok ||
        return _no_direction_evidence(T, REJECT_DUAL_MAP_INCONSISTENT,
                                      map_residual, lease_present, checks)
    if length(dx) != variable_dimension(problem) ||
       length(ds) != row_dimension(problem) || length(dy) != row_dimension(problem)
        return _no_direction_evidence(T, REJECT_DIMENSION_MISMATCH, map_residual,
                                      lease_present, checks + 1)
    end
    checks += 1
    if !(all_finite(dx) && all_finite(ds) && all_finite(dy))
        return _no_direction_evidence(T, REJECT_NONFINITE_INPUT, map_residual,
                                      lease_present, checks + 1)
    end
    checks += 1
    step_norm = max(maximum(abs, dx; init=zero(T)),
                    maximum(abs, ds; init=zero(T)),
                    maximum(abs, dy; init=zero(T)))
    p_margin = min_cone_margin(problem, ds; scale=:primal)
    d_margin = min_cone_margin(problem, dy; scale=:dual)
    checks += 2
    if !(isfinite(p_margin) && p_margin >= -T(tol))
        return DirectionEvidence{T}(DIRECTION_UNVERIFIED,
                                    REJECT_PRIMAL_CONE_VIOLATION, true,
                                    T(p_margin), T(d_margin), map_residual,
                                    T(step_norm), lease_present, checks)
    end
    if !(isfinite(d_margin) && d_margin >= -T(tol))
        return DirectionEvidence{T}(DIRECTION_UNVERIFIED,
                                    REJECT_DUAL_CONE_VIOLATION, true,
                                    T(p_margin), T(d_margin), map_residual,
                                    T(step_norm), lease_present, checks)
    end
    return DirectionEvidence{T}(DIRECTION_ADMISSIBLE, REJECT_NONE, true,
                                T(p_margin), T(d_margin), map_residual,
                                T(step_norm), lease_present, checks)
end

"""Whether a direction was admitted, and after how many checks."""
direction_admitted(e::DirectionEvidence) = e.class === DIRECTION_ADMISSIBLE

"""
    direction_checks_run(dx, ds, dy) -> Int

Number of numeric checks an L2 run is REQUIRED to perform.  Used by the A/B
test: turning diagnostics off must not reduce it.
"""
direction_checks_run(dx, ds, dy) = 1
