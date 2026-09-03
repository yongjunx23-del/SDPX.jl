#=====================================================================#
# Sparse-preserving HSD equality/row-space setup (performance plan P2).
#
# The dense route keeps the existing pivoted RRQR row-space reduction
# (`_hsd_rowspace_reduction(::AbstractMatrix, ...)` in src/hsd/hsd.jl),
# reachable only when the caller passes an explicitly dense matrix.
# This file owns the sparse-eligible path.  It never materializes a dense
# copy of A for rank analysis, never builds a dense QR Q factor, and never
# materializes a dense null-space basis.  SuiteSparse SPQR
# (`SparseArrays.qr`, Float64-only) supplies the sparse rank analysis; the
# original-coordinate identity basis is used whenever no reduction is
# performed.
#
# Outcome policy, frozen in the typed result:
#   * Float64 full-rank sparse     -> preserve original coordinates (V = I).
#   * Float64 rank-deficient       -> explicit original-coordinate expanded
#     route selection (V = I, no dense null basis); the caller must use the
#     expanded route or fail closed.
#   * non-Float64 sparse           -> sparse rank analysis is unsupported
#     (SPQR is Float64-only); same expanded-original selection recorded as
#     unsupported precision.  Never densifies, never downcasts.
#   * near-threshold SPQR diagonal -> rank ambiguous; fail closed exactly
#     like the dense RRQR ambiguity band.
#=====================================================================#

"""Typed outcome of the sparse-preserving HSD equality/row-space setup."""
@enum SparseEqualityReductionStatus::UInt8 begin
    SparseEqualityReady                # Float64 full rank; preserve original coordinates
    SparseEqualityExpandedRequired     # Float64 rank-deficient; original-coordinate expanded route
    SparseEqualityRankAmbiguous        # near-threshold sparse rank; fail closed
    SparseEqualityUnsupportedPrecision # no sparse rank provider for T; expanded-original selection
    SparseEqualityFailClosed           # planner refused (e.g., expanded route unavailable)
end


"""Capability gate: SuiteSparse SPQR rank analysis is Float64-only."""
@inline sparse_equality_rank_supported(::Type{T}) where {T<:AbstractFloat} =
    T === Float64

"""
    SparseEqualityReduction{T}

Cold-path result of the sparse-preserving HSD equality setup.  Property
compatible with the dense `_hsd_rowspace_reduction` NamedTuple (same `Ar`,
`cr`, `V`, `cnull`, `rank`, `rank_tolerance`, `objective_tolerance`,
`ambiguous`, `incompatible`, `ray`), so `_hsd_state_from_reduction`
consumes both representations unchanged.

Typed policy/result metadata:
  * `status`      — one of the `SparseEqualityReductionStatus` values.
  * `mode`        — `:preserve_original` (identity V; any route may solve)
                    or `:expanded_original` (identity V; only the expanded
                    route is valid, otherwise fail closed).
  * `sparse_rank` — rank reported by the sparse provider; `-1` when no
                    sparse rank analysis ran (unsupported precision).
"""
struct SparseEqualityReduction{T<:AbstractFloat}
    status::SparseEqualityReductionStatus
    mode::Symbol
    sparse_rank::Int
    Ar::SparseMatrixCSC{T,Int}
    cr::Vector{T}
    V::Union{SparseMatrixCSC{T,Int},IdentityRankBasis{T}}
    cnull::Vector{T}
    rank::Int
    rank_tolerance::T
    objective_tolerance::T
    ambiguous::Bool
    incompatible::Bool
    ray::Vector{T}
end

# `_hsd_state_from_reduction` stores the typed setup metadata on the
# workspace.  The dense RRQR reduction carries the dense-route markers;
# the sparse reduction carries its own typed status and mode.
@inline _hsd_reduction_mode(::NamedTuple) = :dense_rrqr
@inline _hsd_reduction_status(::NamedTuple) = SparseEqualityReady
@inline _hsd_reduction_mode(reduction::SparseEqualityReduction) = reduction.mode
@inline _hsd_reduction_status(reduction::SparseEqualityReduction) = reduction.status

@inline _hsd_is_identity_basis(::AbstractMatrix) = false

@inline function _hsd_sparse_maxabs(values::AbstractVector{T}) where {T}
    scale = zero(T)
    @inbounds for value in values
        magnitude = value < zero(T) ? -value : value
        magnitude > scale && (scale = magnitude)
    end
    return scale
end

"""Max row-sum (Inf-norm) scale of a sparse matrix, dense-compatible order."""
function _hsd_sparse_scaleA(A::SparseMatrixCSC{T,Int}) where {T}
    m, n = size(A)
    row_sums = alloc_zeros(T, m)
    values = nonzeros(A)
    @inbounds for column in 1:n
        for pointer in nzrange(A, column)
            row = rowvals(A)[pointer]
            value = values[pointer]
            row_sums[row] += value < zero(T) ? -value : value
        end
    end
    scale = zero(T)
    @inbounds for row in 1:m
        row_sums[row] > scale && (scale = row_sums[row])
    end
    return scale
end

"""
    _hsd_sparse_rank_analysis(A) -> NamedTuple

Sparse rank analysis of the equality map via SuiteSparse SPQR on `A'`
(no dense materialization, no Q-factor materialization).  Returns
`(rank, ambiguous, cutoff, scaleA)`.  The cutoff and ambiguity band mirror
the dense RRQR semantics (same formulas, provider-reported diagonal):
  * diagonal `> cutoff` counts toward the rank;
  * diagonal inside `(cutoff, 4·cutoff]` or `(noise_hi, cutoff]` is
    `ambiguous` (fail closed, never guessed).
"""
function _hsd_sparse_rank_analysis(A::SparseMatrixCSC{Float64,Int})
    m, n = size(A)
    scaleA = max(_hsd_sparse_scaleA(A), one(Float64))
    rank_tol = Float64(max(m, n)) * eps(Float64) * scaleA
    # SPQR's default tolerance (20*sum(size)*eps*maxcolnorm) drops any
    # pivot below the dense RRQR cutoff, so the ambiguity band below would
    # never trigger.  Pass the scalar arithmetic floor: SPQR then resolves
    # every pivot the dense path would inspect, and the same cutoff/band
    # formulas below reproduce the dense fail-closed ambiguity semantics.
    factor = LinearAlgebra.qr(
        SparseArrays.sparse(transpose(A)); tol=eps(Float64),
    )
    diagonal = abs.(SparseArrays.diag(factor.R))
    dmax = maximum(diagonal; init=0.0)
    cutoff = dmax > 0.0 ? max(rank_tol, rank_tol * dmax / scaleA) : rank_tol
    noise_hi = Float64(10) * eps(Float64) * scaleA
    ambiguity_hi = cutoff * Float64(4)
    ambiguous = false
    for d in diagonal
        if (d > cutoff && d <= ambiguity_hi) ||
           (d > noise_hi && d <= cutoff)
            ambiguous = true
            break
        end
    end
    rank = count(>(cutoff), diagonal)
    return (rank=rank, ambiguous=ambiguous, cutoff=cutoff, scaleA=scaleA)
end

"""
    SparseEqualityPolicy

Narrow planner policy for the sparse-preserving equality setup.
`status`/`mode` mirror `SparseEqualityReduction`; `reason` records the
decision.  This is the planner hook: route/memory feasibility is decided by
the caller from these typed facts.
"""
struct SparseEqualityPolicy
    status::SparseEqualityReductionStatus
    mode::Symbol
    reason::Symbol
end

"""
    select_sparse_equality_policy(A; expanded_available=true)
        -> (SparseEqualityPolicy, analysis_or_nothing)

Narrow planner settings for the sparse-preserving HSD equality setup.
Returns the typed policy and (when a Float64 sparse rank analysis ran) the
raw analysis facts:
  * `SparseEqualityReady`              — Float64 full rank; preserve original
    coordinates; any route may solve.
  * `SparseEqualityExpandedRequired`   — Float64 rank-deficient; keep
    original coordinates and select the expanded route (or fail closed).
  * `SparseEqualityRankAmbiguous`      — SPQR diagonal inside the fail-closed
    ambiguity band.
  * `SparseEqualityUnsupportedPrecision` — SPQR is Float64-only; no sparse
    rank analysis and no densification/downcast; expanded-original
    selection.
  * `SparseEqualityFailClosed`         — `expanded_available=false`; the
    caller must refuse this problem rather than densify.
"""
function select_sparse_equality_policy(
    A::SparseMatrixCSC{T,Int};
    expanded_available::Bool=true,
) where {T<:AbstractFloat}
    n = size(A, 2)
    if !sparse_equality_rank_supported(T)
        return (
            expanded_available ?
            SparseEqualityPolicy(
                SparseEqualityUnsupportedPrecision,
                :expanded_original,
                :sparse_rank_unsupported_precision,
            ) :
            SparseEqualityPolicy(
                SparseEqualityFailClosed,
                :expanded_original,
                :expanded_route_unavailable,
            ),
            nothing,
        )
    end
    analysis = _hsd_sparse_rank_analysis(A)
    if analysis.ambiguous
        policy = SparseEqualityPolicy(
            SparseEqualityRankAmbiguous,
            :expanded_original,
            :sparse_rank_ambiguous,
        )
    elseif analysis.rank == n
        policy = SparseEqualityPolicy(
            SparseEqualityReady,
            :preserve_original,
            :sparse_full_rank,
        )
    else
        policy = SparseEqualityPolicy(
            SparseEqualityExpandedRequired,
            :expanded_original,
            :sparse_rank_deficient,
        )
    end
    return policy, analysis
end

"""
    hsd_sparse_rowspace_reduction(A, c) -> SparseEqualityReduction

Sparse-preserving HSD equality/row-space setup.  Every outcome keeps the
original coordinates: `V = I`, `Ar = sparse(A)`, `cr = c`, `rank = n`.
No dense `Matrix(A)` copy, no dense QR Q factor, and no dense null-space
basis are ever materialized.  The typed `status`/`mode`/`sparse_rank`
metadata tells the caller which route may consume the result.
"""
function hsd_sparse_rowspace_reduction(
    A::SparseMatrixCSC{T,Int},
    c::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(A)
    n == length(c) || throw(DimensionMismatch(
        "canonical A/c dimensions disagree",
    ))
    if n == 0
        return SparseEqualityReduction{T}(
            SparseEqualityReady, :preserve_original, 0,
            SparseArrays.sparse(alloc_zeros(T, m, 0)),
            Vector{T}(undef, 0),
            IdentityRankBasis(T, 0), alloc_zeros(T, 0), 0,
            zero(T), zero(T), false, false, alloc_zeros(T, 0),
        )
    end

    policy, analysis = select_sparse_equality_policy(A)
    policy.status === SparseEqualityFailClosed && throw(ArgumentError(
        "sparse-preserving equality setup failed closed: $(policy.reason)",
    ))

    # Original coordinates: identity is an orthonormal row-space basis that
    # preserves the established bitwise path for full-rank systems and keeps
    # every coordinate (no dense null-space basis) otherwise.
    # A zero-payload IdentityRankBasis is stored: no O(n) sparse identity
    # values/indices and no dense n×n identity are ever materialized.
    V = IdentityRankBasis(T, n)
    Ar = SparseArrays.sparse(A)
    cr = copy(c)
    scaleC = max(_hsd_sparse_maxabs(c), one(T))
    objective_tolerance = T(100 * max(m, n)) * eps(T) * scaleC
    if analysis === nothing
        # No sparse rank analysis ran (unsupported precision).  Record only
        # the would-be cutoff as metadata; no analysis consumed it and no
        # densification/downcast ever happens.
        rank_tolerance = T(max(m, n)) * eps(T) *
                         max(_hsd_sparse_scaleA(A), one(T))
        sparse_rank = -1
    else
        rank_tolerance = T(analysis.cutoff)
        sparse_rank = analysis.rank
    end
    ambiguous = policy.status === SparseEqualityRankAmbiguous
    return SparseEqualityReduction{T}(
        policy.status,
        policy.mode,
        sparse_rank,
        Ar, cr, V, alloc_zeros(T, n), n,
        rank_tolerance, objective_tolerance,
        ambiguous, false, alloc_zeros(T, n),
    )
end

"""Construct a full-rank sparse reduction from an external structural proof.

The caller must prove that the sparse operator has full column rank (the
fixed-trace route does so from disjoint invertible local tail maps).  No
numerical rank inference or precision downcast occurs here.
"""
function hsd_structural_full_rank_reduction(
    A::SparseMatrixCSC{T,Int}, c::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(A)
    length(c) == n || throw(DimensionMismatch("structural rank A/c"))
    scaleA = max(_hsd_sparse_scaleA(A), one(T))
    scaleC = max(_hsd_sparse_maxabs(c), one(T))
    return SparseEqualityReduction{T}(
        SparseEqualityReady, :preserve_original, n,
        SparseArrays.sparse(A), copy(c), IdentityRankBasis(T, n),
        alloc_zeros(T, n), n,
        T(max(m,n)) * eps(T) * scaleA,
        T(100 * max(m,n)) * eps(T) * scaleC,
        false, false, alloc_zeros(T, n),
    )
end

"""Sparse-eligible dispatch: never densifies (see `hsd_sparse_rowspace_reduction`)."""
function _hsd_rowspace_reduction(
    A::SparseMatrixCSC{T,Int},
    c::AbstractVector{T},
) where {T<:AbstractFloat}
    return hsd_sparse_rowspace_reduction(A, c)
end
