#=====================================================================
    Null-space formulation (plan §12.2)

    With `m` variables and `n` equality constraints `Bᵀx = b`, the
    range-space formulation carries all `m` variables and enforces the
    equalities through Lagrange multipliers. The null-space formulation
    instead parameterises the feasible set directly:

        x = x_particular + Z·z,     Bᵀ Z = 0,

    reducing the unknowns from `m` to `m − rank(B)`. That is a large win
    when the equalities are numerous, and a waste when they are few —
    `Z` has `m − rank(B)` columns, so with a handful of equalities the
    reduced system is barely smaller than the original while `Z` itself
    costs `m × (m − rank)` storage.

    Safeguards §12.2 requires, all implemented here: rank-revealing
    construction, a basis quality estimate, a memory estimate available
    *before* materialisation, and validation in the original
    coordinates.
=====================================================================#

"""
    NullSpaceBasis{T}

A basis `Z` for `null(Bᵀ)` together with a particular solution of `Bᵀx = b` and
the diagnostics needed to decide whether to trust and use it.

`quality` is the ratio of smallest to largest pivot magnitude in the
rank-revealing factorization — a condition-number proxy for the basis. A value
near one means a well-conditioned basis; a tiny value means `Z` is nearly
rank-deficient and the reduced system will inherit that.
"""
struct NullSpaceBasis{T}
    Z::Matrix{T}
    particular::Vector{T}
    rank::Int
    reduced_dimension::Int
    quality::T
    consistent::Bool
end

"""
    nullspace_memory_bytes(variables, equalities, ::Type{T}) -> Int

Bytes `Z` will occupy, computable *before* materialising it.

§12.2 requires this because the basis is the one object in the formulation that
can be larger than the problem: with few equalities `Z` is `m × (m − n)`, which
approaches a dense `m × m` matrix. Checking first avoids allocating a basis only
to discover it does not fit.

Assumes full column rank, i.e. the largest `Z` that could result.
"""
function nullspace_memory_bytes(variables::Integer, equalities::Integer, ::Type{T}) where {T}
    reduced = max(Int(variables) - Int(equalities), 0)
    return ExtendedPrecisionBLAS._element_storage_bytes(T) * Int(variables) * reduced
end

"""
    nullspace_reduction_ratio(variables, equalities) -> Float64

Fraction of the original dimension the reduced system retains, assuming full
rank. Near zero means the formulation removes most of the problem; near one
means it removes almost nothing and only adds the cost of carrying `Z`.
"""
function nullspace_reduction_ratio(variables::Integer, equalities::Integer)
    variables <= 0 && return 1.0
    return max(Int(variables) - Int(equalities), 0) / Int(variables)
end

"""
    build_nullspace_basis(B, b; tolerance) -> NullSpaceBasis

Rank-revealing construction of `Z` and a particular solution of `Bᵀx = b`.

Uses a column-pivoted QR in the problem's own arithmetic — never a `Float64`
downcast — for the same reason the equality presolve does: a rank decision made
in narrower arithmetic can silently change the feasible set.

With `B = QR` (pivoted, `B` is `m × n`), the leading `rank` columns of `Q` span
`range(B)` and the trailing columns span `null(Bᵀ)`, giving `Z` directly. The
minimum-norm particular solution follows from `BᵀB = RᵀR`, so
`x_p = Q(Rᵀ \\ b)` without forming a normal-equations matrix.

`consistent` reports whether `Bᵀ x_p ≈ b` actually holds. It can fail when the
equalities are inconsistent, and a caller must not proceed as though the
feasible set were non-empty in that case.
"""
function build_nullspace_basis(B::AbstractMatrix{T}, b::AbstractVector{T};
                               tolerance::Real=0) where {T}
    variables, equalities = size(B)
    if equalities == 0
        # No equalities: every direction is feasible and `Z` is the identity.
        return NullSpaceBasis{T}(Matrix{T}(LinearAlgebra.I, variables, variables),
            zeros(T, variables), 0, variables, one(T), true)
    end

    factor = qr(B, ColumnNorm())
    diagonal_count = min(variables, equalities)
    diagonal = [abs(factor.R[i, i]) for i in 1:diagonal_count]
    largest = isempty(diagonal) ? zero(T) : maximum(diagonal)
    threshold = max(T(tolerance), T(max(variables, equalities)) * eps(T)) * largest
    rank = count(>(threshold), diagonal)

    quality = (rank == 0 || largest == 0) ? zero(T) : diagonal[rank] / largest

    # Trailing columns of the FULL Q span null(Bᵀ). `Matrix(factor.Q)` returns
    # the *thin* factor (`m × min(m,n)`), which omits exactly the columns we
    # need, so the full orthogonal factor is formed explicitly.
    Q = factor.Q * Matrix{T}(LinearAlgebra.I, variables, variables)
    Z = rank < variables ? Q[:, (rank + 1):variables] : zeros(T, variables, 0)

    # Minimum-norm particular solution from the leading triangular block, using
    # only the numerically independent rows.
    particular = zeros(T, variables)
    if rank > 0
        permuted = b[factor.p][1:rank]
        upper = UpperTriangular(factor.R[1:rank, 1:rank])
        particular = Q[:, 1:rank] * (transpose(upper) \ permuted)
    end

    residual = equalities > 0 ? knrmInf(transpose(B) * particular - b) : zero(T)
    scale = max(one(T), knrmInf(b))
    consistent = residual <= sqrt(eps(T)) * scale

    return NullSpaceBasis{T}(Z, particular, rank, size(Z, 2), quality, consistent)
end

"""
    nullspace_residual(basis, B, b) -> (orthogonality, feasibility)

Validation in the original coordinates, as §12.2 requires:

* `orthogonality` — `‖BᵀZ‖∞`, which must be at round-off. A basis that is not
  orthogonal to the constraints does not parameterise the feasible set, and
  every reduced solve built on it silently violates the equalities.
* `feasibility` — `‖Bᵀx_p − b‖∞`, whether the particular solution satisfies the
  equalities it is supposed to.
"""
function nullspace_residual(basis::NullSpaceBasis{T}, B::AbstractMatrix{T},
                            b::AbstractVector{T}) where {T}
    orthogonality = size(basis.Z, 2) == 0 ? zero(T) :
                    knrmInf(transpose(B) * basis.Z)
    feasibility = isempty(b) ? zero(T) :
                  knrmInf(transpose(B) * basis.particular - b)
    return (orthogonality=orthogonality, feasibility=feasibility)
end

"""
    recover_full_solution(basis, z) -> x

Map a reduced solution back: `x = x_particular + Z·z`. Any `z` gives an `x`
satisfying the equalities exactly (up to round-off), which is the property that
makes the formulation worth using.
"""
function recover_full_solution(basis::NullSpaceBasis{T}, z::AbstractVector{T}) where {T}
    length(z) == basis.reduced_dimension ||
        throw(DimensionMismatch("reduced solution has length $(length(z)); expected $(basis.reduced_dimension)"))
    basis.reduced_dimension == 0 && return copy(basis.particular)
    return basis.particular + basis.Z * z
end

"""Below this reduction ratio the null-space formulation removes enough of the
problem to be worth its cost; above it, `Z` costs more than the reduction saves."""
const NULLSPACE_MAXIMUM_REDUCTION_RATIO = 0.5

"""Bases with a quality below this are treated as untrustworthy and the
formulation is declined rather than producing a badly conditioned reduced
system."""
const NULLSPACE_MINIMUM_QUALITY = 1e-8

"""
    should_use_nullspace(; variables, equalities, arithmetic, memory_budget_bytes) -> Bool

Whether the null-space formulation is worth forming, decided *before* building
`Z`.

Declines when the equalities are too few to reduce the problem materially, and
when the basis would not fit the memory budget. Both are cheap checks against
the expensive step, which is materialising `Z`.
"""
function should_use_nullspace(; variables::Integer, equalities::Integer,
                              arithmetic::Type=Float64,
                              memory_budget_bytes::Integer=typemax(Int))
    equalities > 0 || return false
    equalities < variables || return false
    nullspace_reduction_ratio(variables, equalities) <= NULLSPACE_MAXIMUM_REDUCTION_RATIO ||
        return false
    return nullspace_memory_bytes(variables, equalities, arithmetic) <= memory_budget_bytes
end
