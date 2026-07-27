#=====================================================================
    EXPERIMENTAL / OPT-IN — not reachable from `solve`.

    Tested building blocks, deliberately not wired into the automatic
    pipeline: no benchmark in this repository qualifies (see the Known
    limitations section of the README). Call the functions here directly.

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
    saturating_bytes(elements..., element_bytes) -> Int

Product of the arguments, clamped to `typemax(Int)` instead of wrapping.

A memory estimate that overflows is worse than no estimate at all: it comes
back negative, compares as less than any budget, and the guard it feeds then
approves precisely the allocation it exists to refuse. Measured before this
was added, `nullspace_memory_bytes(2_000_000_000, 1, Float64)` returned
-4893488163419103232.
"""
function saturating_bytes(factors::Integer...)
    total = big(1)
    for factor in factors
        factor <= 0 && return 0
        total *= big(factor)
        total > typemax(Int) && return typemax(Int)
    end
    return Int(total)
end

"""
    saturating_sum_bytes(terms...) -> Int

Sum of already-saturated byte counts, clamped to `typemax(Int)`. The companion
to [`saturating_bytes`](@ref): products saturate individually, but a sum of
several near-limit products can still wrap, and a wrapped total defeats the
same budget comparisons.
"""
function saturating_sum_bytes(terms::Integer...)
    total = big(0)
    for term in terms
        total += big(max(term, 0))
        total > typemax(Int) && return typemax(Int)
    end
    return Int(total)
end

"""
    nullspace_memory_bytes(variables, equalities, ::Type{T}; rank=nothing) -> Int

Bytes the basis will occupy, computable *before* materialising it.

§12.2 requires this because the basis is the one object in the formulation that
can be larger than the problem: `Z` is `m × (m − rank(B))`, which approaches a
dense `m × m` matrix as the equalities become dependent.

`rank` is the numerical rank of `B`, which is what actually sets the width. It
is usually unknown at the point this is called, and **the equality count is not
a safe substitute for it**: rank can only be lower, which makes `Z` only wider.
So with `rank` omitted this returns the worst case, `m × m`.

That correction matters. The previous version used `m × (m − equalities)` and
described it as "the largest `Z` that could result", which is backwards — full
column rank gives the *smallest* `Z`. Measured on a 100-variable problem with
80 equality columns of rank 1, it estimated 16,000 bytes against an actual
79,200, and the gate approved the reduction under a 20,000-byte budget.

The figure also covers only `Z` itself. `build_nullspace_basis` additionally
forms the full `m × m` orthogonal factor, so the worst case returned here is
the right order for the peak as well.
"""
function nullspace_memory_bytes(variables::Integer, equalities::Integer,
                                ::Type{T}; rank::Union{Nothing,Integer}=nothing) where {T}
    variables <= 0 && return 0
    width = rank === nothing ? Int(variables) :
            max(Int(variables) - Int(rank), 0)
    return saturating_bytes(
        ExtendedPrecisionBLAS._element_storage_bytes(T),
        Int(variables),
        width,
    )
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
                               tolerance::Real=0,
                               memory_budget_bytes::Integer=typemax(Int)) where {T}
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

    # The budget is enforced here, between knowing the rank and spending the
    # memory. This is the only point where both facts are available: before the
    # factorization the rank is unknown and the caller can only be given the
    # `m × m` worst case, and after the next line the allocation has already
    # happened. Returning a zero-rank, inconsistent basis signals refusal
    # without partially mutating anything the caller owns.
    required = nullspace_memory_bytes(variables, equalities, T; rank=rank)
    if required > memory_budget_bytes
        return NullSpaceBasis{T}(zeros(T, variables, 0), zeros(T, variables),
            rank, 0, quality, false)
    end

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

#=====================================================================
    Reduction to the null-space coordinates (plan §12.2, §12.6)
=====================================================================#

"""
    NullSpaceReduction{T}

A problem rewritten in null-space coordinates together with everything needed
to map a solution back.

`offset` is the constant `cᵀx_p` dropped from the reduced objective. It must be
added back, or the reported objective is wrong by a value that looks plausible
and is not.
"""
struct NullSpaceReduction{T}
    problem::SDPProblem{T}
    basis::NullSpaceBasis{T}
    offset::T
end

"""
    nullspace_reduce(prob; tolerance) -> Union{Nothing,NullSpaceReduction}

Rewrite `prob` in the coordinates `x = x_p + Z·z`, eliminating the equality
constraints entirely, or return `nothing` when the basis cannot be trusted.

The reduced problem has `rank(B)` fewer variables and no equality rows, so its
Schur complement is `k × k` rather than `m × m`. Since factorization is cubic,
that is the whole point: with `k = 0.3 m` the per-iteration factorization drops
by roughly thirty times.

What it costs is a change of basis on the constraint tensor,
`Ã_j = Σᵢ Z[i,j] Aᵢ`, which is `O(m · k · Σ_l k_l²)` once. That is why this is
not always worth doing even when the equalities are numerous — the transform is
paid up front whether the solve then takes five iterations or fifty. See
[`should_use_nullspace`](@ref) for the gate.

Returns `nothing` when the equalities are inconsistent or the basis is too
ill-conditioned to build on, rather than producing a reduced problem whose
solutions do not correspond to the original's.
"""
function nullspace_reduce(prob::SDPProblem{T}; tolerance::Real=0,
                          memory_budget_bytes::Integer=typemax(Int)) where {T}
    L, m, n, k = prob.dims
    n > 0 || return nothing

    # Enforced here as well as offered to the caller: a reduction that silently
    # trusts an external gate is one bad call site away from allocating a dense
    # `m × m` basis on a problem that cannot hold one.
    basis = build_nullspace_basis(prob.B, prob.b; tolerance=tolerance,
        memory_budget_bytes=memory_budget_bytes)
    basis.consistent || return nothing
    basis.quality >= T(NULLSPACE_MINIMUM_QUALITY) || return nothing
    reduced_dimension = basis.reduced_dimension
    reduced_dimension > 0 || return nothing

    Z = basis.Z
    particular = basis.particular

    # `Ã_j = Σᵢ Z[i,j] Aᵢ` and `C̃ = C − Σᵢ x_p[i] Aᵢ`, block by block.
    coefficients = Vector{Array{T,3}}(undef, L)
    constants = Vector{Matrix{T}}(undef, L)
    for block in 1:L
        dimension = k[block]
        original = _nullspace_block_slices(prob, block)
        transformed = zeros(T, reduced_dimension, dimension, dimension)
        shifted = copy(prob.C[block])
        @inbounds for variable in 1:m
            slice = view(original, variable, :, :)
            weight = particular[variable]
            iszero(weight) || (shifted .-= weight .* slice)
            for reduced in 1:reduced_dimension
                factor = Z[variable, reduced]
                iszero(factor) && continue
                view(transformed, reduced, :, :) .+= factor .* slice
            end
        end
        coefficients[block] = transformed
        constants[block] = shifted
    end

    objective = transpose(Z) * prob.c
    offset = dot(prob.c, particular)

    reduced = ingest(
        objective,
        coefficients,
        constants,
        Matrix{T}(undef, reduced_dimension, 0),
        T[];
        sparse=:auto,
        verbosity=0,
    )
    return NullSpaceReduction{T}(reduced, basis, offset)
end

"""Dense `m × k_l × k_l` view of one block's coefficients, whatever storage the
problem uses. The reduction is a dense change of basis, so a sparse block is
materialised here rather than pretending the transform preserves sparsity —
`Ã_j` is a combination of every `Aᵢ` and is dense in general."""
function _nullspace_block_slices(prob::SDPProblem{T}, block::Integer) where {T}
    L, m, n, k = prob.dims
    dimension = k[block]
    slices = zeros(T, m, dimension, dimension)
    cons = prob.cons
    if cons isa DenseCons{T}
        panel = (cons::DenseCons{T}).Av[block]
        @inbounds for variable in 1:m
            slices[variable, :, :] =
                reshape(view(panel, :, variable), dimension, dimension)
        end
    else
        sparse_cons = cons::SparseCons{T}
        blocks = sparse_cons.Asp[block]
        @inbounds for variable in sparse_cons.active[block]
            matrix = blocks[variable]
            rows = rowvals(matrix)
            values = nonzeros(matrix)
            for column in 1:dimension, index in nzrange(matrix, column)
                slices[variable, rows[index], column] = values[index]
            end
        end
    end
    return slices
end

"""
    nullspace_expand(reduction, result) -> (x, objective)

Map a reduced solution back to the original coordinates, restoring both the
variables and the constant the reduced objective dropped.
"""
function nullspace_expand(reduction::NullSpaceReduction{T}, z::AbstractVector{T},
                          reduced_objective::T) where {T}
    return (
        x=recover_full_solution(reduction.basis, z),
        objective=reduced_objective + reduction.offset,
    )
end

"""
    recover_equality_multiplier(prob, Y_blocks) -> Vector

Least-squares recovery of the equality multiplier `y` that the reduced problem
never forms.

Eliminating the equalities removes their multiplier along with them, but the
final certificate is stated in the *original* problem, where dual feasibility
reads `c − Σ_l A_l*(Y_l) − B y = 0`. So `y` has to be reconstructed before the
result can be validated or warm-started, and the defining equation above is
what reconstructs it: solve `B y ≈ c − Σ_l A_l*(Y_l)` in the least-squares
sense.

Least squares rather than a solve, because `B` need not have full column rank —
the same near-dependence the equality presolve exists to handle. When it does
not, this returns the minimum-norm multiplier, which is the one consistent with
having dropped the dependent directions.
"""
function recover_equality_multiplier(prob::SDPProblem{T}, Y_blocks) where {T}
    L, m, n, k = prob.dims
    n == 0 && return T[]
    residual = copy(prob.c)
    for block in 1:L
        slices = _nullspace_block_slices(prob, block)
        Y = Y_blocks[block]
        @inbounds for variable in 1:m
            residual[variable] -= dot(view(slices, variable, :, :), Y)
        end
    end
    return qr(prob.B, ColumnNorm()) \ residual
end
