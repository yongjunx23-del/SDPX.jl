#=====================================================================#
#    S01 — invertible transform ownership (src/core/transforms.jl).
#
#    Task card: agents/S01.md.  Owner per ADR-001 §2: the
#    `TransformStack` belongs to the COMPILED PROBLEM, which owns the
#    forward *and* inverse maps plus the objective offset. The public/MOI
#    layer must not patch signs.
#
#    This file adds to the already-frozen transform machinery in
#    `src/program/transforms.jl` and `src/program/transforms_rsoc.jl`:
#
#      1. `ObjectiveOffset{T}` — an explicit objective-offset replay.
#         An objective offset is a scalar ADDED to the canonical
#         objective value, so the replay from canonical to original is a
#         sum. This is deliberately NOT the same thing as the frozen
#         `objective_shift(transform)` scalar (that is a shift of the
#         objective's *constant term* at setup); both exist and neither
#         is a synonym for the other.
#
#      2. Whole-chain replay with NO single-transform restriction. The
#         frozen stack methods refuse a stack with more than one entry
#         (a deliberate guard, because a stack may hold block-local
#         maps). The replay functions below instead iterate the chain and
#         size every intermediate from the transform's own declared
#         source/target dimensions.
#
#      3. `EqualityElimination{T}` — the equality-elimination reduction:
#         the transform that was previously absent (no
#         `AbstractProgramReduction` subtype existed anywhere in `src/`).
#         It is rank-revealing-QR driven and carries its own primal,
#         dual, ray and objective-offset replay plus its invalidation.
#
#      4. `forward_primal_ray!` / `forward_dual_ray!` for the whole
#         chain. The frozen interface declares forward primal/dual but
#         only *backward* rays; a certificate ray admitted in original
#         coordinates has to be pushed forward too.
#
#    Cone dimensions are NEVER type parameters here: `EqualityElimination`
#    stores plain `Int`s and matrices/vectors of the working arithmetic.
#    BigFloat values are copied through `owned_arithmetic_copy` under the
#    owned-snapshot rule, so a transform never aliases model storage.
#=====================================================================#

# ---------------------------------------------------------------------------
# Objective offset
# ---------------------------------------------------------------------------

"""
    ObjectiveOffset{T}

The scalar relation between the canonical objective value and the original
one. `offset` is the constant **added** to the canonical objective, so for a
canonical point `(x̂, ŝ)` whose original-coordinate image is `(x, s)`:

    c'x + objective_constant  ==  ĉ'x̂ + offset

`ObjectiveOffset` is immutable and owns its scalar. It is the objective
counterpart of a transform's primal/dual/ray replay: replay without it cannot
recover an objective value once a reduction has moved constants across the
boundary. It carries no cone dimension and no type parameter beyond the
working arithmetic.
"""
struct ObjectiveOffset{T<:AbstractFloat}
    offset::T
end

ObjectiveOffset(value::Real) = ObjectiveOffset{typeof(float(value))}(float(value))

"""The additive objective offset of a transform that adds no constant."""
objective_offset(::AbstractProgramTransform{T}) where {T<:AbstractFloat} = ObjectiveOffset{T}(zero(T))

"""Replay a canonical objective value into original coordinates."""
replay_objective(offset::ObjectiveOffset, canonical_value) = canonical_value + offset.offset

"""Canonical objective value implied by an original-coordinate objective value."""
replay_objective_backward(offset::ObjectiveOffset, original_value) = original_value - offset.offset

Base.zero(::Type{ObjectiveOffset{T}}) where {T} = ObjectiveOffset{T}(zero(T))
Base.:+(a::ObjectiveOffset, b::ObjectiveOffset) = ObjectiveOffset(a.offset + b.offset)
Base.:(==)(a::ObjectiveOffset, b::ObjectiveOffset) = a.offset == b.offset
Base.show(io::IO, offset::ObjectiveOffset{T}) where {T} =
    print(io, "ObjectiveOffset{", T, "}(", offset.offset, ")")

"""The objective offset contributed by one transform (zero unless declared)."""
objective_offset(::RotatedSOCToSOC{T}) where {T<:AbstractFloat} = ObjectiveOffset{T}(zero(T))

# ---------------------------------------------------------------------------
# Forward ray replay for the extracted coordinate transforms
# ---------------------------------------------------------------------------
#
# The frozen interface declares `forward_primal!`/`forward_dual!` but only
# BACKWARD rays. A certificate ray that is admitted in original coordinates
# (or in an intermediate coordinate system) has to be pushed forward too, and
# the forward ray map of an invertible coordinate change is its forward primal
# resp. dual map — not a fresh convention. These methods are declared
# explicitly per transform so that a transform whose ray map differs from its
# point map cannot silently inherit the wrong one.

"""
    forward_primal_ray!(::NonpositiveToNonnegative, dest, src)

Forward primal ray map. The sign transform is its own inverse, so this is the
sign map; it is length-agnostic and applies to a whole vector correctly.
"""
forward_primal_ray!(transform::NonpositiveToNonnegative, dest, src) =
    _nonpositive_sign_map!(dest, src)

"""Forward dual ray map of the sign transform (same map; `T⁻ᵀ = -I`)."""
forward_dual_ray!(transform::NonpositiveToNonnegative, dest, src) =
    _nonpositive_sign_map!(dest, src)

"""Forward primal ray map of the RSOC isometry (orthogonal: `M = M⁻¹ = Mᵀ`)."""
function forward_primal_ray!(
    transform::RotatedSOCToSOC, destination::AbstractVector, source::AbstractVector,
)
    return _rsoc_transform_apply!(destination, transform, source)
end

"""Forward dual ray map of the RSOC isometry (same orthogonal map)."""
function forward_dual_ray!(
    transform::RotatedSOCToSOC, destination::AbstractVector, source::AbstractVector,
)
    return _rsoc_transform_apply!(destination, transform, source)
end

# ---------------------------------------------------------------------------
# A whole-vector sign transform
# ---------------------------------------------------------------------------

"""
    StackedSignTransform{T}

The structural sign map that the canonicalizer applies to one `Nonpositive`
block, carrying the **length of the vector it acts on**.

`NonpositiveToNonnegative` is deliberately length-agnostic: it is a block-local
map whose `dimension` is genuinely undefined (the frozen interface throws for
it). That is correct for blockwise reconstruction, but it means the map cannot
be placed in a whole-vector chain, where every entry must declare the length it
consumes and produces. This wrapper supplies only that missing declaration: the
primal, dual and ray operations are the sign map, delegated unchanged, so there
is exactly one implementation of the sign convention.

The length is a stored value rather than a type parameter, so a stack of these
is homogeneous in `T` and adding one never recompiles the vector holding it.
"""
struct StackedSignTransform{T<:AbstractFloat} <: AbstractCoordinateTransform{T}
    dimension::Int

    function StackedSignTransform{T}(dimension::Integer) where {T<:AbstractFloat}
        length_ = Int(dimension)
        length_ >= 1 || throw(ArgumentError(
            "StackedSignTransform requires a positive length, got $length_",
        ))
        # The default inner constructor, invoked explicitly: `StackedSignTransform{T}(x)`
        # here would recurse into this very method.
        return new{T}(length_)
    end
end

StackedSignTransform(dimension::Integer, ::Type{T}) where {T<:AbstractFloat} =
    StackedSignTransform{T}(dimension)

sign_map(transform::StackedSignTransform) = NonpositiveToNonnegative{eltype(transform)}()

# This is the declaration `NonpositiveToNonnegative` deliberately does not make,
# and the reason the wrapper exists.
dimension(transform::StackedSignTransform) = transform.dimension
scratch_requirements(::StackedSignTransform) = (primal=0, dual=0)

@inline function _stacked_sign_length_check(transform::StackedSignTransform, dest, src)
    length(dest) == transform.dimension && length(src) == transform.dimension ||
        throw(DimensionMismatch(
            "StackedSignTransform acts on length $(transform.dimension), got " *
            "destination $(length(dest)) and source $(length(src))",
        ))
    return nothing
end

for operation in (:forward_primal!, :backward_primal!, :forward_dual!,
                  :backward_dual!, :forward_primal_ray!, :backward_primal_ray!,
                  :forward_dual_ray!, :backward_dual_ray!)
    @eval function $operation(transform::StackedSignTransform, dest, src)
        _stacked_sign_length_check(transform, dest, src)
        return _nonpositive_sign_map!(dest, src)
    end
end

objective_offset(::StackedSignTransform{T}) where {T<:AbstractFloat} =
    ObjectiveOffset{T}(zero(T))

forward_affine!(transform::StackedSignTransform, A_dest, b_dest, A, b) =
    forward_affine!(sign_map(transform), A_dest, b_dest, A, b)
backward_affine!(transform::StackedSignTransform, A_dest, b_dest, A, b) =
    backward_affine!(sign_map(transform), A_dest, b_dest, A, b)

# ---------------------------------------------------------------------------
# Whole-chain replay
# ---------------------------------------------------------------------------

"""Reverse-order view of a chain, used for every canonical -> source replay."""
@inline _replay_reverse(transforms::Vector) = Iterators.reverse(transforms)

"""
    replay_primal!(stack, dest, src) -> dest

Original-coordinate primal from canonical coordinates, applying every
transform's `backward_primal!` in reverse chain order. Intermediate buffers are
sized from each transform's declared source/target dimensions, so a
dimension-changing reduction and a dimension-preserving coordinate map can
share one chain.
"""
function replay_primal!(stack::ReconstructionStack, dest, src)
    transforms = stack.transforms
    isempty(transforms) && (copyto!(dest, src); return dest)
    current = src
    owned = false
    for transform in _replay_reverse(transforms)
        # Walking canonical -> source, this transform's INPUT is its TARGET
        # space and its output is its SOURCE space: `backward_primal!` undoes
        # the transform.
        expected = target_primal_dimension(transform)
        expected == length(current) || throw(DimensionMismatch(
            "primal replay: transform $(typeof(transform)) consumes length " *
            "$expected but received $(length(current))",
        ))
        buffer = Vector{eltype(src)}(undef, source_primal_dimension(transform))
        backward_primal!(transform, buffer, current)
        current = buffer
        owned = true
    end
    length(dest) == length(current) || throw(DimensionMismatch(
        "primal replay destination length $(length(dest)) != reconstructed " *
        "length $(length(current))",
    ))
    owned && copyto!(dest, current)
    return dest
end

"""Original-coordinate dual from canonical coordinates (reverse chain order)."""
function replay_dual!(stack::ReconstructionStack, dest, src)
    transforms = stack.transforms
    isempty(transforms) && (copyto!(dest, src); return dest)
    current = src
    for transform in _replay_reverse(transforms)
        expected = target_dual_dimension(transform)
        expected == length(current) || throw(DimensionMismatch(
            "dual replay: transform $(typeof(transform)) consumes length " *
            "$expected but received $(length(current))",
        ))
        buffer = Vector{eltype(src)}(undef, source_dual_dimension(transform))
        backward_dual!(transform, buffer, current)
        current = buffer
    end
    length(dest) == length(current) || throw(DimensionMismatch(
        "dual replay destination length $(length(dest)) != reconstructed " *
        "length $(length(current))",
    ))
    copyto!(dest, current)
    return dest
end

"""Canonical primal from original coordinates (chain order, dimension-aware)."""
function replay_primal_forward!(stack::ReconstructionStack, dest, src)
    transforms = stack.transforms
    isempty(transforms) && (copyto!(dest, src); return dest)
    current = src
    for transform in transforms
        expected = source_primal_dimension(transform)
        expected == length(current) || throw(DimensionMismatch(
            "forward primal replay: transform $(typeof(transform)) expects " *
            "source length $expected but received $(length(current))",
        ))
        buffer = Vector{eltype(src)}(undef, target_primal_dimension(transform))
        forward_primal!(transform, buffer, current)
        current = buffer
    end
    length(dest) == length(current) || throw(DimensionMismatch(
        "forward primal replay destination length $(length(dest)) != " *
        "reconstructed length $(length(current))",
    ))
    copyto!(dest, current)
    return dest
end

"""Canonical dual from original coordinates (chain order, dimension-aware)."""

"""Replay an original-coordinate primal ray into canonical coordinates."""

"""Replay a canonical primal-infeasibility ray into original coordinates."""

"""Replay an original-coordinate dual ray into canonical coordinates."""

"""Replay a canonical dual-infeasibility ray into original coordinates."""

"""Compose every transform's objective offset in chain order."""
function objective_offset(stack::ReconstructionStack{T}) where {T<:AbstractFloat}
    total = zero(ObjectiveOffset{T})
    for transform in stack.transforms
        total = total + objective_offset(transform)
    end
    return total
end

# ---------------------------------------------------------------------------
# Equality elimination (ZeroCone rows with a rank-revealing QR)
# ---------------------------------------------------------------------------

"""
    EqualityElimination{T}

Exact elimination of `r` independent canonical equality (`:zero`) rows by
substituting the `r` determined primal variables.

The coordinate split is stored, never recomputed: `free_variables` and
`determined_variables` index the ORIGINAL canonical variable order, so the
transform is a permutation composed with an affine substitution and both
directions are exact replay. With `E x = f` the independent rows and

    x_D = x_p + R x_F,

the reduction is the coordinate change `x = x_p + Z x̂` with

    x̂ = x_F,   x_p[determined] = particular,   x_p[free] = 0,
    Z[free, :] = I,   Z[determined, :] = R,

and the canonical equality rows become `Â = A Z`, `b̂ = b - A x_p`
(the eliminated rows are structurally satisfied for every `x̂`).

Fields
- `rank`, `determined_variables`, `free_variables`: the split.
- `particular`, `substitution`: the affine map `x_D = particular + R x_F`.
- `row_transform`, `row_rhs`: the reduced equality rows `Â`, `b̂`.
- `free_embedding`: dual embedding — `y` in reduced row coordinates to full
  canonical row coordinates (zeros on the eliminated rows).
- `objective_offset_value`: the constant the elimination adds to the
  objective, `c[determined]' * particular`.

The coefficient matrix `A` may be `nothing` when the caller needs only the
coordinate split; every replay operation below works without it.
"""
struct EqualityElimination{T<:AbstractFloat} <: AbstractProgramReduction{T}
    original_variables::Int
    reduced_variables::Int
    original_rows::Int
    reduced_rows::Int
    rank::Int
    determined_variables::Vector{Int}
    free_variables::Vector{Int}
    particular::Vector{T}
    substitution::Matrix{T}
    row_transform::Union{Nothing,SparseMatrixCSC{T,Int}}
    row_rhs::Union{Nothing,Vector{T}}
    free_embedding::Vector{Int}
    objective_offset_value::T
end

function Base.show(io::IO, transform::EqualityElimination{T}) where {T}
    print(io, "EqualityElimination{", T, "}(rank=", transform.rank,
          ", variables ", transform.original_variables, " -> ",
          transform.reduced_variables, ", rows ", transform.original_rows,
          " -> ", transform.reduced_rows, ")")
end

# --- parity helper -----------------------------------------------------
#
# A rank decision must be the SAME in the transform and in the replay, so the
# tolerance is stored (it is a field of the transform, never re-derived at
# replay time). Nothing here relaxes a caller tolerance: the default is the
# standard `max(m,n) * eps(T) * largest-pivot` rank threshold, and a caller
# may supply their own.

function _elimination_rank_tolerance(::Type{T}, count::Int, largest) where {T}
    return T(max(count, 1)) * eps(T) * largest
end

"""
    _elimination_row_space(E, tol) -> (rank, permutation, upper)

Pivoted QR of `transpose(E)`; returns `(rank, permutation, upper)`.

`E` is `nrows x nvars` (one row per eliminated equality, one column per
variable). `permutation` is a permutation of ALL `nvars` variables: the `rank`
leading entries are the determined set and the rest are free.

Two traps this function exists to avoid, both of which silently produce a
WRONG transform rather than a rounding effect:

1. `LinearAlgebra.QRPivoted.p` is not a full permutation of the columns. Here
   `transpose(E)` is `nvars x nrows`, so `p` pivots the `nrows` ROWS of `E`
   (which variables carry the independent rows) and has length at most
   `nrows`; the remaining variables are appended explicitly.
2. `size(E)` is `(nrows, nvars)`, not `(nvars, nrows)`. Reading them the wrong
   way round makes the append loop iterate over the row count and the free set
   come out empty. That is asserted below rather than assumed.
"""
function _elimination_row_space(E::AbstractMatrix{T}, tol) where {T}
    nrows, nvars = size(E)
    transpose_e = Matrix{T}(transpose(E))
    decomposition = qr(transpose_e, ColumnNorm())
    upper = Matrix{T}(decomposition.R)
    pivoted = collect(Int, decomposition.p)
    length(pivoted) <= nrows || throw(AssertionError(
        "pivoted QR returned $(length(pivoted)) pivots for $nrows rows",
    ))
    permutation = copy(pivoted)
    for variable in 1:nvars
        variable in pivoted || push!(permutation, variable)
    end
    length(permutation) == nvars || throw(AssertionError(
        "equality-elimination pivot construction produced $(length(permutation)) " *
        "entries for $nvars variables",
    ))
    length(unique(permutation)) == nvars || throw(AssertionError(
        "equality-elimination pivot construction is not a permutation of 1:$nvars",
    ))
    if nrows == 0 || nvars == 0
        return 0, permutation, upper
    end
    diagonal = [abs(upper[i, i]) for i in 1:min(size(upper)...)]
    isempty(diagonal) && return 0, permutation, upper
    threshold = tol === nothing ?
        _elimination_rank_tolerance(T, max(nrows, nvars), maximum(diagonal)) : T(tol)
    rank = 0
    for value in diagonal
        value > threshold || break
        rank += 1
    end
    rank <= min(nrows, nvars) || throw(AssertionError(
        "equality-elimination rank $rank exceeds min($nrows, $nvars)",
    ))
    return rank, permutation, upper
end

"""
    equality_elimination(problem; tolerance=nothing) -> EqualityElimination or nothing

Build the equality-elimination transform for the canonical problem's `:zero`
rows. Returns `nothing` when there is nothing to eliminate (no `:zero` row, or
the independent set is empty), and throws when the independent rows are
inconsistent — an inconsistent equality set is a solver outcome, not a
transform, and must not be silently dropped.

The `:zero` rows come from the canonical cone layout, preserving the current
input math form: the caller's `A`, `b`, `c` are not modified.
"""
function equality_elimination(
    problem::CanonicalConicProgram{T};
    tolerance::Union{Nothing,Real}=nothing,
) where {T<:AbstractFloat}
    bits = problem.precision_bits
    A = problem.A
    b = problem.b
    n = canonical_num_variables(problem)
    m = canonical_num_slack(problem)
    zero_rows = Int[]
    for block in problem.cone_layout.blocks
        if block.cone === :zero
            for position in 0:(block.length - 1)
                push!(zero_rows, block.offset + position)
            end
        end
    end
    isempty(zero_rows) && return nothing

    E = Matrix{T}(A[zero_rows, :])
    f = owned_vector_copy(T, b[zero_rows]; precision_bits=bits)
    rank, permutation, upper = _elimination_row_space(E, tolerance)
    rank == 0 && return nothing
    rank <= min(length(zero_rows), n) || throw(ArgumentError(
        "equality elimination rank $rank exceeds the independent set",
    ))

    determined = permutation[1:rank]
    free = permutation[(rank + 1):end]
    E1 = E[:, free]
    E2 = E[:, determined]

    # x_D = particular + substitution * x_F, solved in the working arithmetic
    # through an LU factorization of the stored square block (reused for every
    # right-hand side, so the elimination costs one factorization).
    square = lu(E2)
    substitution = -(square \ E1)
    particular = square \ f

    offset_value = zero(T)
    for position in eachindex(determined)
        offset_value = _owned_arithmetic_eval(
            T,
            () -> offset_value + problem.c[determined[position]] * particular[position];
            precision_bits=bits,
        )
    end

    # Z is the primal embedding of the reduced variables into the full ones.
    Z = zeros(T, n, n - rank)
    for (column, variable) in enumerate(free)
        Z[variable, column] = one(T)
    end
    for (column, variable) in enumerate(determined)
        for (inner, free_variable) in enumerate(free)
            Z[variable, inner] = substitution[column, inner]
        end
    end

    reduced_rows = setdiff(collect(1:m), zero_rows)
    row_transform = owned_sparse_copy(
        T, sparse(A[reduced_rows, :] * Z); precision_bits=bits,
    )
    reduced_rhs_values = Vector{T}(undef, length(reduced_rows))
    for (position, row) in enumerate(reduced_rows)
        accumulator = b[row]
        for (inner, variable) in enumerate(determined)
            accumulator = _owned_arithmetic_eval(
                T,
                () -> accumulator - A[row, variable] * particular[inner];
                precision_bits=bits,
            )
        end
        reduced_rhs_values[position] = accumulator
    end

    free_embedding = collect(1:length(reduced_rows))

    return EqualityElimination{T}(
        n, n - rank, m, length(reduced_rows), rank,
        determined, free, particular, substitution,
        row_transform, reduced_rhs_values, free_embedding, offset_value,
    )
end

# --- dimensions --------------------------------------------------------

source_primal_dimension(transform::EqualityElimination) = transform.original_variables
target_primal_dimension(transform::EqualityElimination) = transform.reduced_variables
source_dual_dimension(transform::EqualityElimination) = transform.original_rows
target_dual_dimension(transform::EqualityElimination) = transform.reduced_rows
dimension(transform::EqualityElimination) = transform.reduced_variables
scratch_requirements(::EqualityElimination) = (primal=1, dual=0)

# --- primal ------------------------------------------------------------

@inline function _elimination_expand_primal(
    transform::EqualityElimination{T}, dest, src; affine::Bool=true,
) where {T}
    length(src) == transform.reduced_variables || throw(DimensionMismatch(
        "equality-elimination primal source length $(length(src)) != " *
        "$(transform.reduced_variables)",
    ))
    length(dest) == transform.original_variables || throw(DimensionMismatch(
        "equality-elimination primal destination length $(length(dest)) != " *
        "$(transform.original_variables)",
    ))
    for (column, variable) in enumerate(transform.free_variables)
        dest[variable] = src[column]
    end
    for (row, variable) in enumerate(transform.determined_variables)
        # A RAY is a direction: the affine offset `x_p` belongs to the point
        # map only. Adding it here would move the ray off the recession cone
        # (measured: 0.642857 where the homogeneous map gives 0.142857).
        accumulator = affine ? transform.particular[row] : zero(T)
        for (column, free_variable) in enumerate(transform.free_variables)
            coefficient = transform.substitution[row, column]
            iszero(coefficient) && continue
            accumulator += coefficient * src[column]
        end
        dest[variable] = accumulator
    end
    return dest
end

@inline function _elimination_reduce_primal(transform::EqualityElimination{T}, dest, src) where {T}
    length(src) == transform.original_variables || throw(DimensionMismatch(
        "equality-elimination primal source length $(length(src)) != " *
        "$(transform.original_variables)",
    ))
    length(dest) == transform.reduced_variables || throw(DimensionMismatch(
        "equality-elimination primal destination length $(length(dest)) != " *
        "$(transform.reduced_variables)",
    ))
    for (column, variable) in enumerate(transform.free_variables)
        dest[column] = src[variable]
    end
    return dest
end

backward_primal!(transform::EqualityElimination, dest, src) =
    _elimination_expand_primal(transform, dest, src)
forward_primal!(transform::EqualityElimination, dest, src) =
    _elimination_reduce_primal(transform, dest, src)

# Rays are homogeneous: same linear map, without the affine offset.
backward_primal_ray!(transform::EqualityElimination, dest, src) =
    _elimination_expand_primal(transform, dest, src; affine=false)
forward_primal_ray!(transform::EqualityElimination, dest, src) =
    _elimination_reduce_primal(transform, dest, src)

# --- dual --------------------------------------------------------------
#
# Eliminating `r` equality rows removes `r` ROW coordinates from the reduced
# problem (`reduced_rows = m - r`), so the dual map embeds the reduced dual
# back into the full row space with structural zeros on the eliminated rows
# and is the identity on the retained rows.

@inline function _elimination_embed_dual(transform::EqualityElimination{T}, dest, src) where {T}
    length(src) == transform.reduced_rows || throw(DimensionMismatch(
        "equality-elimination dual source length $(length(src)) != " *
        "$(transform.reduced_rows)",
    ))
    length(dest) == transform.original_rows || throw(DimensionMismatch(
        "equality-elimination dual destination length $(length(dest)) != " *
        "$(transform.original_rows)",
    ))
    fill!(dest, zero(T))
    for (position, row) in enumerate(transform.free_embedding)
        dest[row] = src[position]
    end
    return dest
end

@inline function _elimination_restrict_dual(transform::EqualityElimination{T}, dest, src) where {T}
    length(src) == transform.original_rows || throw(DimensionMismatch(
        "equality-elimination dual source length $(length(src)) != " *
        "$(transform.original_rows)",
    ))
    length(dest) == transform.reduced_rows || throw(DimensionMismatch(
        "equality-elimination dual destination length $(length(dest)) != " *
        "$(transform.reduced_rows)",
    ))
    for (position, row) in enumerate(transform.free_embedding)
        dest[position] = src[row]
    end
    return dest
end

backward_dual!(transform::EqualityElimination, dest, src) =
    _elimination_embed_dual(transform, dest, src)
forward_dual!(transform::EqualityElimination, dest, src) =
    _elimination_restrict_dual(transform, dest, src)

backward_dual_ray!(transform::EqualityElimination, dest, src) =
    _elimination_embed_dual(transform, dest, src)
forward_dual_ray!(transform::EqualityElimination, dest, src) =
    _elimination_restrict_dual(transform, dest, src)

# --- objective offset --------------------------------------------------

objective_offset(transform::EqualityElimination{T}) where {T<:AbstractFloat} =
    ObjectiveOffset{T}(transform.objective_offset_value)

"""
    objective_shift(transform::EqualityElimination)

The frozen `objective_shift` scalar is the setup-time shift of the objective's
constant term, which equality elimination does not perform (it performs a
variable substitution). The constant it moves is reported through
[`objective_offset`](@ref) instead, and this method is zero so that a stack
mixer cannot silently add the two.
"""
objective_shift(::EqualityElimination{T}) where {T<:AbstractFloat} = zero(T)

# --- invalidation ------------------------------------------------------

"""
    invalidation(transform::EqualityElimination) -> NamedTuple

What a mutation of the admitted problem invalidates, per transform field. The
distinction is load-bearing: an elimination built only from `E` and `f`
(columns 1-3) survives a change to the objective or to the non-equality rows,
because the coordinate split does not depend on either.
"""
function invalidation(transform::EqualityElimination)
    return (
        equality_coefficients = true,
        equality_rhs = true,
        cone_layout = true,
        objective = false,
        non_equality_rows = false,
    )
end

"""True when the stored elimination is still valid for the given canonical data."""
function elimination_is_valid(
    transform::EqualityElimination{T}, A, b, c, layout,
) where {T<:AbstractFloat}
    transform.original_rows == size(A, 1) || return false
    transform.original_variables == size(A, 2) || return false
    length(b) == size(A, 1) || return false
    length(c) == size(A, 2) || return false
    layout.dimension == size(A, 1) || return false
    return true
end

# ---------------------------------------------------------------------------
# The owned transform stack
# ---------------------------------------------------------------------------

"""
    TransformStack{T}

The compiled problem's owned chain of reversible transforms, in
source-to-canonical order (`ReconstructionStack` semantics). A type alias
would not add ownership; this constructor makes the ownership explicit: the
stack and every transform in it are immutable, and the vector is a fresh copy
that the caller cannot alias.
"""
function TransformStack(
    transforms::AbstractVector{<:AbstractProgramTransform{T}},
) where {T<:AbstractFloat}
    return ReconstructionStack{T}(transforms)
end

TransformStack(::Type{T}) where {T<:AbstractFloat} = ReconstructionStack{T}()

"""
    stack_objective_replay(stack, canonical_value) -> value

Original-coordinate objective value from a canonical one.
"""
function stack_objective_replay(stack::ReconstructionStack, canonical_value)
    return replay_objective(objective_offset(stack), canonical_value)
end

"""
    stack_objective_replay_backward(stack, original_value) -> value

Canonical objective value from an original-coordinate one.
"""
function stack_objective_replay_backward(stack::ReconstructionStack, original_value)
    return replay_objective_backward(objective_offset(stack), original_value)
end

"""
    replay_invariants(transform, primal, dual, transformed_primal, transformed_dual;
                      atol, rtol) -> NamedTuple

The two invariants an invertible transform must preserve, checked without
choosing a tolerance for the caller:

- `pairing`: `<s, y>` in source coordinates equals `<ŝ, ŷ>` in target
  coordinates (a transform that changes the pairing scale is not admissible);
- `roundtrip`: `backward(forward(x)) == x` on both sides.
"""
function replay_invariants(
    transform::AbstractProgramTransform{T}, primal, dual,
    transformed_primal, transformed_dual;
    atol=nothing, rtol=nothing, tol=nothing,
) where {T<:AbstractFloat}
    pairing = verify_pairing_invariant(
        transform, primal, dual, transformed_primal, transformed_dual;
        atol=atol, rtol=rtol, tol=tol,
    )
    reversed_primal = similar(transformed_primal)
    reversed_dual = similar(transformed_dual)
    backward_primal!(transform, reversed_primal, transformed_primal)
    backward_dual!(transform, reversed_dual, transformed_dual)
    primal_roundtrip = all(isapprox.(reversed_primal, primal; atol=atol === nothing ? sqrt(eps(T)) : atol,
                                   rtol=rtol === nothing ? sqrt(eps(T)) : rtol))
    dual_roundtrip = all(isapprox.(reversed_dual, dual; atol=atol === nothing ? sqrt(eps(T)) : atol,
                                   rtol=rtol === nothing ? sqrt(eps(T)) : rtol))
    return (pairing=pairing, primal_roundtrip=primal_roundtrip, dual_roundtrip=dual_roundtrip)
end
