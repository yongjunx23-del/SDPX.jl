#=
    SDPX <-> QDLDL symmetric companion inertia candidate adapter.

    Candidate-adapter scope — docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md:
      * QDLDL 0.4.x is registered as an optional SDPX weak dependency /
        extension in Project.toml, and this module is the candidate adapter:
        QDLDL provides the generic sparse LDLT used to certify the inertia
        of the *symmetric signed-regularized quasidefinite companion* of the
        frozen HSD Newton system (src/kkt/system.jl and
        src/kkt/expanded_quasidefinite.jl).
      * This module is inertia/regularization evidence ONLY. It never solves
        the exact nonsymmetric condensed operator, whose (x,tau) blocks are
        skew adjoints (src/kkt/expanded_quasidefinite.jl:1-7). The adapter
        refuses nonsymmetric input instead of letting QDLDL silently factor
        the wrong upper triangle.
      * The symbolic companion pattern is captured once
        (`companion_matrix`); numeric refactor reuses it through
        `QDLDL.update_values!` + `QDLDL.refactor!`.
      * The expected inertia is recomputed from each `NewtonSystem` through
        `SDPX.expected_expanded_inertia`; it is never inferred from the
        observed factor.
      * The adapter is still NOT wired into any SDPX KKT or HSD route and is
        not part of the public API; this registration task adds the
        Project.toml weakdep/extension entries and the capability facts in
        src/la/sparse_capabilities.jl only. The extension calls the package
        directly (no LinearSolve/SciMLBase).

    QDLDL 0.4.1 API facts this adapter pins (see the evidence doc):
      * `qdldl(A; Dsigns, regularize_eps, regularize_delta)` preserves the
        scalar type of A for Float64, MultiFloat, and BigFloat.
      * `Dsigns` is a sign vector in *input* coordinates; QDLDL permutes it
        internally, so post-factor `workspace.D` signs must be compared in
        `Dsigns[F.perm]` order.
      * A pivot with `Dsigns[k]*D[k] < regularize_eps` is replaced by
        `regularize_delta * Dsigns[k]` and counted by `regularized_entries`.
      * `update_values!` addresses the matrix in *input upper-triangular
        linear indices* (mapped internally through `AtoPAPt`); this adapter
        records the (row,col) -> linear index map at symbolic setup.
      * `solve!` is single-vector only in 0.4.1: a Matrix RHS throws
        `ReadOnlyMemoryError` (with AMD perm) or silently solves only the
        first column (with perm=nothing); the adapter loops per column.
      * Exactly-zero pivots and structurally empty columns abort with
        `ErrorException`; the adapter classifies and fails closed.
=#
module SDPXQDLDLExt

using SDPX
using QDLDL
using SparseArrays
using LinearAlgebra

# Registered in src/la/sparse_capabilities.jl: loading this extension is
# the only way the QDLDL provider becomes available; until then core
# reports it absent (fail closed / honest unavailable).
SDPX.sparse_provider_loaded(::Val{:qdldl}) = true

export QDLDLCompanion,
    QDLDL_COMPANION_READY,
    QDLDL_COMPANION_FACTORED,
    QDLDL_COMPANION_FACTOR_FAILED,
    QDLDL_COMPANION_WRONG_INERTIA,
    qdldl_companion,
    qdldl_companion_raw,
    companion_matrix,
    companion_inertia,
    companion_dsigns_match,
    companion_regularized_entries,
    companion_status,
    companion_failure,
    companion_update!,
    companion_refactor!,
    companion_solve!,
    companion_residual

"""Companion factor state; mirrors the fail-closed ladder of the dense route."""
@enum QDLDLCompanionStatus begin
    QDLDL_COMPANION_READY
    QDLDL_COMPANION_FACTORED
    QDLDL_COMPANION_FACTOR_FAILED
    QDLDL_COMPANION_WRONG_INERTIA
end

"""
    QDLDLCompanion{T}

Narrow sparse adapter handle for the signed-regularized symmetric
quasidefinite companion of a `NewtonSystem`. `matrix` is the full symmetric
companion used for residual certification, `triu_matrix` is the fixed
upper-triangular pattern handed to QDLDL, and `entry_index` maps upper
triangle (row, col) coordinates to the linear indices consumed by
`QDLDL.update_values!`.
"""
mutable struct QDLDLCompanion{T<:AbstractFloat}
    n::Int
    m::Int
    dimension::Int
    matrix::SparseMatrixCSC{T,Int}
    triu_matrix::SparseMatrixCSC{T,Int}
    entry_index::Dict{Tuple{Int,Int},Int}
    matrix_entry_index::Dict{Tuple{Int,Int},Int}
    factor::Union{Nothing,QDLDL.QDLDLFactorisation{T,Int}}
    Dsigns::Vector{Int}
    regularize_eps::T
    regularize_delta::T
    expected::SDPX.KKTInertia
    status::QDLDLCompanionStatus
    failure::Symbol
    regularized_entries::Int
end

companion_status(companion::QDLDLCompanion) = companion.status
companion_failure(companion::QDLDLCompanion) = companion.failure
companion_regularized_entries(companion::QDLDLCompanion) =
    companion.status == QDLDL_COMPANION_FACTORED &&
    companion.factor !== nothing ? companion.regularized_entries : nothing

function _classify_qdldl_error(err)
    message = sprint(showerror, err)
    occursin("Zero entry in D", message) && return :zero_pivot
    occursin("empty column", message) && return :non_quasidefinite
    return :unexpected
end

function _matrix_row_scale(matrix::SparseMatrixCSC{T}) where {T}
    scale = zero(T)
    row_sums = zeros(T, size(matrix, 1))
    @inbounds for column in axes(matrix, 2)
        for position in matrix.colptr[column]:(matrix.colptr[column + 1] - 1)
            row_sums[matrix.rowval[position]] += abs(matrix.nzval[position])
        end
    end
    @inbounds for row in eachindex(row_sums)
        scale = max(scale, row_sums[row])
    end
    return scale
end

"""
    companion_matrix(system; regularization=nothing)

Assemble the signed-regularized symmetric quasidefinite companion of the
frozen `NewtonSystem`, mirroring `assemble_expanded_kkt!` +
`_assemble_regularized!` + `_freeze_symmetric_companion!` in
src/kkt/expanded_quasidefinite.jl with sparse storage:

    [  δI    A'    c ]
    [  A    -H-δI  -b ]
    [  c'   -b'   -κ/τ-δ ]

The (x,tau) coupling is symmetric here (the exact condensed operator keeps
the skew `-c'` and is out of scope for this adapter). Returns the full
symmetric matrix, the fixed upper-triangular pattern, the (row,col) -> linear
index map for `update_values!`, and the operator scale.
"""
function companion_matrix(
    system::SDPX.NewtonSystem{T}; regularization::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}
    m, n = size(system.A)
    n >= 1 && m >= 1 || throw(ArgumentError(
        "companion prototype requires n >= 1 and m >= 1",
    ))
    dimension = n + m + 1
    cone_dimension = SDPX.cone_dimension(system.cone)
    cone_dimension == m || throw(DimensionMismatch(
        "cone dimension does not match rows of A",
    ))
    operator = system.cone.operator
    eltype(operator) === T || throw(ArgumentError(
        "cone operator scalar type does not match the Newton system",
    ))

    rows = Int[]
    cols = Int[]
    values = T[]
    push_entry(row::Int, col::Int, value) = begin
        push!(rows, row)
        push!(cols, col)
        push!(values, value)
    end

    A_sparse = system.A isa SparseMatrixCSC ? system.A : sparse(system.A)
    eltype(A_sparse) === T || throw(ArgumentError(
        "A scalar type does not match the Newton system",
    ))
    # (x, y) coupling: upper-triangle coordinate (j, n+i) holds A[i,j].
    @inbounds for column in axes(A_sparse, 2)
        for position in A_sparse.colptr[column]:(A_sparse.colptr[column + 1] - 1)
            push_entry(column, n + A_sparse.rowval[position], A_sparse.nzval[position])
        end
    end
    # (x, tau) coupling: symmetric companion mirrors the upper x/tau sign.
    @inbounds for j in 1:n
        push_entry(j, dimension, system.c[j])
    end
    # (y, y) block: -H upper triangle, and (y, tau) coupling: -b.
    @inbounds for i in 1:m
        push_entry(n + i, dimension, -system.b[i])
        for j in i:m
            value = -operator[i, j]
            iszero(value) || push_entry(n + i, n + j, value)
        end
    end
    # (tau, tau): -kappa/tau.
    push_entry(dimension, dimension, -system.kappa / system.tau)

    unregularized = sparse(rows, cols, values, dimension, dimension)
    raw_full = unregularized +
               transpose(unregularized) -
               Diagonal(diag(unregularized))
    scale = max(_matrix_row_scale(raw_full), one(T))
    regularization_value = regularization === nothing ?
        sqrt(eps(T)) * scale : T(regularization)
    isfinite(regularization_value) || throw(ArgumentError(
        "companion regularization must be finite",
    ))

    # Signed regularization: +delta on the x block, -delta on (y,tau).
    # `sparse` sums duplicate coordinates, so the diagonal entries combine
    # with the -H and -kappa/tau entries already pushed.
    @inbounds for j in 1:n
        push_entry(j, j, regularization_value)
    end
    @inbounds for index in (n + 1):dimension
        push_entry(index, index, -regularization_value)
    end

    triu_matrix = sparse(rows, cols, values, dimension, dimension)
    matrix = triu_matrix + transpose(triu_matrix) - Diagonal(diag(triu_matrix))
    issymmetric(matrix) || throw(ArgumentError(
        "companion assembly produced a nonsymmetric matrix",
    ))

    entry_index = Dict{Tuple{Int,Int},Int}()
    @inbounds for column in axes(triu_matrix, 2)
        for position in
            triu_matrix.colptr[column]:(triu_matrix.colptr[column + 1] - 1)
            entry_index[(triu_matrix.rowval[position], column)] = position
        end
    end
    # Same linear-index map for the full symmetric matrix, used by
    # companion_update! to keep the residual-certification matrix in sync
    # without ever changing the explicit CSC pattern.
    matrix_entry_index = Dict{Tuple{Int,Int},Int}()
    @inbounds for column in axes(matrix, 2)
        for position in matrix.colptr[column]:(matrix.colptr[column + 1] - 1)
            matrix_entry_index[(matrix.rowval[position], column)] = position
        end
    end
    return (
        matrix=matrix, triu_matrix=triu_matrix, entry_index=entry_index,
        matrix_entry_index=matrix_entry_index,
        scale=scale, regularization=regularization_value,
    )
end

function _certify!(companion::QDLDLCompanion)
    F = companion.factor
    F === nothing && return false
    companion.regularized_entries = QDLDL.regularized_entries(F)
    positive = QDLDL.positive_inertia(F)
    observed = SDPX.KKTInertia(positive, companion.dimension - positive, 0)
    if observed != companion.expected
        companion.status = QDLDL_COMPANION_WRONG_INERTIA
        companion.failure = :wrong_inertia
        return false
    end
    companion.status = QDLDL_COMPANION_FACTORED
    companion.failure = :none
    return true
end

function _factor!(companion::QDLDLCompanion{T}) where {T}
    F = nothing
    try
        F = QDLDL.qdldl(
            companion.triu_matrix;
            Dsigns=companion.Dsigns,
            regularize_eps=companion.regularize_eps,
            regularize_delta=companion.regularize_delta,
        )
    catch err
        reason = _classify_qdldl_error(err)
        companion.factor = nothing
        companion.status = QDLDL_COMPANION_FACTOR_FAILED
        companion.failure = reason
        return false
    end
    F === nothing && return false
    companion.factor = F
    return _certify!(companion)
end

"""
    qdldl_companion_raw(matrix, Dsigns, expected; regularize_eps, regularize_delta)

Low-level matrix entry used by the spike to exercise fail-closed modes
(nonsymmetric input, structurally empty columns, exactly-zero pivots, and a
policy Dsigns vector that contradicts the expected inertia). Always returns a
status; never throws for QDLDL factorization failures. `Dsigns === nothing`
disables signed regularization so that exactly-zero pivots fail closed with
`:zero_pivot` (with a Dsigns vector they are floored to plus-or-minus
`regularize_delta` and counted, which is QDLDL's documented regularization
contract).
"""
function qdldl_companion_raw(
    matrix::SparseMatrixCSC{T,Int}, Dsigns::Union{Nothing,Vector{Int}},
    expected::SDPX.KKTInertia;
    regularize_eps::T, regularize_delta::T,
) where {T<:AbstractFloat}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension || throw(DimensionMismatch(
        "companion matrix must be square",
    ))
    Dsigns === nothing || length(Dsigns) == dimension || throw(DimensionMismatch(
        "Dsigns length does not match companion dimension",
    ))
    issymmetric(matrix) || return (
        factor=nothing, status=QDLDL_COMPANION_FACTOR_FAILED,
        failure=:nonsymmetric,
    )
    try
        F = QDLDL.qdldl(
            triu(matrix);
            Dsigns=Dsigns,
            regularize_eps=regularize_eps,
            regularize_delta=regularize_delta,
        )
        positive = QDLDL.positive_inertia(F)
        observed = SDPX.KKTInertia(positive, dimension - positive, 0)
        if observed != expected
            return (
                factor=F, status=QDLDL_COMPANION_WRONG_INERTIA,
                failure=:wrong_inertia,
            )
        end
        return (
            factor=F, status=QDLDL_COMPANION_FACTORED, failure=:none,
        )
    catch err
        reason = _classify_qdldl_error(err)
        return (
            factor=nothing, status=QDLDL_COMPANION_FACTOR_FAILED,
            failure=reason,
        )
    end
end

"""
    qdldl_companion(system; regularization, regularize_eps, regularize_delta)

Symbolic setup + numeric factor of the signed-regularized companion of
`system`. The Dsigns vector is derived from the frozen block structure
(`+1` on the `x` block, `-1` on the combined `(y,tau)` block), and the
expected inertia is recomputed from `system` via
`SDPX.expected_expanded_inertia`. This factor certifies inertia only; it
never replaces the exact nonsymmetric Newton solve.
"""
function qdldl_companion(
    system::SDPX.NewtonSystem{T};
    regularization::Union{Nothing,T}=nothing,
    regularize_eps::Union{Nothing,T}=nothing,
    regularize_delta::Union{Nothing,T}=nothing,
) where {T<:AbstractFloat}
    assembled = companion_matrix(system; regularization=regularization)
    reg = assembled.regularization
    eps_value = regularize_eps === nothing ? reg : T(regularize_eps)
    delta_value = regularize_delta === nothing ? T(10) * reg : T(regularize_delta)
    m, n = size(system.A)
    dimension = n + m + 1
    Dsigns = [ones(Int, n); -ones(Int, m + 1)]
    companion = QDLDLCompanion{T}(
        n, m, dimension, assembled.matrix, assembled.triu_matrix,
        assembled.entry_index, assembled.matrix_entry_index, nothing,
        Dsigns, eps_value, delta_value,
        SDPX.expected_expanded_inertia(system),
        QDLDL_COMPANION_READY, :none, 0,
    )
    _factor!(companion)
    return companion
end

"""
    companion_update!(companion, changes)

Apply (row, col, value) upper-triangle updates to the fixed pattern, then
hand the linear indices to `QDLDL.update_values!`. A coordinate outside the
captured pattern fails closed instead of silently changing the sparsity.

The factor and inertia authority are invalidated **before** any mutation:
the status drops to `QDLDL_COMPANION_READY` with `failure = :stale_factor`,
so solves and inertia reads fail closed until `companion_refactor!`
re-certifies. Values are written through `nzval` at the recorded linear
indices so that zero updates never delete stored entries (the explicit CSC
pattern and both index maps are invariant).
"""
function companion_update!(
    companion::QDLDLCompanion{T}, changes::AbstractVector,
) where {T<:AbstractFloat}
    F = companion.factor
    F === nothing && return false
    isempty(changes) && return true
    # Invalidate factor and inertia authority BEFORE any mutation: between an
    # update and the next refactor the factor values are stale, so solves and
    # inertia reads must fail closed instead of reporting the pre-update
    # factor. The factor object is retained (refactor! needs it) but its
    # authority is revoked until companion_refactor! re-certifies.
    companion.status = QDLDL_COMPANION_READY
    companion.failure = :stale_factor
    indices = Int[]
    values = T[]
    @inbounds for change in changes
        row, col, value = change
        key = row <= col ? (row, col) : (col, row)
        index = get(companion.entry_index, key, 0)
        index == 0 && throw(ArgumentError(
            "update coordinate ($row, $col) is not in the fixed companion pattern",
        ))
        converted = T(value)
        push!(indices, index)
        push!(values, converted)
        # Direct nzval assignment preserves the explicit CSC pattern and the
        # linear-index maps: sparse setindex! with a zero value deletes the
        # stored entry on some Julia versions, which would shift every later
        # linear index consumed by QDLDL.update_values!.
        companion.triu_matrix.nzval[index] = converted
        # The full symmetric matrix stores both triangles; keep both stored
        # entries in sync (the diagonal maps to itself).
        matrix_index = get(companion.matrix_entry_index, key, 0)
        matrix_index == 0 && throw(ArgumentError(
            "update coordinate ($row, $col) is not in the full companion pattern",
        ))
        companion.matrix.nzval[matrix_index] = converted
        if key[1] != key[2]
            mirror_index = get(companion.matrix_entry_index, (key[2], key[1]), 0)
            mirror_index == 0 && throw(ArgumentError(
                "update coordinate ($row, $col) mirror is not in the full companion pattern",
            ))
            companion.matrix.nzval[mirror_index] = converted
        end
    end
    QDLDL.update_values!(F, indices, values)
    return true
end

"""
    companion_refactor!(companion)

Numeric refactor reusing the fixed symbolic pattern
(`QDLDL.refactor!`), then re-certify expected inertia.
"""
function companion_refactor!(companion::QDLDLCompanion)
    F = companion.factor
    F === nothing && return false
    try
        QDLDL.refactor!(F)
    catch err
        reason = _classify_qdldl_error(err)
        companion.status = QDLDL_COMPANION_FACTOR_FAILED
        companion.failure = reason
        return false
    end
    return _certify!(companion)
end

"""Certified inertia `(positive, negative, zero)` from the QDLDL D signs."""
function companion_inertia(companion::QDLDLCompanion)
    # Inertia authority is revoked between update and refactor: only a
    # certified factor may report inertia.
    companion.status == QDLDL_COMPANION_FACTORED || return SDPX.KKTInertia(
        0, 0, companion.dimension,
    )
    F = companion.factor
    F === nothing && return SDPX.KKTInertia(0, 0, companion.dimension)
    positive = QDLDL.positive_inertia(F)
    return SDPX.KKTInertia(positive, companion.dimension - positive, 0)
end

"""
    companion_dsigns_match(companion)

Verify that every post-factor `workspace.D` sign agrees with `Dsigns` in
QDLDL's internal post-permutation order. This is the D-sign evidence that the
signed regularization enforced the expected block signs.
"""
function companion_dsigns_match(companion::QDLDLCompanion)
    companion.status == QDLDL_COMPANION_FACTORED || return false
    F = companion.factor
    F === nothing && return false
    D = F.workspace.D
    perm = F.perm
    expected_signs =
        perm === nothing ? companion.Dsigns : companion.Dsigns[perm]
    @inbounds for index in 1:companion.dimension
        positive = D[index] > zero(eltype(D))
        positive == (expected_signs[index] > 0) || return false
    end
    return true
end

"""Solve one RHS vector through the certified factor."""
function companion_solve!(
    destination::AbstractVector{T}, companion::QDLDLCompanion{T},
    rhs::AbstractVector{T},
) where {T<:AbstractFloat}
    F = companion.factor
    F === nothing && return false
    companion.status == QDLDL_COMPANION_FACTORED || return false
    length(destination) == length(rhs) == companion.dimension ||
        throw(DimensionMismatch("companion solve vector dimension mismatch"))
    copyto!(destination, rhs)
    QDLDL.solve!(F, destination)
    return all(isfinite, destination)
end

"""
Solve a multi-RHS panel. QDLDL 0.4.1 `solve!` is single-vector only (a
Matrix RHS throws `ReadOnlyMemoryError` or silently solves only the first
column), so this adapter loops per column.
"""
function companion_solve!(
    destination::AbstractMatrix{T}, companion::QDLDLCompanion{T},
    rhs::AbstractMatrix{T},
) where {T<:AbstractFloat}
    F = companion.factor
    F === nothing && return false
    companion.status == QDLDL_COMPANION_FACTORED || return false
    size(destination) == size(rhs) || throw(DimensionMismatch(
        "companion solve panel dimensions disagree",
    ))
    size(rhs, 1) == companion.dimension || throw(DimensionMismatch(
        "companion solve panel row dimension mismatch",
    ))
    # Owned copy of the RHS panel: destination and rhs may alias or overlap
    # (e.g. shifted views of one buffer), and a per-column in-place solve
    # would otherwise read RHS columns that an earlier column already
    # overwrote. The vector solve! is unaffected (copyto! handles overlap).
    rhs_owned = copy(rhs)
    @inbounds for column in axes(rhs, 2)
        destination_view = view(destination, :, column)
        rhs_view = view(rhs_owned, :, column)
        copyto!(destination_view, rhs_view)
        QDLDL.solve!(F, destination_view)
    end
    return all(isfinite, destination)
end

"""
    companion_residual(companion, solution, rhs)

Infinity-norm residual of the full symmetric companion against the original
scalar type. `solution` may be a vector or a matrix panel.
"""
function companion_residual(
    companion::QDLDLCompanion{T}, solution::AbstractVecOrMat{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    residual = companion.matrix * solution - rhs
    return norm(residual, Inf)
end

end # module SDPXQDLDLExt
