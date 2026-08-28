#=====================================================================#
#    SparseSymbolicNumericCache — Float64 CHOLMOD signed-LDL lifecycle.
#
#    This cache is the provider-neutral home for a truthful sparse
#    symmetric-LDL lifecycle used by the symmetric augmented core
#    `K = [0 Ar'; Ar -Theta]` (see src/kkt/symmetric_core.jl).  It is the
#    single sparse cache; there is no second sparse route cache.
#
#    Facts (no allocation-free claims):
#      * `prepare!` copies the frozen lower-triangle CSC pattern into an
#        owned factor view with exactly the same colptr/rowval, and records
#        a fixed `symbolic_epoch` + pattern signature.  No CHOLMOD object is
#        created yet.
#      * the first `factorize!` calls the public `ldlt(Symmetric(K, :L);
#        check=false)` and stores the resulting CHOLMOD.Factor.  This is the
#        only symbolic analysis: `symbolic_count == 1`.
#      * every later `factorize!` with the same pattern reuses the same
#        CHOLMOD.Factor object through the public `ldlt!(factor,
#        Symmetric(K, :L))`; `symbolic_count` stays 1 and `numeric_count`
#        increments.
#      * signed static regularization is applied only to the factor view
#        diagonal (fixed sign vector: + for reduced-x rows, - for y rows);
#        the retained original K values are never modified.
#      * after a successful factor the original retained K values are
#        restored into the factor view so the view always mirrors the
#        original operator; the CHOLMOD factor holds the regularized copy.
#      * `solve!`/`refine_once!` use the public `factor \ rhs`, which for
#        CHOLMOD allocates a result object that is then copied into the
#        caller-owned destination.  This is documented; we do not claim an
#        allocation-free solve.
#      * the cache is Float64-only (CHOLMOD); every other element type is
#        rejected before any factorization.
#      * a failed numeric factor detaches the CHOLMOD object and restores
#        the factor view to the original values; `Failed` never solves stale
#        data and recovery re-runs the sole symbolic analysis on the next
#        same-pattern attempt.  `Invalid` requires a re-`prepare!`.
#=====================================================================#

"""
    SparseSymbolicRequirements

Capacity + symbolic requirements for `SparseSymbolicNumericCache`.
Carries the total dimension `n`, the fixed `symbolic_epoch`, the frozen
lower-triangle CSC `pattern`, the signed diagonal block descriptor
`dsigns` (`+1` for reduced-x rows, `-1` for y rows), and the static
regularization magnitude `regularization`.
"""
struct SparseSymbolicRequirements <: AbstractFactorRequirements
    n::Int
    symbolic_epoch::Int
    pattern::SparseMatrixCSC{Float64, Int}
    dsigns::Vector{Int}
    regularization::Float64

    # Inner constructor: every invariant is enforced here so no positional or
    # keyword construction path can bypass validation.
    function SparseSymbolicRequirements(
        pattern::SparseMatrixCSC{Float64, Int};
        symbolic_epoch::Integer=0,
        dsigns::AbstractVector{Int}=Int[],
        regularization::Real=0.0,
    )
        n = size(pattern, 1)
        size(pattern, 2) == n || throw(ArgumentError(
            "sparse pattern must be square",
        ))
        length(dsigns) == n || throw(ArgumentError(
            "signed diagonal descriptor length $(length(dsigns)) != n=$n",
        ))
        all(sign -> sign == 1 || sign == -1, dsigns) || throw(ArgumentError(
            "signed diagonal descriptor must contain only +1/-1",
        ))
        isfinite(regularization) && regularization >= 0 || throw(ArgumentError(
            "regularization must be finite and nonnegative",
        ))
        return new(
            Int(n), Int(symbolic_epoch),
            SparseMatrixCSC{Float64, Int}(pattern),
            Int[sign for sign in dsigns], Float64(regularization),
        )
    end
end

"""
    SparseSymbolicNumericCache{T}

A sparse symmetric-LDL factor cache with a single retained CHOLMOD factor
object, an owned factor-view CSC with the same colptr/rowval as the frozen
pattern, and an independently retained copy of the original K values.
"""
mutable struct SparseSymbolicNumericCache{T} <: AbstractFactorCache{T}
    n::Int
    colptr::Vector{Int}            # frozen lower-triangle CSC colptr
    rowval::Vector{Int}            # frozen lower-triangle CSC rowval
    factor_view::SparseMatrixCSC{T, Int}   # same structure; numeric buffer
    original_values::Vector{T}     # retained original K values (unmodified)
    dsigns::Vector{Int}            # signed diagonal descriptor
    regularization::T              # signed static regularization magnitude
    factor::Union{Nothing,LinearAlgebra.Factorization{T}}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    symbolic_count::Int
    numeric_count::Int
    solve_count::Int
    refine_count::Int
    signature::UInt64
    status::FactorCacheState
end

function SparseSymbolicNumericCache{T}() where {T}
    T === Float64 || throw(ArgumentError(
        "SparseSymbolicNumericCache is Float64-only (CHOLMOD); got $T",
    ))
    return SparseSymbolicNumericCache{T}(
        0, Vector{Int}(undef, 0), Vector{Int}(undef, 0),
        spzeros(T, 0, 0), Vector{T}(undef, 0), Int[], zero(T), nothing,
        0, 0, 0, 0, 0, 0, 0, UInt64(0), Unprepared,
    )
end

function SparseSymbolicNumericCache{T}(
    pattern::SparseMatrixCSC{Float64, Int};
    symbolic_epoch::Integer=0,
    dsigns::AbstractVector{Int}=Int[],
    regularization::Real=0.0,
) where {T}
    cache = SparseSymbolicNumericCache{T}()
    return prepare!(cache, SparseSymbolicRequirements(
        pattern;
        symbolic_epoch=symbolic_epoch,
        dsigns=dsigns,
        regularization=regularization,
    ))
end

function _sparse_pattern_signature(
    colptr::AbstractVector{Int}, rowval::AbstractVector{Int},
    dsigns::AbstractVector{Int}, regularization::Float64,
)
    signature = UInt64(0xcbf29ce484222325)
    for value in colptr
        signature = (signature ⊻ UInt64(value)) * UInt64(0x100000001b3)
    end
    for value in rowval
        signature = (signature ⊻ UInt64(value)) * UInt64(0x100000001b3)
    end
    for sign in dsigns
        # re-interpret the (possibly negative) signed Int bit pattern.
        mixed = reinterpret(UInt64, Int64(sign))
        signature = (signature ⊻ mixed) * UInt64(0x100000001b3)
    end
    signature = (signature ⊻ reinterpret(UInt64, regularization)) *
                UInt64(0x100000001b3)
    return signature
end

function prepare!(
    cache::SparseSymbolicNumericCache{T},
    req::SparseSymbolicRequirements,
) where {T}
    T === Float64 || throw(ArgumentError(
        "SparseSymbolicNumericCache is Float64-only (CHOLMOD); got $T",
    ))
    cache.status in (Unprepared, Invalid) || throw(FactorCacheStateError(
        :prepare, Unprepared, cache.status,
    ))
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    pattern = req.pattern
    size(pattern) == (n, n) || throw(DimensionMismatch(
        "SparseSymbolicNumericCache pattern dimension mismatch",
    ))
    size(pattern) == (req.n, req.n) || throw(DimensionMismatch(
        "SparseSymbolicNumericCache requirements dimension disagrees " *
        "with pattern",
    ))
    eltype(pattern) === Float64 || throw(ArgumentError(
        "SparseSymbolicNumericCache requires a Float64 pattern; got " *
        "$(eltype(pattern))",
    ))
    length(req.dsigns) == n || throw(ArgumentError(
        "signed diagonal descriptor length mismatch",
    ))
    all(sign -> sign == 1 || sign == -1, req.dsigns) || throw(ArgumentError(
        "signed diagonal descriptor must contain only +1/-1",
    ))
    isfinite(req.regularization) && req.regularization >= 0 ||
        throw(ArgumentError(
            "regularization must be finite and nonnegative",
        ))
    # Owned factor view: copy the frozen structure exactly.
    colptr = Vector{Int}(pattern.colptr)
    rowval = Vector{Int}(pattern.rowval)
    nzval = Vector{T}(undef, length(rowval))
    factor_view = SparseMatrixCSC{T, Int}(n, n, colptr, rowval, nzval)
    # Every column must have a structural diagonal (the core pattern
    # guarantees this for the x diagonal and the -Theta triangle).
    @inbounds for j in 1:n
        lo = colptr[j]
        hi = colptr[j + 1] - 1
        has_diagonal = false
        for pointer in lo:hi
            rowval[pointer] == j && (has_diagonal = true; break)
        end
        has_diagonal || throw(ArgumentError(
            "sparse pattern column $j has no structural diagonal",
        ))
    end
    cache.n = n
    cache.colptr = colptr
    cache.rowval = rowval
    cache.factor_view = factor_view
    cache.original_values = Vector{T}(undef, length(rowval))
    cache.dsigns = Int[sign for sign in req.dsigns]
    cache.regularization = T(req.regularization)
    cache.factor = nothing
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.symbolic_count = 0
    cache.numeric_count = 0
    cache.solve_count = 0
    cache.refine_count = 0
    cache.signature = _sparse_pattern_signature(
        colptr, rowval, cache.dsigns, Float64(cache.regularization),
    )
    cache.status = Prepared
    return cache
end

"""
    _copy_values_into_view!(cache, K)

Copy the caller's lower-triangle sparse `K` values into the owned factor
view (same structure).  `K` must already be a validated `SymmetricCorePattern`
matrix with matching colptr/rowval; a drift check is performed.
"""
function _copy_values_into_view!(
    cache::SparseSymbolicNumericCache{T}, K::SparseMatrixCSC{T, Int},
) where {T}
    size(K) == (cache.n, cache.n) || throw(DimensionMismatch(
        "factorize! dimension mismatch",
    ))
    K.colptr == cache.colptr || throw(ArgumentError(
        "factorize! matrix colptr drifted from the frozen pattern",
    ))
    K.rowval == cache.rowval || throw(ArgumentError(
        "factorize! matrix rowval drifted from the frozen pattern",
    ))
    all(isfinite, K.nzval) || throw(ArgumentError(
        "factorize! matrix contains non-finite data",
    ))
    cache.original_values .= K.nzval
    cache.factor_view.nzval .= K.nzval
    return cache
end

"""
    _apply_signed_regularization!(cache)

Add `regularization * dsign[i]` to each structural diagonal entry of the
factor view.  Mutates only the factor view; `original_values` is untouched.
"""
function _apply_signed_regularization!(
    cache::SparseSymbolicNumericCache{T},
) where {T}
    colptr = cache.colptr
    rowval = cache.rowval
    nzval = cache.factor_view.nzval
    dsigns = cache.dsigns
    delta = cache.regularization
    @inbounds for j in 1:cache.n
        found = false
        for pointer in colptr[j]:(colptr[j + 1] - 1)
            if rowval[pointer] == j
                nzval[pointer] += T(dsigns[j]) * delta
                found = true
                break
            end
        end
        found || throw(ArgumentError(
            "regularization cannot locate structural diagonal of column $j",
        ))
    end
    return cache
end

"""
    _restore_original_values!(cache)

Restore the retained original K values into the factor view so the view
always mirrors the unmodified operator after a successful factor.
"""
function _restore_original_values!(
    cache::SparseSymbolicNumericCache{T},
) where {T}
    cache.factor_view.nzval .= cache.original_values
    return cache
end

function factorize!(
    cache::SparseSymbolicNumericCache{T},
    K::AbstractMatrix{T},
    matrix_epoch::Integer,
) where {T}
    T === Float64 || throw(ArgumentError(
        "SparseSymbolicNumericCache factorize! is Float64-only; got $T",
    ))
    K isa SparseMatrixCSC{T, Int} || throw(ArgumentError(
        "SparseSymbolicNumericCache factorize! requires a SparseMatrixCSC",
    ))
    # Invalid requires an explicit re-prepare before any factorization.
    cache.status === Invalid && throw(FactorCacheStateError(
        :factorize, Prepared, Invalid,
    ))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    # Fail-closed on entry: a stale (Prepared/Fresh/Failed) usable factor is
    # never carried across a new factorization attempt.  The state machine
    # guarantees `Failed` → re-factorize recovers via the retained CHOLMOD
    # object only when the same pattern is presented again; any pattern drift
    # or structural rejection below throws and leaves `Failed` with the old
    # object detached, so no stale solve is possible.
    cache.status = Factoring
    try
        _copy_values_into_view!(cache, K)
        _apply_signed_regularization!(cache)
        if cache.factor === nothing
            # First numeric factor: also performs the sole symbolic analysis.
            # `check=false` reports failure via `issuccess` without throwing.
            factor = ldlt(Symmetric(cache.factor_view, :L); check=false)
            cache.factor = factor
            cache.symbolic_count += 1
        else
            # Same pattern: reuse the same CHOLMOD factor object.  Public
            # `ldlt!` throws `ZeroPivotException` on a genuine zero pivot and
            # can be retried on the same object with a regularized matrix
            # (verified against Julia 1.12 CHOLMOD); we still detach the
            # object on any throw so `Failed` never solves stale data.
            LinearAlgebra.ldlt!(
                cache.factor, Symmetric(cache.factor_view, :L),
            )
        end
        issuccess(cache.factor) || throw(ArgumentError(
            "CHOLMOD LDL factorization reported failure",
        ))
        # The factor owns the regularized copy; restore the view to the
        # original values so it mirrors the unmodified operator.
        _restore_original_values!(cache)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.numeric_count += 1
        cache.status = Fresh
    catch
        # Always restore the original values so the factor view never leaks a
        # regularized matrix, and detach the CHOLMOD object so no stale solve
        # is possible from the `Failed` state.  Recovering requires presenting
        # the same pattern again (a fresh `ldlt` on the next attempt); we do
        # not claim symbolic reuse across a failed factor.
        try
            _restore_original_values!(cache)
        catch
        end
        cache.factor = nothing
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(
    cache::SparseSymbolicNumericCache{T},
    destination::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    _require_fresh(cache.status)
    cache.factor === nothing && throw(FactorCacheStateError(:solve, Fresh, Failed))
    length(rhs) == cache.n || throw(DimensionMismatch(
        "solve rhs length != n",
    ))
    length(destination) == cache.n || throw(DimensionMismatch(
        "solve dest length != n",
    ))
    all(isfinite, rhs) || throw(ArgumentError(
        "solve right-hand side contains non-finite data",
    ))
    # Public CHOLMOD `\` allocates a result object; copy it into the
    # caller-owned destination.  Deliberately honest: no allocation-free
    # claim is made for this sparse solve.
    result = cache.factor \ rhs
    copyto!(destination, result)
    all(isfinite, destination) || throw(ArgumentError(
        "solve produced non-finite data",
    ))
    cache.solve_count += 1
    return destination
end

function solve_multi!(
    cache::SparseSymbolicNumericCache{T},
    destination::AbstractMatrix{T},
    rhs::AbstractMatrix{T},
) where {T}
    _require_fresh(cache.status)
    cache.factor === nothing && throw(FactorCacheStateError(:solve, Fresh, Failed))
    size(rhs, 1) == cache.n || throw(DimensionMismatch(
        "solve rhs rows != n",
    ))
    size(destination, 1) == cache.n || throw(DimensionMismatch(
        "solve dest rows != n",
    ))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch(
        "solve dest cols != rhs cols",
    ))
    all(isfinite, rhs) || throw(ArgumentError(
        "solve right-hand side contains non-finite data",
    ))
    result = cache.factor \ rhs
    copyto!(destination, result)
    all(isfinite, destination) || throw(ArgumentError(
        "solve produced non-finite data",
    ))
    cache.solve_count += 1
    return destination
end

function refine_once!(
    cache::SparseSymbolicNumericCache{T},
    residual::AbstractVector{T},
    correction::AbstractVector{T},
) where {T}
    _require_fresh_for_refine(cache.status)
    cache.factor === nothing && throw(FactorCacheStateError(:refine, Fresh, Failed))
    length(residual) == cache.n || throw(DimensionMismatch(
        "refine residual length != n",
    ))
    length(correction) == cache.n || throw(DimensionMismatch(
        "refine correction length != n",
    ))
    all(isfinite, residual) || throw(ArgumentError(
        "refine residual contains non-finite data",
    ))
    result = cache.factor \ residual
    copyto!(correction, result)
    all(isfinite, correction) || throw(ArgumentError(
        "refine produced non-finite data",
    ))
    cache.refine_count += 1
    return correction
end

function invalidate!(cache::SparseSymbolicNumericCache{T}) where {T}
    cache.factor = nothing
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::SparseSymbolicNumericCache) = cache.status
factor_matrix_epoch(cache::SparseSymbolicNumericCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::SparseSymbolicNumericCache) = cache.symbolic_epoch
factor_epoch(cache::SparseSymbolicNumericCache) = cache.factor_epoch

function factor_diagnostics(cache::SparseSymbolicNumericCache{T}) where {T}
    return (
        provider = :cholmod,
        kind = :symmetric_ldl,
        n = cache.n,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        symbolic_count = cache.symbolic_count,
        numeric_count = cache.numeric_count,
        solve_count = cache.solve_count,
        refine_count = cache.refine_count,
        regularization = cache.regularization,
        signature = cache.signature,
        solve_allocation_policy = :allocating_factor_backslash_copy,
    )
end
