#=
    SDPX provider-neutral FactorCache protocol (Wave A-3)

A FactorCache owns the storage and bookkeeping for a matrix factorization so
that the KKT hot path can factor and solve without re-allocating per
iteration.  The protocol is deliberately provider-neutral: any backend
(dense Cholesky, sparse LDLt, block-arrow, multifloat, ...) may implement it
by overriding the generic methods below.  Nothing in this file is wired into
the KKT hot path yet (that is Wave C); this is the protocol definition plus a
dense reference implementation used to pin the contract.

Storage is split into three concrete structs so a provider can keep symbolic
structure, numeric factors, and solve scratch in separate lifetimes:

  * `SymbolicCache{T}`      -- dimension / symbolic structure (no numeric data)
  * `NumericFactorCache{T}` -- the numeric factor plus its epoch/status bookkeeping
  * `SolveScratch{T}`       -- preallocated rhs / solution buffers for solves

All fields are concrete (no `Any`, no abstract backend fields) so the hot path
stays type-stable and allocation-free.
=#

"""
    AbstractFactorCache{T}

Abstract supertype for a provider-neutral matrix-factor cache.  `T` is the
element type of the factorized matrix.  Concrete providers subtype this and
override the protocol methods below.
"""
abstract type AbstractFactorCache{T} end

# ---------------------------------------------------------------------------
# Protocol.  Each generic method is defined on `AbstractFactorCache` and throws
# `MethodError` unless a concrete provider overrides it.  This makes a missing
# implementation fail loudly at the call site instead of silently no-op'ing.
# ---------------------------------------------------------------------------

"""
    prepare!(cache, plan)

Prepare `cache` for the factorization described by `plan` (a provider-specific
plan object).  Providers that need symbolic analysis or workspace sizing do it
here.  Returns `cache`.
"""
function prepare!(cache::AbstractFactorCache, plan)
    throw(MethodError(prepare!, (cache, plan)))
end

"""
    reserve!(cache, requirements)

Reserve the numeric storage required by `requirements` (a provider-specific
requirements object).  Returns `cache`.
"""
function reserve!(cache::AbstractFactorCache, requirements)
    throw(MethodError(reserve!, (cache, requirements)))
end

"""
    factorize!(cache, A, matrix_epoch)

Factor `A` into `cache`, reusing existing storage (must not allocate new
matrix storage).  `matrix_epoch` is a caller-supplied monotone counter that
identifies the current matrix; if it equals the epoch already stored in
`cache` and the factor is `:fresh`, the factorization is skipped (same-epoch
reuse).  Returns `cache`.
"""
function factorize!(cache::AbstractFactorCache, A, matrix_epoch)
    throw(MethodError(factorize!, (cache, A, matrix_epoch)))
end

"""
    solve!(cache, destination, rhs)

Solve the factorized system for a single right-hand side `rhs`, writing the
solution into `destination`.  Returns `destination`.
"""
function solve!(cache::AbstractFactorCache, destination, rhs)
    throw(MethodError(solve!, (cache, destination, rhs)))
end

"""
    solve_multi!(cache, destination, rhs)

Solve the factorized system for multiple right-hand sides (columns of `rhs`),
writing the solutions into `destination`.  Returns `destination`.
"""
function solve_multi!(cache::AbstractFactorCache, destination, rhs)
    throw(MethodError(solve_multi!, (cache, destination, rhs)))
end

"""
    refine_once!(cache, residual, correction)

Perform one step of iterative refinement: given a `residual`, compute a
`correction` using the factorized system.  Returns `correction`.
"""
function refine_once!(cache::AbstractFactorCache, residual, correction)
    throw(MethodError(refine_once!, (cache, residual, correction)))
end

"""
    invalidate!(cache)

Mark the cached factor as invalid (e.g. the matrix changed out-of-band).
Returns `cache`.
"""
function invalidate!(cache::AbstractFactorCache)
    throw(MethodError(invalidate!, (cache,)))
end

"""
    factor_status(cache) -> Symbol

Return the current factor status: `:fresh`, `:stale`, or `:invalid`.
"""
function factor_status(cache::AbstractFactorCache)
    throw(MethodError(factor_status, (cache,)))
end

"""
    factor_diagnostics(cache)

Return a named tuple of diagnostics for the cached factor (dimension, epoch,
status, factor info, ...).  Provider-specific.
"""
function factor_diagnostics(cache::AbstractFactorCache)
    throw(MethodError(factor_diagnostics, (cache,)))
end

"""
    factor_matrix_epoch(cache) -> Int

Return the matrix epoch currently stored in `cache`.
"""
function factor_matrix_epoch(cache::AbstractFactorCache)
    throw(MethodError(factor_matrix_epoch, (cache,)))
end

# ---------------------------------------------------------------------------
# Concrete storage structs
# ---------------------------------------------------------------------------

"""
    SymbolicCache{T}

Symbolic / structural storage for a factorization.  For the dense reference
this is just the matrix dimension; sparse providers would hold sparsity
patterns and fill-reducing orderings here.
"""
struct SymbolicCache{T}
    n::Int
end

"""
    NumericFactorCache{T}

Numeric factor storage plus epoch/status bookkeeping.  `factor` is the
concrete factor object (for the dense reference, a
`LinearAlgebra.Cholesky{T,Matrix{T}}`), `matrix_epoch` is the epoch of the
currently stored factor, and `status` is `:fresh` / `:stale` / `:invalid`.
"""
mutable struct NumericFactorCache{T}
    factor::Cholesky{T,Matrix{T}}
    matrix_epoch::Int
    status::Symbol
end

"""
    SolveScratch{T}

Preallocated solve buffers: a right-hand-side copy and a solution vector.
Reused across `solve!` calls so solves do not allocate.
"""
struct SolveScratch{T}
    rhs::Vector{T}
    solution::Vector{T}
end

# ---------------------------------------------------------------------------
# Dense reference implementation
# ---------------------------------------------------------------------------

"""
    DenseFactorCache{T}

Provider-neutral dense reference FactorCache backed by an in-place Cholesky
factorization.  Owns a `NumericFactorCache` holding a
`LinearAlgebra.Cholesky{T,Matrix{T}}` (whose `factors` matrix is reused as the
factorization buffer) and a `SolveScratch` holding preallocated rhs/solution
vectors.  `factorize!` reuses the buffer and does not allocate new matrix
storage; `factor_status` returns `:fresh` only while the stored epoch matches
the epoch used for the last factorization.
"""
mutable struct DenseFactorCache{T}
    symbolic::SymbolicCache{T}
    numeric::NumericFactorCache{T}
    scratch::SolveScratch{T}
end

"""
    DenseFactorCache{T}(n::Int)

Construct a dense factor cache for an `n × n` matrix of element type `T`.
All storage (factor buffer, rhs, solution) is allocated up front.
"""
function DenseFactorCache{T}(n::Int) where {T}
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    buffer = Matrix{T}(undef, n, n)
    factor = Cholesky(buffer, 'U', 0)
    numeric = NumericFactorCache{T}(factor, -1, :invalid)
    scratch = SolveScratch{T}(Vector{T}(undef, n), Vector{T}(undef, n))
    return DenseFactorCache{T}(SymbolicCache{T}(n), numeric, scratch)
end

# Convenience constructor inferring the element type from the matrix.
DenseFactorCache(A::AbstractMatrix{T}) where {T} = DenseFactorCache{T}(size(A, 1))

# --- protocol implementation ------------------------------------------------

function prepare!(cache::DenseFactorCache{T}, plan) where {T}
    # Dense reference: symbolic structure and storage are fixed at construction,
    # so there is nothing to prepare beyond validating the plan dimension.
    if plan isa Integer
        plan == cache.symbolic.n || throw(ArgumentError(
            "plan dimension $plan does not match cache dimension $(cache.symbolic.n)",
        ))
    end
    return cache
end

function reserve!(cache::DenseFactorCache{T}, requirements) where {T}
    # Dense reference: all storage is reserved at construction.
    return cache
end

function factorize!(cache::DenseFactorCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    # Same-epoch reuse: if the factor is already fresh for this epoch, skip.
    if cache.numeric.matrix_epoch == matrix_epoch && cache.numeric.status === :fresh
        return cache
    end
    size(A, 1) == cache.symbolic.n || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.symbolic.n)",
    ))
    size(A, 2) == cache.symbolic.n || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    # Reuse the factor buffer: copy A in place, then factor in place.  No new
    # matrix storage is allocated.  `cholesky!` mutates `factor.factors` and
    # returns a Cholesky wrapping the same data, so we keep our existing object.
    copyto!(cache.numeric.factor.factors, A)
    cholesky!(Symmetric(cache.numeric.factor.factors, :U))
    cache.numeric.matrix_epoch = Int(matrix_epoch)
    cache.numeric.status = :fresh
    return cache
end

function solve!(cache::DenseFactorCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    length(rhs) == cache.symbolic.n || throw(DimensionMismatch(
        "rhs length $(length(rhs)) does not match cache dimension $(cache.symbolic.n)",
    ))
    length(destination) == cache.symbolic.n || throw(DimensionMismatch(
        "destination length $(length(destination)) does not match cache dimension $(cache.symbolic.n)",
    ))
    # Reuse scratch buffers: copy rhs, solve into scratch.solution, copy out.
    copyto!(cache.scratch.rhs, rhs)
    ldiv!(cache.scratch.solution, cache.numeric.factor, cache.scratch.rhs)
    copyto!(destination, cache.scratch.solution)
    return destination
end

function solve_multi!(cache::DenseFactorCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    size(rhs, 1) == cache.symbolic.n || throw(DimensionMismatch(
        "rhs rows $(size(rhs, 1)) do not match cache dimension $(cache.symbolic.n)",
    ))
    size(destination, 1) == cache.symbolic.n || throw(DimensionMismatch(
        "destination rows $(size(destination, 1)) do not match cache dimension $(cache.symbolic.n)",
    ))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch(
        "destination columns $(size(destination, 2)) do not match rhs columns $(size(rhs, 2))",
    ))
    ldiv!(destination, cache.numeric.factor, rhs)
    return destination
end

function refine_once!(cache::DenseFactorCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    length(residual) == cache.symbolic.n || throw(DimensionMismatch(
        "residual length $(length(residual)) does not match cache dimension $(cache.symbolic.n)",
    ))
    length(correction) == cache.symbolic.n || throw(DimensionMismatch(
        "correction length $(length(correction)) does not match cache dimension $(cache.symbolic.n)",
    ))
    # One step of iterative refinement: correction = F \ residual.
    ldiv!(correction, cache.numeric.factor, residual)
    return correction
end

function invalidate!(cache::DenseFactorCache{T}) where {T}
    cache.numeric.status = :invalid
    return cache
end

function factor_status(cache::DenseFactorCache{T}) where {T}
    return cache.numeric.status
end

function factor_diagnostics(cache::DenseFactorCache{T}) where {T}
    return (
        n = cache.symbolic.n,
        matrix_epoch = cache.numeric.matrix_epoch,
        status = cache.numeric.status,
        uplo = cache.numeric.factor.uplo,
        info = cache.numeric.factor.info,
    )
end

function factor_matrix_epoch(cache::DenseFactorCache{T}) where {T}
    return cache.numeric.matrix_epoch
end
