#=====================================================================#
#    Provider-neutral FactorCache protocol (Subagent D — reworked).
#
#    A FactorCache owns the storage and bookkeeping for a matrix
#    factorization so the KKT hot path can factor and solve without
#    re-allocating per iteration.  The protocol is deliberately
#    provider-neutral: any backend (dense Cholesky, sparse LDLt,
#    block-arrow, multifloat, ...) may implement it by overriding the
#    generic methods below.
#
#    Storage is split into three concrete structs with separate
#    lifetimes:
#      * `SymbolicCache{T}`      -- structural / sparsity / permutation plan
#      * `NumericFactorCache{T}` -- owns the numeric factors + epochs/state
#      * `SolveScratch{T}`       -- preallocated rhs / solution buffers
#
#    All fields are concrete (no `Any`, no abstract backend fields) so the
#    hot path stays type-stable and allocation-free.
#
#    State is the isbits `FactorCacheState` enum (see state.jl).  Key rules:
#      * `prepare!(cache, requirements)` is the single, exact allocation
#        point; nothing may resize/grow after it returns.
#      * `factorize!` invalidates the old factor immediately on entry and is
#        fail-closed: any exception leaves the cache `Failed`.
#      * `solve!` / `refine_once!` are legal only from `Fresh`.
#      * three epoch counters: symbolic_epoch (fixed by prepare!), matrix_epoch
#        (only a change forces re-factorization), factor_epoch (stamped each
#        time a factor is actually produced).
#=====================================================================#

"""
    AbstractFactorCache{T}

Abstract supertype for a provider-neutral matrix-factor cache.  `T` is the
element type of the factorized matrix.  Concrete providers subtype this and
override the protocol methods below.
"""
abstract type AbstractFactorCache{T} end

# ---------------------------------------------------------------------------
# Protocol.  Each generic method is defined on `AbstractFactorCache` and throws
# `MethodError` unless a concrete provider overrides it.  A missing
# implementation thus fails loudly at the call site instead of silently
# no-op'ing.
# ---------------------------------------------------------------------------

"""
    prepare!(cache, requirements)

Allocate ALL capacity required by `requirements` (the single, exact capacity
source).  After this returns, no resize / growth is permitted anywhere.  Also
fixes the symbolic structure and `symbolic_epoch`.  Returns `cache`.
"""
function prepare!(cache::AbstractFactorCache, requirements)
    throw(MethodError(prepare!, (cache, requirements)))
end

"""
    factorize!(cache, A, matrix_epoch)

Factor `A` into `cache`, reusing existing storage (must not allocate new matrix
storage).  `matrix_epoch` identifies the current matrix; for a fixed symbolic
structure, only a change in `matrix_epoch` forces a numeric re-factorization.
If the factor is already `Fresh` for this `matrix_epoch`, the call is a no-op.
The old factor is invalidated immediately on entry (fail-closed); any exception
leaves the state `Failed`.  Returns `cache`.
"""
function factorize!(cache::AbstractFactorCache, A, matrix_epoch)
    throw(MethodError(factorize!, (cache, A, matrix_epoch)))
end

"""
    solve!(cache, destination, rhs)

Solve the factorized system for a single right-hand side `rhs`, writing the
solution into `destination`.  Legal only when the state is `Fresh`; otherwise
throws a `FactorCacheStateError`.  Returns `destination`.
"""
function solve!(cache::AbstractFactorCache, destination, rhs)
    throw(MethodError(solve!, (cache, destination, rhs)))
end

"""
    solve_multi!(cache, destination, rhs)

Solve the factorized system for multiple right-hand sides (columns of `rhs`).
Legal only when the state is `Fresh`.  Returns `destination`.
"""
function solve_multi!(cache::AbstractFactorCache, destination, rhs)
    throw(MethodError(solve_multi!, (cache, destination, rhs)))
end

"""
    refine_once!(cache, residual, correction)

Perform one step of iterative refinement: given a `residual`, compute a
`correction` using the factorized system.  Legal only when the state is `Fresh`.
Returns `correction`.
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
    factor_status(cache) -> FactorCacheState

Return the current `FactorCacheState` (`Unprepared`, `Prepared`, `Factoring`,
`Fresh`, `Failed`, `Invalid`).
"""
function factor_status(cache::AbstractFactorCache)
    throw(MethodError(factor_status, (cache,)))
end

"""
    factor_matrix_epoch(cache) -> Int

Return the matrix epoch currently stored in `cache`.
"""
function factor_matrix_epoch(cache::AbstractFactorCache)
    throw(MethodError(factor_matrix_epoch, (cache,)))
end

"""
    factor_symbolic_epoch(cache) -> Int

Return the symbolic epoch fixed by `prepare!`.
"""
function factor_symbolic_epoch(cache::AbstractFactorCache)
    throw(MethodError(factor_symbolic_epoch, (cache,)))
end

"""
    factor_epoch(cache) -> Int

Return the factor epoch: incremented each time a numeric factor is actually
produced.
"""
function factor_epoch(cache::AbstractFactorCache)
    throw(MethodError(factor_epoch, (cache,)))
end

"""
    factor_diagnostics(cache)

Return a named tuple of diagnostics for the cached factor (dimension, epochs,
state, factor info, ...).  Provider-specific.  This method constructs
user-facing objects on the COLD path only; it is never called inside `solve!`.
"""
function factor_diagnostics(cache::AbstractFactorCache)
    throw(MethodError(factor_diagnostics, (cache,)))
end

# ---------------------------------------------------------------------------
# Concrete storage structs
# ---------------------------------------------------------------------------

"""
    SymbolicCache{T}

Symbolic / structural storage for a factorization.  Holds the dimension and the
`symbolic_epoch` (identity of the structure / permutation plan).  For the dense
reference this is just the dimension; sparse providers would hold sparsity
patterns and fill-reducing orderings here.
"""
struct SymbolicCache{T}
    n::Int
    symbolic_epoch::Int
end

"""
    NumericFactorCache{T}

Numeric factor storage plus epochs and state.  `factor` is the concrete factor
object (for the dense reference, a `LinearAlgebra.Cholesky{T,Matrix{T}}` whose
`factors` buffer is reused as the factorization scratch), `matrix_epoch` is the
epoch of the currently stored factor, `factor_epoch` is stamped each time a new
factor is produced, and `status` is the `FactorCacheState`.
"""
mutable struct NumericFactorCache{T}
    factor::Cholesky{T,Matrix{T}}
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
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
factorization.  Owns a `SymbolicCache`, a `NumericFactorCache` holding a
`LinearAlgebra.Cholesky{T,Matrix{T}}` (whose `factors` matrix is reused as the
factorization buffer), and a `SolveScratch` holding preallocated rhs/solution
vectors.

All capacity is allocated in `prepare!` from a `FactorRequirements`; after that
nothing grows.  `factorize!` reuses the buffer (no new matrix storage),
invalidates the old factor on entry, and is fail-closed (an exception leaves the
state `Failed`).  `solve!` / `refine_once!` require state `Fresh`.
"""
mutable struct DenseFactorCache{T}
    symbolic::SymbolicCache{T}
    numeric::NumericFactorCache{T}
    scratch::SolveScratch{T}
end

"""
    DenseFactorCache{T}() -> DenseFactorCache{T}

Construct an unprepared dense factor cache with no capacity.  Call
`prepare!(cache, requirements)` to allocate.
"""
function DenseFactorCache{T}() where {T}
    buffer = Matrix{T}(undef, 0, 0)
    numeric = NumericFactorCache{T}(Cholesky(buffer, 'U', 0), 0, 0, Unprepared)
    scratch = SolveScratch{T}(Vector{T}(undef, 0), Vector{T}(undef, 0))
    return DenseFactorCache{T}(SymbolicCache{T}(0, 0), numeric, scratch)
end

"""
    DenseFactorCache{T}(n::Int) -> DenseFactorCache{T}

Construct a dense factor cache prepared for an `n × n` matrix of element type
`T`.  All storage (factor buffer, rhs, solution) is allocated up front.
"""
function DenseFactorCache{T}(n::Int) where {T}
    cache = DenseFactorCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

# Convenience constructor inferring the element type from the matrix.
DenseFactorCache(A::AbstractMatrix{T}) where {T} = DenseFactorCache{T}(size(A, 1))

# --- protocol implementation ------------------------------------------------

function prepare!(cache::DenseFactorCache{T}, requirements::FactorRequirements) where {T}
    n = requirements.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    buffer = Matrix{T}(undef, n, n)
    cache.symbolic = SymbolicCache{T}(n, requirements.symbolic_epoch)
    cache.numeric = NumericFactorCache{T}(Cholesky(buffer, 'U', 0), 0, 0, Prepared)
    cache.scratch = SolveScratch{T}(Vector{T}(undef, n), Vector{T}(undef, n))
    return cache
end

function factorize!(cache::DenseFactorCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A, 1) == cache.symbolic.n || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.symbolic.n)",
    ))
    size(A, 2) == cache.symbolic.n || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    # Same-epoch reuse: already fresh for this matrix epoch -> skip entirely.
    if cache.numeric.status === Fresh && cache.numeric.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    # Fail-closed: invalidate the old factor immediately, then factorize.
    # Reuse the factor buffer: copy A in place, then factor in place.  No new
    # matrix storage is allocated.  On any exception the state is `Failed`.
    cache.numeric.status = Factoring
    try
        copyto!(cache.numeric.factor.factors, A)
        cholesky!(Symmetric(cache.numeric.factor.factors, :U))
        cache.numeric.matrix_epoch = Int(matrix_epoch)
        cache.numeric.factor_epoch += 1
        cache.numeric.status = Fresh
    catch
        cache.numeric.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::DenseFactorCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.numeric.status)
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
    _require_fresh(cache.numeric.status)
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
    _require_fresh_for_refine(cache.numeric.status)
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
    cache.numeric.status = Invalid
    return cache
end

function factor_status(cache::DenseFactorCache{T}) where {T}
    return cache.numeric.status
end

function factor_matrix_epoch(cache::DenseFactorCache{T}) where {T}
    return cache.numeric.matrix_epoch
end

function factor_symbolic_epoch(cache::DenseFactorCache{T}) where {T}
    return cache.symbolic.symbolic_epoch
end

function factor_epoch(cache::DenseFactorCache{T}) where {T}
    return cache.numeric.factor_epoch
end

# Cold-path only: constructs user-facing diagnostic objects.  Never called
# inside `solve!` / the hot path.
function factor_diagnostics(cache::DenseFactorCache{T}) where {T}
    return (
        n = cache.symbolic.n,
        symbolic_epoch = cache.symbolic.symbolic_epoch,
        matrix_epoch = cache.numeric.matrix_epoch,
        factor_epoch = cache.numeric.factor_epoch,
        status = cache.numeric.status,
        uplo = cache.numeric.factor.uplo,
        info = cache.numeric.factor.info,
    )
end
