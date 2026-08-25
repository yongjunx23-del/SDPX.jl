#=====================================================================#
#    DenseSchurCholeskyCache — dense Cholesky of the Schur complement
#    (SDP route).  (Subagent E.)
#
#    Owns a dense `n × n` factor buffer (upper-triangular Cholesky factor
#    stored in place), a solve scratch vector, and the three epoch counters.
#    All capacity is committed in `prepare!`; `factorize!` reuses the buffer
#    (no new matrix storage), invalidates the old factor on entry, and is
#    fail-closed.  `solve!` / `refine_once!` require `Fresh` and use only
#    owned triangular solves (no backslash, no factor-wrapper per epoch).
#=====================================================================#

"""
    DenseSchurCholeskyCache{T}

A dense Cholesky factor cache for the SDP Schur-complement route.  The cache
owns the `n × n` factor buffer `U` (upper-triangular factor of A = U'U), a
preallocated rhs scratch, and the `symbolic_epoch` / `matrix_epoch` /
`factor_epoch` counters plus the `FactorCacheState`.
"""
mutable struct DenseSchurCholeskyCache{T} <: AbstractFactorCache{T}
    n::Int
    U::Matrix{T}                # owned factor buffer (upper-triangular Cholesky)
    scratch::Vector{T}          # preallocated rhs/solution scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function DenseSchurCholeskyCache{T}() where {T}
    return DenseSchurCholeskyCache{T}(0, Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
        0, 0, 0, Unprepared)
end

function DenseSchurCholeskyCache{T}(n::Int) where {T}
    cache = DenseSchurCholeskyCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

DenseSchurCholeskyCache(A::AbstractMatrix{T}) where {T} = DenseSchurCholeskyCache{T}(size(A, 1))

function prepare!(cache::DenseSchurCholeskyCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.U = Matrix{T}(undef, n, n)
    cache.scratch = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::DenseSchurCholeskyCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "DenseSchurCholeskyCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        copyto!(cache.U, A)
        _chol_factor!(cache.U)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::DenseSchurCholeskyCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length $(length(rhs)) != n=$(cache.n)"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _chol_solve!(cache.scratch, cache.U, rhs)
    copyto!(destination, cache.scratch)
    return destination
end

function solve_multi!(cache::DenseSchurCholeskyCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        _chol_solve!(view(destination, :, j), cache.U, view(rhs, :, j))
    end
    return destination
end

function refine_once!(cache::DenseSchurCholeskyCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _chol_solve!(correction, cache.U, residual)
    return correction
end

function invalidate!(cache::DenseSchurCholeskyCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::DenseSchurCholeskyCache) = cache.status
factor_matrix_epoch(cache::DenseSchurCholeskyCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::DenseSchurCholeskyCache) = cache.symbolic_epoch
factor_epoch(cache::DenseSchurCholeskyCache) = cache.factor_epoch

function factor_diagnostics(cache::DenseSchurCholeskyCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end
