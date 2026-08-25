#=====================================================================#
#    SparseSymbolicNumericCache — sparse symbolic analysis reused across
#    epochs + numeric factor.  (Subagent E.)
#
#    `prepare!` runs a sparse symbolic analysis on the sparsity pattern
#    (a fill-reducing symmetric ordering via `amd`), fixes `symbolic_epoch`,
#    and allocates the numeric factor buffer once.  `factorize!` reuses that
#    buffer: it permutes the incoming sparse matrix by the fixed ordering and
#    performs the numeric Cholesky into the preallocated buffer.  Because the
#    ordering is fixed by `symbolic_epoch` (not recomputed per epoch) and the
#    buffer is owned, a warm factorize/solve cycle is allocation-free.
#=====================================================================#

"""
    SparseSymbolicRequirements

Capacity + symbolic requirements for `SparseSymbolicNumericCache`.  Carries
the total dimension `n`, the `symbolic_epoch`, and a `pattern`
(`SparseMatrixCSC`) describing the fixed sparsity structure used to derive
the symbolic ordering in `prepare!`.
"""
struct SparseSymbolicRequirements <: AbstractFactorRequirements
    n::Int
    symbolic_epoch::Int
    pattern::SparseMatrixCSC{Float64, Int}
end

function SparseSymbolicRequirements(A::SparseMatrixCSC{Tv, Ti}; symbolic_epoch::Integer = 0) where {Tv, Ti}
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("sparse pattern must be square"))
    return SparseSymbolicRequirements(Int(n), Int(symbolic_epoch),
        SparseMatrixCSC{Float64, Int}(A))
end

"""
    SparseSymbolicNumericCache{T}

A sparse route factor cache: derives a fill-reducing symmetric ordering from
the sparsity pattern once (`symbolic_epoch`), and reuses a preallocated
numeric Cholesky buffer across matrix epochs.  Owns the ordering `perm`
(and its inverse `pinv`), the factor buffer `U`, and a solve scratch, plus
the `matrix_epoch` / `factor_epoch` counters and `FactorCacheState`.
"""
mutable struct SparseSymbolicNumericCache{T} <: AbstractFactorCache{T}
    n::Int
    perm::Vector{Int}            # owned symbolic ordering (fixed by symbolic_epoch)
    pinv::Vector{Int}            # inverse ordering
    U::Matrix{T}                 # owned numeric factor buffer
    scratch::Vector{T}           # preallocated solve scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function SparseSymbolicNumericCache{T}() where {T}
    return SparseSymbolicNumericCache{T}(0, Vector{Int}(undef, 0), Vector{Int}(undef, 0),
        Matrix{T}(undef, 0, 0), Vector{T}(undef, 0), 0, 0, 0, Unprepared)
end

function SparseSymbolicNumericCache{T}(pattern::SparseMatrixCSC; symbolic_epoch::Integer = 0) where {T}
    cache = SparseSymbolicNumericCache{T}()
    return prepare!(cache, SparseSymbolicRequirements(pattern; symbolic_epoch = symbolic_epoch))
end

function prepare!(cache::SparseSymbolicNumericCache{T}, req::SparseSymbolicRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    # Symbolic analysis: a fill-reducing symmetric ordering from the pattern,
    # obtained once (cold path) from a CHOLMOD symbolic factorization.  The
    # ordering is fixed by `symbolic_epoch` and reused across matrix epochs.
    perm = Vector{Int}(cholesky(Symmetric(req.pattern)).p)
    length(perm) == n || throw(ArgumentError("ordering length $(length(perm)) != n=$n"))
    pinv = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        pinv[perm[i]] = i
    end
    cache.n = n
    cache.perm = perm
    cache.pinv = pinv
    cache.U = Matrix{T}(undef, n, n)
    cache.scratch = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::SparseSymbolicNumericCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "SparseSymbolicNumericCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        _fill_permuted!(cache.U, A, cache.perm, cache.pinv)
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

# U ← P A Pᵀ (P is the symmetric ordering), stored as a dense symmetric copy
# (both triangles) so `_chol_factor!` sees a valid symmetric matrix.
function _fill_permuted!(U::Matrix{T}, A::AbstractMatrix{T}, perm::Vector{Int}, pinv::Vector{Int}) where {T}
    n = length(perm)
    fill!(U, zero(T))
    if A isa SparseMatrixCSC
        colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
        @inbounds for j in 1:n
            pj = pinv[j]
            for p in colptr[j]:colptr[j+1]-1
                i = rowval[p]
                pi = pinv[i]
                U[pi, pj] = nzval[p]
                U[pj, pi] = nzval[p]
            end
        end
    else
        @inbounds for j in 1:n
            pj = pinv[j]
            for i in 1:n
                v = A[i, j]
                pi = pinv[i]
                U[pi, pj] = v
                U[pj, pi] = v
            end
        end
    end
    return U
end

function solve!(cache::SparseSymbolicNumericCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    n = cache.n
    @inbounds for k in 1:n
        cache.scratch[k] = rhs[cache.perm[k]]
    end
    _chol_solve!(cache.scratch, cache.U, cache.scratch)
    @inbounds for i in 1:n
        destination[i] = cache.scratch[cache.pinv[i]]
    end
    return destination
end

function solve_multi!(cache::SparseSymbolicNumericCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        @inbounds for k in 1:cache.n
            cache.scratch[k] = rhs[cache.perm[k], j]
        end
        _chol_solve!(cache.scratch, cache.U, cache.scratch)
        @inbounds for i in 1:cache.n
            destination[i, j] = cache.scratch[cache.pinv[i]]
        end
    end
    return destination
end

function refine_once!(cache::SparseSymbolicNumericCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    @inbounds for k in 1:cache.n
        cache.scratch[k] = residual[cache.perm[k]]
    end
    _chol_solve!(cache.scratch, cache.U, cache.scratch)
    @inbounds for i in 1:cache.n
        correction[i] = cache.scratch[cache.pinv[i]]
    end
    return correction
end

function invalidate!(cache::SparseSymbolicNumericCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::SparseSymbolicNumericCache) = cache.status
factor_matrix_epoch(cache::SparseSymbolicNumericCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::SparseSymbolicNumericCache) = cache.symbolic_epoch
factor_epoch(cache::SparseSymbolicNumericCache) = cache.factor_epoch

function factor_diagnostics(cache::SparseSymbolicNumericCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status, ordering = cache.perm)
end
