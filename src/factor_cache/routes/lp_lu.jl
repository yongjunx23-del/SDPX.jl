#=====================================================================#
#    LPLUCache — dense LU for the LP route.  (Subagent E.)
#
#    Owns the `n × n` factor buffer `F` (PA = LU, factored in place), the
#    pivot vector `ipiv`, and a solve scratch — all committed in `prepare!`.
#    `factorize!` reuses the buffer (no new matrix storage), invalidates the
#    old factor on entry, is fail-closed, and stamps `factor_epoch` exactly at
#    the numeric factor call.  `solve!` / `refine_once!` require `Fresh` and
#    use the owned LU (LAPACK `getrs!` fast path for Float32/64/Complex,
#    generic forward/back substitution otherwise) — no backslash, no
#    throw-away LU wrapper per epoch.
#=====================================================================#

"""
    LPLUCache{T}

A dense pivoted-LU factor cache for the LP route.  Owns the factor buffer
`factors`, the pivot vector `ipiv`, a solve scratch, and the
`symbolic_epoch` / `matrix_epoch` / `factor_epoch` counters plus the
`FactorCacheState`.
"""
mutable struct LPLUCache{T} <: AbstractFactorCache{T}
    n::Int
    factors::Matrix{T}           # owned LU factor buffer
    ipiv::Vector{Int}            # owned pivot vector
    scratch::Vector{T}           # preallocated solve scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function LPLUCache{T}() where {T}
    return LPLUCache{T}(0, Matrix{T}(undef, 0, 0), Vector{Int}(undef, 0),
        Vector{T}(undef, 0), 0, 0, 0, Unprepared)
end

function LPLUCache{T}(n::Int) where {T}
    cache = LPLUCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

LPLUCache(A::AbstractMatrix{T}) where {T} = LPLUCache{T}(size(A, 1))

function prepare!(cache::LPLUCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.factors = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.scratch = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::LPLUCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "LPLUCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        copyto!(cache.factors, A)
        _lu_factor!(cache.factors, cache.ipiv)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::LPLUCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    copyto!(cache.scratch, rhs)
    _lu_solve!(cache.factors, cache.ipiv, cache.scratch)
    copyto!(destination, cache.scratch)
    return destination
end

function solve_multi!(cache::LPLUCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        for i in 1:cache.n
            cache.scratch[i] = rhs[i, j]
        end
        _lu_solve!(cache.factors, cache.ipiv, cache.scratch)
        for i in 1:cache.n
            destination[i, j] = cache.scratch[i]
        end
    end
    return destination
end

function refine_once!(cache::LPLUCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    for i in 1:cache.n
        correction[i] = residual[i]
    end
    _lu_solve!(cache.factors, cache.ipiv, correction)
    return correction
end

function invalidate!(cache::LPLUCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::LPLUCache) = cache.status
factor_matrix_epoch(cache::LPLUCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::LPLUCache) = cache.symbolic_epoch
factor_epoch(cache::LPLUCache) = cache.factor_epoch

function factor_diagnostics(cache::LPLUCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end
