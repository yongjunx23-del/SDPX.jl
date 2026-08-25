#=====================================================================#
#    ArrowLocalCache — block-arrow cache storing the diagonal D block and
#    the arrow B.  (Subagent E.)
#
#    The matrix is  [ D  B ; Bᵀ  C ]  with D d×d, B d×c, C c×c (c = n−d).
#    `factorize!` factors D once, precomputes T = D⁻¹B once, forms and factors
#    the reduced Schur S = C − BᵀT once.  predictor / corrector / refinement
#    (`solve!`, `refine_once!`) then perform only owned triangular solves —
#    they never recompute D⁻¹B and never re-factor the Schur for a new RHS.
#    All storage lives in the owned `ArrowWS` (committed in `prepare!`), so
#    the warm path is allocation-free.
#=====================================================================#

"""
    ArrowLocalCache{T}

A block-arrow factor cache: stores the diagonal block `D`, the arrow block
`B`, and the factored reduced Schur complement `S = C − BᵀD⁻¹B`.  Factor-once
design: `solve!` / `refine_once!` only do triangular solves.
"""
mutable struct ArrowLocalCache{T} <: AbstractFactorCache{T}
    n::Int
    ws::ArrowWS{T}               # owned arrow workspace (D, B, T, S, scratch)
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function ArrowLocalCache{T}() where {T}
    return ArrowLocalCache{T}(0, ArrowWS{T}(0, 0), 0, 0, 0, Unprepared)
end

function ArrowLocalCache{T}(n::Int, d::Int) where {T}
    cache = ArrowLocalCache{T}()
    return prepare!(cache, ArrowRequirements(n, d))
end

function prepare!(cache::ArrowLocalCache{T}, req::ArrowRequirements) where {T}
    n, d = req.n, req.d
    n >= 0 && d >= 0 && d <= n || throw(ArgumentError("bad arrow split n=$n, d=$d"))
    cache.n = n
    cache.ws = ArrowWS{T}(d, n - d)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::ArrowLocalCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "ArrowLocalCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        _arrow_factorize!(cache.ws, A, cache.n)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::ArrowLocalCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _arrow_solve!(destination, cache.ws, rhs)
    return destination
end

function solve_multi!(cache::ArrowLocalCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        _arrow_solve!(view(destination, :, j), cache.ws, view(rhs, :, j))
    end
    return destination
end

function refine_once!(cache::ArrowLocalCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _arrow_solve!(correction, cache.ws, residual)
    return correction
end

function invalidate!(cache::ArrowLocalCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::ArrowLocalCache) = cache.status
factor_matrix_epoch(cache::ArrowLocalCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::ArrowLocalCache) = cache.symbolic_epoch
factor_epoch(cache::ArrowLocalCache) = cache.factor_epoch

function factor_diagnostics(cache::ArrowLocalCache{T}) where {T}
    return (n = cache.n, d = cache.ws.d, c = cache.ws.c,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end
