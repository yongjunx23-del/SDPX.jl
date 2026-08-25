#=====================================================================#
#    ArrowReducedCache — block-arrow cache with a precomputed reduced Schur
#    factor.  (Subagent E.)
#
#    This variant of the block-arrow cache keeps the same factor-once design
#    as `ArrowLocalCache` but is organised around the reduced Schur factor:
#    `factorize!` factors D, precomputes T = D⁻¹B, forms and factors the
#    reduced Schur S = C − BᵀT exactly once, and the solve is driven through
#    that reduced factor (`v = S⁻¹(r₂ − Tᵀr₁)`), after which only a back
#    solve with D recovers `u`.  predictor / corrector / refinement reuse the
#    factored S and precomputed T; they never recompute D⁻¹B or re-factor the
#    Schur.  All storage lives in the owned `ArrowWS`, so the warm path is
#    allocation-free.
#=====================================================================#

"""
    ArrowReducedCache{T}

A block-arrow factor cache with a precomputed reduced Schur factor.  The
reduced complement `S = C − BᵀD⁻¹B` is formed and factored once in
`factorize!`; `solve!` / `refine_once!` solve through `S` and use only
triangular solves.
"""
mutable struct ArrowReducedCache{T} <: AbstractFactorCache{T}
    n::Int
    ws::ArrowWS{T}               # owned arrow workspace (D, B, T, S, scratch)
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function ArrowReducedCache{T}() where {T}
    return ArrowReducedCache{T}(0, ArrowWS{T}(0, 0), 0, 0, 0, Unprepared)
end

function ArrowReducedCache{T}(n::Int, d::Int) where {T}
    cache = ArrowReducedCache{T}()
    return prepare!(cache, ArrowRequirements(n, d))
end

function prepare!(cache::ArrowReducedCache{T}, req::ArrowRequirements) where {T}
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

function factorize!(cache::ArrowReducedCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "ArrowReducedCache factorize! dimension mismatch"))
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

function solve!(cache::ArrowReducedCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _arrow_solve!(destination, cache.ws, rhs)
    return destination
end

function solve_multi!(cache::ArrowReducedCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        _arrow_solve!(view(destination, :, j), cache.ws, view(rhs, :, j))
    end
    return destination
end

function refine_once!(cache::ArrowReducedCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _arrow_solve!(correction, cache.ws, residual)
    return correction
end

function invalidate!(cache::ArrowReducedCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::ArrowReducedCache) = cache.status
factor_matrix_epoch(cache::ArrowReducedCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::ArrowReducedCache) = cache.symbolic_epoch
factor_epoch(cache::ArrowReducedCache) = cache.factor_epoch

function factor_diagnostics(cache::ArrowReducedCache{T}) where {T}
    return (n = cache.n, d = cache.ws.d, c = cache.ws.c,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end
