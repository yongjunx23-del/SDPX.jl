#=====================================================================#
#    DenseAugmentedLDLTCache — dense augmented-system LDLᵀ.  (Subagent E.)
#
#    Factors the symmetric augmented system A = L D Lᵀ (L unit lower
#    triangular, D diagonal) using the owned, allocation-free kernel in
#    common.jl.  The cache owns the `L` factor buffer, the diagonal `d`, and
#    a solve scratch — all committed in `prepare!`.  `factorize!` reuses the
#    buffer, invalidates on entry, is fail-closed, and stamps `factor_epoch`.
#    `solve!` / `refine_once!` require `Fresh` and do only unit-lower /
#    diagonal / upper triangular solves (no backslash).
#
#    The kernel is pivot-free: valid when all leading principal minors are
#    nonzero — satisfied by the regularized augmented KKT system
#    `[H B; Bᵀ −δI]` with the leading block SPD.  A zero pivot throws and
#    leaves the cache `Failed` (fail-closed).
#=====================================================================#

"""
    DenseAugmentedLDLTCache{T}

A dense LDLᵀ factor cache for the augmented-system route.  Owns the `n × n`
unit-lower factor buffer `L`, the diagonal `d`, a solve scratch, and the
`symbolic_epoch` / `matrix_epoch` / `factor_epoch` counters plus the
`FactorCacheState`.
"""
mutable struct DenseAugmentedLDLTCache{T} <: AbstractFactorCache{T}
    n::Int
    L::Matrix{T}                # owned unit-lower factor buffer (holds A on entry)
    d::Vector{T}                 # owned LDLᵀ diagonal
    scratch::Vector{T}           # preallocated solve scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function DenseAugmentedLDLTCache{T}() where {T}
    return DenseAugmentedLDLTCache{T}(0, Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
        Vector{T}(undef, 0), 0, 0, 0, Unprepared)
end

function DenseAugmentedLDLTCache{T}(n::Int) where {T}
    cache = DenseAugmentedLDLTCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

DenseAugmentedLDLTCache(A::AbstractMatrix{T}) where {T} = DenseAugmentedLDLTCache{T}(size(A, 1))

function prepare!(cache::DenseAugmentedLDLTCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.L = Matrix{T}(undef, n, n)
    cache.d = Vector{T}(undef, n)
    cache.scratch = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::DenseAugmentedLDLTCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "DenseAugmentedLDLTCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        copyto!(cache.L, A)
        _ldlt_factor!(cache.L, cache.d)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::DenseAugmentedLDLTCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _ldlt_solve!(cache.scratch, cache.L, cache.d, rhs)
    copyto!(destination, cache.scratch)
    return destination
end

function solve_multi!(cache::DenseAugmentedLDLTCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        _ldlt_solve!(view(destination, :, j), cache.L, cache.d, view(rhs, :, j))
    end
    return destination
end

function refine_once!(cache::DenseAugmentedLDLTCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _ldlt_solve!(correction, cache.L, cache.d, residual)
    return correction
end

function invalidate!(cache::DenseAugmentedLDLTCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::DenseAugmentedLDLTCache) = cache.status
factor_matrix_epoch(cache::DenseAugmentedLDLTCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::DenseAugmentedLDLTCache) = cache.symbolic_epoch
factor_epoch(cache::DenseAugmentedLDLTCache) = cache.factor_epoch

function factor_diagnostics(cache::DenseAugmentedLDLTCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status, nonzero_pivot = all(!iszero, cache.d))
end
