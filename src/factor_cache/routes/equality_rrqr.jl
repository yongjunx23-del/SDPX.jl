#=====================================================================#
#    EqualityRRQRCache — rank-revealing QR (RRQR) for the
#    equality-constrained reduction route.  (Subagent E.)
#
#    Factors  A·P = Q R  by Householder QR with column pivoting, entirely in
#    owned, preallocated storage (`R`/householder buffer, `tau`, `jpvt`, and
#    a `col` norm workspace + a solve scratch), so a warm factorize/solve
#    cycle is allocation-free.  `factorize!` reuses the buffers, invalidates
#    the old factor on entry, is fail-closed, and stamps `factor_epoch` at the
#    numeric factor call.  `solve!` / `refine_once!` require `Fresh` and apply
#    only the accumulated reflectors plus a triangular solve — no backslash.
#=====================================================================#

"""
    EqualityRRQRCache{T}

A rank-revealing QR (column pivoting) factor cache for the equality-constrained
reduction route.  Owns the `n × n` factor buffer `F` (R's upper triangle and
Householder vectors), the reflector coefficients `tau`, the pivot `jpvt`,
and two workspaces, plus the `symbolic_epoch` / `matrix_epoch` /
`factor_epoch` counters and `FactorCacheState`.
"""
mutable struct EqualityRRQRCache{T} <: AbstractFactorCache{T}
    n::Int
    F::Matrix{T}                 # owned factors buffer
    tau::Vector{T}               # owned Householder coefficients
    jpvt::Vector{Int}            # owned column pivot permutation
    col::Vector{T}               # owned column-norm workspace
    work::Vector{T}              # owned solve scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function EqualityRRQRCache{T}() where {T}
    return EqualityRRQRCache{T}(0, Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
        Vector{Int}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0),
        0, 0, 0, Unprepared)
end

function EqualityRRQRCache{T}(n::Int) where {T}
    cache = EqualityRRQRCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

EqualityRRQRCache(A::AbstractMatrix{T}) where {T} = EqualityRRQRCache{T}(size(A, 1))

function prepare!(cache::EqualityRRQRCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.F = Matrix{T}(undef, n, n)
    cache.tau = Vector{T}(undef, n)
    cache.jpvt = Vector{Int}(undef, n)
    cache.col = Vector{T}(undef, n)
    cache.work = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

function factorize!(cache::EqualityRRQRCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "EqualityRRQRCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        copyto!(cache.F, A)
        _rrqr_factor!(cache.F, cache.tau, cache.jpvt, cache.col)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::EqualityRRQRCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _rrqr_solve!(destination, cache.F, cache.tau, cache.jpvt, cache.work, rhs)
    return destination
end

function solve_multi!(cache::EqualityRRQRCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        _rrqr_solve!(view(destination, :, j), cache.F, cache.tau, cache.jpvt, cache.work, view(rhs, :, j))
    end
    return destination
end

function refine_once!(cache::EqualityRRQRCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _rrqr_solve!(correction, cache.F, cache.tau, cache.jpvt, cache.work, residual)
    return correction
end

function invalidate!(cache::EqualityRRQRCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::EqualityRRQRCache) = cache.status
factor_matrix_epoch(cache::EqualityRRQRCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::EqualityRRQRCache) = cache.symbolic_epoch
factor_epoch(cache::EqualityRRQRCache) = cache.factor_epoch

function factor_diagnostics(cache::EqualityRRQRCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end
