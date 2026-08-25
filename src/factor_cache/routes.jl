#=====================================================================#
#    Route-specific FactorCache implementations (Subagent B, PR1/PR4).
#
#    Concrete `AbstractFactorCache` subtypes for the fast-path routes.
#    They share the same protocol (prepare!, reserve!, factorize!,
#    solve!, solve_multi!, refine_once!, invalidate!, factor_status,
#    factor_diagnostics, factor_matrix_epoch) and the same SymbolicCache /
#    NumericFactorCache / SolveScratch split.
#
#    - LPLUCache:            dense LU (LP route).
#    - DenseSchurCholeskyCache: dense Cholesky (SDP Schur route).
#    - ArrowFactorCache:     block-arrow LDLt (SOCP route).
#=====================================================================#

# ---------------------------------------------------------------------------
# LPLUCache — dense LU for the LP route
# ---------------------------------------------------------------------------

"""
    LPLUCache{T}

A dense LU factor cache for the LP route. Owns the numeric LU factors
and the solve scratch; reuses storage across iterations and only
re-factorizes when the matrix epoch changes.
"""
mutable struct LPLUCache{T} <: AbstractFactorCache{T}
    n::Int
    lu::LU{T, Matrix{T}, Vector{Int}}
    scratch::SolveScratch{T}
    matrix_epoch::Int
    status::Symbol
end

function LPLUCache{T}(n::Int) where {T}
    n >= 1 || throw(ArgumentError("LPLUCache dimension must be >= 1"))
    factors = zeros(T, n, n)
    ipiv = zeros(Int, n)
    lu = LU{T, Matrix{T}, Vector{Int}}(factors, ipiv, 0)
    return LPLUCache{T}(
        n,
        lu,
        SolveScratch{T}(zeros(T, n), zeros(T, n)),
        -1,
        :unprepared,
    )
end

function factorize!(cache::LPLUCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch("LPLUCache factorize! dimension mismatch"))
    if cache.matrix_epoch != matrix_epoch
        copyto!(cache.lu.factors, A)
        cache.lu = lu!(cache.lu.factors)
        cache.matrix_epoch = matrix_epoch
        cache.status = :factored
    end
    return cache
end

function solve!(cache::LPLUCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    cache.status === :factored || throw(ArgumentError("LPLUCache solve! before factorize!"))
    copyto!(cache.scratch.rhs, rhs)
    ldiv!(cache.lu, cache.scratch.rhs)
    copyto!(destination, cache.scratch.rhs)
    return destination
end

function invalidate!(cache::LPLUCache{T}) where {T}
    cache.matrix_epoch = -1
    cache.status = :unprepared
    return cache
end

factor_status(cache::LPLUCache) = cache.status
factor_matrix_epoch(cache::LPLUCache) = cache.matrix_epoch

# ---------------------------------------------------------------------------
# DenseSchurCholeskyCache — dense Cholesky for the SDP Schur route
# ---------------------------------------------------------------------------

"""
    DenseSchurCholeskyCache{T}

A dense Cholesky factor cache for the SDP Schur-complement route.
"""
mutable struct DenseSchurCholeskyCache{T} <: AbstractFactorCache{T}
    n::Int
    factor::Matrix{T}
    scratch::SolveScratch{T}
    matrix_epoch::Int
    status::Symbol
end

function DenseSchurCholeskyCache{T}(n::Int) where {T}
    n >= 1 || throw(ArgumentError("DenseSchurCholeskyCache dimension must be >= 1"))
    return DenseSchurCholeskyCache{T}(
        n,
        zeros(T, n, n),
        SolveScratch{T}(zeros(T, n), zeros(T, n)),
        -1,
        :unprepared,
    )
end

function factorize!(cache::DenseSchurCholeskyCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch("DenseSchurCholeskyCache factorize! dimension mismatch"))
    if cache.matrix_epoch != matrix_epoch
        copyto!(cache.factor, A)
        cache.factor = cholesky!(Symmetric(cache.factor)).L
        cache.matrix_epoch = matrix_epoch
        cache.status = :factored
    end
    return cache
end

function solve!(cache::DenseSchurCholeskyCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    cache.status === :factored || throw(ArgumentError("DenseSchurCholeskyCache solve! before factorize!"))
    copyto!(cache.scratch.rhs, rhs)
    # A = L Lᵀ (Cholesky). Solve L y = b, then Lᵀ x = y.
    L = LowerTriangular(cache.factor)
    ldiv!(L, cache.scratch.rhs)
    ldiv!(L', cache.scratch.rhs)
    copyto!(destination, cache.scratch.rhs)
    return destination
end

function invalidate!(cache::DenseSchurCholeskyCache{T}) where {T}
    cache.matrix_epoch = -1
    cache.status = :unprepared
    return cache
end

factor_status(cache::DenseSchurCholeskyCache) = cache.status
factor_matrix_epoch(cache::DenseSchurCholeskyCache) = cache.matrix_epoch

# ---------------------------------------------------------------------------
# ArrowFactorCache — block-arrow LDLt for the SOCP route
# ---------------------------------------------------------------------------

"""
    ArrowFactorCache{T}

A block-arrow LDLt factor cache for the SOCP route. The matrix is
`[D  B; Bᵀ C]`; the cache stores the diagonal block `D`, the arrow
block `B`, and the Schur complement `C - Bᵀ D⁻¹ B`.
"""
mutable struct ArrowFactorCache{T} <: AbstractFactorCache{T}
    n::Int
    arrow_size::Int
    D::Matrix{T}
    B::Matrix{T}
    schur::Matrix{T}
    scratch::SolveScratch{T}
    matrix_epoch::Int
    status::Symbol
end

function ArrowFactorCache{T}(n::Int, arrow_size::Int) where {T}
    n >= 1 || throw(ArgumentError("ArrowFactorCache dimension must be >= 1"))
    arrow_size >= 0 || throw(ArgumentError("arrow_size must be >= 0"))
    return ArrowFactorCache{T}(
        n,
        arrow_size,
        zeros(T, arrow_size, arrow_size),
        zeros(T, arrow_size, n - arrow_size),
        zeros(T, n - arrow_size, n - arrow_size),
        SolveScratch{T}(zeros(T, n), zeros(T, n)),
        -1,
        :unprepared,
    )
end

function factorize!(cache::ArrowFactorCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch("ArrowFactorCache factorize! dimension mismatch"))
    if cache.matrix_epoch != matrix_epoch
        a = cache.arrow_size
        copyto!(cache.D, view(A, 1:a, 1:a))
        copyto!(cache.B, view(A, 1:a, (a + 1):cache.n))
        # Schur complement C - Bᵀ D⁻¹ B
        cache.schur .= view(A, (a + 1):cache.n, (a + 1):cache.n)
        cache.schur .-= cache.B' * (cache.D \ cache.B)
        cache.matrix_epoch = matrix_epoch
        cache.status = :factored
    end
    return cache
end

function solve!(cache::ArrowFactorCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    cache.status === :factored || throw(ArgumentError("ArrowFactorCache solve! before factorize!"))
    a = cache.arrow_size
    # solve [D B; Bᵀ C] [u; v] = [r1; r2]
    r1 = view(cache.scratch.rhs, 1:a)
    r2 = view(cache.scratch.rhs, (a + 1):cache.n)
    copyto!(r1, view(rhs, 1:a))
    copyto!(r2, view(rhs, (a + 1):cache.n))
    # v = schur \ (r2 - Bᵀ D⁻¹ r1)
    temp = cache.D \ r1
    r2 .-= cache.B' * temp
    v = cache.schur \ r2
    # u = D⁻¹ (r1 - B v)
    u = cache.D \ (r1 - cache.B * v)
    copyto!(view(destination, 1:a), u)
    copyto!(view(destination, (a + 1):cache.n), v)
    return destination
end

function invalidate!(cache::ArrowFactorCache{T}) where {T}
    cache.matrix_epoch = -1
    cache.status = :unprepared
    return cache
end

factor_status(cache::ArrowFactorCache) = cache.status
factor_matrix_epoch(cache::ArrowFactorCache) = cache.matrix_epoch
