#=
#    Provider-neutral FactorCache (Phase 3)
#
#    A `FactorCache` owns the factorization lifecycle of one KKT / Newton
#    linear system, split into three responsibilities:
#
#      * `SymbolicCache`       - structural facts (dimension, ordering,
#                                permutation) fixed before any numeric work
#                                and reused while the sparsity pattern is
#                                unchanged.
#      * `NumericFactorCache`  - the current numeric factor for a given KKT
#                                matrix *epoch* (the value of the matrix for
#                                the current iterate), reused by the
#                                predictor, corrector, and refinement solves.
#      * `SolveScratch`        - buffers shared by single-RHS, multi-RHS, and
#                                refinement solves.
#
#    `matrix_epoch` contract: `factorize!(cache, A, epoch)` re-factors only
#    when `epoch` differs from the cached epoch (or the cache is invalid).
#    Every solve reuses the current numeric factor; the same factor therefore
#    serves predictor, corrector, and refinement with no refactorization and
#    no solution-object allocation.  `invalidate!(cache)` drops the numeric
#    factor so the next `factorize!` always refactors.
#
#    This is deliberately a *provider-neutral interface*: routes that need
#    route-specific numerics implement their own `<: FactorCache` and keep
#    their specialized factor storage.  `DenseFactorCache` is the reference
#    dense implementation over the `AbstractLABackend` seam, so it works
#    across Float64, MultiFloat, and BigFloat providers.
=#

abstract type FactorCache{T} end

"""Structural facts fixed before any numeric work; reused across iterations."""
struct SymbolicCache{T}
    dimension::Int
    order::Vector{Int}     # row/column permutation (identity by default)
    invorder::Vector{Int}  # inverse permutation
end
function SymbolicCache(::Type{T}, dimension::Integer) where {T}
    dimension >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    order = collect(1:Int(dimension))
    return SymbolicCache{T}(Int(dimension), order, copy(order))
end

# Closed set of concrete dense Cholesky factor handles produced by the
# provider seam.  Typed union (not Any) keeps every factor/solve dispatch
# statically resolvable, which is the hot-state requirement.
const _CholeskyFactor{T} = Union{
    Nothing,
    ProviderLACholeskyFactor{T},
    StandardLACholeskyFactor{T},
    LegacyLACholeskyFactor{T},
}

"""The current numeric factor and the KKT matrix epoch it corresponds to."""
mutable struct NumericFactorCache{T}
    factor::_CholeskyFactor{T}
    factor_kind::Symbol      # :cholesky | :none
    matrix_epoch::Int        # -1 => invalid (must refactor)
    dimension::Int
end
NumericFactorCache{T}() where {T} = NumericFactorCache{T}(nothing, :none, -1, 0)

"""Scratch buffers shared by predictor/corrector/multi-RHS/refinement."""
mutable struct SolveScratch{T}
    rhs::Vector{T}
    solution::Vector{T}
    residual::Vector{T}
    correction::Vector{T}
    multi_rhs::Matrix{T}
    multi_solution::Matrix{T}
end
function SolveScratch(::Type{T}, n::Integer) where {T}
    n = Int(n)
    return SolveScratch{T}(
        alloc_zeros(T, n),
        alloc_zeros(T, n),
        alloc_zeros(T, n),
        alloc_zeros(T, n),
        alloc_zeros(T, n, 0),
        alloc_zeros(T, n, 0),
    )
end

"""Reference dense `FactorCache` implementation over `AbstractLABackend`."""
mutable struct DenseFactorCache{T} <: FactorCache{T}
    symbolic::SymbolicCache{T}
    numeric::NumericFactorCache{T}
    scratch::SolveScratch{T}
    backend::AbstractLABackend
end

function DenseFactorCache(
    ::Type{T},
    n::Integer;
    requested::Symbol=:auto,
    threads::Int=1,
    backend::Union{Nothing,AbstractLABackend}=nothing,
) where {T}
    la_backend = backend === nothing ?
        instantiate_la_backend(
            plan_la_backend(T; requested=requested, route=:dense_cholesky, threads=threads),
            T,
            threads,
        ) :
        backend
    return DenseFactorCache{T}(
        SymbolicCache(T, n),
        NumericFactorCache{T}(),
        SolveScratch(T, n),
        la_backend,
    )
end

# ---------------------------------------------------------------------------
#  Public interface
# ---------------------------------------------------------------------------

function prepare!(
    cache::DenseFactorCache{T},
    dimension::Integer,
) where {T}
    n = Int(dimension)
    n == cache.symbolic.dimension && return cache
    n >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    cache.symbolic = SymbolicCache(T, n)
    cache.scratch = SolveScratch(T, n)
    invalidate!(cache)
    return cache
end

function reserve!(
    cache::DenseFactorCache{T},
    dimension::Integer,
) where {T}
    return prepare!(cache, dimension)
end

function factorize!(
    cache::DenseFactorCache{T},
    A::AbstractMatrix{T},
    matrix_epoch::Integer,
) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("factorize! requires a square matrix"))
    n == cache.symbolic.dimension || prepare!(cache, n)
    if cache.numeric.matrix_epoch == Int(matrix_epoch) &&
       cache.numeric.factor !== nothing
        return (ok=true, refactorized=false, matrix_epoch=Int(matrix_epoch))
    end
    work = _owned_array_copy(T, A)
    factor = la_cholesky_factor!(cache.backend, work)
    if factor === nothing
        invalidate!(cache)
        return (ok=false, refactorized=true, matrix_epoch=Int(matrix_epoch))
    end
    cache.numeric.factor = factor
    cache.numeric.factor_kind = :cholesky
    cache.numeric.matrix_epoch = Int(matrix_epoch)
    cache.numeric.dimension = n
    return (ok=true, refactorized=true, matrix_epoch=Int(matrix_epoch))
end

isvalid(cache::FactorCache) = cache.numeric.factor !== nothing
matrix_epoch(cache::FactorCache) = cache.numeric.matrix_epoch
factor_kind(cache::FactorCache) = cache.numeric.factor_kind
factor_dimension(cache::FactorCache) = cache.numeric.dimension

function invalidate!(cache::FactorCache)
    cache.numeric.factor = nothing
    cache.numeric.factor_kind = :none
    cache.numeric.matrix_epoch = -1
    return cache
end

@inline function _require_valid_factor(cache::FactorCache)
    isvalid(cache) || throw(ArgumentError(
        "FactorCache has no valid numeric factor; call factorize! first",
    ))
    return cache.numeric.factor
end

function solve!(cache::DenseFactorCache{T}, rhs::AbstractVector{T}) where {T}
    factor = _require_valid_factor(cache)
    copyto!(cache.scratch.solution, rhs)
    la_factor_solve!(factor, cache.scratch.solution)
    return cache.scratch.solution
end

function solve!(
    cache::DenseFactorCache{T},
    rhs::AbstractVector{T},
    out::AbstractVector{T},
) where {T}
    factor = _require_valid_factor(cache)
    copyto!(out, rhs)
    la_factor_solve!(factor, out)
    return out
end

function solve_multi!(
    cache::DenseFactorCache{T},
    rhs::AbstractMatrix{T},
) where {T}
    factor = _require_valid_factor(cache)
    for column in 1:size(rhs, 2)
        la_factor_solve!(factor, view(rhs, :, column))
    end
    return rhs
end

function refine_once!(
    cache::DenseFactorCache{T},
    residual::AbstractVector{T},
    correction::AbstractVector{T},
) where {T}
    factor = _require_valid_factor(cache)
    copyto!(correction, residual)
    la_factor_solve!(factor, correction)
    return correction
end
