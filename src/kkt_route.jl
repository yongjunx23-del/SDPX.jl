#=====================================================================#
#    Route-cache KKT driver (Subagent G).
#
#    Wires the real route FactorCaches (src/factor_cache/routes) into the
#    production KKT solve.  The contract is deliberately small:
#
#      * `kkt_epoch_factorize!` bumps the matrix epoch and asks the route
#        cache to `factorize!` the assembled matrix ONCE.  Because every step
#        uses a fresh matrix epoch, this is exactly one real numeric
#        factorization per KKT epoch (the cache skips when already `Fresh` for
#        the same epoch, but a new epoch always re-factors).
#      * the predictor solve, corrector solve, and zero-or-more refinement
#        solves then all run through the SAME cache (shared factor identity).
#        `kkt_solve!` / `kkt_refine!` are thin wrappers over the cache's own
#        `solve!` / `refine_once!`; allocation behavior is cache-specific
#        (e.g. the Float64 CHOLMOD cache uses a public allocating
#        `factor \\ rhs`), so the driver itself does not promise
#        allocation-free solves.
#      * `factorizations(driver)` is the actual-factorization-count statistic:
#        it counts REAL numeric factor calls (the route's `factor_epoch`
#        delta), so per KKT epoch it is exactly 1.
#      * line search never touches the factor cache (see step_hot.jl).
#
#    The driver is generic over the route cache element type `T` and the
#    concrete route `R <: AbstractFactorCache{T}` (DenseSchurCholeskyCache,
#    DenseAugmentedLDLTCache, LPLUCache, EqualityRRQRCache, ArrowLocalCache,
#    ArrowReducedCache, SparseSymbolicNumericCache).  All fields are concrete;
#    no `Any`, no `Symbol`, no closure capture.
#=====================================================================#

"""
    HotRouteCache{T, R<:AbstractFactorCache{T}}

A thin KKT driver around a concrete route FactorCache `R`.  Owns the matrix
epoch bookkeeping and the actual-factorization-count statistic.  `n` is the
dimension of the square KKT / Schur matrix this driver factors.
"""
mutable struct HotRouteCache{T, R<:AbstractFactorCache{T}}
    route::R
    n::Int
    matrix_epoch::Int      # bumped once per KKT epoch
    factorizations::Int    # real numeric factor calls (route.factor_epoch delta)
end

function HotRouteCache(route::R; n::Integer) where {T, R<:AbstractFactorCache{T}}
    return HotRouteCache{T, R}(route, Int(n), 0, 0)
end

"""
    kkt_epoch_factorize!(driver, M) -> driver

Bump the matrix epoch and factor the freshly-assembled matrix `M` exactly once
through the wrapped route cache.  Because the epoch is new every call, the
route always re-factors (never a same-epoch skip).  Returns `driver`.
"""
@inline function kkt_epoch_factorize!(
    driver::HotRouteCache{T, R}, M::AbstractMatrix{T},
) where {T, R}
    driver.matrix_epoch += 1
    prev = factor_epoch(driver.route)
    factorize!(driver.route, M, driver.matrix_epoch)
    driver.factorizations += factor_epoch(driver.route) - prev
    return driver
end

"""
    kkt_solve!(driver, destination, rhs)

Solve the (shared) factorized system for one right-hand side.  Legal only when
the route cache is `Fresh` (fail-closed); reuses the factor produced by
[`kkt_epoch_factorize!`](@ref).  Allocation behavior is cache-specific (for
    example the Float64 CHOLMOD cache uses a public allocating `factor \\ rhs`);
    this driver does not promise an allocation-free solve.
"""
@inline function kkt_solve!(
    driver::HotRouteCache{T, R}, destination::AbstractVector{T}, rhs::AbstractVector{T},
) where {T, R}
    return solve!(driver.route, destination, rhs)
end

"""
    kkt_refine!(driver, residual, correction)

One step of iterative refinement through the shared factor.  Legal only when
the route is `Fresh`; reuses the factor produced by
[`kkt_epoch_factorize!`](@ref).  Allocation behavior is cache-specific; this
    driver does not promise an allocation-free refinement.
"""
@inline function kkt_refine!(
    driver::HotRouteCache{T, R}, correction::AbstractVector{T}, residual::AbstractVector{T},
) where {T, R}
    return refine_once!(driver.route, residual, correction)
end

"""
    kkt_factor_count(driver) -> Int

The actual-factorization-count statistic: the number of real numeric factor
calls performed so far.  Per KKT epoch this is exactly 1.
"""
@inline kkt_factor_count(driver::HotRouteCache) = driver.factorizations

"""
    kkt_matrix_epoch(driver) -> Int
    kkt_factor_epoch(driver) -> Int

Cold-path epoch accessors (cheap, allocation-free).
"""
@inline kkt_matrix_epoch(driver::HotRouteCache) = driver.matrix_epoch
@inline kkt_factor_epoch(driver::HotRouteCache) = factor_epoch(driver.route)

"""
    kkt_route_status(driver) -> FactorCacheState

Expose the wrapped route's state-machine status (fail-closed discipline).
"""
@inline kkt_route_status(driver::HotRouteCache) = factor_status(driver.route)

"""
    kkt_route_diagnostics(driver) -> NamedTuple

COLD-path only.  Builds a human-readable diagnostics NamedTuple for the wrapped
route cache.  Never called on the hot path (see step_hot.jl).
"""
function kkt_route_diagnostics(driver::HotRouteCache{T, R}) where {T, R}
    return (
        route = nameof(R),
        T = T,
        n = driver.n,
        matrix_epoch = driver.matrix_epoch,
        factor_epoch = factor_epoch(driver.route),
        factorizations = driver.factorizations,
        status = factor_status(driver.route),
    )
end
