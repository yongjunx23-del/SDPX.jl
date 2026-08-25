#=
    Optional LinearSolve bridge (Wave A-2 A/B spike).

This extension exposes a minimal, explicit-algorithm factor cache over
LinearSolve.  It deliberately does NOT use LinearSolve's automatic
polyalgorithm (`defaultalg`): the caller must pass the algorithm explicitly
(e.g. `LinearSolve.LUFactorization()`), so the factorization path is fully
deterministic and comparable against SDPX's native dense Cholesky solve.

The wrapper is a thin, allocation-light adapter around a `LinearSolve.LinearCache`:

  * `prepare!(cache, A)`   (re)initialise the cache for matrix `A`; marks stale.
  * `factorize!(cache, A)` factorise `A`; marks fresh.
  * `solve!(cache, dest, rhs)` solve `A * x = rhs` into `dest`, reusing the
    cached factorization when it is fresh.
  * `solve_multi!(cache, dest, rhs)` solve a batch of right-hand sides
    (column-wise, reusing the factorization).
  * `invalidate!(cache)`   mark the cached factorization stale.
  * `factor_status(cache)` returns `:fresh` / `:stale`.
  * `factor_matrix_epoch(cache)` returns the epoch of the matrix currently
    loaded in the cache.

No SDPX solver hot path is touched: this is purely an optional, opt-in wrapper.
=#
module SDPXLinearSolveExt

using SDPX
using LinearSolve
using SciMLBase

export LinearSolveFactorCache,
    prepare!, factorize!, solve!, solve_multi!, invalidate!,
    factor_status, factor_matrix_epoch

"""
    LinearSolveFactorCache(alg)

A minimal explicit-algorithm factor cache backed by LinearSolve. `alg` must be
a concrete `SciMLLinearSolveAlgorithm` (e.g. `LinearSolve.LUFactorization()`);
the automatic polyalgorithm is never used.
"""
struct LinearSolveFactorCache{Alg,Cache}
    alg::Alg
    cache::Cache
end

"""
    LinearSolveCacheState

Mutable bookkeeping for a `LinearSolveFactorCache`: the underlying
`LinearSolve.LinearCache` (or `nothing` before the first `prepare!`), the
fresh/stale status, and the matrix epoch counters.
"""
mutable struct LinearSolveCacheState
    lc::Any
    status::Symbol
    epoch::Int
    matrix_epoch::Int
end

function LinearSolveFactorCache(alg::LinearSolve.SciMLLinearSolveAlgorithm)
    state = LinearSolveCacheState(nothing, :stale, 0, 0)
    return LinearSolveFactorCache(alg, state)
end

# --- status / epoch queries -------------------------------------------------

factor_status(cache::LinearSolveFactorCache) = cache.cache.status
factor_matrix_epoch(cache::LinearSolveFactorCache) = cache.cache.matrix_epoch

# --- lifecycle --------------------------------------------------------------

"""
    prepare!(cache, A)

(Re)initialise the cache for matrix `A`. A fresh `LinearSolve.LinearCache` is
built with the explicit algorithm; the cached factorization is marked stale and
the matrix epoch is advanced.
"""
function prepare!(cache::LinearSolveFactorCache, A::AbstractMatrix)
    state = cache.cache
    state.epoch += 1
    b0 = zeros(eltype(A), size(A, 1))
    state.lc = init(LinearProblem(A, b0), cache.alg)
    state.matrix_epoch = state.epoch
    state.status = :stale
    return cache
end

"""
    factorize!(cache, A)

Factorise `A` with the explicit algorithm. If the cache is not yet prepared for
`A` (or was prepared for a different matrix), it is re-initialised first.
Afterwards the cached factorization is fresh.
"""
function factorize!(cache::LinearSolveFactorCache, A::AbstractMatrix)
    state = cache.cache
    if state.lc === nothing || state.matrix_epoch != state.epoch
        prepare!(cache, A)
    else
        # Same matrix epoch: point the cache at `A` and mark it for
        # refactorization on the next `solve!`.
        state.lc.A = A
    end
    # `solve!` factorizes when the cache is fresh; the zero RHS makes the solve
    # part negligible so this call is dominated by the factorization.
    LinearSolve.solve!(state.lc)
    state.status = :fresh
    return cache
end

"""
    solve!(cache, dest, rhs)

Solve `A * x = rhs` into `dest`, reusing the cached factorization when it is
fresh. If the factorization is stale, it is recomputed first. The solution is
always copied into `dest`, so `dest` may be a plain vector or a view.
"""
function solve!(cache::LinearSolveFactorCache, dest::AbstractVector, rhs::AbstractVector)
    state = cache.cache
    state.lc.b = rhs
    sol = LinearSolve.solve!(state.lc)
    copyto!(dest, sol.u)
    state.status = :fresh
    return dest
end

"""
    solve_multi!(cache, dest, rhs)

Solve `A * X = RHS` for a batch of right-hand sides (columns of `rhs`) into the
columns of `dest`, reusing the cached factorization across columns.
"""
function solve_multi!(
    cache::LinearSolveFactorCache,
    dest::AbstractMatrix,
    rhs::AbstractMatrix,
)
    size(dest) == size(rhs) ||
        throw(DimensionMismatch("dest and rhs must have the same size"))
    for j in axes(rhs, 2)
        solve!(cache, view(dest, :, j), view(rhs, :, j))
    end
    return dest
end

"""
    invalidate!(cache)

Mark the cached factorization stale so the next `solve!` refactorizes.
"""
function invalidate!(cache::LinearSolveFactorCache)
    cache.cache.status = :stale
    return cache
end

end # module SDPXLinearSolveExt
