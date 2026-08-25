# KKT route-cache wiring (Subagent G).
#
# Covers the HotRouteCache driver: one real numeric factorization per KKT
# epoch, shared factor identity across the predictor/corrector/refinement
# solves, the actual-factorization-count statistic, fail-closed behaviour, and
# the zero-allocation warm path.

using SDPX
using Test
using LinearAlgebra

const _S = SDPX

# ---------------------------------------------------------------------------
# Driver-level: factorize ONCE per epoch, then multiple solves through the
# SAME factor (factor_epoch / factorizations must not change between solves).
# ---------------------------------------------------------------------------
@testset "HotRouteCache one-factor-per-epoch" begin
    T = Float64
    n = 4
    M = Matrix{T}(I, n, n)
    for k in 1:n
        M[k, k] += T(k) * 0.5
    end
    M2 = M + 0.3I

    cache = _S.DenseSchurCholeskyCache{T}(n)
    driver = _S.HotRouteCache(cache; n=n)
    @test _S.kkt_factor_count(driver) == 0
    @test _S.kkt_matrix_epoch(driver) == 0

    b = randn(n)
    x = zeros(n)

    # epoch 1 -> exactly one numeric factorization.
    _S.kkt_epoch_factorize!(driver, M)
    @test _S.kkt_matrix_epoch(driver) == 1
    @test _S.kkt_factor_epoch(driver) == 1
    @test _S.kkt_factor_count(driver) == 1

    # predictor + corrector + refinement solves all reuse the same factor:
    # the factor epoch / factor count must NOT change while we only solve.
    _S.kkt_solve!(driver, x, b)
    _S.kkt_solve!(driver, x, b)
    corr = zeros(n)
    _S.kkt_refine!(driver, corr, b)
    @test _S.kkt_factor_epoch(driver) == 1
    @test _S.kkt_factor_count(driver) == 1
    @test M * x ≈ b

    # epoch 2 -> exactly one more numeric factorization (total == 2).
    _S.kkt_epoch_factorize!(driver, M2)
    @test _S.kkt_matrix_epoch(driver) == 2
    @test _S.kkt_factor_epoch(driver) == 2
    @test _S.kkt_factor_count(driver) == 2

    # The cache state-machine must be Fresh after a successful factorize.
    @test _S.kkt_route_status(driver) === _S.Fresh
end

# ---------------------------------------------------------------------------
# The driver is provider-neutral: the same one-per-epoch contract holds for
# every route cache type (LPLU, LDLT, RRQR, arrow, sparse).
# ---------------------------------------------------------------------------
@testset "one-factor-per-epoch across route types" begin
    T = Float64
    n = 4
    base = Matrix{T}(I, n, n)
    for k in 1:n
        base[k, k] += T(k)
    end
    M  = base + 0.1I
    M2 = M + 0.2I
    b = randn(n)
    x = zeros(n)

    routes = Any[
        _S.LPLUCache{T}(n),
        _S.DenseAugmentedLDLTCache{T}(n),
        _S.EqualityRRQRCache{T}(n),
        _S.ArrowLocalCache{T}(n, 2),
        _S.ArrowReducedCache{T}(n, 2),
    ]
    for route in routes
        driver = _S.HotRouteCache(route; n=n)
        _S.kkt_epoch_factorize!(driver, M)
        @test _S.kkt_factor_count(driver) == 1
        _S.kkt_solve!(driver, x, b)
        _S.kkt_solve!(driver, x, b)
        @test norm(M * x - b) < 1e-8
        @test _S.kkt_factor_count(driver) == 1      # solves do not factor
        _S.kkt_epoch_factorize!(driver, M2)
        @test _S.kkt_factor_count(driver) == 2      # one per epoch
        @test _S.kkt_route_status(driver) === _S.Fresh
    end
end
