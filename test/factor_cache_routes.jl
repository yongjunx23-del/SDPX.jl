# Route-specific FactorCache implementations (Subagent E).
#
# Concrete Float64 route caches sharing the provider-neutral FactorCache
# protocol: DenseSchurCholeskyCache, DenseAugmentedLDLTCache, LPLUCache,
# EqualityRRQRCache, ArrowLocalCache, ArrowReducedCache,
# SparseSymbolicNumericCache.  Each battery covers
#   * factorize! + solve! on a nontrivial matrix,
#   * multi-RHS solve_multi!,
#   * refine_once!,
#   * same-epoch reuse (factor_epoch unchanged),
#   * new-epoch re-factorize (factor_epoch bumped, storage reused),
#   * the 0-byte warm factorize/solve/refine/skip contract.

using SDPX
using Test
using LinearAlgebra
using SparseArrays

const _S = SDPX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A nontrivial symmetric-positive-definite matrix of size n.
function _spd(n)
    M = randn(n, n)
    return M * M' + n * I
end

# A block-arrow SOC-style SPD matrix of total size n with a diagonal-block D
# of size d (the arrow's stem).  D is SPD block-diagonal, B an arrow block,
# and C is chosen so the whole matrix is SPD (C ≈ SPD + B'D⁻¹B).
function _arrow_matrix(n::Int, d::Int)
    d > 0 && d < n || throw(ArgumentError("d must satisfy 0 < d < n"))
    D = Matrix{Float64}(I, d, d)
    for b in 1:div(d, 2)
        M = randn(2, 2)
        D[2b-1:2b, 2b-1:2b] += M * M' + 6I
    end
    B = randn(d, n - d)
    C = _spd(n - d) + B' * (D \ B)
    return Matrix([D B; B' C])
end

# Full battery: prepare state, factorize/solve/multi/refine correctness,
# same-epoch reuse, new-epoch re-factorize, invalidate.
function _run_battery(name, cache, A, A2)
    n = size(A, 1)
    @testset "$name" begin
        @test _S.factor_status(cache) === _S.Prepared

        @test _S.factorize!(cache, A, 1) === cache
        @test _S.factor_status(cache) === _S.Fresh
        @test _S.factor_matrix_epoch(cache) == 1
        @test _S.factor_epoch(cache) == 1

        x_true = randn(n)
        b = A * x_true
        x = zeros(n)
        _S.solve!(cache, x, b)
        @test norm(x - x_true) < 1e-8

        k = 3
        X_true = randn(n, k)
        B = A * X_true
        X = zeros(n, k)
        _S.solve_multi!(cache, X, B)
        @test norm(X - X_true) < 1e-8

        residual = randn(n)
        correction = zeros(n)
        _S.refine_once!(cache, correction, residual)
        @test norm(A * correction - residual) < 1e-8

        # same-epoch reuse: no re-factorization.
        @test _S.factorize!(cache, A, 1) === cache
        @test _S.factor_status(cache) === _S.Fresh
        @test _S.factor_matrix_epoch(cache) == 1
        @test _S.factor_epoch(cache) == 1

        # new-epoch re-factorize: factor_epoch bumps, storage reused.
        @test _S.factorize!(cache, A2, 2) === cache
        @test _S.factor_status(cache) === _S.Fresh
        @test _S.factor_matrix_epoch(cache) == 2
        @test _S.factor_epoch(cache) == 2
        b2 = A2 * x_true
        _S.solve!(cache, x, b2)
        @test norm(x - x_true) < 1e-8

        _S.invalidate!(cache)
        @test _S.factor_status(cache) === _S.Invalid
        @test_throws _S.FactorCacheStateError _S.solve!(cache, x, b)
    end
end

# 0-byte warm-path contract: on an already-prepared, warmed cache, a new-epoch
# factorize!, solve!, refine_once!, a same-epoch skip and solve_multi! must
# each allocate zero Julia bytes.
function _run_zero_bytes(name, cache, A, A2, A3)
    n = size(A, 1)
    @testset "$name 0-byte warm path" begin
        _S.factorize!(cache, A, 1)
        _S.factorize!(cache, A2, 2)
        b = randn(n)
        x = zeros(n)
        corr = zeros(n)
        _S.solve!(cache, x, b)
        _S.refine_once!(cache, corr, b)
        _S.factorize!(cache, A2, 2)                # warm same-epoch skip
        Xm = zeros(n, 2)
        Bm = hcat(b, b)
        _S.solve_multi!(cache, Xm, Bm)             # warm multi

        af = @allocated _S.factorize!(cache, A3, 3)
        as = @allocated _S.solve!(cache, x, b)
        ar = @allocated _S.refine_once!(cache, corr, b)
        askip = @allocated _S.factorize!(cache, A3, 3)
        am = @allocated _S.solve_multi!(cache, Xm, Bm)
        @test af == 0
        @test as == 0
        @test ar == 0
        @test askip == 0
        @test am == 0
    end
end

# ---------------------------------------------------------------------------
# 1. DenseSchurCholeskyCache
# ---------------------------------------------------------------------------
n = 16
A  = _spd(n)
A2 = A + 0.25I
A3 = A2 + 0.5I
_run_battery("DenseSchurCholeskyCache", _S.DenseSchurCholeskyCache{Float64}(n), A, A2)
_run_zero_bytes("DenseSchurCholeskyCache", _S.DenseSchurCholeskyCache{Float64}(n), A, A2, A3)

# ---------------------------------------------------------------------------
# 2. DenseAugmentedLDLTCache (regularized augmented system)
# ---------------------------------------------------------------------------
B0 = randn(n, 2)
Aug  = [A B0; B0' -1.0 * Matrix{Float64}(I, 2, 2)]
Aug2 = Aug  + 0.05I
Aug3 = Aug2  + 0.05I
_run_battery("DenseAugmentedLDLTCache", _S.DenseAugmentedLDLTCache{Float64}(n + 2), Aug, Aug2)
_run_zero_bytes("DenseAugmentedLDLTCache", _S.DenseAugmentedLDLTCache{Float64}(n + 2), Aug, Aug2, Aug3)

# ---------------------------------------------------------------------------
# 3. LPLUCache (dense LU for the LP route)
# ---------------------------------------------------------------------------
_run_battery("LPLUCache", _S.LPLUCache{Float64}(n), A, A2)
_run_zero_bytes("LPLUCache", _S.LPLUCache{Float64}(n), A, A2, A3)

# ---------------------------------------------------------------------------
# 4. EqualityRRQRCache (rank-revealing QR with column pivoting)
# ---------------------------------------------------------------------------
_run_battery("EqualityRRQRCache", _S.EqualityRRQRCache{Float64}(n), A, A2)
_run_zero_bytes("EqualityRRQRCache", _S.EqualityRRQRCache{Float64}(n), A, A2, A3)

# ---------------------------------------------------------------------------
# 5. ArrowLocalCache — block-arrow with an SOC-style diagonal block
# ---------------------------------------------------------------------------
d_arrow = 8
narrow  = d_arrow + 8
Aarrow  = _arrow_matrix(narrow, d_arrow)
Aarrow2 = Aarrow  + 0.05I
Aarrow3 = Aarrow2  + 0.05I
_run_battery("ArrowLocalCache", _S.ArrowLocalCache{Float64}(narrow, d_arrow), Aarrow, Aarrow2)
_run_zero_bytes("ArrowLocalCache", _S.ArrowLocalCache{Float64}(narrow, d_arrow), Aarrow, Aarrow2, Aarrow3)

# ---------------------------------------------------------------------------
# 6. ArrowReducedCache — block-arrow with a precomputed reduced Schur factor
# ---------------------------------------------------------------------------
_run_battery("ArrowReducedCache", _S.ArrowReducedCache{Float64}(narrow, d_arrow), Aarrow, Aarrow2)
_run_zero_bytes("ArrowReducedCache", _S.ArrowReducedCache{Float64}(narrow, d_arrow), Aarrow, Aarrow2, Aarrow3)

# ---------------------------------------------------------------------------
# 7. SparseSymbolicNumericCache — sparse symbolic analysis + numeric factor
# ---------------------------------------------------------------------------
function _laplacian(nn)
    I, J, V = Int[], Int[], Float64[]
    for i in 1:nn, j in 1:nn
        k = (i - 1) * nn + j
        push!(I, k); push!(J, k); push!(V, 4.0)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            if 1 <= ii <= nn && 1 <= jj <= nn
                l = (ii - 1) * nn + jj
                push!(I, k); push!(J, l); push!(V, -1.0)
            end
        end
    end
    return sparse(I, J, V, nn * nn, nn * nn)
end
spn = 8
As  = _laplacian(spn) + sparse(1:spn*spn, 1:spn*spn, 1.0)
Am  = Matrix(As)
Am2 = Am  + 0.1I
Am3 = Am2  + 0.2I
_run_battery("SparseSymbolicNumericCache", _S.SparseSymbolicNumericCache{Float64}(As), Am, Am2)
_run_zero_bytes("SparseSymbolicNumericCache", _S.SparseSymbolicNumericCache{Float64}(As), Am, Am2, Am3)

# ---------------------------------------------------------------------------
# Fail-closed / state-machine behaviour across the route caches
# ---------------------------------------------------------------------------
@testset "route caches fail closed" begin
    bad = -Matrix{Float64}(I, 4, 4)
    c = _S.DenseSchurCholeskyCache{Float64}(4)
    @test_throws Exception _S.factorize!(c, bad, 1)
    @test _S.factor_status(c) === _S.Failed
    @test_throws _S.FactorCacheStateError _S.solve!(c, zeros(4), ones(4))

    for C in (_S.DenseSchurCholeskyCache{Float64}, _S.DenseAugmentedLDLTCache{Float64},
              _S.LPLUCache{Float64}, _S.EqualityRRQRCache{Float64})
        @test_throws _S.FactorCacheStateError _S.solve!(C(2), zeros(2), ones(2))
    end

    c2 = _S.DenseAugmentedLDLTCache{Float64}(2)
    @test_throws Exception _S.factorize!(c2, zeros(2, 2), 1)
    @test _S.factor_status(c2) === _S.Failed
end

# BigFloat still works through the generic LU fallback (LP route).
@testset "LPLUCache BigFloat" begin
    setprecision(BigFloat, 256) do
        M = BigFloat[4 3; 6 3]
        cache = _S.LPLUCache{BigFloat}(2)
        _S.factorize!(cache, M, 1)
        b = BigFloat[1, 2]
        x = zeros(BigFloat, 2)
        _S.solve!(cache, x, b)
        @test M * x ≈ b
    end
end
