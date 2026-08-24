# Phase 3 FactorCache contract tests.
#
# Exercises the provider-neutral FactorCache interface across the full
# arithmetic family: Float64, Float64x2/x3/x4 (MultiFloats), and BigFloat.
# Verifies the matrix_epoch contract (factor once, reuse for predictor/
# corrector/refinement), single- and multi-RHS solve reuse, refinement, and
# invalidation.
using SDPX
using Test
using LinearAlgebra
using Random

# Load MultiFloats when available so the full arithmetic family is exercised.
const _MULTIFLOATS_AVAILABLE = try
    @eval import MultiFloats
    true
catch
    false
end


function _pd_matrix(::Type{T}, rng, n::Int) where {T}
    R = T.(randn(rng, n, n))
    A = transpose(R) * R + T(n) * Matrix{T}(I, n, n)
    return A
end

function _residual_ratio(A, x, b)
    return maximum(abs.(A * x .- b)) / max(maximum(abs.(b)), one(eltype(b)))
end

function _ref_tol(::Type{T}) where {T}
    T === Float64 && return T(1e-8)
    return T(1e-12)
end

function _exercise_factor_cache(::Type{T}) where {T}
    rng = MersenneTwister(2026_0819)
    n = 4
    A = _pd_matrix(T, rng, n)
    b1 = T.(randn(rng, n))
    b2 = T.(randn(rng, n))
    cache = SDPX.DenseFactorCache(T, n)
    @test cache isa SDPX.FactorCache{T}
    @test !SDPX.isvalid(cache)
    @test SDPX.matrix_epoch(cache) == -1

    # First factorization at epoch 1 must actually factor.
    first = SDPX.factorize!(cache, A, 1)
    @test first.ok
    @test first.refactorized
    @test SDPX.isvalid(cache)
    @test SDPX.factor_kind(cache) === :cholesky
    @test SDPX.matrix_epoch(cache) == 1

    # Same epoch must NOT refactor (factor-once reuse).
    reuse = SDPX.factorize!(cache, A, 1)
    @test reuse.ok
    @test !reuse.refactorized

    # A new epoch refactors.
    refactor = SDPX.factorize!(cache, A, 2)
    @test refactor.ok
    @test refactor.refactorized
    @test SDPX.matrix_epoch(cache) == 2

    # Single-RHS solve.
    x1 = SDPX.solve!(cache, b1)
    @test _residual_ratio(A, x1, b1) <= _ref_tol(T)
    # solve into a caller buffer.
    out = SDPX.alloc_zeros(T, n)
    SDPX.solve!(cache, b2, out)
    @test _residual_ratio(A, out, b2) <= _ref_tol(T)

    # Multi-RHS solve reuses the one factor.
    RHS = hcat(b1, b2)
    solved = SDPX.solve_multi!(cache, copy(RHS))
    @test _residual_ratio(A, view(solved, :, 1), b1) <= _ref_tol(T)
    @test _residual_ratio(A, view(solved, :, 2), b2) <= _ref_tol(T)

    # Iterative refinement: one correction step from a perturbed solution.
    x_perturbed = T.(collect(1:n))
    residual = SDPX.alloc_zeros(T, n)
    correction = SDPX.alloc_zeros(T, n)
    # residual = rhs - A*x_perturbed
    for i in 1:n
        residual[i] = b1[i]
    end
    ax = A * x_perturbed
    for i in 1:n
        residual[i] -= ax[i]
    end
    SDPX.refine_once!(cache, residual, correction)
    improved = x_perturbed .+ correction
    @test _residual_ratio(A, improved, b1) <= _residual_ratio(A, x_perturbed, b1)

    # Invalidate forces a refactor on the next factorize!.
    SDPX.invalidate!(cache)
    @test !SDPX.isvalid(cache)
    @test SDPX.matrix_epoch(cache) == -1
    again = SDPX.factorize!(cache, A, 1)
    @test again.ok
    @test again.refactorized

    # solve! before factorize! must error.
    SDPX.invalidate!(cache)
    @test_throws ArgumentError SDPX.solve!(cache, b1)
    return cache
end

@testset "Phase 3 FactorCache" begin
    for T in (Float64, BigFloat)
        @testset "FactorCache $T" begin
            setprecision(BigFloat, 256) do
                _exercise_factor_cache(T)
            end
        end
    end
    if _MULTIFLOATS_AVAILABLE
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
            @testset "FactorCache $T" begin
                _exercise_factor_cache(T)
            end
        end
    end
end
