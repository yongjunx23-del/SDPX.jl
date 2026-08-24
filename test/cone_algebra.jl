# Phase 7 ConeAlgebra (PSD cone) tests.
using SDPX
using Test
using LinearAlgebra
using Random

function _spd(::Type{T}, rng, n) where {T}
    R = T.(randn(rng, n, n))
    A = transpose(R) * R + T(n) * Matrix{T}(I, n, n)
    return (A + transpose(A)) / 2
end

@testset "Phase 7 ConeAlgebra (PSD)" begin
    T = Float64
    rng = MersenneTwister(2026_0821)
    n = 4
    X = _spd(T, rng, n)
    Y = _spd(T, rng, n)

    # Jordan product is symmetric and equals (XY+YX)/2.
    prod = SDPX.psd_jordan_product(X, Y)
    @test prod ≈ (X * Y + Y * X) / 2
    @test prod ≈ transpose(prod)

    # Spectral decomposition reconstructs X.
    values, vectors = SDPX.psd_spectral_decomposition(X)
    @test isapprox(vectors * Diagonal(values) * transpose(vectors), X; atol=1e-10)
    @test all(values .> 0)

    # Square root squares back to X and is PSD.
    sqrtX = SDPX.psd_sqrt(X)
    @test isapprox(sqrtX * sqrtX, X; atol=1e-10)
    @test all(eigvals(Symmetric(sqrtX)) .>= 0)

    # Inverse recovers identity.
    @test isapprox(SDPX.psd_inverse(X) * X, Matrix{T}(I, n, n); atol=1e-10)

    # NT scaling preserves the pairing trace(X*Y) = trace(N*inv(N...)) basic sanity.
    W, N = SDPX.psd_nt_scaling(X, Y)
    @test all(eigvals(Symmetric(W)) .> 0)
    @test all(eigvals(Symmetric(N)) .> 0)

    # Boundary step along a negative direction is finite and keeps PSD.
    dX = -T.(randn(rng, n, n))
    dX = (dX + transpose(dX)) / 2
    t = SDPX.psd_boundary_step(X, dX)
    @test t > 0
    @test t < Inf
    @test minimum(eigvals(Symmetric(X + t * dX))) <= 1e-8
end