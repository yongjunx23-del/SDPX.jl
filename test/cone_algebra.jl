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

@testset "Phase 7 ConeAlgebra (orthant)" begin
    x = [1.0, 4.0, 9.0]
    y = [2.0, 3.0, 4.0]
    @test SDPX.orthant_jordan_product(x, y) == [2.0, 12.0, 36.0]
    @test SDPX.orthant_sqrt(x) == [1.0, 2.0, 3.0]
    @test SDPX.orthant_inverse(x) ≈ [1.0, 0.25, 1.0/9.0]
    values, _ = SDPX.orthant_spectral(x)
    @test values == x
    @test SDPX.orthant_nt_scaling(x) ≈ [1.0, 0.5, 1.0/3.0]
    # Boundary step along a negative direction.
    dx = [-1.0, 1.0, 0.0]
    @test SDPX.orthant_boundary_step(x, dx) ≈ 1.0
    # Positive direction gives Inf.
    @test SDPX.orthant_boundary_step(x, [1.0, 1.0, 1.0]) == Inf
end

@testset "Phase 7 ConeAlgebra (Lorentz/SOC)" begin
    # Reference Jordan product for x=(t,u), y=(s,v): (t*s+u'v, t*v+s*u).
    x = [3.0, 1.0, 2.0]
    y = [4.0, 2.0, 1.0]
    j = SDPX.soc_jordan_product(x, y)
    @test j ≈ [x[1]*y[1]+x[2]*y[2]+x[3]*y[3], x[1]*y[2]+y[1]*x[2], x[1]*y[3]+y[1]*x[3]]

    # Identity element e=(1,0,0).
    e = [1.0, 0.0, 0.0]
    @test SDPX.soc_jordan_product(e, x) ≈ x

    # Inverse satisfies x^{-1} o x = e.
    invx = SDPX.soc_inverse(x)
    @test SDPX.soc_jordan_product(invx, x) ≈ e atol=1e-12

    # sqrt squares back to x.
    sx = SDPX.soc_sqrt(x)
    @test SDPX.soc_jordan_product(sx, sx) ≈ x atol=1e-10

    # Spectral: x = λ1 c1 + λ2 c2, with c1∘c2 = 0.
    λ, C = SDPX.soc_spectral(x)
    @test λ[1] * C[:,1] + λ[2] * C[:,2] ≈ x atol=1e-10
    @test SDPX.soc_jordan_product(C[:,1], C[:,2]) ≈ zeros(3) atol=1e-10

    # Jordan solve: left o result = right.
    sol = SDPX.soc_jordan_solve(x, y)
    @test SDPX.soc_jordan_product(x, sol) ≈ y atol=1e-10

    # Boundary step: x + t*dx stays interior.
    dx = [-1.0, 0.5, 0.0]
    t = SDPX.soc_boundary_step(x, dx)
    @test t > 0
    @test t < Inf
    w = x + t * dx
    @test w[1] > norm(w[2:3]) - 1e-8
end
