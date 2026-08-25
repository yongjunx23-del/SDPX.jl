# test/cone_algebra.jl
#
# Wave A-5: ConeAlgebra math validation (orthant / SOC / PSD).
# Covers Float64, Float64x2, Float64x4, and BigFloat256.

using Test
using LinearAlgebra

const _HAVE_MULTIFLOATS = try
    @eval import MultiFloats
    true
catch
    false
end

const _CA = SDPX.ConeAlgebra

function _test_orthant(::Type{T}) where {T}
    @testset "orthant ($T)" begin
        x = T[2, 4, 8]
        s = T[1, 2, 4]
        w = _CA.orthant_nt_scaling(x, s)
        @test w ≈ sqrt.(x ./ s)
        # NT identity W*Y*W = X with Y = s
        @test w .* s .* w ≈ x
        @test _CA.orthant_jordan_product(x, s) ≈ x .* s
        @test _CA.orthant_sqrt(x) ≈ sqrt.(x)
        @test _CA.orthant_inverse(x) ≈ 1 ./ x
        @test _CA.orthant_boundary_step(T[1, 2], T[-1, 0.5]) ≈ T(1)
        @test _CA.orthant_boundary_step(T[1, 2], T[1, 0.5]) == T(Inf)
        # mutating workspace API
        w1 = zeros(T, 1)
        @test _CA.orthant_boundary_step!(w1, T[1, 2], T[-1, 0.5]) ≈ T(1)
        @test w1[1] ≈ T(1)
    end
end

function _test_soc(::Type{T}) where {T}
    @testset "SOC ($T)" begin
        # analytic boundary-step cases
        @test _CA.soc_boundary_step(T[2, 0], T[-1, 0]) ≈ T(2)
        @test _CA.soc_boundary_step(T[2, 0], T[-1, 1]) ≈ T(1)
        # zero-tail idempotents: x=(t,0,0) -> two (1/2,0,0) idempotents
        lam1, lam2, c1, c2 = _CA.soc_spectral(T[3, 0, 0])
        @test lam1 ≈ T(3)
        @test lam2 ≈ T(3)
        @test c1 ≈ T[0.5, 0, 0]
        @test c2 ≈ T[0.5, 0, 0]
        # spectral reconstruction x = λ1*c1 + λ2*c2
        x = T[3, 1, 0.5]
        lam1, lam2, c1, c2 = _CA.soc_spectral(x)
        @test lam1 * c1 + lam2 * c2 ≈ x
        # sqrt roundtrip
        r = _CA.soc_sqrt(x)
        @test _CA.soc_jordan_product(r, r) ≈ x
        # zero-tail sqrt
        r0 = _CA.soc_sqrt(T[4, 0, 0])
        @test _CA.soc_jordan_product(r0, r0) ≈ T[4, 0, 0]
        # inverse roundtrip (identity is (1,0,..))
        invx = _CA.soc_inverse(x)
        @test _CA.soc_jordan_product(x, invx) ≈ T[1, 0, 0]
        # edge cases: tangential, outside-cone, boundary-outward
        @test _CA.soc_boundary_step(T[1, 0], T[0, 1]) ≈ T(1)
        @test _CA.soc_boundary_step(T[-1, 0], T[1, 0]) == T(0)
        @test _CA.soc_boundary_step(T[1, 1], T[0, 1]) == T(0)
        # mutating workspace API
        w1 = zeros(T, 1)
        @test _CA.soc_boundary_step!(w1, T[2, 0], T[-1, 0]) ≈ T(2)
        @test w1[1] ≈ T(2)
    end
end

function _test_psd(::Type{T}) where {T}
    @testset "PSD ($T)" begin
        X = T[3 1; 1 2]
        Y = T[2 0.5; 0.5 1.5]
        # NT-scaling identity W*Y*W ≈ X
        W = _CA.psd_nt_scaling(X, Y)
        @test W * Y * W ≈ X
        # sqrt roundtrip
        R = _CA.psd_sqrt(X)
        @test R * R ≈ X
        # inverse roundtrip
        Xi = _CA.psd_inverse(X)
        @test X * Xi ≈ Matrix{T}(I, 2, 2)
        # jordan product
        @test _CA.psd_jordan_product(X, Y) ≈ (X * Y + Y * X) / 2
        # boundary step on a concrete example
        B = T[2 0; 0 1]
        dB = T[-1 0; 0 1]
        @test _CA.psd_boundary_step(B, dB) ≈ T(2)
        # mutating workspace API
        w1 = zeros(T, 1)
        @test _CA.psd_boundary_step!(w1, B, dB) ≈ T(2)
        @test w1[1] ≈ T(2)
    end
end

@testset "ConeAlgebra" begin
    @testset "dispatched API" begin
        x = [3.0, 1.0, 0.5]
        @test _CA.sqrt(_CA.LorentzCone(), x) ≈ _CA.soc_sqrt(x)
        @test _CA.inverse(_CA.LorentzCone(), x) ≈ _CA.soc_inverse(x)
        @test _CA.jordan_product(_CA.LorentzCone(), x, x) ≈
            _CA.soc_jordan_product(x, x)
        @test _CA.boundary_step(_CA.LorentzCone(), [2.0, 0.0], [-1.0, 0.0]) ≈ 2.0
        X = [3.0 1.0; 1.0 2.0]
        @test _CA.sqrt(_CA.PSDCone(), X) ≈ _CA.psd_sqrt(X)
        @test _CA.inverse(_CA.PSDCone(), X) ≈ _CA.psd_inverse(X)
        @test _CA.nt_scaling(_CA.PSDCone(), X, X) ≈ _CA.psd_nt_scaling(X, X)
        o = [2.0, 4.0]
        @test _CA.sqrt(_CA.OrthantCone(), o) ≈ _CA.orthant_sqrt(o)
        @test _CA.nt_scaling(_CA.OrthantCone(), o, [1.0, 2.0]) ≈
            _CA.orthant_nt_scaling(o, [1.0, 2.0])
    end

    _test_orthant(Float64)
    _test_soc(Float64)
    _test_psd(Float64)

    setprecision(BigFloat, 256) do
        _test_orthant(BigFloat)
        _test_soc(BigFloat)
        _test_psd(BigFloat)
    end

    if _HAVE_MULTIFLOATS
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x4)
            _test_orthant(T)
            _test_soc(T)
            _test_psd(T)
        end
    end
end

@testset "ConeAlgebra PSD inverse identity" begin
    for T in (Float64, BigFloat)
        T === BigFloat && setprecision(BigFloat, 256)
        X = T[3 1; 1 2]
        I2 = Matrix{T}(I, 2, 2)
        Xi = _CA.psd_inverse(X)
        @test isapprox(Xi * X, I2; atol=T(1e-10))
        @test isapprox(_CA.psd_jordan_product(Xi, X), I2; atol=T(1e-10))
    end
end

@testset "ConeAlgebra SOC commutativity + inverse identity" begin
    for T in (Float64, BigFloat)
        T === BigFloat && setprecision(BigFloat, 256)
        x = T[2, 1, 0]
        y = T[3, -1, 1]
        # Jordan product is commutative.
        @test isapprox(_CA.soc_jordan_product(x, y), _CA.soc_jordan_product(y, x); atol=T(1e-10))
        # x o x^{-1} = e = (1,0,0).
        e = T[1, 0, 0]
        @test isapprox(_CA.soc_jordan_product(x, _CA.soc_inverse(x)), e; atol=T(1e-10))
    end
end
