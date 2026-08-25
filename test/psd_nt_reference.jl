using SDPX
using Test
using LinearAlgebra
using Random

const PSDNT_SC = SDPX.SymmetricCones

function _psdnt_svec(X::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    out = Vector{T}(undef, div(n * (n + 1), 2))
    sqrt2 = sqrt(T(2))
    k = 1
    @inbounds for j in 1:n, i in j:n
        out[k] = i == j ? X[i, j] : sqrt2 * X[i, j]
        k += 1
    end
    return out
end

function _psdnt_unsvec(x::AbstractVector{T}, n::Int) where {T}
    X = zeros(T, n, n)
    invsqrt2 = one(T) / sqrt(T(2))
    k = 1
    @inbounds for j in 1:n, i in j:n
        value = i == j ? x[k] : invsqrt2 * x[k]
        X[i, j] = value
        X[j, i] = value
        k += 1
    end
    return X
end

function _psdnt_rawpack(X::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    out = Vector{T}(undef, div(n * (n + 1), 2))
    k = 1
    @inbounds for j in 1:n, i in j:n
        out[k] = X[i, j]
        k += 1
    end
    return out
end

function _psdnt_spd(::Type{T}, n::Int, shift::Int) where {T}
    L = zeros(T, n, n)
    @inbounds for j in 1:n, i in j:n
        L[i, j] = i == j ? T(i + shift + 1) : T(i + j + shift) / T(11n)
    end
    return L * transpose(L) + Matrix{T}(I, n, n)
end

function _psdnt_reference_case(::Type{T}, n::Int; rtol, atol) where {T}
    cone = PSDNT_SC.PSDTriangleCone{T}(n)
    state = PSDNT_SC.PSDNTScaling{T}(n)
    S = _psdnt_spd(T, n, 1)
    Y = _psdnt_spd(T, n, 2)
    s = _psdnt_svec(S)
    y = _psdnt_svec(Y)
    Z = _psdnt_spd(T, n, 3)
    z = _psdnt_svec(Z)
    out = zeros(T, length(s))
    tmp = similar(out)
    tmp2 = similar(out)

    PSDNT_SC.nt_scaling!(cone, state, s, y)
    @test state.P * Y * state.P ≈ S rtol=rtol atol=atol
    @test state.Pinv * S * state.Pinv ≈ Y rtol=rtol atol=atol

    PSDNT_SC.theta_apply!(cone, out, state, y)
    @test out ≈ s rtol=rtol atol=atol
    PSDNT_SC.g_apply!(cone, out, state, s)
    @test out ≈ y rtol=rtol atol=atol
    PSDNT_SC.r_apply!(cone, out, state, y)
    PSDNT_SC.r_inverse_apply!(cone, tmp, state, s)
    @test out ≈ tmp rtol=rtol atol=atol
    @test _psdnt_unsvec(out, n) ≈ state.Lambda rtol=rtol atol=atol

    PSDNT_SC.theta_apply!(cone, out, state, z)
    PSDNT_SC.g_apply!(cone, tmp, state, out)
    @test tmp ≈ z rtol=rtol atol=atol
    PSDNT_SC.r_apply!(cone, tmp, state, z)
    PSDNT_SC.r_apply!(cone, tmp2, state, tmp)
    @test tmp2 ≈ out rtol=rtol atol=atol

    rhs = _psdnt_svec(S + Y / T(n + 1))
    PSDNT_SC.solve_Llambda!(cone, out, state, rhs)
    Xsol = _psdnt_unsvec(out, n)
    RHS = _psdnt_unsvec(rhs, n)
    @test (state.Lambda * Xsol + Xsol * state.Lambda) / T(2) ≈ RHS rtol=rtol atol=atol

    d1 = _psdnt_svec(S / T(n + 2))
    d2 = _psdnt_svec(Y / T(n + 3))
    d3 = z
    PSDNT_SC.third_order_correction!(cone, state, out, d1, d2, d3)
    D1, D2, D3 = _psdnt_unsvec(d1, n), _psdnt_unsvec(d2, n), Z
    inner = (D2 * D3 + D3 * D2) / T(2)
    expected = (D1 * inner + inner * D1) / T(2)
    @test _psdnt_unsvec(out, n) ≈ expected rtol=rtol atol=atol

    W = _psdnt_spd(T, n, 4)
    X = S - Y / T(n + 2)
    Wraw, Xraw = _psdnt_rawpack(W), _psdnt_rawpack(X)
    PSDNT_SC.scaling_inverse_apply!(cone, out, Wraw, Xraw)
    # Legacy packed coordinates have no sqrt(2) scaling.
    Zsol = begin
        Zraw = zeros(T, n, n)
        k = 1
        @inbounds for j in 1:n, i in j:n
            Zraw[i, j] = out[k]
            Zraw[j, i] = out[k]
            k += 1
        end
        Zraw
    end
    @test (W * Zsol + Zsol * W) / T(2) ≈ X rtol=rtol atol=atol
end

function _psdnt_seeded_pairs(n::Int, rng::AbstractRNG)
    cone = PSDNT_SC.PSDTriangleCone{Float64}(n)
    state = PSDNT_SC.PSDNTScaling{Float64}(n; eigen_route=:setup_jacobi)
    for _ in 1:10
        LS = randn(rng, n, n)
        LY = randn(rng, n, n)
        S = LS * transpose(LS) + Matrix{Float64}(I, n, n)
        Y = LY * transpose(LY) + 2 * Matrix{Float64}(I, n, n)
        s, y = _psdnt_svec(S), _psdnt_svec(Y)
        z = _psdnt_svec((S + Y) / 2)
        out, tmp = zeros(length(s)), zeros(length(s))
        PSDNT_SC.nt_scaling!(cone, state, s, y)
        @test state.P * Y * state.P ≈ S rtol=5e-11 atol=5e-11
        PSDNT_SC.theta_apply!(cone, out, state, y)
        @test out ≈ s rtol=5e-11 atol=5e-11
        PSDNT_SC.g_apply!(cone, out, state, s)
        @test out ≈ y rtol=5e-11 atol=5e-11
        PSDNT_SC.theta_apply!(cone, out, state, z)
        PSDNT_SC.g_apply!(cone, tmp, state, out)
        @test tmp ≈ z rtol=5e-11 atol=5e-11
    end
end

@testset "pair-dependent PSD NT reference in svec coordinates" begin
    seeded_rng = MersenneTwister(0x51d05d)
    for n in (2, 3, 5)
        _psdnt_reference_case(Float64, n; rtol=2e-11, atol=2e-11)
        _psdnt_seeded_pairs(n, seeded_rng)
    end

    A = [2.0 0.75 -0.2; 0.75 3.0 0.4; -0.2 0.4 1.5]
    B = [1.0 -0.3 0.2; -0.3 2.0 0.6; 0.2 0.6 4.0]
    @test dot(_psdnt_svec(A), _psdnt_svec(B)) ≈ tr(A * B)
    @test dot(_psdnt_rawpack(A), _psdnt_rawpack(B)) != tr(A * B)

    cone = PSDNT_SC.PSDTriangleCone{Float64}(2)
    state = PSDNT_SC.PSDNTScaling{Float64}(2)
    @test state.eigen_route === :setup_jacobi
    @test_throws ArgumentError PSDNT_SC.PSDNTScaling{Float64}(2; eigen_route=:hidden_fallback)
    @test_throws ArgumentError PSDNT_SC.theta_apply!(cone, zeros(3), state, ones(3))
    @test_throws DomainError PSDNT_SC.nt_scaling!(cone, state, _psdnt_svec([1.0 0.0; 0.0 -1.0]), _psdnt_svec([2.0 0.0; 0.0 3.0]))
    @test_throws ArgumentError PSDNT_SC.theta_apply!(cone, zeros(3), state, ones(3))
    @test_throws DomainError PSDNT_SC.nt_scaling!(cone, state, _psdnt_svec([2.0 0.0; 0.0 3.0]), [NaN, 0.0, 1.0])

    setprecision(BigFloat, 192) do
        _psdnt_reference_case(BigFloat, 2; rtol=big"1e-45", atol=big"1e-45")
        _psdnt_reference_case(BigFloat, 3; rtol=big"1e-42", atol=big"1e-42")
    end
end
