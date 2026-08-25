using SDPX
using Test
using LinearAlgebra
using Random

const SOCNT_SC = SDPX.SymmetricCones

function _orthantnt_reference_case(::Type{T}; rtol, atol) where {T}
    cone = SOCNT_SC.NonnegativeCone(5)
    state = SOCNT_SC.OrthantNTScaling{T}(5)
    s = T[2, 3, 5, 7, 11]
    y = T[13, 11, 7, 5, 3]
    z = T[1, 2, 3, 4, 5]
    out = zeros(T, 5)
    tmp = zeros(T, 5)
    SOCNT_SC.nt_scaling!(cone, state, s, y)
    SOCNT_SC.theta_apply!(cone, out, state, y)
    @test out ≈ s rtol=rtol atol=atol
    SOCNT_SC.g_apply!(cone, out, state, s)
    @test out ≈ y rtol=rtol atol=atol
    SOCNT_SC.r_apply!(cone, out, state, y)
    SOCNT_SC.r_inverse_apply!(cone, tmp, state, s)
    @test out ≈ tmp rtol=rtol atol=atol
    @test out ≈ state.lambda rtol=rtol atol=atol
    SOCNT_SC.theta_apply!(cone, out, state, z)
    SOCNT_SC.g_apply!(cone, tmp, state, out)
    @test tmp ≈ z rtol=rtol atol=atol
    SOCNT_SC.solve_Llambda!(cone, out, state, z)
    @test state.lambda .* out ≈ z rtol=rtol atol=atol
end

function _socnt_pair(::Type{T}, n::Int) where {T}
    s = zeros(T, n)
    y = zeros(T, n)
    s[1] = T(n + 3)
    y[1] = T(n + 2)
    @inbounds for i in 2:n
        s[i] = T(i - 1) / T(5n)
        y[i] = (isodd(i) ? -one(T) : one(T)) * T(i) / T(7n)
    end
    return s, y
end

function _socnt_qmatrix(w::AbstractVector{T}) where {T}
    n = length(w)
    Q = zeros(T, n, n)
    w0 = w[1]
    tail2 = sum(abs2, @view w[2:end])
    Q[1, 1] = w0 * w0 + tail2
    @inbounds for i in 2:n
        Q[1, i] = T(2) * w0 * w[i]
        Q[i, 1] = Q[1, i]
        for j in 2:n
            Q[i, j] = T(2) * w[i] * w[j]
        end
        Q[i, i] += w0 * w0 - tail2
    end
    return Q
end

function _socnt_reference_case(::Type{T}, n::Int; rtol, atol) where {T}
    cone = SOCNT_SC.SOCone(n)
    state = SOCNT_SC.SOCNTScaling{T}(n)
    s, y = _socnt_pair(T, n)
    z = T[T(i + 1) / T(n + 4) for i in 1:n]
    out = zeros(T, n)
    tmp = zeros(T, n)
    tmp2 = zeros(T, n)

    SOCNT_SC.nt_scaling!(cone, state, s, y)
    SOCNT_SC.theta_apply!(cone, out, state, y)
    @test out ≈ s rtol=rtol atol=atol
    SOCNT_SC.g_apply!(cone, out, state, s)
    @test out ≈ y rtol=rtol atol=atol

    SOCNT_SC.r_apply!(cone, out, state, y)
    SOCNT_SC.r_inverse_apply!(cone, tmp, state, s)
    @test out ≈ tmp rtol=rtol atol=atol
    @test out ≈ state.lambda rtol=rtol atol=atol

    SOCNT_SC.theta_apply!(cone, out, state, z)
    SOCNT_SC.g_apply!(cone, tmp, state, out)
    @test tmp ≈ z rtol=rtol atol=atol
    SOCNT_SC.r_apply!(cone, tmp, state, z)
    SOCNT_SC.r_apply!(cone, tmp2, state, tmp)
    @test tmp2 ≈ out rtol=rtol atol=atol

    SOCNT_SC.quadratic_apply!(cone, tmp, state.w, z)
    @test tmp ≈ _socnt_qmatrix(state.w) * z rtol=rtol atol=atol
    SOCNT_SC.quadratic_inverse_apply!(cone, tmp2, state.winv, tmp)
    @test tmp2 ≈ z rtol=rtol atol=atol

    rhs = T[T(i) / T(n + 3) for i in 1:n]
    SOCNT_SC.solve_Llambda!(cone, out, state, rhs)
    SOCNT_SC.jordan_product!(cone, tmp, state.lambda, out)
    @test tmp ≈ rhs rtol=rtol atol=atol

    W = copy(y)
    SOCNT_SC.scaling_inverse_apply!(cone, out, W, rhs)
    SOCNT_SC.scaling_apply!(cone, tmp, W, out)
    @test tmp ≈ rhs rtol=rtol atol=atol

    d1 = copy(z)
    d2 = reverse(z)
    d3 = s ./ T(n + 1)
    SOCNT_SC.third_order_correction!(cone, state, out, d1, d2, d3)
    SOCNT_SC.jordan_product!(cone, tmp, d2, d3)
    SOCNT_SC.jordan_product!(cone, tmp2, d1, tmp)
    @test out ≈ tmp2 rtol=rtol atol=atol
end

function _socnt_seeded_pairs(n::Int, rng::AbstractRNG)
    cone = SOCNT_SC.SOCone(n)
    state = SOCNT_SC.SOCNTScaling{Float64}(n)
    out = zeros(n)
    tmp = zeros(n)
    for _ in 1:20
        s = randn(rng, n)
        y = randn(rng, n)
        z = randn(rng, n)
        s[2:end] ./= n
        y[2:end] ./= n
        s[1] = norm(@view s[2:end]) + 1 + rand(rng)
        y[1] = norm(@view y[2:end]) + 1 + rand(rng)
        SOCNT_SC.nt_scaling!(cone, state, s, y)
        SOCNT_SC.theta_apply!(cone, out, state, y)
        @test out ≈ s rtol=3e-12 atol=3e-12
        SOCNT_SC.g_apply!(cone, out, state, s)
        @test out ≈ y rtol=3e-12 atol=3e-12
        SOCNT_SC.theta_apply!(cone, out, state, z)
        SOCNT_SC.g_apply!(cone, tmp, state, out)
        @test tmp ≈ z rtol=3e-12 atol=3e-12
        SOCNT_SC.r_apply!(cone, out, state, y)
        SOCNT_SC.r_inverse_apply!(cone, tmp, state, s)
        @test out ≈ tmp rtol=3e-12 atol=3e-12
    end
end

@testset "pair-dependent SOC NT reference" begin
    _orthantnt_reference_case(Float64; rtol=2e-15, atol=2e-15)
    orthant = SOCNT_SC.NonnegativeCone(2)
    orthant_state = SOCNT_SC.OrthantNTScaling{Float64}(2)
    @test_throws ArgumentError SOCNT_SC.theta_apply!(orthant, zeros(2), orthant_state, ones(2))
    @test_throws DomainError SOCNT_SC.nt_scaling!(orthant, orthant_state, [1.0, 0.0], [1.0, 1.0])
    @test_throws ArgumentError SOCNT_SC.theta_apply!(orthant, zeros(2), orthant_state, ones(2))
    @test_throws DomainError SOCNT_SC.nt_scaling!(orthant, orthant_state, [1.0, 1.0], [1.0, Inf])
    seeded_rng = MersenneTwister(0x51d0c)
    for n in (3, 4, 8, 16)
        _socnt_reference_case(Float64, n; rtol=2e-12, atol=2e-12)
        _socnt_seeded_pairs(n, seeded_rng)
    end

    cone = SOCNT_SC.SOCone(4)
    zero_tail = [4.0, 0.0, 0.0, 0.0]
    lambda1, lambda2, c1, c2 = SOCNT_SC.spectrum(cone, zero_tail)
    @test lambda1 == lambda2 == 4.0
    @test c1 == [0.5, 0.5, 0.0, 0.0]
    @test c2 == [0.5, -0.5, 0.0, 0.0]
    tmp = zeros(4)
    SOCNT_SC.jordan_product!(cone, tmp, c1, c1)
    @test tmp == c1
    SOCNT_SC.jordan_product!(cone, tmp, c2, c2)
    @test tmp == c2
    SOCNT_SC.jordan_product!(cone, tmp, c1, c2)
    @test tmp == zeros(4)
    @test lambda1 * c1 + lambda2 * c2 == zero_tail

    alpha = Ref(0.0)
    SOCNT_SC.boundary_step!(cone, [2.0, 0.5, 0.0, 0.0], alpha, [-1.0, 0.0, 0.0, 0.0])
    @test alpha[] ≈ 1.5
    SOCNT_SC.boundary_step!(cone, [2.0, 0.0, 0.0, 0.0], alpha, [1.0, 0.0, 0.0, 0.0])
    @test isinf(alpha[])
    SOCNT_SC.boundary_step!(cone, [1.0, 1.0, 0.0, 0.0], alpha, [1.0, 1.0, 0.0, 0.0])
    @test isinf(alpha[])
    SOCNT_SC.boundary_step!(cone, [1.0, 1.0, 0.0, 0.0], alpha, [0.0, 0.0, 1.0, 0.0])
    @test iszero(alpha[])
    SOCNT_SC.boundary_step!(cone, [0.5, 1.0, 0.0, 0.0], alpha, [1.0, 0.0, 0.0, 0.0])
    @test iszero(alpha[])

    state = SOCNT_SC.SOCNTScaling{Float64}(4)
    @test_throws ArgumentError SOCNT_SC.theta_apply!(cone, zeros(4), state, zero_tail)
    @test_throws DomainError SOCNT_SC.nt_scaling!(cone, state, [1.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0])
    @test_throws ArgumentError SOCNT_SC.theta_apply!(cone, zeros(4), state, zero_tail)
    @test_throws DomainError SOCNT_SC.nt_scaling!(cone, state, [2.0, 0.0, 0.0, 0.0], [NaN, 0.0, 0.0, 0.0])

    setprecision(BigFloat, 192) do
        _orthantnt_reference_case(BigFloat; rtol=big"1e-55", atol=big"1e-55")
        _socnt_reference_case(BigFloat, 5; rtol=big"1e-50", atol=big"1e-50")
    end
end
