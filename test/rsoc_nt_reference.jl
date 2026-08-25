using SDPX
using Test
using LinearAlgebra

const RSOCNT_SC = SDPX.SymmetricCones

function _rsocnt_transform(::Type{T}, n::Int) where {T}
    n >= 3 || throw(ArgumentError("rotated SOC dimension must be at least three"))
    A = zeros(T, n, n)
    invsqrt2 = one(T) / sqrt(T(2))
    A[1, 1] = invsqrt2
    A[1, 2] = invsqrt2
    A[2, 1] = invsqrt2
    A[2, 2] = -invsqrt2
    @inbounds for i in 3:n
        A[i, i] = one(T)
    end
    return A
end

function _rsocnt_case(::Type{T}; rtol, atol) where {T}
    n = 5
    A = _rsocnt_transform(T, n)
    Iref = Matrix{T}(I, n, n)
    @test A == transpose(A)
    @test transpose(A) * A ≈ Iref rtol=rtol atol=atol
    @test A * A ≈ Iref rtol=rtol atol=atol

    soc_s = T[6, 1 // 4, -1 // 5, 1 // 6, -1 // 7]
    soc_y = T[5, -1 // 6, 1 // 7, -1 // 8, 1 // 9]
    rsoc_s = transpose(A) * soc_s
    rsoc_y = transpose(A) * soc_y
    @test rsoc_s[1] > zero(T) && rsoc_s[2] > zero(T)
    @test T(2) * rsoc_s[1] * rsoc_s[2] > sum(abs2, @view rsoc_s[3:end])

    cone = RSOCNT_SC.SOCone(n)
    @test RSOCNT_SC.membership(cone, A * rsoc_s)
    bad_rsoc = T[one(T), one(T), T(2), T(2), zero(T)]
    @test (T(2) * bad_rsoc[1] * bad_rsoc[2] >= sum(abs2, @view bad_rsoc[3:end])) ==
          RSOCNT_SC.membership(cone, A * bad_rsoc)
    state = RSOCNT_SC.SOCNTScaling{T}(n)
    RSOCNT_SC.nt_scaling!(cone, state, A * rsoc_s, A * rsoc_y)

    soc_out = zeros(T, n)
    RSOCNT_SC.theta_apply!(cone, soc_out, state, A * rsoc_y)
    rsoc_out = transpose(A) * soc_out
    @test rsoc_out ≈ rsoc_s rtol=rtol atol=atol

    RSOCNT_SC.g_apply!(cone, soc_out, state, A * rsoc_s)
    @test transpose(A) * soc_out ≈ rsoc_y rtol=rtol atol=atol

    z = T[2, 3, 1 // 8, -1 // 9, 1 // 10]
    RSOCNT_SC.theta_apply!(cone, soc_out, state, A * z)
    rsoc_theta_z = transpose(A) * soc_out
    RSOCNT_SC.g_apply!(cone, soc_out, state, A * rsoc_theta_z)
    @test transpose(A) * soc_out ≈ z rtol=rtol atol=atol
end

@testset "RSOC is orthogonally conjugate to SOC NT" begin
    _rsocnt_case(Float64; rtol=2e-12, atol=2e-12)
    setprecision(BigFloat, 192) do
        _rsocnt_case(BigFloat; rtol=big"1e-50", atol=big"1e-50")
    end
end
