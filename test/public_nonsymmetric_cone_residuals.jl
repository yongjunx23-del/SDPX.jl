using SDPX
using Test
using MultiFloats

const _PUBLIC_NS_TYPES = (
    Float64,
    Float64x2,
    Float64x3,
    Float64x4,
)

@inline function _public_ns_tol(::Type{T}) where {T}
    return T(4096) * eps(one(T))
end

function _check_public_nonsymmetric_residuals(::Type{T}) where {T<:AbstractFloat}
    exp_domain = SDPX.ExponentialCone()
    alpha = T(3) / T(10)
    power_domain = SDPX.PowerCone(alpha)
    tol = _public_ns_tol(T)

    exp_primal = T[zero(T), one(T), one(T)]
    exp_dual = T[-one(T), -one(T), one(T)]
    @test SDPX._public_primal_cone_residual(exp_primal, exp_domain, 3) <= tol
    @test SDPX._public_dual_cone_residual(exp_dual, exp_domain, 3) <= tol
    @test SDPX._public_primal_cone_residual(T[one(T), zero(T), zero(T)], exp_domain, 3) > tol
    @test SDPX._public_dual_cone_residual(T[one(T), zero(T), zero(T)], exp_domain, 3) > tol

    beta = one(T) - alpha
    power_primal = T[one(T), one(T), one(T)]
    power_dual = T[alpha, beta, one(T)]
    @test SDPX._public_primal_cone_residual(power_primal, power_domain, 3) <= tol
    @test SDPX._public_dual_cone_residual(power_dual, power_domain, 3) <= tol
    @test SDPX._public_primal_cone_residual(T[-one(T), one(T), zero(T)], power_domain, 3) > tol
    @test SDPX._public_dual_cone_residual(T[-one(T), one(T), zero(T)], power_domain, 3) > tol

    # Near-boundary violations remain finite, positive, and homogeneous.
    delta = sqrt(eps(one(T)))
    exp_outside = T[delta, one(T), one(T)]
    exp_violation = SDPX._public_primal_cone_residual(exp_outside, exp_domain, 3)
    @test isfinite(exp_violation) && exp_violation > zero(T)
    @test SDPX.exp_primal_membership(exp_outside...; tol=T(2) * delta)
    @test !SDPX.exp_primal_membership(exp_outside...; tol=delta / T(4))
    @test isapprox(
        SDPX._public_primal_cone_residual(T(7) .* exp_outside, exp_domain, 3),
        T(7) * exp_violation;
        atol=tol,
        rtol=tol,
    )

    power_outside = T[one(T), one(T), one(T) + delta]
    power_violation = SDPX._public_primal_cone_residual(power_outside, power_domain, 3)
    @test isfinite(power_violation) && power_violation > zero(T)
    @test SDPX.power_membership(
        power_outside..., alpha; tol=T(2) * delta,
    )
    @test !SDPX.power_membership(
        power_outside..., alpha; tol=delta / T(4),
    )
    @test isapprox(
        SDPX._public_primal_cone_residual(T(5) .* power_outside, power_domain, 3),
        T(5) * power_violation;
        atol=tol,
        rtol=tol,
    )

    @test isinf(SDPX._public_primal_cone_residual(T[zero(T), one(T)], exp_domain, 2))
    @test isinf(SDPX._public_dual_cone_residual(T[zero(T), one(T)], power_domain, 2))
    @test isinf(SDPX._public_primal_cone_residual(T[zero(T), one(T), T(Inf)], exp_domain, 3))
    return nothing
end

@testset "public Exp/Power primal-dual cone residuals" begin
    for T in _PUBLIC_NS_TYPES
        @testset "$T" begin
            _check_public_nonsymmetric_residuals(T)
        end
    end
    for bits in (256, 512, 1024)
        @testset "BigFloat$bits" begin
            setprecision(BigFloat, bits) do
                _check_public_nonsymmetric_residuals(BigFloat)
            end
        end
    end
end
