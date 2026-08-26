# Double-secant/BFGS and explicit conjugate-Hessian fallback gates.

using SDPX
using LinearAlgebra
using Test

if !isdefined(SDPX, :NonsymmetricConjugateWorkspace)
    Base.include(
        SDPX,
        joinpath(@__DIR__, "..", "src", "cones", "nonsymmetric", "types.jl"),
    )
end
if !isdefined(SDPX, :conjugate_shadow!)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric", "conjugate3.jl",
        ),
    )
end
if !isdefined(SDPX, :try_update_nonsymmetric_scaling!)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric", "scaling3.jl",
        ),
    )
end

const _NS_SCALING_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

function _nss_double_cases(::Type{T}) where {T}
    exp_case = (
        SDPX.ExpConjugateTag(),
        (zero(T), one(T), T(2)),
        (-one(T), one(T), one(T)),
    )
    p01 = T(1) / T(10)
    p05 = T(1) / T(2)
    p09 = T(9) / T(10)
    power_case(alpha) = (
        SDPX.PowerConjugateTag(alpha),
        (one(T), one(T), zero(T)),
        (alpha, one(T) - alpha, T(1) / T(2)),
    )
    return (
        exp_case,
        power_case(p01),
        power_case(p05),
        power_case(p09),
    )
end

function _nss_centered_case(::Type{T}, ::SDPX.ExpConjugateTag) where {T}
    primal = (zero(T), one(T), T(2))
    gradient = SDPX.exp_barrier_gradient(primal...)
    dual = (-gradient[1], -gradient[2], -gradient[3])
    return primal, dual
end

function _nss_centered_case(
    ::Type{T}, tag::SDPX.PowerConjugateTag,
) where {T}
    primal = (one(T), one(T), zero(T))
    gradient = SDPX.power_barrier_gradient(primal..., tag.alpha)
    dual = (-gradient[1], -gradient[2], -gradient[3])
    return primal, dual
end

function _nss_update!(workspace, policy, tag, primal, dual)
    return SDPX.try_update_nonsymmetric_scaling!(
        workspace, policy, tag, primal, dual,
    )
end

function _nss_allocated(workspace, policy, tag, primal, dual)
    _nss_update!(workspace, policy, tag, primal, dual)
    return @allocated _nss_update!(workspace, policy, tag, primal, dual)
end

function _nss_matrix_tolerance(::Type{T}) where {T}
    return T(32768) * eps(one(T))
end

function _nss_check_double(::Type{T}, tag, primal, dual) where {T}
    workspace = SDPX.NonsymmetricScalingWorkspace(T)
    policy = SDPX.StrictDoubleSecantScaling()
    result = _nss_update!(workspace, policy, tag, primal, dual)
    tolerance = _nss_matrix_tolerance(T)
    @test result.status === SDPX.NS_SCALING_DOUBLE_SECANT
    @test result.reason === SDPX.NS_SCALING_CONVERGED
    @test result.fallback_reason === SDPX.NS_SCALING_NO_FALLBACK
    @test workspace.valid
    @test !workspace.used_fallback
    @test isapprox(result.mu, dot(collect(primal), collect(dual)) / T(3);
                   atol=tolerance, rtol=tolerance)

    stilde = workspace.conjugate.shadow
    ytilde = workspace.dual_shadow
    S = hcat(collect(primal), stilde)
    Y = hcat(collect(dual), ytilde)
    M = transpose(Y) * S
    @test isapprox(M, transpose(M); atol=tolerance, rtol=tolerance)
    @test isposdef(Symmetric(M))
    @test isapprox(workspace.g * S, Y; atol=tolerance, rtol=tolerance)
    @test isapprox(workspace.theta * Y, S; atol=tolerance, rtol=tolerance)
    @test isapprox(
        workspace.g * workspace.theta, Matrix{T}(I, 3, 3);
        atol=tolerance, rtol=tolerance,
    )
    @test isposdef(Symmetric(workspace.g))
    @test isposdef(Symmetric(workspace.theta))
    @test all(isfinite, workspace.g)
    @test all(isfinite, workspace.theta)
    @test result.secant_error <= workspace.settings.validation_tolerance
    @test result.inverse_error <= workspace.settings.validation_tolerance
    return workspace, result
end

function _nss_reference_double(workspace)
    s = workspace.primal
    y = workspace.dual
    stilde = workspace.conjugate.shadow
    ytilde = workspace.dual_shadow
    S = hcat(s, stilde)
    Y = hcat(y, ytilde)
    M = transpose(Y) * S
    mu = workspace.mu
    G0 = mu * workspace.primal_hessian
    GB = Y * inv(M) * transpose(Y) + G0 -
         G0 * S * inv(transpose(S) * G0 * S) * transpose(S) * G0
    z = cross(s, stilde)
    z /= norm(z)
    ycross = cross(y, ytilde)
    r = ycross / dot(z, ycross)
    tG = dot(z, GB * z)
    G = Y * inv(M) * transpose(Y) + tG * z * transpose(z)
    theta = S * inv(M) * transpose(S) + inv(tG) * r * transpose(r)
    return GB, G, theta, tG
end

@testset "nonsymmetric scaling typed ABI" begin
    @test isbitstype(SDPX.StrictDoubleSecantScaling)
    @test isbitstype(SDPX.DoubleSecantWithDualHessianFallback)
    @test isbitstype(SDPX.NonsymmetricScalingStatus)
    @test isbitstype(SDPX.NonsymmetricScalingReason)
    @test isbitstype(SDPX.NonsymmetricScalingResult{Float64})
    @test isbitstype(SDPX.NonsymmetricScalingSettings{Float64})
end

@testset "double-secant BFGS identities and fixed-width zero allocation" begin
    for T in (
        Float64,
        _NS_SCALING_MF.Float64x2,
        _NS_SCALING_MF.Float64x3,
        _NS_SCALING_MF.Float64x4,
    )
        @testset "$T" begin
            for (tag, primal, dual) in _nss_double_cases(T)
                workspace, result = _nss_check_double(T, tag, primal, dual)
                policy = SDPX.StrictDoubleSecantScaling()
                @test _nss_allocated(
                    workspace, policy, tag, primal, dual,
                ) == 0
                @test isbits(result)
            end
        end
    end
end

@testset "independent BFGS formula reference" begin
    for T in (Float64, BigFloat)
        runner = function ()
            tolerance = T === Float64 ? T(2e-12) : big"1e-60"
            for (tag, primal, dual) in _nss_double_cases(T)
                workspace, _ = _nss_check_double(T, tag, primal, dual)
                GB, G, theta, tG = _nss_reference_double(workspace)
                @test tG > zero(T)
                @test isapprox(
                    workspace.g_bfgs, GB; atol=tolerance, rtol=tolerance,
                )
                @test isapprox(workspace.g, G; atol=tolerance, rtol=tolerance)
                @test isapprox(
                    workspace.theta, theta; atol=tolerance, rtol=tolerance,
                )
            end
        end
        if T === BigFloat
            setprecision(runner, BigFloat, 256)
        else
            runner()
        end
    end
end

@testset "explicit dual-Hessian one-secant fallback" begin
    for T in (
        Float64,
        _NS_SCALING_MF.Float64x2,
        _NS_SCALING_MF.Float64x3,
        _NS_SCALING_MF.Float64x4,
    )
        tolerance = _nss_matrix_tolerance(T)
        for tag in (
            SDPX.ExpConjugateTag(),
            SDPX.PowerConjugateTag(T(1) / T(10)),
            SDPX.PowerConjugateTag(T(1) / T(2)),
            SDPX.PowerConjugateTag(T(9) / T(10)),
        )
            primal, dual = _nss_centered_case(T, tag)
            strict_workspace = SDPX.NonsymmetricScalingWorkspace(T)
            strict = _nss_update!(
                strict_workspace, SDPX.StrictDoubleSecantScaling(),
                tag, primal, dual,
            )
            @test strict.status === SDPX.NS_SCALING_FAILED
            @test strict.reason === SDPX.NS_SCALING_SECOND_SECANT_DEGENERATE
            @test !strict_workspace.valid

            workspace = SDPX.NonsymmetricScalingWorkspace(T)
            policy = SDPX.DoubleSecantWithDualHessianFallback()
            fallback = _nss_update!(workspace, policy, tag, primal, dual)
            @test fallback.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
            @test fallback.reason === SDPX.NS_SCALING_CONVERGED
            @test fallback.fallback_reason ===
                  SDPX.NS_SCALING_SECOND_SECANT_DEGENERATE
            @test workspace.valid
            @test workspace.used_fallback
            @test isposdef(Symmetric(workspace.theta))
            @test isposdef(Symmetric(workspace.g))
            @test isapprox(
                workspace.theta * collect(dual), collect(primal);
                atol=tolerance, rtol=tolerance,
            )
            @test isapprox(
                workspace.g * workspace.theta, Matrix{T}(I, 3, 3);
                atol=tolerance, rtol=tolerance,
            )

            theta0 = fallback.mu * workspace.conjugate.inverse_hessian
            y = collect(dual)
            s = collect(primal)
            q = theta0 * y
            theta_reference = theta0 - q * transpose(q) / dot(y, q) +
                              s * transpose(s) / dot(s, y)
            @test isapprox(
                workspace.theta, theta_reference;
                atol=tolerance, rtol=tolerance,
            )
            @test _nss_allocated(workspace, policy, tag, primal, dual) == 0
        end
    end
end

@testset "BigFloat256 double-secant and fallback rungs" begin
    setprecision(BigFloat, 256) do
        for (tag, primal, dual) in _nss_double_cases(BigFloat)
            workspace, _ = _nss_check_double(BigFloat, tag, primal, dual)
            @test workspace.valid
        end
        tag = SDPX.ExpConjugateTag()
        primal, dual = _nss_centered_case(BigFloat, tag)
        workspace = SDPX.NonsymmetricScalingWorkspace(BigFloat)
        result = _nss_update!(
            workspace, SDPX.DoubleSecantWithDualHessianFallback(),
            tag, primal, dual,
        )
        @test result.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
        @test result.fallback_reason ===
              SDPX.NS_SCALING_SECOND_SECANT_DEGENERATE
    end
end

@testset "mapped dual Hessian is not the fallback conjugate Hessian" begin
    tag = SDPX.ExpConjugateTag()
    primal, dual = _nss_centered_case(Float64, tag)
    workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
    result = _nss_update!(
        workspace, SDPX.DoubleSecantWithDualHessianFallback(),
        tag, primal, dual,
    )
    @test result.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
    mapped_hessian = SDPX.exp_dual_hessian(dual...)
    @test norm(mapped_hessian - workspace.conjugate.inverse_hessian, Inf) > 0.1
    @test norm(mapped_hessian - workspace.conjugate.inverse_hessian, Inf) >
          1.0e10 * result.inverse_error
end

@testset "nonsymmetric scaling fail-closed reasons" begin
    policy = SDPX.StrictDoubleSecantScaling()
    exp_tag = SDPX.ExpConjugateTag()
    good_primal = (0.0, 1.0, 2.0)
    good_dual = (-1.0, 1.0, 1.0)

    workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
    nonfinite = _nss_update!(
        workspace, policy, exp_tag, (NaN, 1.0, 2.0), good_dual,
    )
    @test nonfinite.reason === SDPX.NS_SCALING_NONFINITE_INPUT
    @test !workspace.valid

    primal_boundary = _nss_update!(
        workspace, policy, exp_tag, (0.0, 1.0, 1.0), good_dual,
    )
    @test primal_boundary.reason === SDPX.NS_SCALING_PRIMAL_NOT_INTERIOR

    dual_boundary = _nss_update!(
        workspace, policy, exp_tag, good_primal,
        (-1.0, 1.0, exp(-2.0)),
    )
    @test dual_boundary.reason === SDPX.NS_SCALING_DUAL_NOT_INTERIOR

    invalid_alpha = _nss_update!(
        workspace, policy, SDPX.PowerConjugateTag(1.0),
        (1.0, 1.0, 0.0), (1.0, 1.0, 0.0),
    )
    @test invalid_alpha.reason === SDPX.NS_SCALING_INVALID_PARAMETER

    wrong_dimension = _nss_update!(
        workspace, policy, exp_tag, (1.0, 2.0), good_dual,
    )
    @test wrong_dimension.reason === SDPX.NS_SCALING_INVALID_PARAMETER

    unresolved_pairing = SDPX.NonsymmetricScalingWorkspace(
        Float64; degeneracy_tolerance=2.0,
    )
    pairing = _nss_update!(
        unresolved_pairing, policy, exp_tag, good_primal, good_dual,
    )
    @test pairing.reason === SDPX.NS_SCALING_NONPOSITIVE_PAIRING

    limited = SDPX.NonsymmetricScalingWorkspace(
        Float64; max_iterations=0,
    )
    conjugate_failure = _nss_update!(
        limited, policy, exp_tag, good_primal, good_dual,
    )
    @test conjugate_failure.reason === SDPX.NS_SCALING_CONJUGATE_FAILED
    @test conjugate_failure.conjugate_reason ===
          SDPX.NS_CONJUGATE_ITERATION_LIMIT

    @test_throws ArgumentError SDPX.apply_nonsymmetric_G!(
        zeros(3), workspace, ones(3),
    )
    @test_throws ArgumentError SDPX.apply_nonsymmetric_Theta!(
        zeros(3), workspace, ones(3),
    )
end

@testset "scaling numerical-gate fault injection" begin
    tag = SDPX.ExpConjugateTag()
    primal = (0.0, 1.0, 2.0)
    dual = (-1.0, 1.0, 1.0)

    function fresh_workspace()
        workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
        result = _nss_update!(
            workspace, SDPX.StrictDoubleSecantScaling(), tag, primal, dual,
        )
        @test result.status === SDPX.NS_SCALING_DOUBLE_SECANT
        return workspace
    end

    # M remains symmetric SPD, but S is deliberately unresolved at the
    # arithmetic scale.  This reaches the axis gate rather than the Gram gate.
    axis = fresh_workspace()
    delta = eps(Float64)
    axis.primal .= (1.0, 0.0, 0.0)
    axis.conjugate.shadow .= (1.0, delta, 0.0)
    axis.dual .= (3.0, 0.0, 0.0)
    axis.dual_shadow .= (3.0, inv(delta), 0.0)
    axis.mu = 1.0
    axis.primal_hessian .= Matrix{Float64}(I, 3, 3)
    @test SDPX._ns_scaling_double_secant!(axis) ===
          SDPX.NS_SCALING_AXIS_DEGENERATE

    # The secant Gram is well conditioned, while huge components orthogonal
    # to S make z'cross(Y) numerically unresolved relative to cross(Y).
    axis_pairing = fresh_workspace()
    large = inv(eps(Float64))
    axis_pairing.primal .= (1.0, 0.0, 0.0)
    axis_pairing.conjugate.shadow .= (0.0, 1.0, 0.0)
    axis_pairing.dual .= (4.0, 3.0, large)
    axis_pairing.dual_shadow .= (3.0, 4.0, large)
    axis_pairing.mu = 4.0 / 3.0
    axis_pairing.primal_hessian .= Matrix{Float64}(I, 3, 3)
    @test SDPX._ns_scaling_double_secant!(axis_pairing) ===
          SDPX.NS_SCALING_AXIS_PAIRING_DEGENERATE

    denominator = fresh_workspace()
    fill!(denominator.primal_hessian, 0.0)
    @test SDPX._ns_scaling_double_secant!(denominator) ===
          SDPX.NS_SCALING_BFGS_DENOMINATOR

    bfgs_spd = fresh_workspace()
    z = cross(bfgs_spd.primal, bfgs_spd.conjugate.shadow)
    z /= norm(z)
    bfgs_spd.primal_hessian .= Matrix{Float64}(I, 3, 3) -
                                  1.0e6 * z * transpose(z)
    @test SDPX._ns_scaling_double_secant!(bfgs_spd) ===
          SDPX.NS_SCALING_BFGS_NOT_SPD

    metric_spd = fresh_workspace()
    metric_spd.g[1, 1] = -abs(metric_spd.g[1, 1])
    reason, _, _ = SDPX._ns_scaling_validate_metric!(metric_spd, true)
    @test reason === SDPX.NS_SCALING_METRIC_NOT_SPD

    fallback_denominator = fresh_workspace()
    fallback_denominator.settings = SDPX.NonsymmetricScalingSettings(
        Float64; degeneracy_tolerance=2.0,
    )
    @test SDPX._ns_scaling_dual_hessian_fallback!(fallback_denominator) ===
          SDPX.NS_SCALING_FALLBACK_DENOMINATOR

    fallback_spd = fresh_workspace()
    fallback_spd.conjugate.inverse_hessian[1, 1] = -1.0
    @test SDPX._ns_scaling_dual_hessian_fallback!(fallback_spd) ===
          SDPX.NS_SCALING_FALLBACK_NOT_SPD
end
