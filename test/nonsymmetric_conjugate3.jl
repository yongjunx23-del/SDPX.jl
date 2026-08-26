# Fenchel-conjugate inverse-gradient tests.  This file may be run standalone
# before the integration owner mounts the two Phase-4 source files.

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

const _NS_CONJUGATE_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

@inline function _nc_primal_gradient!(destination, ::SDPX.ExpConjugateTag, point)
    return SDPX.exp_primal_gradient!(destination, point[1], point[2], point[3])
end

@inline function _nc_primal_gradient!(
    destination, tag::SDPX.PowerConjugateTag, point,
)
    return SDPX.power_primal_gradient!(
        destination, point[1], point[2], point[3], tag.alpha,
    )
end

@inline function _nc_primal_hessian(::SDPX.ExpConjugateTag, point)
    return SDPX.exp_barrier_hessian(point[1], point[2], point[3])
end

@inline function _nc_primal_hessian(tag::SDPX.PowerConjugateTag, point)
    return SDPX.power_barrier_hessian(
        point[1], point[2], point[3], tag.alpha,
    )
end

@inline function _nc_primal_interior(::SDPX.ExpConjugateTag, point)
    return SDPX.exp_primal_interior(point[1], point[2], point[3])
end

@inline function _nc_primal_interior(tag::SDPX.PowerConjugateTag, point)
    return SDPX.power_primal_interior(
        point[1], point[2], point[3], tag.alpha,
    )
end

function _nc_cases(::Type{T}) where {T}
    return (
        (
            SDPX.ExpConjugateTag(),
            (-one(T), zero(T), T(3) / T(2)),
            (T(1) / T(4), T(6) / T(5), T(3)),
        ),
        map(T[1 // 10, 1 // 2, 9 // 10]) do alpha
            beta = one(T) - alpha
            (
                SDPX.PowerConjugateTag(alpha),
                (T(2) * alpha, T(3) * beta, T(1) / T(4)),
                (T(2), T(3), T(1) / T(4)),
            )
        end...,
    )
end

function _nc_call!(workspace, tag, dual)
    return SDPX.conjugate_shadow!(workspace, tag, dual)
end

function _nc_allocated(workspace, tag, dual)
    _nc_call!(workspace, tag, dual)
    return @allocated _nc_call!(workspace, tag, dual)
end

function _nc_check_identities(::Type{T}, tag, dual, primal; tolerance) where {T}
    workspace = SDPX.NonsymmetricConjugateWorkspace(T)
    result = _nc_call!(workspace, tag, dual)
    @test result.status === SDPX.NS_CONJUGATE_SUCCESS
    @test result.reason === SDPX.NS_CONJUGATE_CONVERGED
    @test workspace.valid
    @test _nc_primal_interior(tag, workspace.shadow)

    gradient = zeros(T, 3)
    _nc_primal_gradient!(gradient, tag, workspace.shadow)
    @test isapprox(-gradient, collect(dual); atol=tolerance, rtol=tolerance)

    # Fenchel Hessian is the inverse primal Hessian at the inverse-gradient
    # shadow, not the Hessian of the mapped dual-cone barrier.
    primal_hessian = _nc_primal_hessian(tag, workspace.shadow)
    identity3 = Matrix{T}(I, 3, 3)
    @test isapprox(
        primal_hessian * workspace.inverse_hessian,
        identity3;
        atol=tolerance,
        rtol=tolerance,
    )
    @test isapprox(
        workspace.inverse_hessian,
        transpose(workspace.inverse_hessian);
        atol=tolerance,
        rtol=tolerance,
    )

    # The two cross pairings used by the double-secant Gram matrix both equal
    # the degree by logarithmic homogeneity.
    @test isapprox(
        dot(workspace.shadow, collect(dual)), T(3);
        atol=tolerance, rtol=tolerance,
    )
    _nc_primal_gradient!(gradient, tag, primal)
    @test isapprox(
        dot(collect(primal), -gradient), T(3);
        atol=tolerance, rtol=tolerance,
    )

    original_shadow = copy(workspace.shadow)
    original_inverse = copy(workspace.inverse_hessian)
    scale = T(7) / T(3)
    scaled_dual = (
        scale * dual[1], scale * dual[2], scale * dual[3],
    )
    scaled = _nc_call!(workspace, tag, scaled_dual)
    @test scaled.status === SDPX.NS_CONJUGATE_SUCCESS
    @test isapprox(
        workspace.shadow, original_shadow / scale;
        atol=tolerance, rtol=tolerance,
    )
    @test isapprox(
        workspace.inverse_hessian,
        original_inverse / (scale * scale);
        atol=tolerance, rtol=tolerance,
    )
    return workspace
end

@testset "nonsymmetric Fenchel shadow ABI" begin
    @test isbitstype(SDPX.ExpConjugateTag)
    @test isbitstype(SDPX.PowerConjugateTag{Float64})
    @test isbitstype(SDPX.NonsymmetricConjugateStatus)
    @test isbitstype(SDPX.NonsymmetricConjugateReason)
    @test isbitstype(SDPX.NonsymmetricConjugateResult{Float64})
    @test isbitstype(SDPX.NonsymmetricConjugateSettings{Float64})
end

@testset "Float64/MultiFloat conjugacy, homogeneity, and zero allocation" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        tolerance = T(4096) * eps(one(T))
        @testset "$T" begin
            for (tag, dual, primal) in _nc_cases(T)
                workspace = _nc_check_identities(
                    T, tag, dual, primal; tolerance=tolerance,
                )
                original_dual = (
                    dual[1], dual[2], dual[3],
                )
                @test _nc_allocated(workspace, tag, original_dual) == 0
                result = _nc_call!(workspace, tag, original_dual)
                @test isbits(result)
            end
        end
    end
end

@testset "BigFloat256 conjugacy for Exp and Power alpha rungs" begin
    setprecision(BigFloat, 256) do
        for (tag, dual, primal) in _nc_cases(BigFloat)
            _nc_check_identities(
                BigFloat, tag, dual, primal; tolerance=big"1e-65",
            )
        end
    end
end

@testset "mapped dual barrier is not the Fenchel conjugate" begin
    # This point is generated exactly by the primal inverse-gradient relation:
    # -gradient(F_exp, (0,1,2)) = (-1,0,3/2).
    dual = (-1.0, 0.0, 1.5)
    workspace = SDPX.NonsymmetricConjugateWorkspace(Float64)
    result = _nc_call!(workspace, SDPX.ExpConjugateTag(), dual)
    @test result.status === SDPX.NS_CONJUGATE_SUCCESS
    @test workspace.shadow ≈ [0.0, 1.0, 2.0] atol=2e-14 rtol=2e-14

    mapped_gradient = collect(SDPX.exp_dual_gradient(dual...))
    @test norm(-mapped_gradient - workspace.shadow, Inf) > 0.1
    @test norm(-mapped_gradient - workspace.shadow, Inf) >
          1.0e10 * result.residual
end

@testset "Fenchel shadow fail-closed reasons" begin
    workspace = SDPX.NonsymmetricConjugateWorkspace(Float64)

    nonfinite = _nc_call!(
        workspace, SDPX.ExpConjugateTag(), (-1.0, NaN, 2.0),
    )
    @test nonfinite.status === SDPX.NS_CONJUGATE_FAILED
    @test nonfinite.reason === SDPX.NS_CONJUGATE_NONFINITE_DUAL
    @test !workspace.valid

    exp_boundary = _nc_call!(
        workspace,
        SDPX.ExpConjugateTag(),
        (-1.0, 1.0, exp(-2.0)),
    )
    @test exp_boundary.reason === SDPX.NS_CONJUGATE_DUAL_NOT_INTERIOR

    alpha = 0.3
    power_boundary = _nc_call!(
        workspace,
        SDPX.PowerConjugateTag(alpha),
        (alpha, 1.0 - alpha, 1.0),
    )
    @test power_boundary.reason === SDPX.NS_CONJUGATE_DUAL_NOT_INTERIOR

    invalid_alpha = _nc_call!(
        workspace,
        SDPX.PowerConjugateTag(1.0),
        (1.0, 1.0, 0.0),
    )
    @test invalid_alpha.reason === SDPX.NS_CONJUGATE_INVALID_PARAMETER

    wrong_dimension = _nc_call!(
        workspace, SDPX.ExpConjugateTag(), (-1.0, 1.0),
    )
    @test wrong_dimension.reason === SDPX.NS_CONJUGATE_INVALID_PARAMETER

    limited = SDPX.NonsymmetricConjugateWorkspace(
        Float64; max_iterations=0,
    )
    iteration_limit = _nc_call!(
        limited, SDPX.ExpConjugateTag(), (-1.0, 0.0, 1.5),
    )
    @test iteration_limit.status === SDPX.NS_CONJUGATE_FAILED
    @test iteration_limit.reason === SDPX.NS_CONJUGATE_ITERATION_LIMIT
    @test !limited.valid
end

