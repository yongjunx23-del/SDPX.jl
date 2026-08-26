# Three-dimensional nonsymmetric higher-order corrector gates.

using LinearAlgebra
using SDPX
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
if !isdefined(SDPX, :NonsymmetricCorrectorWorkspace)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric", "corrector3.jl",
        ),
    )
end
if !isdefined(SDPX, :NonsymmetricFullNewtonResult)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric",
            "full_newton_reference.jl",
        ),
    )
end

const _NS_CORRECTOR_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

function _nsc_cases(::Type{T}) where {T}
    power_case(alpha) = (
        SDPX.PowerConjugateTag(alpha),
        (one(T), one(T), zero(T)),
        (alpha, one(T) - alpha, T(1) / T(2)),
    )
    return (
        (
            SDPX.ExpConjugateTag(),
            (zero(T), one(T), T(2)),
            (-one(T), one(T), one(T)),
        ),
        power_case(T(1) / T(10)),
        power_case(T(1) / T(2)),
        power_case(T(9) / T(10)),
    )
end

@inline function _nsc_directions(::Type{T}) where {T}
    return (
        (T(1) / T(10), -T(1) / T(5), T(1) / T(20)),
        (-T(3) / T(10), T(3) / T(20), T(1) / T(5)),
    )
end

@noinline function _nsc_higher_allocated(workspace, tag, primal, ds, dy)
    SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, ds, dy,
    )
    return @allocated SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, ds, dy,
    )
end

@noinline function _nsc_affine_allocated(workspace, scaling, primal, dual)
    SDPX.nonsymmetric_affine_shift!(workspace, scaling, primal, dual)
    return @allocated SDPX.nonsymmetric_affine_shift!(
        workspace, scaling, primal, dual,
    )
end

@noinline function _nsc_combined_allocated(
    workspace, scaling, tag, primal, dual, ds, dy, target,
)
    SDPX.nonsymmetric_combined_shift!(
        workspace, scaling, tag, primal, dual, ds, dy, target,
    )
    return @allocated SDPX.nonsymmetric_combined_shift!(
        workspace, scaling, tag, primal, dual, ds, dy, target,
    )
end

function _nsc_max3(vector)
    return max(abs(vector[1]), abs(vector[2]), abs(vector[3]))
end

function _nsc_max_newton_residual(result)
    residuals = result.residuals
    return max(
        maximum(abs, residuals.primal),
        maximum(abs, residuals.dual),
        abs(residuals.gap),
        maximum(abs, residuals.complementarity),
        abs(residuals.tau),
    )
end

function _nsc_reference_data(tag, primal)
    block = tag isa SDPX.ExpConjugateTag ?
            SDPX.NewtonExpBlock() : SDPX.NewtonPowerBlock(tag.alpha)
    gradient, hessian = SDPX._ns_newton_barrier_data(block, primal...)
    return block, gradient, hessian
end

function _nsc_centered_pair(tag, primal)
    gradient = if tag isa SDPX.ExpConjugateTag
        SDPX.exp_barrier_gradient(primal...)
    else
        SDPX.power_barrier_gradient(primal..., tag.alpha)
    end
    return (-gradient[1], -gradient[2], -gradient[3])
end

@testset "nonsymmetric corrector typed ABI" begin
    @test isbitstype(SDPX.NonsymmetricCorrectorStatus)
    @test isbitstype(SDPX.NonsymmetricCorrectorReason)
    @test isbitstype(SDPX.NonsymmetricCorrectorResult{Float64})
end

@testset "Exp/Power correction identities and fixed-width zero allocation" begin
    for T in (
        Float64,
        _NS_CORRECTOR_MF.Float64x2,
        _NS_CORRECTOR_MF.Float64x3,
        _NS_CORRECTOR_MF.Float64x4,
    )
        @testset "$T" begin
            tolerance = T(262144) * eps(one(T))
            ds, dy = _nsc_directions(T)
            target = T(2) / T(5)
            for (tag, primal, dual) in _nsc_cases(T)
                scaling = SDPX.NonsymmetricScalingWorkspace(T)
                scaling_result = SDPX.try_update_nonsymmetric_scaling!(
                    scaling, SDPX.StrictDoubleSecantScaling(),
                    tag, primal, dual,
                )
                @test scaling_result.status === SDPX.NS_SCALING_DOUBLE_SECANT
                workspace = SDPX.NonsymmetricCorrectorWorkspace(T)

                affine = SDPX.nonsymmetric_affine_shift!(
                    workspace, scaling, primal, dual,
                )
                @test affine.status === SDPX.NS_CORRECTOR_AFFINE_READY
                @test affine.reason === SDPX.NS_CORRECTOR_CONVERGED
                @test isbits(affine)
                @test isapprox(
                    workspace.rho, -collect(dual); atol=tolerance, rtol=tolerance,
                )
                @test isapprox(
                    workspace.h, -collect(primal); atol=tolerance, rtol=tolerance,
                )
                @test isapprox(
                    scaling.theta * workspace.rho, workspace.h;
                    atol=tolerance, rtol=tolerance,
                )

                combined = SDPX.nonsymmetric_combined_shift!(
                    workspace, scaling, tag, primal, dual, ds, dy, target,
                )
                @test combined.status === SDPX.NS_CORRECTOR_COMBINED_READY
                @test combined.reason === SDPX.NS_CORRECTOR_CONVERGED
                @test isbits(combined)
                @test isapprox(
                    dot(collect(primal), workspace.chi),
                    dot(collect(ds), collect(dy));
                    atol=tolerance, rtol=tolerance,
                )
                rho_reference = target .* scaling.dual_shadow .-
                                collect(dual) .- workspace.chi
                @test isapprox(
                    workspace.rho, rho_reference;
                    atol=tolerance, rtol=tolerance,
                )
                @test isapprox(
                    scaling.theta * workspace.rho, workspace.h;
                    atol=tolerance, rtol=tolerance,
                )
                @test isapprox(
                    scaling.g * workspace.h, workspace.rho;
                    atol=tolerance, rtol=tolerance,
                )

                @test _nsc_higher_allocated(
                    workspace, tag, primal, ds, dy,
                ) == 0
                @test _nsc_affine_allocated(
                    workspace, scaling, primal, dual,
                ) == 0
                @test _nsc_combined_allocated(
                    workspace, scaling, tag, primal, dual,
                    ds, dy, target,
                ) == 0
            end
        end
    end
end

@testset "independent BigFloat Hessian derivative fixes chi sign" begin
    setprecision(BigFloat, 256) do
        ds, dy = _nsc_directions(BigFloat)
        step = BigFloat(2)^(-80)
        tolerance = BigFloat("2e-40")
        for (tag, primal, _) in _nsc_cases(BigFloat)
            workspace = SDPX.NonsymmetricCorrectorWorkspace(BigFloat)
            result = SDPX.try_nonsymmetric_higher_correction!(
                workspace, tag, primal, ds, dy,
            )
            @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY

            block, _, hessian = _nsc_reference_data(tag, primal)
            plus = ntuple(i -> primal[i] + step * ds[i], 3)
            minus = ntuple(i -> primal[i] - step * ds[i], 3)
            _, hessian_plus = SDPX._ns_newton_barrier_data(block, plus...)
            _, hessian_minus = SDPX._ns_newton_barrier_data(block, minus...)
            u_reference = hessian \ collect(dy)
            chi_reference = -(
                (hessian_plus - hessian_minus) * u_reference
            ) / (step + step + step + step)
            @test maximum(abs, workspace.u - u_reference) <= tolerance
            @test maximum(abs, workspace.chi - chi_reference) <= tolerance
            @test maximum(abs, workspace.chi + chi_reference) > BigFloat("1e-5")
            @test abs(dot(collect(primal), workspace.chi) -
                      dot(collect(ds), collect(dy))) <= tolerance
        end
    end
end

@testset "BigFloat256 affine and combined identities" begin
    setprecision(BigFloat, 256) do
        tolerance = BigFloat("2e-66")
        ds, dy = _nsc_directions(BigFloat)
        target = BigFloat("0.4")
        for (tag, primal, dual) in _nsc_cases(BigFloat)
            scaling = SDPX.NonsymmetricScalingWorkspace(BigFloat)
            scaling_result = SDPX.try_update_nonsymmetric_scaling!(
                scaling, SDPX.StrictDoubleSecantScaling(), tag, primal, dual,
            )
            @test scaling_result.status === SDPX.NS_SCALING_DOUBLE_SECANT
            workspace = SDPX.NonsymmetricCorrectorWorkspace(BigFloat)
            affine = SDPX.nonsymmetric_affine_shift!(
                workspace, scaling, primal, dual,
            )
            @test affine.status === SDPX.NS_CORRECTOR_AFFINE_READY
            combined = SDPX.nonsymmetric_combined_shift!(
                workspace, scaling, tag, primal, dual, ds, dy, target,
            )
            @test combined.status === SDPX.NS_CORRECTOR_COMBINED_READY
            @test combined.euler_error <= tolerance
            @test combined.linearization_error <= tolerance
            @test maximum(abs, scaling.g * workspace.h - workspace.rho) <=
                  tolerance
        end
    end
end

@testset "combined cone step reduces the nonlinear shadow residual" begin
    ds, dy = _nsc_directions(Float64)
    target = 0.4
    for (tag, primal, dual) in _nsc_cases(Float64)
        scaling = SDPX.NonsymmetricScalingWorkspace(Float64)
        SDPX.try_update_nonsymmetric_scaling!(
            scaling, SDPX.StrictDoubleSecantScaling(), tag, primal, dual,
        )
        workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
        result = SDPX.nonsymmetric_combined_shift!(
            workspace, scaling, tag, primal, dual, ds, dy, target,
        )
        @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY

        # A cone-only direction (ds=0, dy=rho) obeys the accepted linearized
        # equation exactly.  Its full step leaves s strictly interior and moves
        # the actual dual point from y to target*ytilde-chi, not merely a
        # synthetic residual vector.
        dual_trial = collect(dual) + workspace.rho
        residual_before = collect(dual) - target .* scaling.dual_shadow
        residual_after = dual_trial - target .* scaling.dual_shadow
        @test norm(residual_after) < norm(residual_before)
        @test SDPX._ns_conjugate_dual_interior(
            tag, dual_trial[1], dual_trial[2], dual_trial[3],
        )
    end
end

function _nsc_full_newton_fixture(tag, primal)
    T = eltype(primal)
    dual = _nsc_centered_pair(tag, primal)
    scaling = SDPX.NonsymmetricScalingWorkspace(T)
    scaling_result = SDPX.try_update_nonsymmetric_scaling!(
        scaling, SDPX.DoubleSecantWithDualHessianFallback(),
        tag, primal, dual,
    )
    @test scaling_result.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
    block, gradient, _ = _nsc_reference_data(tag, primal)
    A = T[
        1 1 / 5
        -3 / 10 7 / 10
        1 / 2 -11 / 10
    ]
    b = T[2 / 5, -1 / 5, 3 / 10]
    c = T[3 / 5, -1 / 2]
    x = T[1 / 5, -3 / 20]
    tau = T(13) / T(10)
    kappa = T(4) / T(5)
    return (
        scaling=scaling,
        block=block,
        gradient=gradient,
        A=A,
        b=b,
        c=c,
        x=x,
        dual=dual,
        tau=tau,
        kappa=kappa,
    )
end

@testset "independent coupled full-Newton affine/corrector directions" begin
    setprecision(BigFloat, 256) do
        cases = (
            (SDPX.ExpConjugateTag(),
             (big"0.0", big"1.0", big"2.0")),
            (SDPX.PowerConjugateTag(big"0.1"),
             (big"1.0", big"1.0", big"0.0")),
            (SDPX.PowerConjugateTag(big"0.5"),
             (big"1.0", big"1.0", big"0.0")),
            (SDPX.PowerConjugateTag(big"0.9"),
             (big"1.0", big"1.0", big"0.0")),
        )
        residual_tolerance = BigFloat("1e-70")
        direction_tolerance = BigFloat("2e-62")
        for (tag, primal) in cases
            fixture = _nsc_full_newton_fixture(tag, primal)
            scaling = fixture.scaling
            mu = scaling.mu
            affine_target = mu .* fixture.gradient
            affine = SDPX.nonsymmetric_hsd_full_newton_reference(
                fixture.A, fixture.b, fixture.c, fixture.x,
                collect(fixture.dual), collect(primal), mu,
                (fixture.block,), fixture.tau, fixture.kappa;
                cone_target=affine_target,
                precision_bits=384,
            )
            @test affine.status === SDPX.NS_NEWTON_SOLVED
            @test _nsc_max_newton_residual(affine) <= residual_tolerance

            workspace = SDPX.NonsymmetricCorrectorWorkspace(BigFloat)
            affine_shift = SDPX.nonsymmetric_affine_shift!(
                workspace, scaling, primal, fixture.dual,
            )
            @test affine_shift.status === SDPX.NS_CORRECTOR_AFFINE_READY
            @test maximum(abs, affine.dy + affine.G * affine.ds -
                               workspace.rho) <= direction_tolerance
            @test maximum(abs, affine.ds + scaling.theta * affine.dy -
                               workspace.h) <= direction_tolerance

            sigma_mu = BigFloat("0.4")
            combined_shift = SDPX.nonsymmetric_combined_shift!(
                workspace, scaling, tag, primal, fixture.dual,
                affine.ds, affine.dy, sigma_mu,
            )
            @test combined_shift.status === SDPX.NS_CORRECTOR_COMBINED_READY
            combined_target = workspace.rho .+ collect(fixture.dual) .+
                              mu .* fixture.gradient
            scalar_target = sigma_mu - affine.dtau * affine.dkappa
            combined = SDPX.nonsymmetric_hsd_full_newton_reference(
                fixture.A, fixture.b, fixture.c, fixture.x,
                collect(fixture.dual), collect(primal), mu,
                (fixture.block,), fixture.tau, fixture.kappa;
                cone_target=combined_target,
                scalar_target=scalar_target,
                precision_bits=384,
            )
            @test combined.status === SDPX.NS_NEWTON_SOLVED
            @test _nsc_max_newton_residual(combined) <= residual_tolerance
            @test maximum(abs, combined.jacobian * combined.solution -
                               combined.rhs) <= residual_tolerance
            @test maximum(abs, combined.dy + combined.G * combined.ds -
                               workspace.rho) <= direction_tolerance
            @test maximum(abs, combined.ds + scaling.theta * combined.dy -
                               workspace.h) <= direction_tolerance

            # The frozen higher-order central residual is eliminated by the
            # independently solved coupled direction; this guards the sign of
            # both chi and the metric orientation.
            central_before = collect(fixture.dual) .+
                             sigma_mu .* fixture.gradient .+ workspace.chi
            central_after = central_before .+ combined.dy .+
                            combined.G * combined.ds
            @test maximum(abs, central_before) > BigFloat("1e-6")
            @test maximum(abs, central_after) <= direction_tolerance
        end
    end
end

@testset "corrector failures are typed and fail closed" begin
    tag, primal, dual = first(_nsc_cases(Float64))
    ds, dy = _nsc_directions(Float64)
    workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
    scaling = SDPX.NonsymmetricScalingWorkspace(Float64)

    invalid_scaling = SDPX.nonsymmetric_affine_shift!(
        workspace, scaling, primal, dual,
    )
    @test invalid_scaling.status === SDPX.NS_CORRECTOR_FAILED
    @test invalid_scaling.reason === SDPX.NS_CORRECTOR_SCALING_INVALID
    @test !workspace.valid

    SDPX.try_update_nonsymmetric_scaling!(
        scaling, SDPX.StrictDoubleSecantScaling(), tag, primal, dual,
    )
    mismatch = SDPX.nonsymmetric_affine_shift!(
        workspace, scaling, (0.0, 1.0, 2.1), dual,
    )
    @test mismatch.reason === SDPX.NS_CORRECTOR_SCALING_POINT_MISMATCH
    nonfinite = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, ds, (NaN, dy[2], dy[3]),
    )
    @test nonfinite.reason === SDPX.NS_CORRECTOR_NONFINITE_INPUT
    boundary = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, (0.0, 1.0, 1.0), ds, dy,
    )
    @test boundary.reason === SDPX.NS_CORRECTOR_PRIMAL_NOT_INTERIOR
    bad_tag = SDPX.try_nonsymmetric_higher_correction!(
        workspace, SDPX.PowerConjugateTag(0.0),
        (1.0, 1.0, 0.0), ds, dy,
    )
    @test bad_tag.reason === SDPX.NS_CORRECTOR_INVALID_PARAMETER
    bad_target = SDPX.nonsymmetric_combined_shift!(
        workspace, scaling, tag, primal, dual, ds, dy, Inf,
    )
    @test bad_target.reason === SDPX.NS_CORRECTOR_INVALID_PARAMETER

    not_spd = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, (0.0, 1.0e200, 2.0e200), ds, dy,
    )
    @test not_spd.status === SDPX.NS_CORRECTOR_FAILED
    @test not_spd.reason === SDPX.NS_CORRECTOR_HESSIAN_NOT_SPD
    third_failure = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, (1.0e308, 1.0e308, 1.0e308), dy,
    )
    @test third_failure.status === SDPX.NS_CORRECTOR_FAILED
    @test third_failure.reason === SDPX.NS_CORRECTOR_THIRD_DERIVATIVE_FAILED

    saved_theta = scaling.theta[1, 1]
    scaling.theta[1, 1] = NaN
    metric_failure = SDPX.nonsymmetric_affine_shift!(
        workspace, scaling, primal, dual,
    )
    @test metric_failure.reason === SDPX.NS_CORRECTOR_METRIC_FAILED
    scaling.theta[1, 1] = saved_theta

    saved_g = scaling.g[1, 1]
    scaling.g[1, 1] *= 2.0
    linearization_failure = SDPX.nonsymmetric_combined_shift!(
        workspace, scaling, tag, primal, dual, ds, dy, 0.4,
    )
    @test linearization_failure.reason ===
          SDPX.NS_CORRECTOR_LINEARIZATION_MISMATCH
    scaling.g[1, 1] = saved_g
    @test !workspace.valid
end
