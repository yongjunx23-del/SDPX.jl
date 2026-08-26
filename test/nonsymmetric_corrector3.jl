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
    @test SDPX.NS_CORRECTOR_LINEAR_SOLVE_MISMATCH isa
          SDPX.NonsymmetricCorrectorReason
    @test SDPX.NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH isa
          SDPX.NonsymmetricCorrectorReason
    @test SDPX.NS_CORRECTOR_PROJECTION_TOO_LARGE isa
          SDPX.NonsymmetricCorrectorReason
    @test SDPX.NS_CORRECTOR_FACTOR_FAILED isa
          SDPX.NonsymmetricCorrectorReason
    @test SDPX.NS_CORRECTOR_FACTOR_MISMATCH isa
          SDPX.NonsymmetricCorrectorReason
end

function _nsc_composed_cancellation_fixture(::Type{T}) where {T}
    workspace = SDPX.NonsymmetricCorrectorWorkspace(T)
    scaling = SDPX.NonsymmetricScalingWorkspace(T)
    condition = sqrt(inv(eps(one(T))))
    twenty_five = T(25)
    g11 = (T(9) * condition + T(16)) / twenty_five
    g12 = T(12) * (condition - one(T)) / twenty_five
    g22 = (T(16) * condition + T(9)) / twenty_five
    inverse_condition = inv(condition)
    t11 = (T(9) * inverse_condition + T(16)) / twenty_five
    t12 = T(12) * (inverse_condition - one(T)) / twenty_five
    t22 = (T(16) * inverse_condition + T(9)) / twenty_five
    two = one(T) + one(T)
    fill!(scaling.g, zero(T))
    fill!(scaling.theta, zero(T))
    scaling.g[1, 1] = g11
    scaling.g[1, 2] = g12
    scaling.g[2, 1] = g12
    scaling.g[2, 2] = g22
    scaling.g[3, 3] = inv(two)
    scaling.theta[1, 1] = t11
    scaling.theta[1, 2] = t12
    scaling.theta[2, 1] = t12
    scaling.theta[2, 2] = t22
    scaling.theta[3, 3] = two
    workspace.rho .= (one(T), one(T), T(3) / T(10))
    SDPX._ns_scaling_matvec!(workspace.h, scaling.theta, workspace.rho)
    SDPX._ns_scaling_matvec!(workspace.work, scaling.g, workspace.h)
    return workspace, scaling
end

@testset "composed-map cancellation gate and corruption rejection" begin
    for T in (
        Float64,
        _NS_CORRECTOR_MF.Float64x2,
        _NS_CORRECTOR_MF.Float64x3,
        _NS_CORRECTOR_MF.Float64x4,
    )
        workspace, scaling = _nsc_composed_cancellation_fixture(T)
        naive_error = maximum(abs, workspace.work - workspace.rho)
        @test naive_error > workspace.validation_tolerance
        accepted, reported = SDPX._ns_corrector_composed_map_gate!(
            workspace, scaling,
        )
        @test accepted
        @test isfinite(reported) && reported <= workspace.validation_tolerance

        scaling.g[1, 1] *= T(2)
        SDPX._ns_scaling_matvec!(workspace.h, scaling.theta, workspace.rho)
        SDPX._ns_scaling_matvec!(workspace.work, scaling.g, workspace.h)
        corrupted, _ = SDPX._ns_corrector_composed_map_gate!(
            workspace, scaling,
        )
        @test !corrupted
    end
    setprecision(BigFloat, 256) do
        workspace, scaling = _nsc_composed_cancellation_fixture(BigFloat)
        naive_error = maximum(abs, workspace.work - workspace.rho)
        @test naive_error > workspace.validation_tolerance
        accepted, reported = SDPX._ns_corrector_composed_map_gate!(
            workspace, scaling,
        )
        @test accepted
        @test isfinite(reported) && reported <= workspace.validation_tolerance
        scaling.theta[1, 1] *= BigFloat(2)
        SDPX._ns_scaling_matvec!(workspace.h, scaling.theta, workspace.rho)
        SDPX._ns_scaling_matvec!(workspace.work, scaling.g, workspace.h)
        corrupted, _ = SDPX._ns_corrector_composed_map_gate!(
            workspace, scaling,
        )
        @test !corrupted
    end
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
                @test workspace.factor_valid
                @test isfinite(workspace.factor_error)
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

@inline function _nsc_near_boundary_cases(::Type{T}) where {T}
    margin = T(1) / T(10_000)
    return (
        (SDPX.ExpConjugateTag(), (zero(T), one(T), exp(margin))),
        (
            SDPX.PowerConjugateTag(T(1) / T(10)),
            (one(T), one(T), exp(-margin)),
        ),
        (
            SDPX.PowerConjugateTag(T(1) / T(2)),
            (one(T), one(T), exp(-margin)),
        ),
        (
            SDPX.PowerConjugateTag(T(9) / T(10)),
            (one(T), one(T), exp(-margin)),
        ),
    )
end

@inline function _nsc_hessian(tag, point)
    return tag isa SDPX.ExpConjugateTag ?
        SDPX.exp_barrier_hessian(point...) :
        SDPX.power_barrier_hessian(point..., tag.alpha)
end

@testset "near-boundary directional jet, Euler gate, and zero allocation" begin
    for T in (
        Float64,
        _NS_CORRECTOR_MF.Float64x2,
        _NS_CORRECTOR_MF.Float64x3,
        _NS_CORRECTOR_MF.Float64x4,
    )
        ds, dy = _nsc_directions(T)
        for (tag, primal) in _nsc_near_boundary_cases(T)
            workspace = SDPX.NonsymmetricCorrectorWorkspace(T)
            result = SDPX.try_nonsymmetric_higher_correction!(
                workspace, tag, primal, ds, dy,
            )
            @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
            @test result.euler_error <= T(1024) * sqrt(eps(one(T)))
            @test workspace.solve_error <= T(128) *
                  (T(3) * eps(one(T)) / (one(T) - T(3) * eps(one(T))))
            @test workspace.symmetry_error <= T(512) * sqrt(eps(one(T)))
            @test _nsc_higher_allocated(
                workspace, tag, primal, ds, dy,
            ) == 0
        end
    end

    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            ds, dy = _nsc_directions(BigFloat)
            margin = inv(BigFloat(10_000))
            step = margin * sqrt(sqrt(eps(BigFloat)))
            tolerance = BigFloat(4096) * sqrt(eps(BigFloat))
            for (tag, primal) in _nsc_near_boundary_cases(BigFloat)
                workspace = SDPX.NonsymmetricCorrectorWorkspace(BigFloat)
                result = SDPX.try_nonsymmetric_higher_correction!(
                    workspace, tag, primal, ds, dy,
                )
                @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
                hessian = _nsc_hessian(tag, primal)
                u_reference = hessian \ collect(dy)
                plus = ntuple(i -> primal[i] + step * ds[i], 3)
                minus = ntuple(i -> primal[i] - step * ds[i], 3)
                hessian_plus = _nsc_hessian(tag, plus)
                hessian_minus = _nsc_hessian(tag, minus)
                chi_reference = -(
                    (hessian_plus - hessian_minus) * u_reference
                ) / (step + step + step + step)
                chi_scale = max(one(BigFloat), maximum(abs, chi_reference))
                @test maximum(abs, workspace.chi - chi_reference) <=
                      tolerance * chi_scale
                @test result.euler_error <= tolerance
                @test precision(workspace.chi[1]) == bits
            end
        end
    end
end

function _nsc_raw_contractions(::Type{T}, tag, primal, ds, dy) where {T}
    hessian = _nsc_hessian(tag, primal)
    u = hessian \ T[dy[1], dy[2], dy[3]]
    primary = zeros(T, 3)
    swapped = zeros(T, 3)
    SDPX._ns_corrector_third_contraction!(
        primary, tag, primal, ds, u,
    )
    SDPX._ns_corrector_third_contraction!(
        swapped, tag, primal, u, ds,
    )
    return primary, swapped
end

function _nsc_structural_raw_contractions(
    ::Type{T}, tag, primal, ds, dy,
) where {T}
    factor = zeros(T, 3, 3)
    @assert SDPX._ns_structural_hessian_factor!(
        factor, tag, primal[1], primal[2], primal[3],
    )
    u = zeros(T, 3)
    forward = zeros(T, 3)
    @assert SDPX._ns_structural_hessian_solve!(
        u, factor, T[dy[1], dy[2], dy[3]], forward,
    )
    primary = zeros(T, 3)
    swapped = zeros(T, 3)
    SDPX._ns_corrector_third_contraction!(
        primary, tag, primal, ds, u,
    )
    SDPX._ns_corrector_third_contraction!(
        swapped, tag, primal, u, ds,
    )
    return primary, swapped
end

function _nsc_bigfloat_hessian_difference(tag, primal, ds, dy)
    return setprecision(BigFloat, 256) do
        big_tag = tag isa SDPX.ExpConjugateTag ?
                  SDPX.ExpConjugateTag() :
                  SDPX.PowerConjugateTag(BigFloat(tag.alpha))
        sb = BigFloat.(primal)
        dsb = BigFloat.(ds)
        dyb = BigFloat.(dy)
        hessian = _nsc_hessian(big_tag, sb)
        u = hessian \ collect(dyb)
        step = BigFloat(2)^(-100) /
               max(one(BigFloat), maximum(abs, dsb))
        plus = ntuple(i -> sb[i] + step * dsb[i], 3)
        minus = ntuple(i -> sb[i] - step * dsb[i], 3)
        hessian_plus = _nsc_hessian(big_tag, plus)
        hessian_minus = _nsc_hessian(big_tag, minus)
        return Float64.(-((hessian_plus - hessian_minus) * u) /
                        (step + step + step + step))
    end
end

@testset "release-state jets match independent BigFloat Hessian differences" begin
    fixtures = (
        (
            SDPX.ExpConjugateTag(),
            (0.0, 2.468472436326448, 2.4684801888833774),
            (-0.0, -4.469527126192585e-12, -0.00010133351089104937),
            (-0.006861126333914458, -0.00019674454689309846,
             3.2453886527072154e-11),
        ),
        (
            SDPX.PowerConjugateTag(0.5),
            (0.44702688968365784, 0.44702688968365784,
             -0.44702580381560175),
            (-1.085644632050869e-6, -1.085644632050869e-6,
             -2.221716217883298e-10),
            (-5.60062303807507e-7, -5.600599004666737e-7,
             -3.138694380843711e-7),
        ),
    )
    for (tag, primal, ds, dy) in fixtures
        # These pre-projection jets use production's structural solve.  The
        # reference remains independent: it forms BigFloat analytic Hessians
        # at two displaced primal points and takes a centered difference; it
        # never calls the factor builder or analytic third contraction.
        primary, swapped = _nsc_structural_raw_contractions(
            Float64, tag, primal, ds, dy,
        )
        workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
        result = SDPX.try_nonsymmetric_higher_correction!(
            workspace, tag, primal, ds, dy,
        )
        @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
        @test workspace.solve_error <= 128 *
              (3eps(Float64) / (1 - 3eps(Float64)))
        @test workspace.symmetry_error <= 512sqrt(eps(Float64))
        @test workspace.raw_euler_error <= 1024sqrt(eps(Float64))

        reference = _nsc_bigfloat_hessian_difference(
            tag, primal, ds, dy,
        )
        primary_error = maximum(abs, primary - reference)
        swapped_error = maximum(abs, swapped - reference)
        midpoint_error = maximum(abs, (primary + swapped) / 2 - reference)
        projected_error = maximum(abs, workspace.chi - reference)
        reference_scale = maximum(abs, reference)
        @test primary_error <= 1e-4 * reference_scale
        @test swapped_error <= 1e-4 * reference_scale
        @test midpoint_error <= 1e-4 * reference_scale
        @test projected_error <= 1e-4 * reference_scale
    end
end

@testset "structural factors survive release-state dense SPD loss" begin
    # Frozen pre-step primal points from the analytic Exp/Power release
    # trajectories.  The independently formed Float64 dense Hessians have
    # lost numerical positive definiteness, although the exact barriers are
    # strictly convex.  The certified analytic L must remain authoritative.
    fixtures = (
        (
            SDPX.ExpConjugateTag(),
            (0.0, 3.255596693374307, 3.2555967067500338),
        ),
        (
            SDPX.PowerConjugateTag(0.5),
            (0.47214176985044354, 0.47214176985044354,
             -0.47214176737485286),
        ),
    )
    ds, dy = _nsc_directions(Float64)
    for (tag, primal) in fixtures
        dense = _nsc_hessian(tag, primal)
        @test !isposdef(Symmetric(dense))
        workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
        result = SDPX.try_nonsymmetric_higher_correction!(
            workspace, tag, primal, ds, dy,
        )
        @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
        @test result.reason === SDPX.NS_CORRECTOR_CONVERGED
        @test workspace.factor_valid
        @test SDPX._ns_structural_factor_finite_lower(workspace.factor)
        factor_ok, error =
            SDPX._ns_structural_hessian_factor_certificate!(
                workspace.factor, tag,
                primal[1], primal[2], primal[3],
            )
        @test factor_ok
        @test error == workspace.factor_error
    end
end

@testset "rational directions match independent BigFloat Hessian differences" begin
    for (case_index, (tag, primal, _)) in enumerate(_nsc_cases(Float64))
        for sample in 1:6
            ds = ntuple(i -> Float64(
                mod(7sample + 11i + 3case_index, 29) - 14,
            ) / 37, 3)
            dy = ntuple(i -> Float64(
                mod(13sample + 5i + case_index, 31) - 15,
            ) / 41, 3)
            workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
            result = SDPX.try_nonsymmetric_higher_correction!(
                workspace, tag, primal, ds, dy,
            )
            @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
            reference = _nsc_bigfloat_hessian_difference(
                tag, primal, ds, dy,
            )
            scale = max(maximum(abs, reference), floatmin(Float64))
            @test maximum(abs, workspace.chi - reference) <= 1e-10 * scale
        end
    end
end

@testset "pre-projection corruption is rejected" begin
    tag, primal, _ = _nsc_cases(Float64)[2]
    ds, dy = _nsc_directions(Float64)
    workspace = SDPX.NonsymmetricCorrectorWorkspace(Float64)
    result = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, ds, dy,
    )
    @test result.status === SDPX.NS_CORRECTOR_COMBINED_READY
    scale = maximum(abs, workspace.chi)
    workspace.chi[1] += scale
    reason = SDPX._ns_corrector_euler_projection!(
        workspace, primal..., ds,
    )
    @test reason === SDPX.NS_CORRECTOR_PROJECTION_TOO_LARGE
end

@testset "Euler backward gate is homogeneous in affine directions" begin
    T = Float64
    ds, dy = _nsc_directions(T)
    for (tag, primal) in _nsc_near_boundary_cases(T)
        workspace = SDPX.NonsymmetricCorrectorWorkspace(T)
        baseline = SDPX.try_nonsymmetric_higher_correction!(
            workspace, tag, primal, ds, dy,
        )
        @test baseline.status === SDPX.NS_CORRECTOR_COMBINED_READY
        baseline_error = baseline.euler_error
        for exponent in (-40, 40)
            scale = ldexp(one(T), exponent)
            scaled_ds = ntuple(i -> scale * ds[i], 3)
            scaled_dy = ntuple(i -> scale * dy[i], 3)
            scaled = SDPX.try_nonsymmetric_higher_correction!(
                workspace, tag, primal, scaled_ds, scaled_dy,
            )
            @test scaled.status === SDPX.NS_CORRECTOR_COMBINED_READY
            if iszero(baseline_error)
                @test iszero(scaled.euler_error)
            else
                @test isapprox(
                    scaled.euler_error, baseline_error;
                    atol=zero(T), rtol=T(4096) * eps(T),
                )
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

    factor_failure = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, (0.0, 1.0e200, 2.0e200), ds, dy,
    )
    @test factor_failure.status === SDPX.NS_CORRECTOR_FAILED
    @test factor_failure.reason === SDPX.NS_CORRECTOR_FACTOR_FAILED
    third_failure = SDPX.try_nonsymmetric_higher_correction!(
        workspace, tag, primal, (1.0e308, 1.0e308, 1.0e308), dy,
    )
    @test third_failure.status === SDPX.NS_CORRECTOR_FAILED
    @test third_failure.reason === SDPX.NS_CORRECTOR_THIRD_DERIVATIVE_FAILED
    @test !workspace.factor_valid
    @test isinf(workspace.factor_error)

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
