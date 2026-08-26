# Double-secant/BFGS and explicit conjugate-Hessian fallback gates.

using SDPX
using LinearAlgebra
using SparseArrays
using Test

# The file remains self-contained when run directly (where the full precision
# ladder is useful), while the quick suite keeps the 512/1024 rungs out of the
# edit-test loop.  `runtests.jl` defines TEST_PROFILE in Main before including
# this file.
const _NS_SCALING_PROFILE = isdefined(Main, :TEST_PROFILE) ?
    getfield(Main, :TEST_PROFILE) : :full

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

function _nss_update_global!(workspace, policy, tag, primal, dual, mu)
    return SDPX.try_update_nonsymmetric_scaling!(
        workspace, policy, tag, primal, dual, mu,
    )
end

function _nss_allocated_global(workspace, policy, tag, primal, dual, mu)
    _nss_update_global!(workspace, policy, tag, primal, dual, mu)
    return @allocated _nss_update_global!(
        workspace, policy, tag, primal, dual, mu,
    )
end

function _nss_release_pair(::Type{T}, kind::Symbol) where {T}
    if kind === :exp
        return (
            SDPX.ExpConjugateTag(),
            T.((0.0, 4.06030934303952, 4.060458846531402)),
            T.((-4.06180700823749, -4.060291160015501,
                4.060309343039519)),
            T(2.2695305177613926e-4),
        )
    elseif kind === :power
        return (
            SDPX.PowerConjugateTag(T(1) / T(2)),
            T.((0.4470149812161023, 0.4470149812161023,
                -0.44701389537423586)),
            T.((0.22350750786723314, 0.2235075078668967,
                0.4470142094688258)),
            T(3.366542675584775e-7),
        )
    end
    throw(ArgumentError("unknown release scaling fixture $kind"))
end

"""The Power endpoint state after two accepted native HSD epochs."""
function _nss_power_endpoint_state(alpha::T) where {T<:AbstractFloat}
    layout = SDPX.canonical_layout([
        SDPX.ConeBlockDescriptor(
            T, :power, 3; offset=1, parameter=alpha,
        ),
    ])
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    seed_program = SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(undef, 0),
        spzeros(T, layout.dimension, 0), ones(T, layout.dimension),
        layout, chain,
    )
    seed_state = SDPX.ProductConeHSDState(seed_program)
    SDPX.product_hsd_cold_start!(seed_state)
    program = SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(undef, 0),
        spzeros(T, layout.dimension, 0),
        (T(3) / T(2)) .* seed_state.base.s, layout, chain,
    )
    state = SDPX.ProductConeHSDState(program)
    SDPX.product_hsd_cold_start!(state)
    SDPX.product_hsd_step!(state) === SDPX.HSDStepOK ||
        throw(ErrorException("Power endpoint first HSD step failed"))
    SDPX.product_hsd_step!(state) === SDPX.HSDStepOK ||
        throw(ErrorException("Power endpoint second HSD step failed"))
    return state
end

"""The Power endpoint pair after two accepted native HSD epochs."""
function _nss_power_endpoint_pair(alpha::T) where {T<:AbstractFloat}
    state = _nss_power_endpoint_state(alpha)
    return copy(state.base.s), copy(state.base.y), state.base.mu
end

# Keep the precision-parity assertions from forcing the large fully-inlined
# runtime preflight through the test caller's optimization pipeline.  The
# production method itself remains the one being invoked.
@noinline function _nss_runtime_live_preflight(block, mu)
    return Base.invokelatest(SDPX._runtime_ns_live_block_preflight, block, mu)
end

@noinline function _nss_runtime_checkpoint_preflight(block, mu)
    return Base.invokelatest(
        SDPX._runtime_ns_checkpoint_block_preflight, block, mu,
    )
end

@noinline function _nss_force_fallback!(
    workspace, tag, primal, dual, mu,
)
    workspace.valid = false
    workspace.primal .= primal
    workspace.dual .= dual
    workspace.mu = mu
    candidate = SDPX._ns_conjugate_shadow_hessian_candidate!(
        workspace.conjugate, tag, workspace.dual,
    )
    candidate.status === SDPX.NS_CONJUGATE_SUCCESS || return false
    SDPX._ns_scaling_primal_gradient_hessian!(workspace, tag)
    SDPX._ns_conjugate_ensure_inverse_hessian!(workspace.conjugate) ||
        return false
    return SDPX._ns_scaling_dual_hessian_fallback!(workspace) ===
           SDPX.NS_SCALING_CONVERGED
end

@noinline function _nss_force_fallback_allocated!(
    workspace, tag, primal, dual, mu,
)
    _nss_force_fallback!(workspace, tag, primal, dual, mu)
    return @allocated _nss_force_fallback!(
        workspace, tag, primal, dual, mu,
    )
end

# These are the actual final one-secant metrics at the two release trajectory
# failures.  The Exp conjugate inverse cannot be regenerated deterministically
# from a cold Float64 seed, so its accepted Theta is frozen directly.  The
# validator below tests only the metric authority contract, not the provider.
function _nss_load_release_authority!(workspace, kind::Symbol)
    _, primal, dual, mu = _nss_release_pair(Float64, kind)
    workspace.primal .= primal
    workspace.dual .= dual
    workspace.mu = mu
    if kind === :exp
        workspace.theta .= [
            6.303230208567776 171.9364363201679 178.24122153443437
            171.9364363201679 496075.90472089336 496246.68303177704
            178.24122153443437 496246.68303177704 496423.7677254614
        ]
    elseif kind === :power
        workspace.theta .= [
            581461.6777070523 581457.9414206652 -581459.8583226866
            581457.9414206652 581461.677775377 -581459.8583568492
            -581459.8583226866 -581459.8583568492 581459.907101113
        ]
    else
        throw(ArgumentError("unknown release scaling fixture $kind"))
    end
    return SDPX._ns_scaling_rebuild_fallback_g!(workspace)
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

@testset "explicit product-HSD global mu is authoritative" begin
    for T in (
        Float64,
        _NS_SCALING_MF.Float64x2,
        _NS_SCALING_MF.Float64x3,
        _NS_SCALING_MF.Float64x4,
    )
        tag, primal, dual = first(_nss_double_cases(T))
        local_mu = dot(collect(primal), collect(dual)) / T(3)
        global_mu = T(7) * local_mu / T(5)
        workspace = SDPX.NonsymmetricScalingWorkspace(T)
        policy = SDPX.StrictDoubleSecantScaling()
        result = _nss_update_global!(
            workspace, policy, tag, primal, dual, global_mu,
        )
        tolerance = _nss_matrix_tolerance(T)
        @test result.status === SDPX.NS_SCALING_DOUBLE_SECANT
        @test result.mu == global_mu
        @test workspace.mu == global_mu
        @test workspace.mu != local_mu
        @test isapprox(
            workspace.g * collect(primal), collect(dual);
            atol=tolerance, rtol=tolerance,
        )
        @test isapprox(
            workspace.theta * collect(dual), collect(primal);
            atol=tolerance, rtol=tolerance,
        )
        @test _nss_allocated_global(
            workspace, policy, tag, primal, dual, global_mu,
        ) == 0

        invalid = _nss_update_global!(
            workspace, policy, tag, primal, dual, zero(T),
        )
        @test invalid.status === SDPX.NS_SCALING_FAILED
        @test invalid.reason === SDPX.NS_SCALING_INVALID_PARAMETER
        @test !workspace.valid
    end
end

@testset "Power endpoint inverse authority is typed and policy explicit" begin
    for alpha in (0.1, 0.9)
        primal, dual, mu = _nss_power_endpoint_pair(alpha)
        tag = SDPX.PowerConjugateTag(alpha)

        strict_workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
        strict = _nss_update_global!(
            strict_workspace, SDPX.StrictDoubleSecantScaling(),
            tag, primal, dual, mu,
        )
        @test strict.status === SDPX.NS_SCALING_FAILED
        @test strict.reason === SDPX.NS_SCALING_INVERSE_MISMATCH
        @test strict.fallback_reason === SDPX.NS_SCALING_NO_FALLBACK
        @test !strict_workspace.valid

        workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
        policy = SDPX.DoubleSecantWithDualHessianFallback()
        fallback = _nss_update_global!(
            workspace, policy, tag, primal, dual, mu,
        )
        @test fallback.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
        @test fallback.reason === SDPX.NS_SCALING_CONVERGED
        @test fallback.fallback_reason === SDPX.NS_SCALING_INVERSE_MISMATCH
        @test workspace.valid
        @test workspace.used_fallback
        @test SDPX._runtime_ns_metric_inverse_preflight(
            workspace.g, workspace.theta, workspace.factor,
        )

        saved = workspace.g[1, 1]
        workspace.g[1, 1] = saved + 16.0 * max(1.0, abs(saved))
        @test !SDPX._runtime_ns_metric_inverse_preflight(
            workspace.g, workspace.theta, workspace.factor,
        )
        workspace.g[1, 1] = saved

        @test @allocated(_nss_update_global!(
            workspace, policy, tag, primal, dual, mu,
        )) == 0
    end
end

@testset "Power endpoint runtime/checkpoint parity across precision" begin
    precision_cases = (
        (Float64, (0.1, 0.9)),
        (_NS_SCALING_MF.Float64x2, (_NS_SCALING_MF.Float64x2(0.1),)),
        (_NS_SCALING_MF.Float64x3, (_NS_SCALING_MF.Float64x3(0.1),)),
        (_NS_SCALING_MF.Float64x4, (_NS_SCALING_MF.Float64x4(0.1),)),
    )
    for (T, alphas) in precision_cases
        @testset "$T" begin
            for alpha in alphas
                # The Float64 pair is the actual two-epoch endpoint above;
                # its decimal form is reused as a deterministic high-
                # precision runtime input to keep this parity gate fast.
                primal = [
                    parse(T, value) for value in
                    ("0.6949346097633113", "0.6949346097633113", "0.0")
                ]
                dual = [
                    parse(T, value) for value in (
                        "0.0011865865735625897",
                        "0.0018722641692649478",
                        "-0.00012188993973588635",
                    )
                ]
                mu = parse(T, "0.0009080787889582998")
                layout = SDPX.canonical_layout([
                    SDPX.ConeBlockDescriptor(
                        T, :power, 3; offset=1, parameter=alpha,
                    ),
                ])
                runtime = SDPX.ProductConeRuntime(layout, T)
                @test SDPX.try_update_scaling!(
                    runtime, primal, dual, mu,
                )
                block = only(runtime.power)
                @test runtime.valid
                @test block.last_scaling_reason ===
                      SDPX.NS_SCALING_CONVERGED
                @test _nss_runtime_live_preflight(
                    block, runtime.last_mu,
                )
                @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
                @test runtime.checkpoint_valid
                @test _nss_runtime_checkpoint_preflight(
                    block, runtime.checkpoint_mu,
                )
                @test SDPX.restore_nonsymmetric_scaling_checkpoint!(
                    runtime,
                )
                @test runtime.valid
                if T !== Float64
                    @test @allocated(SDPX.try_update_scaling!(
                        runtime, primal, dual, mu,
                    )) == 0
                end
            end
        end
    end
    setprecision(BigFloat, 256) do
        @testset "BigFloat256" begin
            alpha = BigFloat("0.1")
            primal = [
                parse(BigFloat, value) for value in
                ("0.6949346097633113", "0.6949346097633113", "0.0")
            ]
            dual = [
                parse(BigFloat, value) for value in (
                    "0.0011865865735625897",
                    "0.0018722641692649478",
                    "-0.00012188993973588635",
                )
            ]
            mu = parse(BigFloat, "0.0009080787889582998")
            layout = SDPX.canonical_layout([
                SDPX.ConeBlockDescriptor(
                    BigFloat, :power, 3; offset=1, parameter=alpha,
                ),
            ])
            runtime = SDPX.ProductConeRuntime(layout, BigFloat)
            @test SDPX.try_update_scaling!(
                runtime, primal, dual, mu,
            )
            block = only(runtime.power)
            @test runtime.valid
            @test _nss_runtime_live_preflight(
                block, runtime.last_mu,
            )
            @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
            @test SDPX.restore_nonsymmetric_scaling_checkpoint!(
                runtime,
            )
            @test runtime.valid
        end
    end
end

@testset "final metric validation cannot commit a rejected candidate" begin
    tag = SDPX.ExpConjugateTag()
    old_primal, old_dual = _nss_centered_case(Float64, tag)
    workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
    old = _nss_update!(
        workspace, SDPX.DoubleSecantWithDualHessianFallback(),
        tag, old_primal, old_dual,
    )
    @test old.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK

    conjugate = workspace.conjugate
    accepted_snapshot = (
        dual = copy(conjugate.accepted_dual),
        shadow = copy(conjugate.accepted_shadow),
        hessian = copy(conjugate.accepted_hessian),
        factor = copy(conjugate.accepted_hessian_factor),
        inverse = copy(conjugate.accepted_inverse_hessian),
        valid = conjugate.accepted_valid,
        factor_valid = conjugate.accepted_hessian_factor_valid,
        inverse_valid = conjugate.accepted_inverse_valid,
    )

    # Prepare a new candidate without publishing it.  This mirrors the
    # provider's validate-then-accept transaction and lets the test inject a
    # deterministic failure between the first and final validation.
    _, primal, dual = first(_nss_double_cases(Float64))
    workspace.valid = false
    workspace.primal .= primal
    workspace.dual .= dual
    workspace.mu = dot(collect(primal), collect(dual)) / 3
    candidate = SDPX._ns_conjugate_shadow_hessian_candidate!(
        conjugate, tag, workspace.dual,
    )
    @test candidate.status === SDPX.NS_CONJUGATE_SUCCESS
    SDPX._ns_scaling_primal_gradient_hessian!(workspace, tag)
    @test SDPX._ns_scaling_double_secant!(workspace) ===
          SDPX.NS_SCALING_CONVERGED
    first_reason, _, _ = SDPX._ns_scaling_validate_metric!(workspace, true)
    @test first_reason === SDPX.NS_SCALING_CONVERGED

    workspace.g[1, 1] += 16.0 * max(1.0, abs(workspace.g[1, 1]))
    second_reason, _, _ = SDPX._ns_scaling_validate_metric!(workspace, true)
    @test second_reason === SDPX.NS_SCALING_SECANT_MISMATCH
    @test SDPX._ns_conjugate_restore_accepted!(conjugate)
    @test conjugate.accepted_dual == accepted_snapshot.dual
    @test conjugate.accepted_shadow == accepted_snapshot.shadow
    @test conjugate.accepted_hessian == accepted_snapshot.hessian
    @test conjugate.accepted_hessian_factor == accepted_snapshot.factor
    @test conjugate.accepted_inverse_hessian == accepted_snapshot.inverse
    @test conjugate.accepted_valid == accepted_snapshot.valid
    @test conjugate.accepted_hessian_factor_valid ==
          accepted_snapshot.factor_valid
    @test conjugate.accepted_inverse_valid == accepted_snapshot.inverse_valid
end

@testset "strict double-secant does not consume the optional inverse" begin
    for T in (
        Float64,
        _NS_SCALING_MF.Float64x2,
        _NS_SCALING_MF.Float64x3,
        _NS_SCALING_MF.Float64x4,
    )
        tag, primal, dual = first(_nss_double_cases(T))
        workspace = SDPX.NonsymmetricScalingWorkspace(T)
        fill!(workspace.conjugate.inverse_hessian, T(NaN))
        policy = SDPX.StrictDoubleSecantScaling()
        result = _nss_update!(workspace, policy, tag, primal, dual)
        @test result.status === SDPX.NS_SCALING_DOUBLE_SECANT
        @test workspace.conjugate.valid
        @test workspace.conjugate.accepted_valid
        @test !workspace.conjugate.inverse_valid
        @test !workspace.conjugate.accepted_inverse_valid
        @test all(isnan, workspace.conjugate.inverse_hessian)
        @test_throws ArgumentError SDPX.conjugate_inverse_hessian(
            workspace.conjugate,
        )
        @test _nss_allocated(workspace, policy, tag, primal, dual) == 0

        centered_primal, centered_dual = _nss_centered_case(T, tag)
        fallback = _nss_update!(
            workspace, SDPX.DoubleSecantWithDualHessianFallback(), tag,
            centered_primal, centered_dual,
        )
        @test fallback.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
        @test workspace.conjugate.inverse_valid
        @test workspace.conjugate.accepted_inverse_valid
        @test all(isfinite, workspace.conjugate.inverse_hessian)
    end
end

@testset "per-secant backward gate cannot be masked by a huge shadow" begin
    workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
    workspace.mu = 1.0
    workspace.primal .= (1.0, 0.0, 0.0)
    workspace.dual .= (1.0, 0.0, 0.0)
    workspace.conjugate.shadow .= (0.0, 1.0e20, 0.0)
    workspace.dual_shadow .= (0.0, 1.0e20, 0.0)
    workspace.g .= Matrix{Float64}(I, 3, 3)
    workspace.theta .= Matrix{Float64}(I, 3, 3)
    workspace.g[1, 3] = 1.0e-3
    workspace.g[3, 1] = 1.0e-3

    old_shared_scale_error = 1.0e-3 / 1.0e20
    @test old_shared_scale_error < workspace.settings.validation_tolerance
    secant_error = SDPX._ns_scaling_secant_error!(workspace, true)
    @test secant_error > 1.0e8 * workspace.settings.validation_tolerance
    reason, checked_secant, inverse_error =
        SDPX._ns_scaling_validate_metric!(workspace, true)
    @test reason === SDPX.NS_SCALING_SECANT_MISMATCH
    @test checked_secant == secant_error
    @test isfinite(inverse_error)
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

@testset "Theta-authoritative fallback near release boundaries" begin
    # Both frozen Float64 metrics were accepted by the one-secant construction
    # but rejected by the old output-relative G*s check.  The direct G secant
    # remains a poor forward diagnostic, while the Theta factor, inverse
    # columns, and propagated consequence all have strict backward proofs.
    for kind in (:exp, :power)
        workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
        @test _nss_load_release_authority!(workspace, kind) ===
              SDPX.NS_SCALING_CONVERGED
        raw_g_error = SDPX._ns_scaling_secant_equation_error!(
            workspace, workspace.g, workspace.primal, workspace.dual,
            workspace.work1,
        )
        @test raw_g_error > workspace.settings.validation_tolerance
        factor_before = copy(workspace.factor)
        reason, secant_error, inverse_error =
            SDPX._ns_scaling_validate_fallback_metric!(workspace)
        @test reason === SDPX.NS_SCALING_CONVERGED
        @test secant_error <= workspace.settings.validation_tolerance
        @test inverse_error <= workspace.settings.validation_tolerance
        @test workspace.factor == factor_before
        @test isposdef(Symmetric(workspace.theta))
        @test isposdef(Symmetric(workspace.g))
    end

    # The Power point reproduces the complete production decision: strict
    # BFGS is unresolved and the explicitly recorded fallback now succeeds.
    tag, primal, dual, mu = _nss_release_pair(Float64, :power)
    workspace = SDPX.NonsymmetricScalingWorkspace(Float64)
    policy = SDPX.DoubleSecantWithDualHessianFallback()
    result = _nss_update_global!(
        workspace, policy, tag, primal, dual, mu,
    )
    @test result.status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
    @test result.reason === SDPX.NS_SCALING_CONVERGED
    @test result.fallback_reason === SDPX.NS_SCALING_BFGS_DENOMINATOR
    @test workspace.factor * transpose(workspace.factor) ≈ workspace.theta
    @test _nss_allocated_global(
        workspace, policy, tag, primal, dual, mu,
    ) == 0

    # The authority certificate is three-way: changing Theta, G, or the
    # retained factor independently must fail closed.
    for corruption in (:theta, :g, :factor)
        corrupted = SDPX.NonsymmetricScalingWorkspace(Float64)
        @test _nss_load_release_authority!(corrupted, :power) ===
              SDPX.NS_SCALING_CONVERGED
        if corruption === :theta
            corrupted.theta[1, 1] *= 1.0 + 1.0e-10
        elseif corruption === :g
            corrupted.g[1, 1] *= 1.0 + 1.0e-10
        else
            corrupted.factor[1, 1] *= 1.0 + 1.0e-10
        end
        reason, _, _ =
            SDPX._ns_scaling_validate_fallback_metric!(corrupted)
        @test reason === SDPX.NS_SCALING_INVERSE_MISMATCH
    end
end

@testset "near-boundary fallback authority across fixed precisions" begin
    for T in (
        _NS_SCALING_MF.Float64x2,
        _NS_SCALING_MF.Float64x3,
        _NS_SCALING_MF.Float64x4,
    )
        for kind in (:exp, :power)
            tag, primal, dual, mu = _nss_release_pair(T, kind)
            workspace = SDPX.NonsymmetricScalingWorkspace(T)
            @test _nss_force_fallback!(
                workspace, tag, primal, dual, mu,
            )
            factor_before = copy(workspace.factor)
            reason, secant_error, inverse_error =
                SDPX._ns_scaling_validate_fallback_metric!(workspace)
            @test reason === SDPX.NS_SCALING_CONVERGED
            @test secant_error <= workspace.settings.validation_tolerance
            @test inverse_error <= workspace.settings.validation_tolerance
            @test workspace.factor == factor_before
            @test _nss_force_fallback_allocated!(
                workspace, tag, primal, dual, mu,
            ) == 0
        end
    end
end

@testset "Power near-boundary fallback BigFloat256/512/1024" begin
    for precision in (256, 512, 1024)
        setprecision(BigFloat, precision) do
            tag, primal, dual, mu = _nss_release_pair(BigFloat, :power)
            workspace = SDPX.NonsymmetricScalingWorkspace(BigFloat)
            @test _nss_force_fallback!(
                workspace, tag, primal, dual, mu,
            )
            factor_before = copy(workspace.factor)
            reason, secant_error, inverse_error =
                SDPX._ns_scaling_validate_fallback_metric!(workspace)
            @test reason === SDPX.NS_SCALING_CONVERGED
            @test secant_error <= workspace.settings.validation_tolerance
            @test inverse_error <= workspace.settings.validation_tolerance
            @test workspace.factor == factor_before
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
    @test SDPX._ns_conjugate_ensure_inverse_hessian!(
        fallback_denominator.conjugate,
    )
    fallback_denominator.settings = SDPX.NonsymmetricScalingSettings(
        Float64; degeneracy_tolerance=2.0,
    )
    @test SDPX._ns_scaling_dual_hessian_fallback!(fallback_denominator) ===
          SDPX.NS_SCALING_FALLBACK_DENOMINATOR

    fallback_spd = fresh_workspace()
    @test SDPX._ns_conjugate_ensure_inverse_hessian!(fallback_spd.conjugate)
    fallback_spd.conjugate.inverse_hessian[1, 1] = -1.0
    @test SDPX._ns_scaling_dual_hessian_fallback!(fallback_spd) ===
          SDPX.NS_SCALING_FALLBACK_NOT_SPD
end
