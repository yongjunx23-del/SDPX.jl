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

@noinline function _nc_gap_root_allocated!(workspace, tag, dual)
    SDPX._ns_conjugate_gap_root(
        workspace, tag, dual[1], dual[2], dual[3],
    )
    return @allocated SDPX._ns_conjugate_gap_root(
        workspace, tag, dual[1], dual[2], dual[3],
    )
end

# A test-only monotone scalar equation exercises the exact-zero-work branch
# without manufacturing an absolute scale inside either physical cone.
struct _NCZeroWorkTag <: SDPX.NonsymmetricConjugateTag end

@inline function SDPX._ns_conjugate_gap_evaluation(
    ::_NCZeroWorkTag, y1::T, ::T, ::T, ::T,
) where {T}
    return y1, one(T), zero(T), zero(T)
end

@inline SDPX._ns_conjugate_gap_derivative_lower_bound(
    ::_NCZeroWorkTag, ::Type{T},
) where {T} = one(T)

@inline function _nc_warm_target(::Type{T}, tag, dual, multiplier=one(T)) where {T}
    q = multiplier / T(100)
    delta1 = tag isa SDPX.ExpConjugateTag ? -q : q
    return (dual[1] + delta1, dual[2] + q / T(5), dual[3] + q)
end

function _nc_check_identities(::Type{T}, tag, dual, primal; tolerance) where {T}
    workspace = SDPX.NonsymmetricConjugateWorkspace(T)
    result = _nc_call!(workspace, tag, dual)
    @test result.status === SDPX.NS_CONJUGATE_SUCCESS
    @test result.reason === SDPX.NS_CONJUGATE_CONVERGED
    @test workspace.valid
    @test workspace.hessian_factor_valid
    @test workspace.accepted_hessian_factor_valid
    @test isfinite(workspace.hessian_factor_error)
    @test isfinite(workspace.accepted_hessian_factor_error)
    @test workspace.inverse_valid
    @test workspace.accepted_inverse_valid
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
    @test isbitstype(SDPX.NonsymmetricConjugateSeedMode)
    @test isbitstype(SDPX.NonsymmetricConjugateResult{Float64})
    @test isbitstype(SDPX.NonsymmetricConjugateSettings{Float64})
    @test SDPX.NS_CONJUGATE_FACTOR_FAILED isa
          SDPX.NonsymmetricConjugateReason
    @test SDPX.NS_CONJUGATE_FACTOR_MISMATCH isa
          SDPX.NonsymmetricConjugateReason
end

@inline function _nc_structural_factor_cases(::Type{T}) where {T}
    return (
        (SDPX.ExpConjugateTag(), (zero(T), T(2), T(4))),
        (
            SDPX.PowerConjugateTag(T(1) / T(2)),
            (one(T), one(T), T(3) / T(5)),
        ),
    )
end

@noinline function _nc_structural_factor_hot!(factor, tag, primal)
    SDPX._ns_structural_hessian_factor!(
        factor, tag, primal[1], primal[2], primal[3],
    ) || return false
    ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
        factor, tag, primal[1], primal[2], primal[3],
    )
    return ok
end

@noinline function _nc_structural_factor_allocated!(factor, tag, primal)
    _nc_structural_factor_hot!(factor, tag, primal)
    return @allocated _nc_structural_factor_hot!(factor, tag, primal)
end

function _nc_factorization_backward_ok(factor, hessian, tolerance)
    T = eltype(factor)
    for j in 1:3, i in 1:3
        action = zero(T)
        work = abs(hessian[i, j])
        for k in 1:min(i, j)
            term = factor[i, k] * factor[j, k]
            action += term
            work += abs(term)
        end
        residual = abs(action - hessian[i, j])
        if iszero(work)
            iszero(residual) || return false
        elseif !(isfinite(residual) && residual <= tolerance * work)
            return false
        end
    end
    return true
end

@testset "structural Hessian factors are certified and allocation free" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        tolerance = T(65_536) * eps(one(T))
        for (tag, primal) in _nc_structural_factor_cases(T)
            factor = zeros(T, 3, 3)
            @test _nc_structural_factor_hot!(factor, tag, primal)
            ok, error = SDPX._ns_structural_hessian_factor_certificate!(
                factor, tag, primal[1], primal[2], primal[3],
            )
            @test ok
            @test isfinite(error)
            @test error <= T(8) * SDPX._ns_structural_factor_gamma(T)
            hessian = _nc_primal_hessian(tag, primal)
            @test _nc_factorization_backward_ok(
                factor, hessian, tolerance,
            )

            scale = T(7) / T(3)
            scaled_primal = ntuple(i -> scale * primal[i], 3)
            scaled_factor = zeros(T, 3, 3)
            @test _nc_structural_factor_hot!(
                scaled_factor, tag, scaled_primal,
            )
            @test isapprox(
                scaled_factor .* scale, factor;
                atol=tolerance, rtol=tolerance,
            )
            @test _nc_structural_factor_allocated!(
                factor, tag, primal,
            ) == 0
        end
    end

    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            tolerance = BigFloat(65_536) * eps(BigFloat)
            for (tag, primal) in _nc_structural_factor_cases(BigFloat)
                factor = zeros(BigFloat, 3, 3)
                @test _nc_structural_factor_hot!(factor, tag, primal)
                @test all(value -> precision(value) == bits, factor)
                hessian = _nc_primal_hessian(tag, primal)
                @test _nc_factorization_backward_ok(
                    factor, hessian, tolerance,
                )
                scale = BigFloat(7) / BigFloat(3)
                scaled_primal = ntuple(i -> scale * primal[i], 3)
                scaled_factor = zeros(BigFloat, 3, 3)
                @test _nc_structural_factor_hot!(
                    scaled_factor, tag, scaled_primal,
                )
                @test isapprox(
                    scaled_factor .* scale, factor;
                    atol=tolerance, rtol=tolerance,
                )
            end
        end
    end
end

@testset "structural factor corruption fails closed" begin
    lower_entries = ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
    upper_entries = ((1, 2), (1, 3), (2, 3))
    for (tag, primal) in _nc_structural_factor_cases(Float64)
        factor = zeros(3, 3)
        @test _nc_structural_factor_hot!(factor, tag, primal)
        for (i, j) in lower_entries
            corrupted = copy(factor)
            corrupted[i, j] *= 1.01
            ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
                corrupted, tag, primal[1], primal[2], primal[3],
            )
            @test !ok
        end
        for (i, j) in upper_entries
            corrupted = copy(factor)
            corrupted[i, j] = 1.0
            ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
                corrupted, tag, primal[1], primal[2], primal[3],
            )
            @test !ok
        end
        corrupted = copy(factor)
        corrupted[2, 2] = -corrupted[2, 2]
        ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
            corrupted, tag, primal[1], primal[2], primal[3],
        )
        @test !ok
        corrupted = copy(factor)
        corrupted[3, 1] = NaN
        ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
            corrupted, tag, primal[1], primal[2], primal[3],
        )
        @test !ok

        # A solve can be internally consistent with a corrupted factor.  The
        # independent analytic identities must still reject that factor.
        corrupted = copy(factor)
        corrupted[2, 1] *= 1.01
        rhs = [1.0, -0.25, 0.5]
        solution = zeros(3)
        forward = zeros(3)
        action = zeros(3)
        @test SDPX._ns_structural_hessian_solve!(
            solution, corrupted, rhs, forward,
        )
        solve_ok, _ = SDPX._ns_structural_hessian_solve_certificate!(
            corrupted, solution, rhs, forward, action,
        )
        @test solve_ok
        factor_ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
            corrupted, tag, primal[1], primal[2], primal[3],
        )
        @test !factor_ok

        scaled_primal = ntuple(i -> 2 * primal[i], 3)
        point_ok, _ = SDPX._ns_structural_hessian_factor_certificate!(
            factor, tag,
            scaled_primal[1], scaled_primal[2], scaled_primal[3],
        )
        @test !point_ok
    end
end

@testset "shadow/H candidate is independent of the optional inverse" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        tag = SDPX.ExpConjugateTag()
        gap = T(1) / T(1_000_000)
        increment = SDPX._nonsymmetric_stable_log1p(gap) +
                    gap / (one(T) - gap)
        dual = (-one(T), zero(T), exp(-one(T) + increment))
        workspace = SDPX.NonsymmetricConjugateWorkspace(T)
        fill!(workspace.inverse_hessian, T(NaN))
        candidate = SDPX._ns_conjugate_shadow_hessian_candidate!(
            workspace, tag, dual,
        )
        @test candidate.status === SDPX.NS_CONJUGATE_SUCCESS
        @test workspace.valid
        @test workspace.hessian_factor_valid
        @test isfinite(workspace.hessian_factor_error)
        @test !workspace.inverse_valid
        @test !workspace.accepted_valid
        @test all(isnan, workspace.inverse_hessian)
        @test_throws ArgumentError SDPX.conjugate_inverse_hessian(workspace)
        # The rounded dense Hessian is diagnostic only.  Optional inverse
        # construction must remain tied to the separately certified L.
        fill!(workspace.hessian, T(NaN))
        @test SDPX._ns_conjugate_ensure_inverse_hessian!(workspace)
        @test workspace.inverse_valid
        @test all(isfinite, workspace.inverse_hessian)
    end
end

@inline function _nc_gap_fixture(
    ::Type{T}, ::SDPX.ExpConjugateTag, gap::T,
) where {T}
    increment = SDPX._nonsymmetric_stable_log1p(gap) +
                gap / (one(T) - gap)
    return (-one(T), zero(T), exp(-one(T) + increment))
end

@inline function _nc_gap_fixture(
    ::Type{T}, tag::SDPX.PowerConjugateTag{T}, gap::T,
) where {T}
    a = tag.alpha
    b = one(T) - a
    two = one(T) + one(T)
    increment = a * SDPX._nonsymmetric_stable_log1p(
        b * gap / (two * a),
    ) + b * SDPX._nonsymmetric_stable_log1p(
        a * gap / (two * b),
    ) - SDPX._nonsymmetric_stable_log1p(-gap) / two
    return (a, b, exp(-increment))
end

@testset "certified 1D near-boundary gap provider" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        gap = T(1) / T(1_000_000)
        tags = (
            SDPX.ExpConjugateTag(),
            SDPX.PowerConjugateTag(T(1) / T(10)),
            SDPX.PowerConjugateTag(T(1) / T(2)),
            SDPX.PowerConjugateTag(T(9) / T(10)),
        )
        for tag in tags
            dual = _nc_gap_fixture(T, tag, gap)
            workspace = SDPX.NonsymmetricConjugateWorkspace(T)
            result = _nc_call!(workspace, tag, dual)
            @test result.status === SDPX.NS_CONJUGATE_SUCCESS
            @test workspace.valid
            @test zero(T) < workspace.gap <= one(T)
            root_tolerance = T(4096) * eps(one(T)) / gap
            @test isapprox(
                workspace.gap, gap;
                rtol=root_tolerance, atol=T(4096) * eps(one(T)),
            )
            @test isapprox(
                dot(workspace.shadow, collect(dual)), T(3);
                rtol=T(4096) * sqrt(eps(one(T))),
                atol=T(4096) * sqrt(eps(one(T))),
            )
            for column in 1:3
                workspace.residual .= zero(T)
                workspace.residual[column] = one(T)
                workspace.direction .= workspace.inverse_hessian[:, column]
                @test SDPX._ns_conjugate_linear_solve_gate(
                    workspace.hessian,
                    workspace.direction,
                    workspace.residual,
                )
            end
            @test _nc_allocated(workspace, tag, dual) == 0
        end
    end

    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            gap = BigFloat("1e-20")
            for tag in (
                SDPX.ExpConjugateTag(),
                SDPX.PowerConjugateTag(BigFloat("0.1")),
                SDPX.PowerConjugateTag(BigFloat("0.5")),
                SDPX.PowerConjugateTag(BigFloat("0.9")),
            )
                dual = _nc_gap_fixture(BigFloat, tag, gap)
                workspace = SDPX.NonsymmetricConjugateWorkspace(BigFloat)
                result = _nc_call!(workspace, tag, dual)
                @test result.status === SDPX.NS_CONJUGATE_SUCCESS
                @test precision(workspace.gap) == bits
                @test isapprox(
                    workspace.gap, gap;
                    rtol=BigFloat(4096) * eps(BigFloat) / gap,
                    atol=BigFloat(4096) * eps(BigFloat),
                )
            end
        end
    end
end

@testset "local derivative certificate resolves an exactly represented root" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        tag = SDPX.PowerConjugateTag(T(1) / T(2))
        for gap in (
            T(2.3841858296463164e-7),
            T(1.1920930034938002e-7),
        )
            dual = _nc_gap_fixture(T, tag, gap)
            workspace = SDPX.NonsymmetricConjugateWorkspace(T)
            workspace.accepted_gap = gap
            workspace.accepted_valid = true
            ok, root, iterations, bisections, residual =
                SDPX._ns_conjugate_gap_root(
                    workspace, tag, dual[1], dual[2], dual[3],
                )
            @test ok
            @test iterations == 1
            @test bisections == 0
            @test root == gap
            @test isfinite(residual)
            @test !workspace.root_resolution_limited
        end
    end
    setprecision(BigFloat, 256) do
        tag = SDPX.PowerConjugateTag(BigFloat("0.5"))
        gap = BigFloat("2.3841858296463164e-7")
        dual = _nc_gap_fixture(BigFloat, tag, gap)
        workspace = SDPX.NonsymmetricConjugateWorkspace(BigFloat)
        workspace.accepted_gap = gap
        workspace.accepted_valid = true
        ok, root, iterations, bisections, residual =
            SDPX._ns_conjugate_gap_root(
                workspace, tag, dual[1], dual[2], dual[3],
            )
        @test ok
        @test iterations == 1
        @test bisections == 0
        @test root == gap
        @test isfinite(residual)
    end
end

@testset "scale-free root, identity, and linear-solve certificates" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        tiny = eps(one(T)) * eps(one(T))
        @test SDPX._ns_conjugate_identity_gate(zero(T), zero(T))
        @test !SDPX._ns_conjugate_identity_gate(tiny, zero(T))
        @test !SDPX._ns_conjugate_identity_gate(tiny, tiny)
        @test !SDPX._ns_conjugate_identity_gate(T(Inf), one(T))

        hessian = zeros(T, 3, 3)
        solution = zeros(T, 3)
        rhs = zeros(T, 3)
        @test SDPX._ns_conjugate_linear_solve_gate(
            hessian, solution, rhs,
        )
        rhs[1] = tiny
        @test !SDPX._ns_conjugate_linear_solve_gate(
            hessian, solution, rhs,
        )
        hessian[1, 1] = tiny
        solution[1] = one(T)
        @test SDPX._ns_conjugate_linear_solve_gate(
            hessian, solution, rhs,
        )
        hessian[2, 2] = T(Inf)
        @test !SDPX._ns_conjugate_linear_solve_gate(
            hessian, solution, rhs,
        )

        workspace = SDPX.NonsymmetricConjugateWorkspace(T)
        exact, root, iterations, bisections, residual =
            SDPX._ns_conjugate_gap_root(
                workspace, _NCZeroWorkTag(),
                zero(T), zero(T), zero(T),
            )
        @test exact
        @test root == inv(T(2))
        @test iterations == 1
        @test bisections == 0
        @test iszero(residual)
        inconsistent = SDPX._ns_conjugate_gap_root(
            workspace, _NCZeroWorkTag(), tiny, zero(T), zero(T),
        )
        @test !inconsistent[1]
    end

    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            tiny = eps(BigFloat)^2
            @test SDPX._ns_conjugate_identity_gate(
                zero(BigFloat), zero(BigFloat),
            )
            @test !SDPX._ns_conjugate_identity_gate(tiny, tiny)
            workspace = SDPX.NonsymmetricConjugateWorkspace(BigFloat)
            exact = SDPX._ns_conjugate_gap_root(
                workspace, _NCZeroWorkTag(),
                zero(BigFloat), zero(BigFloat), zero(BigFloat),
            )
            @test exact[1]
            @test precision(exact[2]) == bits
        end
    end
end

@testset "gap Phi work and roots are homogeneous" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        gap = inv(T(1_000_000))
        for tag in (
            SDPX.ExpConjugateTag(),
            SDPX.PowerConjugateTag(T(1) / T(10)),
            SDPX.PowerConjugateTag(T(1) / T(2)),
            SDPX.PowerConjugateTag(T(9) / T(10)),
        )
            dual = _nc_gap_fixture(T, tag, gap)
            scale = T(8)
            scaled_dual = ntuple(i -> scale * dual[i], 3)
            phi, derivative, work = SDPX._ns_conjugate_gap_data(
                tag, dual[1], dual[2], dual[3], gap,
            )
            _, _, _, roundoff = SDPX._ns_conjugate_gap_evaluation(
                tag, dual[1], dual[2], dual[3], gap,
            )
            scaled_phi, scaled_derivative, scaled_work =
                SDPX._ns_conjugate_gap_data(
                    tag, scaled_dual[1], scaled_dual[2],
                    scaled_dual[3], gap,
                )
            _, _, _, scaled_roundoff =
                SDPX._ns_conjugate_gap_evaluation(
                    tag, scaled_dual[1], scaled_dual[2],
                    scaled_dual[3], gap,
                )
            @test phi == scaled_phi
            @test derivative == scaled_derivative
            @test work == scaled_work
            @test roundoff == scaled_roundoff
            @test isfinite(work) && work > zero(T)
            @test isfinite(roundoff) && roundoff >= zero(T)

            workspace = SDPX.NonsymmetricConjugateWorkspace(T)
            workspace.accepted_gap = gap
            workspace.accepted_valid = true
            original = SDPX._ns_conjugate_gap_root(
                workspace, tag, dual[1], dual[2], dual[3],
            )
            scaled = SDPX._ns_conjugate_gap_root(
                workspace, tag, scaled_dual[1], scaled_dual[2],
                scaled_dual[3],
            )
            @test original[1] && scaled[1]
            @test original[2] == scaled[2] == gap
            @test _nc_gap_root_allocated!(workspace, tag, dual) == 0
        end
    end
end

@testset "Power release-gap roundoff is arithmetic, not an absolute floor" begin
    tag = SDPX.PowerConjugateTag(0.5)
    for gap in (2.3841858296463164e-7, 1.1920930034938002e-7)
        dual = _nc_gap_fixture(Float64, tag, gap)
        phi, derivative, work, roundoff =
            SDPX._ns_conjugate_gap_evaluation(tag, dual..., gap)
        relative_gap_cap = 16sqrt(eps(Float64)) * gap
        @test isfinite(phi)
        @test derivative > 1.0
        # The two near-unity log ratios contribute their real cancellation
        # operands (work≈2); their two arithmetic operations use gamma_8,
        # rather than being multiplied a second time by gamma_64.
        @test 1.9 < work < 2.1
        @test 0.0 <= roundoff < relative_gap_cap
    end

    unresolvable_gap = 8eps(Float64)
    dual = _nc_gap_fixture(Float64, tag, unresolvable_gap)
    _, _, work, roundoff = SDPX._ns_conjugate_gap_evaluation(
        tag, dual..., unresolvable_gap,
    )
    @test 1.9 < work < 2.1
    @test roundoff > 16sqrt(eps(Float64)) * unresolvable_gap
end

@testset "interval derivative floors are valid and tighten global bounds" begin
    T = Float64
    lower = T(1) / T(10)
    upper = T(1) / T(5)
    for tag in (
        SDPX.ExpConjugateTag(),
        SDPX.PowerConjugateTag(T(1) / T(10)),
        SDPX.PowerConjugateTag(T(1) / T(2)),
        SDPX.PowerConjugateTag(T(9) / T(10)),
    )
        local_floor =
            SDPX._ns_conjugate_gap_interval_derivative_lower_bound(
                tag, lower, upper,
            )
        global_floor = SDPX._ns_conjugate_gap_derivative_lower_bound(
            tag, T,
        )
        @test isfinite(local_floor)
        @test local_floor >= global_floor
        dual = _nc_gap_fixture(T, tag, (lower + upper) / T(2))
        previous = zero(T)
        for index in 0:16
            point = lower + (upper - lower) * T(index) / T(16)
            _, derivative, _ = SDPX._ns_conjugate_gap_data(
                tag, dual[1], dual[2], dual[3], point,
            )
            @test derivative >= local_floor
            if tag isa SDPX.ExpConjugateTag
                @test index == 0 || derivative >= previous
            end
            previous = derivative
        end
    end
end


@testset "predicted warm seed, cold comparison, and transactional restore" begin
    for T in (
        Float64,
        _NS_CONJUGATE_MF.Float64x2,
        _NS_CONJUGATE_MF.Float64x3,
        _NS_CONJUGATE_MF.Float64x4,
    )
        @testset "$T" begin
            for (tag, dual, _) in _nc_cases(T)
                warm_workspace = SDPX.NonsymmetricConjugateWorkspace(T)
                base = _nc_call!(warm_workspace, tag, dual)
                @test base.status === SDPX.NS_CONJUGATE_SUCCESS
                @test base.seed_mode === SDPX.NS_CONJUGATE_MAPPED_COLD_SEED

                target = _nc_warm_target(T, tag, dual)
                warm = _nc_call!(warm_workspace, tag, target)
                cold_workspace = SDPX.NonsymmetricConjugateWorkspace(T)
                cold = _nc_call!(cold_workspace, tag, target)
                @test warm.status === SDPX.NS_CONJUGATE_SUCCESS
                @test cold.status === SDPX.NS_CONJUGATE_SUCCESS
                @test warm.seed_mode ===
                      SDPX.NS_CONJUGATE_PREDICTED_WARM_SEED
                @test cold.seed_mode === SDPX.NS_CONJUGATE_MAPPED_COLD_SEED
                @test warm.iterations <= cold.iterations
                @test warm.backtracks <= cold.backtracks

                accepted_dual = copy(warm_workspace.accepted_dual)
                accepted_shadow = copy(warm_workspace.accepted_shadow)
                accepted_inverse = copy(
                    warm_workspace.accepted_inverse_hessian,
                )
                accepted_hessian = copy(warm_workspace.accepted_hessian)
                accepted_hessian_factor = copy(
                    warm_workspace.accepted_hessian_factor,
                )
                accepted_hessian_factor_error =
                    warm_workspace.accepted_hessian_factor_error
                normal_settings = warm_workspace.settings
                warm_workspace.settings = SDPX.NonsymmetricConjugateSettings(
                    T; max_iterations=0,
                )
                fill!(warm_workspace.hessian, T(NaN))
                fill!(warm_workspace.hessian_factor, T(NaN))
                warm_workspace.hessian_factor_error = T(Inf)
                warm_workspace.hessian_factor_valid = false
                rejected_target = _nc_warm_target(T, tag, dual, T(10))
                rejected = _nc_call!(warm_workspace, tag, rejected_target)
                @test rejected.status === SDPX.NS_CONJUGATE_FAILED
                @test rejected.reason === SDPX.NS_CONJUGATE_ITERATION_LIMIT
                @test rejected.restored
                @test warm_workspace.valid
                @test warm_workspace.accepted_dual == accepted_dual
                @test warm_workspace.shadow == accepted_shadow
                @test warm_workspace.hessian == accepted_hessian
                @test warm_workspace.hessian_factor ==
                      accepted_hessian_factor
                @test warm_workspace.hessian_factor_error ==
                      accepted_hessian_factor_error
                @test warm_workspace.hessian_factor_valid
                @test warm_workspace.accepted_hessian_factor_valid
                @test warm_workspace.inverse_hessian == accepted_inverse
                @test warm_workspace.inverse_valid
                @test warm_workspace.accepted_inverse_valid

                warm_workspace.settings = normal_settings
                recovered = _nc_call!(warm_workspace, tag, target)
                @test recovered.status === SDPX.NS_CONJUGATE_SUCCESS
                @test recovered.seed_mode ===
                      SDPX.NS_CONJUGATE_PREDICTED_WARM_SEED
                @test _nc_allocated(warm_workspace, tag, target) == 0
                @test warm_workspace.last_seed_mode ===
                      SDPX.NS_CONJUGATE_PREDICTED_WARM_SEED
            end
        end
    end
end

@testset "BigFloat256/512/1024 warm seed residual and restore" begin
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            for (tag, dual, _) in _nc_cases(BigFloat)
                workspace = SDPX.NonsymmetricConjugateWorkspace(BigFloat)
                base = _nc_call!(workspace, tag, dual)
                @test base.status === SDPX.NS_CONJUGATE_SUCCESS
                target = _nc_warm_target(BigFloat, tag, dual)
                warm = _nc_call!(workspace, tag, target)
                @test warm.status === SDPX.NS_CONJUGATE_SUCCESS
                @test warm.seed_mode ===
                      SDPX.NS_CONJUGATE_PREDICTED_WARM_SEED
                @test isfinite(warm.residual)
                residual_scale = max(
                    one(BigFloat), abs(target[1]), abs(target[2]),
                    abs(target[3]), abs(workspace.gradient[1]),
                    abs(workspace.gradient[2]), abs(workspace.gradient[3]),
                )
                @test warm.residual <=
                      workspace.settings.residual_tolerance * residual_scale
                @test all(value -> precision(value) == bits, workspace.shadow)
                @test all(
                    value -> precision(value) == bits,
                    workspace.inverse_hessian,
                )

                accepted_dual = copy(workspace.accepted_dual)
                accepted_shadow = copy(workspace.accepted_shadow)
                accepted_inverse = copy(workspace.accepted_inverse_hessian)
                normal_settings = workspace.settings
                workspace.settings = SDPX.NonsymmetricConjugateSettings(
                    BigFloat; max_iterations=0,
                )
                rejected = _nc_call!(
                    workspace,
                    tag,
                    _nc_warm_target(BigFloat, tag, dual, BigFloat(10)),
                )
                @test rejected.status === SDPX.NS_CONJUGATE_FAILED
                @test rejected.reason === SDPX.NS_CONJUGATE_ITERATION_LIMIT
                @test rejected.restored
                @test workspace.valid
                @test workspace.accepted_dual == accepted_dual
                @test workspace.shadow == accepted_shadow
                @test workspace.inverse_hessian == accepted_inverse
                workspace.settings = normal_settings
            end
        end
    end
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

    resolution = SDPX.NonsymmetricConjugateWorkspace(Float64)
    resolution_gap = 8eps(Float64)
    resolution_tag = SDPX.PowerConjugateTag(0.5)
    resolution_dual = _nc_gap_fixture(
        Float64, resolution_tag, resolution_gap,
    )
    resolution.accepted_gap = resolution_gap
    resolution.accepted_valid = true
    resolution_limited = SDPX._ns_conjugate_shadow_hessian_candidate!(
        resolution, resolution_tag, resolution_dual,
    )
    @test resolution_limited.status === SDPX.NS_CONJUGATE_FAILED
    @test resolution_limited.reason ===
          SDPX.NS_CONJUGATE_ROOT_RESOLUTION_LIMIT
    @test resolution.root_resolution_limited
end
