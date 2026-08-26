# Mixed symmetric/nonsymmetric setup-built product runtime.

using LinearAlgebra
using SDPX
using Test

if !isdefined(SDPX, :_NONSYMMETRIC_RUNTIME_API_LOADED)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "runtime",
            "nonsymmetric_api.jl",
        ),
    )
end

const _NS_PRODUCT_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

function _nspr_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for spec in specs
        kind = spec[1]
        dimension = spec[2]
        parameter = length(spec) == 3 ? spec[3] : zero(T)
        block = SDPX.ConeBlockDescriptor(
            T, kind, dimension; offset=offset, parameter=parameter,
        )
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _nspr_pure_specs(::Type{T}) where {T}
    return (
        ((:exp, 3, zero(T)),),
        ((:power, 3, T(1) / T(10)),),
        ((:power, 3, T(1) / T(2)),),
        ((:power, 3, T(9) / T(10)),),
    )
end

function _nspr_mixed_specs(::Type{T}) where {T}
    return (
        (:nonnegative, 2, zero(T)),
        (:soc, 3, zero(T)),
        (:psd, 2, zero(T)),
        (:exp, 3, zero(T)),
        (:power, 3, T(1) / T(10)),
    )
end

@inline function _nspr_tolerance(::Type{T}) where {T}
    return T(1_048_576) * eps(one(T))
end

function _nspr_nonsymmetric_blocks(runtime)
    return (runtime.exp..., runtime.power...)
end

function _nspr_set_directions!(ds, dy, runtime)
    fill!(ds, zero(eltype(ds)))
    fill!(dy, zero(eltype(dy)))
    T = eltype(ds)
    for block in runtime.exp
        offset = block.offset
        ds[offset] = T(1) / T(10)
        ds[offset + 1] = -T(1) / T(5)
        ds[offset + 2] = T(1) / T(20)
        dy[offset] = -T(3) / T(10)
        dy[offset + 1] = T(3) / T(20)
        dy[offset + 2] = T(1) / T(5)
    end
    for block in runtime.power
        offset = block.offset
        ds[offset] = T(1) / T(10)
        ds[offset + 1] = -T(1) / T(5)
        ds[offset + 2] = T(1) / T(20)
        dy[offset] = -T(3) / T(10)
        dy[offset + 1] = T(3) / T(20)
        dy[offset + 2] = T(1) / T(5)
    end
    return ds, dy
end

@noinline function _nspr_hot!(
    runtime,
    s,
    y,
    src,
    dst,
    ds,
    dy,
    h,
    chi,
    mu,
    target,
)
    SDPX.checkpoint_nonsymmetric_scaling!(runtime) || return false
    SDPX.try_update_scaling!(runtime, s, y, mu) || return false
    SDPX.apply_G!(runtime, dst, src)
    SDPX.apply_Theta!(runtime, dst, src)
    SDPX.product_strictly_interior(runtime, s, y) || return false
    ap = SDPX.max_step_primal!(runtime, s, ds)
    ad = SDPX.max_step_dual!(runtime, y, dy)
    ap >= zero(mu) && ad >= zero(mu) || return false
    SDPX.affine_shift!(runtime, h, s, y)
    SDPX.corrector_shift!(runtime, h, s, y, ds, dy, target)
    result = SDPX.try_nonsymmetric_runtime_higher_correction!(
        runtime, chi, ds, dy,
    )
    result.status === SDPX.NS_RUNTIME_READY || return false
    return SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
end

@noinline function _nspr_hot_allocated!(args...)
    _nspr_hot!(args...)
    return @allocated _nspr_hot!(args...)
end

function _nspr_check_runtime(::Type{T}, specs) where {T}
    layout = _nspr_layout(T, specs)
    runtime = SDPX.ProductConeRuntime(layout, T)
    s = zeros(T, layout.dimension)
    y = zeros(T, layout.dimension)
    SDPX.initialize_primal_dual!(runtime, s, y)
    tolerance = _nspr_tolerance(T)
    @test runtime.valid
    @test SDPX.product_strictly_interior(runtime, s, y)
    @test runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_READY

    for block in runtime.exp
        @test block.offset >= 1
        @test block.dim == 3
        @test block.tag isa SDPX.ExpConjugateTag
        @test block.policy isa SDPX.DoubleSecantWithDualHessianFallback
        @test block.initialization.scaling === block.scaling
        @test block.primal === block.scaling.primal
        @test block.dual === block.scaling.dual
        @test block.last_scaling_status === SDPX.NS_SCALING_DOUBLE_SECANT
        @test block.last_fallback_reason === SDPX.NS_SCALING_NO_FALLBACK
        @test !block.scaling.conjugate.inverse_valid
        @test !block.scaling.conjugate.accepted_inverse_valid
        @test block.scaling.conjugate.hessian_factor_valid
        @test block.scaling.conjugate.accepted_hessian_factor_valid
    end
    for block in runtime.power
        @test block.offset >= 1
        @test block.dim == 3
        @test block.tag isa SDPX.PowerConjugateTag{T}
        @test block.policy isa SDPX.DoubleSecantWithDualHessianFallback
        @test block.initialization.scaling === block.scaling
        @test block.primal === block.scaling.primal
        @test block.dual === block.scaling.dual
        @test block.last_scaling_status === SDPX.NS_SCALING_DOUBLE_SECANT
        @test block.last_fallback_reason === SDPX.NS_SCALING_NO_FALLBACK
        @test !block.scaling.conjugate.inverse_valid
        @test !block.scaling.conjugate.accepted_inverse_valid
        @test block.scaling.conjugate.hessian_factor_valid
        @test block.scaling.conjugate.accepted_hessian_factor_valid
        @test block.tag.alpha isa T
    end

    g_s = similar(s)
    theta_y = similar(s)
    SDPX.apply_G!(runtime, g_s, s)
    SDPX.apply_Theta!(runtime, theta_y, y)
    @test isapprox(g_s, y; atol=tolerance, rtol=tolerance)
    @test isapprox(theta_y, s; atol=tolerance, rtol=tolerance)

    src = similar(s)
    for index in eachindex(src)
        src[index] = T(index) / T(length(src) + 1)
    end
    g_src = similar(src)
    inverse_src = similar(src)
    SDPX.apply_G!(runtime, g_src, src)
    SDPX.apply_Theta!(runtime, inverse_src, g_src)
    @test isapprox(inverse_src, src; atol=tolerance, rtol=tolerance)

    for block in _nspr_nonsymmetric_blocks(runtime)
        scaling = block.scaling
        @test isapprox(
            scaling.g * scaling.conjugate.shadow,
            scaling.dual_shadow;
            atol=tolerance,
            rtol=tolerance,
        )
        @test isapprox(
            scaling.theta * scaling.dual_shadow,
            scaling.conjugate.shadow;
            atol=tolerance,
            rtol=tolerance,
        )
    end

    h = similar(s)
    SDPX.affine_shift!(runtime, h, s, y)
    for block in _nspr_nonsymmetric_blocks(runtime)
        for local_index in 1:3
            global_index = block.offset + local_index - 1
            @test h[global_index] == -s[global_index]
        end
    end

    ds = zeros(T, layout.dimension)
    dy = zeros(T, layout.dimension)
    _nspr_set_directions!(ds, dy, runtime)
    target = T(2) / T(5)
    SDPX.corrector_shift!(runtime, h, s, y, ds, dy, target)
    chi = zeros(T, layout.dimension)
    higher = SDPX.try_nonsymmetric_runtime_higher_correction!(
        runtime, chi, ds, dy,
    )
    @test higher.status === SDPX.NS_RUNTIME_READY
    for block in _nspr_nonsymmetric_blocks(runtime)
        offset = block.offset
        @test isapprox(
            block.scaling.theta * block.corrector.rho,
            block.corrector.h;
            atol=tolerance,
            rtol=tolerance,
        )
        @test isapprox(
            block.scaling.theta \ block.corrector.h,
            block.corrector.rho;
            atol=tolerance,
            rtol=tolerance,
        )
        lhs = s[offset] * chi[offset] +
              s[offset + 1] * chi[offset + 1] +
              s[offset + 2] * chi[offset + 2]
        rhs = ds[offset] * dy[offset] +
              ds[offset + 1] * dy[offset + 1] +
              ds[offset + 2] * dy[offset + 2]
        @test isapprox(lhs, rhs; atol=tolerance, rtol=tolerance)
    end

    step_primal_direction = zeros(T, layout.dimension)
    step_dual_direction = zeros(T, layout.dimension)
    for block in runtime.exp
        step_primal_direction[block.offset + 2] = -T(4)
        step_dual_direction[block.offset + 2] = -T(2)
    end
    for block in runtime.power
        step_primal_direction[block.offset + 2] = T(2)
        step_dual_direction[block.offset + 2] = T(2)
    end
    ap = SDPX.max_step_primal!(runtime, s, step_primal_direction)
    ad = SDPX.max_step_dual!(runtime, y, step_dual_direction)
    @test isfinite(ap) && zero(T) < ap < one(T)
    @test isfinite(ad) && zero(T) < ad < one(T)
    primal_trial = s + (T(99) / T(100)) * ap * step_primal_direction
    dual_trial = y + (T(99) / T(100)) * ad * step_dual_direction
    @test SDPX.product_strictly_interior(runtime, primal_trial, y)
    @test SDPX.product_strictly_interior(runtime, s, dual_trial)
    @test SDPX.max_step_primal!(
        runtime, s, zeros(T, layout.dimension),
    ) == T(Inf)
    @test SDPX.max_step_dual!(
        runtime, y, zeros(T, layout.dimension),
    ) == T(Inf)

    return runtime, s, y, src, g_src, ds, dy, h, chi, target
end

@testset "mixed runtime typed layout and fixed-width hot paths" begin
    for T in (
        Float64,
        _NS_PRODUCT_MF.Float64x2,
        _NS_PRODUCT_MF.Float64x3,
        _NS_PRODUCT_MF.Float64x4,
    )
        @testset "$T" begin
            fixtures = (_nspr_pure_specs(T)..., _nspr_mixed_specs(T))
            for specs in fixtures
                runtime, s, y, src, dst, ds, dy, h, chi, target =
                    _nspr_check_runtime(T, specs)
                @test _nspr_hot_allocated!(
                    runtime,
                    s,
                    y,
                    src,
                    dst,
                    ds,
                    dy,
                    h,
                    chi,
                    one(T),
                    target,
                ) == 0
                @test all(
                    block -> !block.scaling.conjugate.inverse_valid &&
                             !block.scaling.conjugate.accepted_inverse_valid,
                    _nspr_nonsymmetric_blocks(runtime),
                )
                @test isbits(runtime.last_nonsymmetric)
            end
        end
    end
end

@testset "product runtime propagates one explicit global mu" begin
    for T in (
        Float64,
        _NS_PRODUCT_MF.Float64x2,
        _NS_PRODUCT_MF.Float64x3,
        _NS_PRODUCT_MF.Float64x4,
    )
        layout = _nspr_layout(
            T, ((:exp, 3, zero(T)), (:power, 3, T(1) / T(2))),
        )
        runtime = SDPX.ProductConeRuntime(layout, T)
        s = zeros(T, layout.dimension)
        y = zeros(T, layout.dimension)
        SDPX.initialize_primal_dual!(runtime, s, y)
        global_mu = T(7) / T(19)
        @test SDPX.try_update_scaling!(runtime, s, y, global_mu)
        @test runtime.last_mu == global_mu
        @test runtime.last_nonsymmetric.value == global_mu
        for block in _nspr_nonsymmetric_blocks(runtime)
            local_mu = dot(block.primal, block.dual) / T(3)
            @test block.scaling.mu == global_mu
            @test block.scaling.mu != local_mu
        end
        @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        @test runtime.checkpoint_mu == global_mu
        @test all(
            block -> block.checkpoint.mu == global_mu,
            _nspr_nonsymmetric_blocks(runtime),
        )

        @test !SDPX.try_update_scaling!(runtime, s, y, zero(T))
        @test !runtime.valid
        @test runtime.last_nonsymmetric.reason ===
              SDPX.NS_RUNTIME_INVALID_PARAMETER
        @test SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
        @test runtime.last_mu == global_mu
        @test all(
            block -> block.scaling.mu == global_mu,
            _nspr_nonsymmetric_blocks(runtime),
        )
    end
end

@testset "BigFloat256/512/1024 pure and mixed runtime" begin
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            fixtures = (
                _nspr_pure_specs(BigFloat)...,
                _nspr_mixed_specs(BigFloat),
            )
            for specs in fixtures
                runtime, s, y = _nspr_check_runtime(BigFloat, specs)
                @test all(value -> precision(value) == bits, s)
                @test all(value -> precision(value) == bits, y)
                for block in runtime.power
                    @test precision(block.tag.alpha) == bits
                    @test precision(block.line_search.primal_tag.alpha) == bits
                    @test precision(block.line_search.dual_tag.alpha) == bits
                end
            end
        end
    end
end

@testset "symmetric-only families retain their runtime behavior" begin
    T = Float64
    layout = _nspr_layout(T, _nspr_mixed_specs(T)[1:3])
    runtime = SDPX.ProductConeRuntime(layout, T)
    s = zeros(T, layout.dimension)
    y = zeros(T, layout.dimension)
    SDPX.initialize_primal_dual!(runtime, s, y)
    @test isempty(runtime.exp)
    @test isempty(runtime.power)
    @test runtime.valid
    @test SDPX.product_strictly_interior(runtime, s, y)
    output = similar(s)
    SDPX.apply_G!(runtime, output, s)
    @test output == y
    SDPX.apply_Theta!(runtime, output, y)
    @test output == s
end

@testset "product G action is authoritative Theta solve for every container" begin
    T = Float64
    layout = _nspr_layout(T, ((:exp, 3, zero(T)),))
    runtime = SDPX.ProductConeRuntime(layout, T)
    s = zeros(T, 3)
    y = zeros(T, 3)
    SDPX.initialize_primal_dual!(runtime, s, y)
    block = only(runtime.exp)
    src = T[1 / 7, -2 / 9, 3 / 11]
    expected = block.scaling.theta \ src
    ds = zeros(T, 3)
    dy = zeros(T, 3)
    _nspr_set_directions!(ds, dy, runtime)
    baseline_h = zeros(T, 3)
    SDPX.corrector_shift!(runtime, baseline_h, s, y, ds, dy, T(2) / T(5))

    # Corrupt only the diagnostic explicit inverse. Product execution must
    # remain tied to accepted Theta, while the standalone diagnostic API
    # deliberately retains its historical stored-G semantics.
    fill!(block.scaling.g, zero(T))
    block.scaling.g[1, 1] = T(17)
    block.scaling.g[2, 2] = T(19)
    block.scaling.g[3, 3] = T(23)
    diagnostic = zeros(T, 3)
    SDPX.apply_nonsymmetric_G!(diagnostic, block.scaling, src)
    @test diagnostic == block.scaling.g * src

    authoritative = zeros(T, 3)
    SDPX.apply_G!(runtime, authoritative, src)
    @test authoritative ≈ expected rtol=1e-13 atol=1e-13
    @test block.scaling.theta * authoritative ≈ src rtol=1e-13 atol=1e-13
    @test authoritative != diagnostic

    # The product-specific higher-order shift must likewise avoid the generic
    # stored-G composed-map gate.
    authoritative_h = zeros(T, 3)
    SDPX.corrector_shift!(
        runtime, authoritative_h, s, y, ds, dy, T(2) / T(5),
    )
    @test authoritative_h == baseline_h
    solved_rho = block.scaling.theta \ authoritative_h
    @test block.scaling.theta * solved_rho ≈ authoritative_h rtol=1e-13 atol=1e-13

    padded_src = zeros(T, 5)
    padded_dst = zeros(T, 5)
    padded_src[2:4] .= src
    src_view = @view padded_src[2:4]
    dst_view = @view padded_dst[2:4]
    SDPX.apply_G!(runtime, dst_view, src_view)
    @test collect(dst_view) ≈ expected rtol=1e-13 atol=1e-13

    @test SDPX.try_apply_nonsymmetric_G_reason!(
        block.output, block.scaling, T[Inf, 0, 0],
    ) === SDPX.NS_SCALING_NONFINITE_INPUT
    saved_theta = copy(block.scaling.theta)
    block.scaling.theta[1, 1] = -one(T)
    @test SDPX.try_apply_nonsymmetric_G_reason!(
        block.output, block.scaling, src,
    ) === SDPX.NS_SCALING_METRIC_NOT_SPD
    copyto!(block.scaling.theta, saved_theta)
    @test SDPX.try_apply_nonsymmetric_G_reason!(
        block.output, block.scaling, src,
    ) === SDPX.NS_SCALING_CONVERGED

    @test_throws DomainError SDPX.apply_G!(runtime, authoritative, T[Inf, 0, 0])
    @test !runtime.valid
    @test runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_FAILED
    @test runtime.last_nonsymmetric.reason === SDPX.NS_RUNTIME_NONFINITE_INPUT
end

function _nspr_exp_power_runtime(::Type{T}) where {T}
    layout = _nspr_layout(
        T, ((:exp, 3, zero(T)), (:power, 3, T(1) / T(2))),
    )
    runtime = SDPX.ProductConeRuntime(layout, T)
    s = zeros(T, layout.dimension)
    y = zeros(T, layout.dimension)
    SDPX.initialize_primal_dual!(runtime, s, y)
    return runtime
end

function _nspr_live_metric_snapshot(runtime)
    return map(_nspr_nonsymmetric_blocks(runtime)) do block
        scaling = block.scaling
        conjugate = scaling.conjugate
        (
            primal=copy(scaling.primal),
            dual=copy(scaling.dual),
            dual_shadow=copy(scaling.dual_shadow),
            g=copy(scaling.g),
            theta=copy(scaling.theta),
            scaling_factor=copy(scaling.factor),
            shadow=copy(conjugate.shadow),
            hessian=copy(conjugate.hessian),
            hessian_factor=copy(conjugate.hessian_factor),
            accepted_hessian_factor=copy(
                conjugate.accepted_hessian_factor,
            ),
            factor_error=conjugate.hessian_factor_error,
            accepted_factor_error=conjugate.accepted_hessian_factor_error,
            factor_valid=conjugate.hessian_factor_valid,
            accepted_factor_valid=conjugate.accepted_hessian_factor_valid,
            inverse_valid=conjugate.inverse_valid,
            accepted_inverse_valid=conjugate.accepted_inverse_valid,
        )
    end
end

function _nspr_checkpoint_metric_snapshot(runtime)
    return map(_nspr_nonsymmetric_blocks(runtime)) do block
        checkpoint = block.checkpoint
        (
            valid=checkpoint.valid,
            primal=copy(checkpoint.primal),
            dual=copy(checkpoint.dual),
            dual_shadow=copy(checkpoint.dual_shadow),
            g=copy(checkpoint.g),
            theta=copy(checkpoint.theta),
            scaling_factor=copy(checkpoint.scaling_factor),
            shadow=copy(checkpoint.conjugate_shadow),
            hessian=copy(checkpoint.conjugate_hessian),
            hessian_factor=copy(checkpoint.conjugate_hessian_factor),
            accepted_hessian_factor=copy(
                checkpoint.accepted_hessian_factor,
            ),
            factor_error=checkpoint.conjugate_hessian_factor_error,
            accepted_factor_error=checkpoint.accepted_hessian_factor_error,
            factor_valid=checkpoint.conjugate_hessian_factor_valid,
            accepted_factor_valid=
                checkpoint.accepted_hessian_factor_valid,
        )
    end
end

@noinline function _nspr_checkpoint_restore!(runtime)
    SDPX.checkpoint_nonsymmetric_scaling!(runtime) || return false
    return SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
end

@noinline function _nspr_checkpoint_restore_allocated(runtime)
    _nspr_checkpoint_restore!(runtime)
    return @allocated _nspr_checkpoint_restore!(runtime)
end

@testset "multiblock scaling checkpoints are atomic and certified" begin
    for T in (
        Float64,
        _NS_PRODUCT_MF.Float64x2,
        _NS_PRODUCT_MF.Float64x3,
        _NS_PRODUCT_MF.Float64x4,
    )
        runtime = _nspr_exp_power_runtime(T)
        @test runtime.valid
        @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        for block in _nspr_nonsymmetric_blocks(runtime)
            checkpoint = block.checkpoint
            conjugate = block.scaling.conjugate
            @test checkpoint.scaling_factor == block.scaling.factor
            @test checkpoint.conjugate_hessian_factor ==
                  conjugate.hessian_factor
            @test checkpoint.accepted_hessian_factor ==
                  conjugate.accepted_hessian_factor
            @test SDPX._runtime_ns_checkpoint_block_preflight(
                block, runtime.checkpoint_mu,
            )
        end
        @test SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
        @test runtime.valid
        @test _nspr_checkpoint_restore_allocated(runtime) == 0
    end

    runtime = _nspr_exp_power_runtime(Float64)
    @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
    accepted_live = _nspr_live_metric_snapshot(runtime)
    for block in _nspr_nonsymmetric_blocks(runtime)
        fill!(block.scaling.factor, NaN)
        fill!(block.scaling.conjugate.hessian_factor, NaN)
        fill!(block.scaling.conjugate.accepted_hessian_factor, NaN)
        block.scaling.conjugate.hessian_factor_error = Inf
        block.scaling.conjugate.accepted_hessian_factor_error = Inf
        block.scaling.conjugate.hessian_factor_valid = false
        block.scaling.conjugate.accepted_hessian_factor_valid = false
    end
    @test SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
    @test runtime.valid
    @test _nspr_live_metric_snapshot(runtime) == accepted_live

    # A later block fails live preflight after an earlier globally valid
    # checkpoint.  No block checkpoint may be partially rewritten, and the
    # failed retry must revoke the older global checkpoint.
    runtime = _nspr_exp_power_runtime(Float64)
    @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
    saved_checkpoints = _nspr_checkpoint_metric_snapshot(runtime)
    runtime.exp[1].scaling.conjugate.hessian[1, 1] += 1 / 7
    runtime.power[1].scaling.conjugate.hessian_factor[2, 1] *= 1.01
    @test !SDPX.checkpoint_nonsymmetric_scaling!(runtime)
    @test !runtime.checkpoint_valid
    @test _nspr_checkpoint_metric_snapshot(runtime) == saved_checkpoints
    live_before_failed_restore = _nspr_live_metric_snapshot(runtime)
    @test !SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
    @test !runtime.valid
    @test _nspr_live_metric_snapshot(runtime) == live_before_failed_restore

    # Finite G corruption and a strict second-secant corruption are distinct
    # from Theta/L damage and must independently revoke a checkpoint retry.
    for corrupt_live! in (
        block -> (block.scaling.g[1, 1] *= 1.01),
        block -> (block.scaling.dual_shadow[1] += 0.01),
    )
        runtime = _nspr_exp_power_runtime(Float64)
        @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        saved_checkpoints = _nspr_checkpoint_metric_snapshot(runtime)
        runtime.exp[1].scaling.conjugate.hessian[1, 1] += 1 / 13
        corrupt_live!(runtime.power[1])
        @test !SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        @test !runtime.checkpoint_valid
        @test _nspr_checkpoint_metric_snapshot(runtime) == saved_checkpoints
    end

    # The global product mu and every block mu are one exact epoch value.
    for corrupt_mu! in (
        runtime -> (runtime.last_mu = NaN),
        runtime -> (runtime.last_mu *= 2),
        runtime -> (runtime.power[1].scaling.mu *= 2),
    )
        runtime = _nspr_exp_power_runtime(Float64)
        @test all(
            block -> block.scaling.mu == runtime.last_mu,
            _nspr_nonsymmetric_blocks(runtime),
        )
        @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        saved_checkpoints = _nspr_checkpoint_metric_snapshot(runtime)
        corrupt_mu!(runtime)
        @test !SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        @test !runtime.checkpoint_valid
        @test _nspr_checkpoint_metric_snapshot(runtime) == saved_checkpoints
    end

    # Every saved corruption is placed in the later Power block.  The Exp
    # live metric carries a sentinel change which would expose any one-pass
    # partial restore before the later preflight failure.
    corruptions = (
        checkpoint ->
            (checkpoint.conjugate_hessian_factor[2, 1] *= 1.01),
        checkpoint ->
            (checkpoint.accepted_hessian_factor[3, 2] *= 1.01),
        checkpoint -> (checkpoint.scaling_factor[2, 1] *= 1.01),
        checkpoint -> (checkpoint.theta[1, 1] *= 1.01),
        checkpoint -> (checkpoint.g[1, 1] *= 1.01),
        checkpoint -> (checkpoint.dual_shadow[1] += 0.01),
        checkpoint -> (checkpoint.mu *= 2),
        checkpoint ->
            (checkpoint.accepted_hessian_factor_valid = false),
    )
    for corrupt! in corruptions
        runtime = _nspr_exp_power_runtime(Float64)
        @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
        runtime.exp[1].scaling.g[1, 1] += 1 / 11
        live_before = _nspr_live_metric_snapshot(runtime)
        corrupt!(runtime.power[1].checkpoint)
        @test !SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
        @test !runtime.valid
        @test runtime.last_nonsymmetric.block_offset == 4
        @test _nspr_live_metric_snapshot(runtime) == live_before
    end


    runtime = _nspr_exp_power_runtime(Float64)
    @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
    runtime.exp[1].scaling.conjugate.hessian[1, 1] += 1 / 17
    live_before = _nspr_live_metric_snapshot(runtime)
    runtime.checkpoint_mu = NaN
    @test !SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
    @test !runtime.valid
    @test _nspr_live_metric_snapshot(runtime) == live_before
end

@testset "descriptor and runtime failures are explicit and fail closed" begin
    float_layout = _nspr_layout(
        Float64, ((:power, 3, 0.5),),
    )
    @test_throws ArgumentError SDPX.ProductConeRuntime(float_layout, BigFloat)
    for alpha in (0.0, 1.0, NaN, Inf)
        layout = _nspr_layout(Float64, ((:power, 3, alpha),))
        @test_throws ArgumentError SDPX.ProductConeRuntime(layout, Float64)
    end

    layout = _nspr_layout(Float64, ((:exp, 3, 0.0),))
    runtime = SDPX.ProductConeRuntime(layout, Float64)
    s = zeros(3)
    y = zeros(3)
    SDPX.initialize_primal_dual!(runtime, s, y)

    boundary = copy(s)
    boundary[3] = 1.0
    @test !SDPX.try_update_scaling!(runtime, boundary, y, 1.0)
    @test !runtime.valid
    @test runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_FAILED
    @test runtime.last_nonsymmetric.reason === SDPX.NS_RUNTIME_SCALING_FAILED
    @test runtime.last_nonsymmetric.scaling_reason ===
          SDPX.NS_SCALING_PRIMAL_NOT_INTERIOR

    SDPX.initialize_primal_dual!(runtime, s, y)
    centered_gradient = SDPX.exp_barrier_gradient(s...)
    centered_dual = -collect(centered_gradient)
    @test SDPX.checkpoint_nonsymmetric_scaling!(runtime)
    checkpoint_block = only(runtime.exp)
    checkpoint_g = copy(checkpoint_block.scaling.g)
    checkpoint_theta = copy(checkpoint_block.scaling.theta)
    checkpoint_scaling_factor = copy(checkpoint_block.scaling.factor)
    checkpoint_accepted_dual = copy(
        checkpoint_block.scaling.conjugate.accepted_dual,
    )
    checkpoint_gap = checkpoint_block.scaling.conjugate.gap
    checkpoint_accepted_gap =
        checkpoint_block.scaling.conjugate.accepted_gap
    checkpoint_inverse_valid =
        checkpoint_block.scaling.conjugate.inverse_valid
    checkpoint_accepted_inverse_valid =
        checkpoint_block.scaling.conjugate.accepted_inverse_valid
    checkpoint_hessian = copy(checkpoint_block.scaling.conjugate.hessian)
    checkpoint_accepted_hessian = copy(
        checkpoint_block.scaling.conjugate.accepted_hessian,
    )
    checkpoint_hessian_factor = copy(
        checkpoint_block.scaling.conjugate.hessian_factor,
    )
    checkpoint_accepted_hessian_factor = copy(
        checkpoint_block.scaling.conjugate.accepted_hessian_factor,
    )
    checkpoint_hessian_factor_error =
        checkpoint_block.scaling.conjugate.hessian_factor_error
    checkpoint_accepted_hessian_factor_error =
        checkpoint_block.scaling.conjugate.accepted_hessian_factor_error
    checkpoint_hessian_factor_valid =
        checkpoint_block.scaling.conjugate.hessian_factor_valid
    checkpoint_accepted_hessian_factor_valid =
        checkpoint_block.scaling.conjugate.accepted_hessian_factor_valid
    @test SDPX.try_update_scaling!(runtime, s, centered_dual, 1.0)
    block = only(runtime.exp)
    @test block.last_scaling_status ===
          SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
    @test block.last_scaling_reason === SDPX.NS_SCALING_CONVERGED
    @test block.last_fallback_reason ===
          SDPX.NS_SCALING_SECOND_SECANT_DEGENERATE
    @test block.scaling.used_fallback
    @test runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_READY
    @test runtime.last_nonsymmetric.block_offset == block.offset
    @test runtime.last_nonsymmetric.scaling_status ===
          SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
    @test runtime.last_nonsymmetric.fallback_reason ===
          SDPX.NS_SCALING_SECOND_SECANT_DEGENERATE
    @test block.scaling.conjugate.accepted_dual != checkpoint_accepted_dual
    @test block.scaling.conjugate.accepted_gap != checkpoint_accepted_gap
    fill!(block.scaling.conjugate.hessian_factor, NaN)
    fill!(block.scaling.conjugate.accepted_hessian_factor, NaN)
    block.scaling.conjugate.hessian_factor_error = Inf
    block.scaling.conjugate.accepted_hessian_factor_error = Inf
    block.scaling.conjugate.hessian_factor_valid = false
    block.scaling.conjugate.accepted_hessian_factor_valid = false
    @test SDPX.restore_nonsymmetric_scaling_checkpoint!(runtime)
    @test runtime.valid
    @test block.scaling.g == checkpoint_g
    @test block.scaling.theta == checkpoint_theta
    @test block.scaling.factor == checkpoint_scaling_factor
    @test block.scaling.conjugate.accepted_dual == checkpoint_accepted_dual
    @test block.scaling.conjugate.gap == checkpoint_gap
    @test block.scaling.conjugate.accepted_gap == checkpoint_accepted_gap
    @test block.scaling.conjugate.inverse_valid == checkpoint_inverse_valid
    @test block.scaling.conjugate.accepted_inverse_valid ==
          checkpoint_accepted_inverse_valid
    @test block.scaling.conjugate.hessian == checkpoint_hessian
    @test block.scaling.conjugate.accepted_hessian ==
          checkpoint_accepted_hessian
    @test block.scaling.conjugate.hessian_factor ==
          checkpoint_hessian_factor
    @test block.scaling.conjugate.accepted_hessian_factor ==
          checkpoint_accepted_hessian_factor
    @test block.scaling.conjugate.hessian_factor_error ==
          checkpoint_hessian_factor_error
    @test block.scaling.conjugate.accepted_hessian_factor_error ==
          checkpoint_accepted_hessian_factor_error
    @test block.scaling.conjugate.hessian_factor_valid ==
          checkpoint_hessian_factor_valid
    @test block.scaling.conjugate.accepted_hessian_factor_valid ==
          checkpoint_accepted_hessian_factor_valid
    @test block.scaling.primal == s
    @test block.scaling.dual == y

    SDPX.initialize_primal_dual!(runtime, s, y)
    nonfinite = copy(y)
    nonfinite[1] = NaN
    typed_nonfinite = SDPX.try_update_nonsymmetric_blocks!(
        runtime, s, nonfinite,
    )
    @test typed_nonfinite.status === SDPX.NS_RUNTIME_FAILED
    @test typed_nonfinite.reason === SDPX.NS_RUNTIME_NONFINITE_INPUT

    SDPX.initialize_primal_dual!(runtime, s, y)
    ds = zeros(3)
    dy = zeros(3)
    ds[1] = Inf
    chi = zeros(3)
    higher = SDPX.try_nonsymmetric_runtime_higher_correction!(
        runtime, chi, ds, dy,
    )
    @test higher.status === SDPX.NS_RUNTIME_FAILED
    @test higher.reason === SDPX.NS_RUNTIME_CORRECTOR_FAILED
    @test higher.corrector_reason === SDPX.NS_CORRECTOR_NONFINITE_INPUT
    @test !runtime.valid

    SDPX.initialize_primal_dual!(runtime, s, y)
    bad_current = copy(s)
    bad_current[3] = 1.0
    @test SDPX.max_step_primal!(runtime, bad_current, zeros(3)) == 0.0
    @test runtime.last_nonsymmetric.reason === SDPX.NS_RUNTIME_STEP_FAILED
    @test runtime.last_nonsymmetric.step_status === SDPX.NS_STEP_NOT_INTERIOR

    mixed_layout = _nspr_layout(
        Float64, ((:exp, 3, 0.0), (:power, 3, 0.5)),
    )
    mixed_runtime = SDPX.ProductConeRuntime(mixed_layout, Float64)
    mixed_s = zeros(6)
    mixed_y = zeros(6)
    SDPX.initialize_primal_dual!(mixed_runtime, mixed_s, mixed_y)
    mixed_bad_current = copy(mixed_s)
    mixed_bad_current[3] = 1.0
    @test SDPX.max_step_primal!(
        mixed_runtime, mixed_bad_current, zeros(6),
    ) == 0.0
    @test mixed_runtime.last_nonsymmetric.block_offset == 1
    @test mixed_runtime.last_nonsymmetric.reason === SDPX.NS_RUNTIME_STEP_FAILED
    @test mixed_runtime.last_nonsymmetric.step_status ===
          SDPX.NS_STEP_NOT_INTERIOR

    # Regression: a 1-O(eps) safety factor rounded the final bisection point
    # back onto the curved Exp boundary and returned zero/NO_BRACKET even
    # though alpha=1/2 is strictly interior.
    retreat_runtime = SDPX.ProductConeRuntime(layout, Float64)
    retreat_s = zeros(3)
    retreat_y = zeros(3)
    SDPX.initialize_primal_dual!(retreat_runtime, retreat_s, retreat_y)
    near_boundary = [0.0, 1.1851222655, 1.1882672303]
    near_direction = [0.0, 0.04909735115, 0.04450510181]
    @test SDPX.exp_primal_interior(near_boundary...)
    @test SDPX.exp_primal_interior(
        (near_boundary + 0.5 * near_direction)...,
    )
    retreat_step = SDPX.max_step_primal!(
        retreat_runtime, near_boundary, near_direction,
    )
    @test isfinite(retreat_step) && retreat_step > 0.0
    @test retreat_runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_READY
    @test retreat_runtime.last_nonsymmetric.step_status === SDPX.NS_STEP_ACCEPTED
    @test SDPX.exp_primal_interior(
        (near_boundary + retreat_step * near_direction)...,
    )

    mismatch = copy(s)
    mismatch[3] += 0.1
    h = zeros(3)
    @test_throws DomainError SDPX.affine_shift!(runtime, h, mismatch, y)
    @test runtime.last_nonsymmetric.reason === SDPX.NS_RUNTIME_POINT_MISMATCH
    @test !runtime.valid
end
