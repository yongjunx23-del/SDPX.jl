using Test
using SDPX

function _predictor_rescue_runtime()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    t = SDPX.variable!(model, :t, 1; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :power,
        (t[1], 1.0, x[1]), SDPX.PowerCone(0.5))
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    runtime = SDPX.ProductConeRuntime(canonical.cone_layout, Float64)
    s = ones(runtime.dimension)
    y = ones(runtime.dimension)
    power = only(runtime.power)
    s[power.offset + 2] = 0.25
    SDPX.try_update_scaling!(runtime, s, y, 1.0) || error("scaling setup failed")
    return runtime, s, y
end

@testset "certified affine predictor rescue" begin
    runtime, s, y = _predictor_rescue_runtime()
    block = only(runtime.power)
    failed = SDPX.NonsymmetricCorrectorResult{Float64}(
        SDPX.NS_CORRECTOR_FAILED,
        SDPX.NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH,
        Inf, Inf,
    )
    result = SDPX._runtime_ns_corrector_result!(runtime, block, failed)
    @test result.status === SDPX.NS_RUNTIME_FAILED
    @test runtime.valid
    @test SDPX._runtime_ns_affine_fallback_certified(runtime)

    h = zeros(runtime.dimension)
    @test SDPX._runtime_ns_affine_fallback!(runtime, h, s, y)
    @test runtime.valid
    @test h[block.offset:(block.offset + 2)] ==
          -s[block.offset:(block.offset + 2)]
    @test runtime.last_nonsymmetric.status === SDPX.NS_RUNTIME_READY
    @test runtime.last_nonsymmetric.corrector_reason ===
          SDPX.NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH
end

@testset "rescue rejects stale pairs and corrupted metrics" begin
    runtime, s, y = _predictor_rescue_runtime()
    block = only(runtime.power)
    failed = SDPX.NonsymmetricCorrectorResult{Float64}(
        SDPX.NS_CORRECTOR_FAILED,
        SDPX.NS_CORRECTOR_THIRD_SYMMETRY_MISMATCH,
        Inf, Inf,
    )
    SDPX._runtime_ns_corrector_result!(runtime, block, failed)
    stale_s = copy(s)
    stale_s[block.offset] = nextfloat(stale_s[block.offset])
    @test_throws DomainError SDPX._runtime_ns_affine_fallback!(
        runtime, zeros(runtime.dimension), stale_s, y,
    )
    @test !runtime.valid

    runtime, s, y = _predictor_rescue_runtime()
    block = only(runtime.power)
    block.scaling.theta[1, 1] = NaN
    result = SDPX._runtime_ns_corrector_result!(runtime, block, failed)
    @test result.status === SDPX.NS_RUNTIME_FAILED
    @test !runtime.valid
    @test !SDPX._runtime_ns_optional_corrector_failure(
        SDPX.NS_CORRECTOR_HESSIAN_FAILED,
    )
end
