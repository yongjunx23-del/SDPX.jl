# Opt-in Clarabel-style Power dual-Hessian scaling experiment.
using Test
using SDPX
include(joinpath(@__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

function _dual_power_canonical()
    spec = only(filter(s -> s.id === :power_epigraph_small, inventory(tier=:small)))
    model = build(spec.problem, Float64, spec.params)
    program = SDPX.compile_product_cone_model(model)
    reduction = SDPX.hsd_equality_reduce(SDPX.canonicalize(program))
    reduction.status === SDPX.HSDEqualityReady || error("equality reduction failed")
    return reduction.reduced
end

function _dual_power_state(canonical; prepared::Bool)
    if !prepared
        state = SDPX.ProductConeHSDState(canonical; kkt_route=:bordered)
    else
        base = SDPX.HSDState(canonical)
        blocks = Int[b.length for b in SDPX.layout_blocks(canonical.cone_layout)]
        dimension = size(canonical.A, 2) + SDPX.hsd_num_slack(base)
        estimate = SDPX.symmetric_core_state_prepare_bytes(Float64, dimension, blocks)
        state = SDPX.ProductConeHSDState(
            canonical; kkt_route=:bordered, prepare_symmetric_core=true,
            symmetric_core_memory_limit=estimate + 4096,
            symmetric_core_current_rss=0,
        )
    end
    SDPX.product_hsd_cold_start!(state)
    return state
end

@testset "Power dual-Hessian scaling experiment" begin
    canonical = _dual_power_canonical()

    old = _dual_power_state(canonical; prepared=false)
    old_result = SDPX.product_hsd_solve!(old; max_iterations=200, max_time=30.0)
    @test old_result.status === SDPX.ProductHSDOptimal

    default_core = _dual_power_state(canonical; prepared=true)
    default_result = SDPX.product_hsd_solve!(
        default_core; max_iterations=200, max_time=30.0,
    )
    @test default_result.status === SDPX.ProductHSDOptimal

    dual_core = _dual_power_state(canonical; prepared=true)
    SDPX.hsd_residual!(dual_core.base)
    @test SDPX.force_power_dual_hessian_scaling!(
        dual_core.runtime, dual_core.base.s, dual_core.base.y,
        dual_core.base.mu,
    )
    checkpoint = (
        x=copy(dual_core.base.x), s=copy(dual_core.base.s),
        y=copy(dual_core.base.y), tau=dual_core.base.tau,
        kappa=dual_core.base.kappa,
    )
    @test all(block.force_dual_hessian for block in dual_core.runtime.power)
    @test all(block.forced_dual_hessian_updates == 1 for block in dual_core.runtime.power)
    @test all(block.last_scaling_status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
              for block in dual_core.runtime.power)
    @test dual_core.base.x == checkpoint.x
    @test dual_core.base.s == checkpoint.s
    @test dual_core.base.y == checkpoint.y
    @test dual_core.base.tau == checkpoint.tau
    @test dual_core.base.kappa == checkpoint.kappa

    dual_result = SDPX.product_hsd_solve!(
        dual_core; max_iterations=200, max_time=30.0,
    )
    @test dual_result.status === SDPX.ProductHSDOptimal
    @test all(block.force_dual_hessian for block in dual_core.runtime.power)
    @test all(block.last_scaling_status === SDPX.NS_SCALING_DUAL_HESSIAN_FALLBACK
              for block in dual_core.runtime.power)
    @test SDPX.product_hsd_symmetric_core(dual_core).factor_epoch ==
          dual_result.iterations
    @test dual_core.symmetric_bordered === nothing
    @test dual_core.coupled === nothing
end
