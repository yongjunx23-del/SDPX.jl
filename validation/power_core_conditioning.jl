# Deterministic late-Power raw-dual regression guard.
#
# The symmetric augmented core owns the dual direction returned by its KKT
# solve.  `G(target)` is now diagnostic roundtrip work only and never
# overwrites that raw direction.  With this Clarabel-style ownership,
# `power_epigraph_small` reaches ProductHSDOptimal on both independent old and
# prepared-core states; LP/SOC/PSD/Exp one-step prepared-core probes also pass.
#
# This validation adds no tolerance, fallback, public route switch, QDLDL, or
# PureKLU.  It records the factual fix for the former late-Power trajectory
# regression and keeps the public default old bordered route authoritative.
using Test
using SDPX
using LinearAlgebra
using SparseArrays

include(joinpath(
    @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
))
using .GenericConicBenchmark

const _COND_IDS = (
    :lp_afiro_style,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
)

function _cond_canonical(id::Symbol)
    spec = only(filter(s -> s.id === id, inventory(; tier=:small)))
    model = build(spec.problem, Float64, spec.params)
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    reduction.status === SDPX.HSDEqualityReady || error(
        "equality reduction failed for $id: $(reduction.status)",
    )
    return reduction.reduced
end

function _cond_core_state(canonical)
    base0 = SDPX.HSDState(canonical)
    m = SDPX.hsd_num_slack(base0)
    blocks = Int[block.length for block in SDPX.layout_blocks(canonical.cone_layout)]
    dim = size(canonical.A, 2) + m
    estimate = SDPX.symmetric_core_state_prepare_bytes(Float64, dim, blocks)
    state = SDPX.ProductConeHSDState(
        canonical;
        kkt_route=:bordered,
        prepare_symmetric_core=true,
        symmetric_core_memory_limit=estimate + 4096,
        symmetric_core_current_rss=0,
    )
    SDPX.product_hsd_cold_start!(state)
    return state
end

function _cond_snapshot(state)
    base = state.base
    return (
        x=copy(base.x), s=copy(base.s), y=copy(base.y),
        tau=base.tau, kappa=base.kappa,
        iterations=base.record.iterations, epoch=base.epoch,
    )
end

function _cond_rel_diff(a, b)
    diff = max(
        maximum(abs, a.x - b.x; init=0.0),
        maximum(abs, a.s - b.s; init=0.0),
        maximum(abs, a.y - b.y; init=0.0),
        abs(a.tau - b.tau), abs(a.kappa - b.kappa),
    )
    scale = max(
        1.0, maximum(abs, a.x; init=0.0), maximum(abs, a.s; init=0.0),
        maximum(abs, a.y; init=0.0), abs(a.tau), abs(a.kappa),
    )
    return diff / scale
end

@testset "C8-det Power conditioning trace" begin
    @testset "overall solve on independent fresh states" begin
        canonical = _cond_canonical(:power_epigraph_small)
        # (A) authoritative old bordered route on a fresh state: must solve.
        old = SDPX.ProductConeHSDState(canonical; kkt_route=:bordered)
        SDPX.product_hsd_cold_start!(old)
        result_old = SDPX.product_hsd_solve!(
            old; max_iterations=200, max_time=30.0,
        )
        @test result_old.status === SDPX.ProductHSDOptimal

        # (B) prepared raw-dy symmetric-core route on a fresh state: record
        # its truthful terminal outcome.  No claim is made that the raw core
        # alone is production-qualified; this asserts only what the existing
        # solver (raw-core steps + terminal verifier) actually returns.
        core = _cond_core_state(canonical)
        result_core = SDPX.product_hsd_solve!(
            core; max_iterations=200, max_time=30.0,
        )
        @test result_core.status === SDPX.ProductHSDOptimal
        @test result_core.reason in (
            SDPX.ProductHSDVerifiedAcceptedStep,
            SDPX.ProductHSDVerifiedTerminalNewtonTrial,
        )
    end
end

@testset "One-step prepared core passes LP/SOC/PSD/Exp" begin
    for id in _COND_IDS
        @testset "$id" begin
            core_state = _cond_core_state(_cond_canonical(id))
            @test SDPX.product_hsd_step!(core_state) === SDPX.HSDStepOK
        end
    end
end
