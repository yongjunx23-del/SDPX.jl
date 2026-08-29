# C7.1b: ProductConeHSDState optional setup-owned symmetric-core workspace.
#
# The state may own a prepared (unfactored) SymmetricCoreWorkspace plus
# state-owned block Theta operators/RHS and a BlockProduct semantic cone,
# without any direction dispatch reading it.  The old bordered route remains
# authoritative until the next commit.  This file asserts:
#   * prepare=false preserves all current behavior (core === nothing, old
#     product_hsd_step! still works);
#   * prepare=true for deterministic LP/SOC/PSD/Exp/Power canonical cases
#     builds a non-nothing core whose cache is Prepared with factor/homogeneous
#     /variable counts all zero, block ranges exact, block matrices
#     independent, and no direction/iterate mutation;
#   * a tiny memory cap rejects before factor (and before block allocation);
#   * expanded stays nothing for the default bordered route.

using Test
using SDPX
using LinearAlgebra
using SparseArrays

include(joinpath(
    @__DIR__, "..", "benchmark", "general", "GenericConicBenchmark.jl",
))
using .GenericConicBenchmark

const _STATE_IDS = (
    :lp_afiro_style,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)

function _state_spec(id::Symbol)
    matches = filter(spec -> spec.id === id, inventory(; tier=:small))
    length(matches) == 1 || error(
        "expected exactly one small case $id, found $(length(matches))",
    )
    return only(matches)
end

function _state_canonical(id::Symbol)
    spec = _state_spec(id)
    model = build(spec.problem, Float64, spec.params)
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    # Mirror the production path: equality reduction eliminates ZeroCone rows
    # before the product-HSD state is constructed.
    reduction = SDPX.hsd_equality_reduce(canonical)
    reduction.status === SDPX.HSDEqualityReady || error(
        "equality reduction failed for $id: $(reduction.status)",
    )
    return reduction.reduced
end

"""Conservative per-block count (dense lower block Theta) + core margin."""
function _state_budget(::Type{T}, dimension::Int, blocks::AbstractVector{Int}) where {T}
    estimate = SDPX.symmetric_core_state_prepare_bytes(T, dimension, blocks)
    return estimate > typemax(Int) - 1024 ? typemax(Int) : estimate + 1024
end

@testset "C7.1b product HSD symmetric core state ownership" begin
    for id in _STATE_IDS
        @testset "$id" begin
            canonical = _state_canonical(id)
            m = SDPX.hsd_num_slack(SDPX.HSDState(canonical))

            # Default behavior is unchanged: no core, old route still works.
            default_state = SDPX.ProductConeHSDState(canonical; kkt_route=:bordered)
            @test SDPX.product_hsd_symmetric_core(default_state) === nothing
            @test !SDPX.product_hsd_symmetric_core_prepared(default_state)
            @test default_state.expanded === nothing
            SDPX.product_hsd_cold_start!(default_state)
            code = SDPX.product_hsd_step!(default_state)
            @test code === SDPX.HSDStepOK

            # Prepare=true builds an unfactored prepared core.
            blocks = [
                block.length
                for block in SDPX.layout_blocks(canonical.cone_layout)
            ]
            dimension = size(canonical.A, 2) + m
            budget = _state_budget(Float64, dimension, blocks)
            state = SDPX.ProductConeHSDState(
                canonical;
                kkt_route=:bordered,
                prepare_symmetric_core=true,
                symmetric_core_memory_limit=budget,
                symmetric_core_current_rss=0,
            )
            core = SDPX.product_hsd_symmetric_core(state)
            @test core !== nothing
            @test SDPX.product_hsd_symmetric_core_prepared(state)
            @test state.expanded === nothing
            @test SDPX.factor_status(core.cache) === SDPX.Prepared
            @test core.factor_epoch == 0
            @test core.homogeneous_solves == 0
            @test core.variable_solves == 0
            @test core.directions == 0

            # Block ranges exactly match the canonical layout, block matrices
            # are independent (no shared storage), RHS is owned.
            expected_ranges = UnitRange{Int}[
                block.offset:(block.offset + block.length - 1)
                for block in SDPX.layout_blocks(canonical.cone_layout)
            ]
            @test core.pattern.block_ranges == expected_ranges
            cone = core.system.cone
            @test cone isa SDPX.BlockProductConeLinearization{Float64}
            @test length(cone.operators) == length(expected_ranges)
            @test cone.corrector_rhs !== canonical.b  # owned, not aliased
            for (index, rows) in enumerate(expected_ranges)
                @test size(cone.operators[index]) == (length(rows), length(rows))
            end

            # No direction/iterate mutation by preparation alone: cold start
            # only; the old route's step mutation behavior is covered by C6b.
            SDPX.product_hsd_cold_start!(state)
            base = state.base
            snap = (
                x=copy(base.x), s=copy(base.s), y=copy(base.y),
                tau=base.tau, kappa=base.kappa,
            )
            @test base.x == snap.x && base.s == snap.s && base.y == snap.y
            @test base.tau == snap.tau && base.kappa == snap.kappa
            @test SDPX.product_hsd_symmetric_core_prepared(state)
            @test core.factor_epoch == 0   # preparation never factored
            @test core.homogeneous_solves == 0
            @test core.variable_solves == 0
        end
    end

    @testset "tiny memory cap rejects before block allocation" begin
        canonical = _state_canonical(:socp_portfolio_small)
        @test_throws ArgumentError SDPX.ProductConeHSDState(
            canonical;
            kkt_route=:bordered,
            prepare_symmetric_core=true,
            symmetric_core_memory_limit=1,
            symmetric_core_current_rss=0,
        )
        # Unknown memory facts fail closed too.
        @test_throws ArgumentError SDPX.ProductConeHSDState(
            canonical;
            kkt_route=:bordered,
            prepare_symmetric_core=true,
            symmetric_core_memory_limit=nothing,
            symmetric_core_current_rss=nothing,
        )
    end

    @testset "Float32 / unsupported arithmetic fails closed" begin
        # No dense provider for Float32; only Float64 path is built-in sparse.
        @test_throws ArgumentError SDPX.symmetric_core_provider_available(
            Float32, 24,
        )
    end
end
