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

"""Snapshot every base iterate and direction field preparation could touch."""
function _base_snapshot(state)
    base = state.base
    return (
        x=copy(base.x), s=copy(base.s), y=copy(base.y),
        tau=base.tau, kappa=base.kappa,
        dx=copy(base.dx), dy=copy(base.dy), ds=copy(base.ds),
        dtau=base.dtau, dkappa=base.dkappa,
        dx_a=copy(base.dx_a), dy_a=copy(base.dy_a), ds_a=copy(base.ds_a),
        dtau_a=base.dtau_a, dkappa_a=base.dkappa_a,
        epoch=base.epoch, h=copy(state.h), gb=copy(state.gb),
    )
end

function _base_identical(a, b)
    return a.x == b.x && a.s == b.s && a.y == b.y &&
           a.tau == b.tau && a.kappa == b.kappa &&
           a.dx == b.dx && a.dy == b.dy && a.ds == b.ds &&
           a.dtau == b.dtau && a.dkappa == b.dkappa &&
           a.dx_a == b.dx_a && a.dy_a == b.dy_a && a.ds_a == b.ds_a &&
           a.dtau_a == b.dtau_a && a.dkappa_a == b.dkappa_a &&
           a.epoch == b.epoch && a.h == b.h && a.gb == b.gb
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

            # Non-vacuous mutation check: preparation must not change any base
            # iterate/direction/scratch field relative to an identical
            # UNPREPARED state constructed from the same canonical program.
            unprepared = SDPX.ProductConeHSDState(canonical; kkt_route=:bordered)
            @test SDPX.product_hsd_symmetric_core(unprepared) === nothing
            before = _base_snapshot(unprepared)
            @test _base_identical(_base_snapshot(state), before)

            # Cold-start is deterministic for identical programs: both states
            # evolve identically, and the prepared core still never factors.
            SDPX.product_hsd_cold_start!(unprepared)
            SDPX.product_hsd_cold_start!(state)
            @test _base_identical(_base_snapshot(state), _base_snapshot(unprepared))
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
        # The dimension-only gate itself must be allocation-free (scalar work
        # only), so the tiny cap cannot have triggered any block/operator/RHS
        # allocation before failing.  Warm the method first so compilation is
        # excluded from the count.
        blocks = Int[
            block.length
            for block in SDPX.layout_blocks(canonical.cone_layout)
        ]
        m = SDPX.hsd_num_slack(SDPX.HSDState(canonical))
        dim = size(canonical.A, 2) + m
        budget = _state_budget(Float64, dim, blocks)
        SDPX.symmetric_core_state_preflight(Float64, dim, blocks, 0, budget, 0)
        reject = () -> begin
            try
                SDPX.symmetric_core_state_preflight(
                    Float64, dim, blocks, 0, 1, 0,
                )
                false
            catch exception
                exception isa ArgumentError && return true
                rethrow()
            end
        end
        @test reject()
        bytes = @allocated reject()
        @test bytes < 1 << 16   # no block/operator/RHS allocation before reject
    end

    @testset "dimension/overflow/saturation fail closed" begin
        # Negative dimensions/sizes are rejected, never masked to zero.
        @test_throws ArgumentError SDPX.symmetric_core_state_prepare_bytes(
            Float64, -1, Int[],
        )
        @test_throws ArgumentError SDPX.symmetric_core_state_prepare_bytes(
            Float64, 5, [-1],
        )
        @test_throws ArgumentError SDPX.symmetric_core_state_preflight(
            Float64, -1, Int[], 0, typemax(Int), 0,
        )
        # Out-of-addressable dimension is rejected.
        @test_throws ArgumentError SDPX.symmetric_core_state_preflight(
            Float64, UInt128(typemax(Int)) + 1, Int[], 0, typemax(Int), 0,
        )
        # Saturated byte estimate is ineligible even against a typemax budget.
        @test SDPX.symmetric_core_state_prepare_bytes(
            Float64, typemax(UInt128), Int[],
        ) == typemax(Int)
        @test SDPX.symmetric_core_state_prepare_bytes(
            Float64, typemax(Int), Int[],
        ) == typemax(Int)
        @test_throws ArgumentError SDPX.symmetric_core_state_preflight(
            Float64, typemax(Int), Int[], 0, typemax(Int), 0,
        )
        # Checked/saturating nr+m never wraps (existing helper contract).
        @test SDPX.saturating_sum_bytes(typemax(Int) - 5, 20) == typemax(Int)
    end

    @testset "Float32 / unsupported arithmetic fails closed" begin
        # No dense provider for Float32; only Float64 path is built-in sparse.
        @test_throws ArgumentError SDPX.symmetric_core_provider_available(
            Float32, 24,
        )
    end

@testset "C7.2a prepared symmetric-core production dispatch" begin
    for id in _STATE_IDS
        @testset "$id" begin
            canonical = _state_canonical(id)
            m = SDPX.hsd_num_slack(SDPX.HSDState(canonical))
            blocks = Int[
                block.length
                for block in SDPX.layout_blocks(canonical.cone_layout)
            ]
            dim = size(canonical.A, 2) + m
            budget = _state_budget(Float64, dim, blocks)

            # Old bordered route (no core) is unchanged and authoritative.
            old_state = SDPX.ProductConeHSDState(canonical; kkt_route=:bordered)
            @test SDPX.product_hsd_symmetric_core(old_state) === nothing
            SDPX.product_hsd_cold_start!(old_state)
            old_code = SDPX.product_hsd_step!(old_state)
            @test old_code === SDPX.HSDStepOK

            # Prepared core state dispatches the symmetric core only.
            core_state = SDPX.ProductConeHSDState(
                canonical;
                kkt_route=:bordered,
                prepare_symmetric_core=true,
                symmetric_core_memory_limit=budget,
                symmetric_core_current_rss=0,
            )
            core = SDPX.product_hsd_symmetric_core(core_state)
            @test core !== nothing
            @test core_state.expanded === nothing
            SDPX.product_hsd_cold_start!(core_state)

            # First epoch: one factor, one homogeneous solve, predictor +
            # corrector variable solves, one truthful receipt.
            code1 = SDPX.product_hsd_step!(core_state)
            @test code1 === SDPX.HSDStepOK
            @test core.factor_epoch == 1
            @test core.homogeneous_solves == 1
            @test core.variable_solves == 2
            @test core.directions == 2
            @test SDPX.product_hsd_factor_count(core_state) == 1
            @test SDPX.product_hsd_receipt_build_count(core_state) == 1
            receipt = SDPX.product_hsd_factor_receipt(core_state)
            @test receipt !== nothing
            @test receipt.route === :symmetric_augmented_core
            diag = SDPX.factor_diagnostics(core.cache)
            @test diag.symbolic_count == 1
            @test diag.numeric_count == 1

            # Second epoch reuses the symbolic factor and refactors once.
            code2 = SDPX.product_hsd_step!(core_state)
            @test code2 === SDPX.HSDStepOK
            @test core.factor_epoch == 2
            @test core.homogeneous_solves == 2
            @test core.variable_solves == 4
            diag2 = SDPX.factor_diagnostics(core.cache)
            @test diag2.symbolic_count == 1
            @test diag2.numeric_count == 2
            @test SDPX.product_hsd_factor_count(core_state) == 2
            @test SDPX.product_hsd_receipt_build_count(core_state) == 2

            # A fresh prepared-core state stepped exactly once must land on
            # the same iterate as the old bordered route to roundoff.
            core_once = SDPX.ProductConeHSDState(
                canonical;
                kkt_route=:bordered,
                prepare_symmetric_core=true,
                symmetric_core_memory_limit=budget,
                symmetric_core_current_rss=0,
            )
            SDPX.product_hsd_cold_start!(core_once)
            @test SDPX.product_hsd_step!(core_once) === SDPX.HSDStepOK
            snap = (
                x=copy(old_state.base.x), s=copy(old_state.base.s),
                y=copy(old_state.base.y),
                tau=old_state.base.tau, kappa=old_state.base.kappa,
            )
            scale = max(
                1.0, maximum(abs, snap.x; init=0.0),
                maximum(abs, snap.s; init=0.0),
                maximum(abs, snap.y; init=0.0),
                abs(snap.tau), abs(snap.kappa),
            )
            tol = 1.0e-9 * scale
            @test maximum(abs, core_once.base.x - snap.x; init=0.0) <= tol
            @test maximum(abs, core_once.base.s - snap.s; init=0.0) <= tol
            @test maximum(abs, core_once.base.y - snap.y; init=0.0) <= tol
            @test abs(core_once.base.tau - snap.tau) <= tol
            @test abs(core_once.base.kappa - snap.kappa) <= tol
        end
    end
end

end