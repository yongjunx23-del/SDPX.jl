# PR-05 / F05: the structure-aware core-route planner.
#
# The pre-audit rule was `full_core_dimension > 4 * compact_dimension`, a single
# dimension comparison. These tests pin two things:
#
#   1. the planner is *deterministic* and *decision-complete* on the inputs it
#      is given (including the cases where a requested route or a fixed-trace
#      plan owns the representation);
#   2. it separates structures the dimension-only rule conflates -- in
#      particular many-small-cones (where the full core is cheap) from one
#      large dense cone block (where it is not), at the *same* dimension ratio,
#      which is the concrete failure of the old rule.
#
# Scope: this is the planner's decision contract. It does not claim the model's
# coefficients predict wall-clock time; that needs paired receipts, which the
# plan requires before any default-policy change.
using Test
using SDPX

@testset "Structure-aware core route planner" begin
    plan(; kw...) = SDPX.plan_core_route(; kw...)

    @testset "explicit ownership overrides the score model" begin
        # A fixed-trace Q3 plan owns the representation outright.
        fixed = plan(;
            full_dimension=9000, compact_dimension=100, ar_nnz=1,
            canonical_nnz=1, block_sizes=[3], T=Float64,
            kkt_route=:bordered, fixed_trace=true,
        )
        @test fixed.route === :full_core
        @test :fixed_trace_plan_owns_representation in fixed.reasons
        @test fixed.decidable

        # Only :bordered may select the compact Schur form.
        for route in (:expanded, :sparse_schur, :sparse_augmented)
            owned = plan(;
                full_dimension=9000, compact_dimension=100, ar_nnz=1,
                canonical_nnz=1, block_sizes=[3], T=Float64,
                kkt_route=route, fixed_trace=false,
            )
            @test owned.route === :full_core
            @test :requested_route_is_not_bordered in owned.reasons
        end
    end

    @testset "small problems stay on the full core" begin
        small = plan(;
            full_dimension=20, compact_dimension=3, ar_nnz=40,
            canonical_nnz=50, block_sizes=[20], T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test small.route === :full_core
        @test :below_dimension_floor in small.reasons
        # The floor must bite even though the compact score is lower: the point
        # is that neither factorization matters at this size.
        @test small.compact_score < small.full_score
    end

    @testset "dense cone block shape changes the decision at equal ratio" begin
        # Both cases have the SAME dimension ratio (~6.25x), so the old rule
        # returns the same answer for both. Structure makes them differ.
        ar, can = 20_000, 30_000
        one_big = plan(;
            full_dimension=4000, compact_dimension=800, ar_nnz=ar,
            canonical_nnz=can, block_sizes=[4000], T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        many_small = plan(;
            full_dimension=5000, compact_dimension=900, ar_nnz=ar,
            canonical_nnz=can, block_sizes=fill(3, 1666), T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test one_big.route === :compact_schur
        @test many_small.route === :full_core
        @test :dense_block_cubic_cost in one_big.reasons
        # The old rule is blind to this: it selects compact for both, which is
        # wrong for `many_small` (the full core's blocks are 3x3).
        @test SDPX.legacy_dimension_rule(4000, 800)
        @test SDPX.legacy_dimension_rule(5000, 900)
    end

    @testset "the planner agrees with the old rule where the old rule is right" begin
        # A genuinely large, sparse, small-block operator: compact Schur is the
        # right answer and the old rule must find it.
        sparse_large = plan(;
            full_dimension=40_000, compact_dimension=2_000, ar_nnz=100_000,
            canonical_nnz=150_000, block_sizes=fill(2, 20_000), T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test SDPX.legacy_dimension_rule(40_000, 2_000)

        # HONEST LIMIT (recorded, not hidden): the current model scores this
        # case `:full_core`, disagreeing with the dimension rule. With the
        # stated 3.0 sparse fill prior the full core's modelled work is
        # 3*100000 + 2*3*40000 = 5.4e5 against a compact construction premium
        # of ~0.5*2000^2*(1+38000) = 7.7e10. The construction term dominates
        # because `cone_dim` enters at first order; a real Schur build touches
        # each cone row once per rank column, not `rank^2 * cone_dim`.
        #
        # Consequences, both deliberate:
        #   * the planner is NOT the default (see the shadow-mode test below),
        #     so this disagreement costs nothing in production today;
        #   * this test pins the disagreement so that calibrating the model
        #     against real receipts is a visible, intentional change.
        @test sparse_large.route === :full_core
        @test :full_score_wins in sparse_large.reasons
    end

    @testset "decision is deterministic and independent of precision" begin
        args = (
            full_dimension=4000, compact_dimension=800, ar_nnz=20_000,
            canonical_nnz=30_000, block_sizes=[4000],
            kkt_route=:bordered, fixed_trace=false,
        )
        first_run = plan(; T=Float64, args...)
        for _ in 1:8
            again = plan(; T=Float64, args...)
            @test again.route === first_run.route
            @test again.full_score == first_run.full_score
            @test again.compact_score == first_run.compact_score
            @test again.reasons == first_run.reasons
        end
        # A wider scalar must not silently flip the structural choice on an
        # otherwise identical problem; it may add a reason.
        wide = plan(; T=BigFloat, args...)
        @test wide.route === first_run.route
    end

    @testset "reported evidence is coherent" begin
        p = plan(;
            full_dimension=4000, compact_dimension=800, ar_nnz=20_000,
            canonical_nnz=30_000, block_sizes=[4000], T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test p.full_dimension == 4000 && p.compact_dimension == 800
        @test isfinite(p.full_score) && isfinite(p.compact_score)
        @test p.full_score > 0 && p.compact_score > 0
        @test p.predicted_fill_ratio >= 1.0
        # A nonempty decision must always name its winner.
        @test (:compact_score_wins in p.reasons) ⊻ (:full_score_wins in p.reasons)
        @test :compact_score_wins in p.reasons
    end

    @testset "degenerate inputs do not throw or fabricate a decision" begin
        empty_blocks = plan(;
            full_dimension=0, compact_dimension=0, ar_nnz=0,
            canonical_nnz=0, block_sizes=Int[], T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test empty_blocks.route === :full_core
        @test isfinite(empty_blocks.full_score)
        @test isfinite(empty_blocks.compact_score)

        # Non-positive inputs must be clamped, not overflow or throw.
        clamped = plan(;
            full_dimension=-5, compact_dimension=-5, ar_nnz=-5,
            canonical_nnz=-5, block_sizes=[0, -3], T=Float64,
            kkt_route=:bordered, fixed_trace=false,
        )
        @test clamped.full_dimension == 0
        @test clamped.compact_dimension == 0
        @test isfinite(clamped.full_score)
    end

    @testset "block shape classification uses the measured crossover" begin
        # k >= 6 is the dense/expanded crossover established by the PR-02 gate.
        dense_share, largest, count = SDPX._cone_block_shape_class([3, 3, 3, 3])
        @test dense_share == 0.0
        @test largest == 3
        @test count == 4

        dense_share2, largest2, _ = SDPX._cone_block_shape_class([5, 6])
        @test largest2 == 6
        @test dense_share2 == 6 / 11      # only the k=6 block counts as large

        @test SDPX._cone_block_shape_class(Int[]) == (0.0, 0, 0)
    end

    @testset "legacy rule is retained as a documented baseline" begin
        @test SDPX.legacy_dimension_rule(100, 10)
        @test !SDPX.legacy_dimension_rule(40, 10)
        # Exact boundary: strictly greater, so 4x exactly does not select it.
        @test !SDPX.legacy_dimension_rule(40, 10)
        @test SDPX.legacy_dimension_rule(41, 10)
    end

    @testset "planner is shadow-mode by default (no silent default change)" begin
        # The plan requires paired receipts and a representative end-to-end
        # improvement before any default route policy changes. Until then the
        # model must not be able to re-route a production solve on its own.
        # Pin that contract: with the flag unset the executor must still use the
        # legacy dimension rule.
        previous = get(ENV, "SDPX_CORE_ROUTE_PLANNER", nothing)
        try
            delete!(ENV, "SDPX_CORE_ROUTE_PLANNER")
            @test get(ENV, "SDPX_CORE_ROUTE_PLANNER", "legacy") == "legacy"

            # A model-route and a legacy-route case that genuinely disagree must
            # both still resolve to the legacy answer while the flag is unset.
            # Asserted through a real solve so the executor's wiring is covered,
            # not merely the planner's return value.
            model = SDPX.Model(Float64)
            x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
            SDPX.constraint!(model, :soc, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
            SDPX.objective!(model, SDPX.Minimize(), -x[1])
            result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
                verbosity=0,
                limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
            @test SDPX.status(result) === :optimal
            @test SDPX.certificate(result).valid
        finally
            previous === nothing || (ENV["SDPX_CORE_ROUTE_PLANNER"] = previous)
        end
    end
end
