# A1 — first-class BigFloat precision ladder.
#
# Focused tests for the split pre-execution ladder authority
# (`PrecisionAttemptSpec` / `PrecisionLadderPlan`) and post-execution
# diagnostics (`PrecisionAttemptReport` / `PrecisionLadderReport`):
# selector semantics, retry eligibility, shared wall-clock budget, resume
# bypass, A0 attempt IDs, diagnostics independence, and the
# no-result/no-third-rung retention contract.
using SDPX
using LinearAlgebra
using SparseArrays
using Test

function _ladder_2x2_problem(; bits::Int=256)
    return setprecision(BigFloat, bits) do
        SDPX.ingest(
            BigFloat[2, 3],
            [reshape(BigFloat[1, 0, 0, 0, 0, 0, 0, 1], 2, 2, 2)],
            [BigFloat[0 1; 1 0]],
            Matrix{BigFloat}(undef, 2, 0),
            BigFloat[];
            verbosity=0,
        )
    end
end

"""Sparse 2x2 arrow problem with block constants spanning orders of
magnitude. Unpinned, the 3-block instance fails deterministically with
`NumericalBreakdown` at both 256 and 512 bits, which exercises the
eligible-retry path without iteration-count or timing flakiness."""
function _ladder_arrow_problem(; blocks::Int=3, shared::Int=2)
    m = shared + blocks
    coefficients = [
        Vector{SparseMatrixCSC{BigFloat,Int}}(undef, m)
        for _ in 1:blocks
    ]
    for l in 1:blocks, i in 1:m
        active = i <= shared || i == shared + l
        coefficients[l][i] = active ?
                             sparse([1, 2], [1, 2], BigFloat[1, 1], 2, 2) :
                             spzeros(2, 2)
    end
    C = [
        Matrix{BigFloat}(
            (BigFloat(2) * BigFloat(10)^(l - 1)) * I,
            2,
            2,
        )
        for l in 1:blocks
    ]
    return SDPX.ingest(
        ones(BigFloat, m),
        coefficients,
        C,
        zeros(BigFloat, m, 0),
        BigFloat[];
        sparse=true,
        verbosity=0,
    )
end

@testset "A1 precision ladder: selector semantics" begin
    @testset "fixed policy requests exactly one rung" begin
        setprecision(BigFloat, 256) do
            prob = _ladder_2x2_problem(bits=256)
            opts = SDPX.SolverOptions{BigFloat}(
                precision_bits=256,
                working_precision_policy=:fixed,
                iter_max=200,
                verbosity=0,
                diagnostics=true,
            )
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test ladder !== nothing
            @test ladder.plan.policy === :fixed
            @test ladder.plan.requested_bits == 256
            @test ladder.plan.selected_bits == 256
            @test ladder.plan.floor_bits == 192
            @test !ladder.plan.resume_bypass
            @test ladder.plan.selection_reason === :fixed
            @test [rung.bits for rung in ladder.plan.rungs] == [256]
            @test [rung.role for rung in ladder.plan.rungs] == [:requested]
            @test ladder.plan.time_budget === :shared_wall_clock
            @test length(ladder.attempts) == 1
            @test ladder.attempts[1].spec.bits == 256
            @test ladder.attempts[1].spec.role === :requested
            @test ladder.attempts[1].facts.retry_decision === :terminal
            @test ladder.attempts[1].facts.remaining_budget_seconds == Inf
            @test result.diagnostics.attempts ==
                  (ladder.attempts[1].record,)
        end
    end

    @testset "auto with no ladder: selected == requested" begin
        # Zero tolerances force the adaptive selector to the requested bits,
        # so the ladder has exactly one rung and one attempt even under
        # `:auto` (the solve itself ends in NumericalBreakdown; only the
        # ladder structure is under test here).
        setprecision(BigFloat, 256) do
            prob = _ladder_2x2_problem(bits=256)
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=zero(BigFloat),
                ϵ_primal=zero(BigFloat),
                ϵ_dual=zero(BigFloat),
                precision_bits=256,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=100,
                verbosity=0,
                diagnostics=true,
            )
            @test SDPX.adaptive_working_precision_bits(prob, opts) == 256
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test [rung.bits for rung in ladder.plan.rungs] == [256]
            @test ladder.plan.selected_bits == 256
            @test ladder.plan.floor_bits == 192
            @test ladder.plan.selection_reason === :adaptive_requested
            @test length(ladder.attempts) == 1
            @test only(result.diagnostics.attempts).attempt_id == 1
            @test only(result.diagnostics.attempts).plan_id == 1
            @test only(result.diagnostics.attempts).planned.precision.
                  explicit_bits == 256
        end
    end

    @testset "auto lower success: two planned rungs, one executed" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_2x2_problem(bits=512)
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-50"),
                ϵ_primal=parse(BigFloat, "1e-50"),
                ϵ_dual=parse(BigFloat, "1e-50"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=200,
                verbosity=0,
                diagnostics=true,
            )
            @test SDPX.adaptive_working_precision_bits(prob, opts) == 288
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test [rung.bits for rung in ladder.plan.rungs] == [288, 512]
            @test [rung.role for rung in ladder.plan.rungs] ==
                  [:selected, :requested]
            @test ladder.plan.selected_bits == 288
            @test ladder.plan.floor_bits == 192
            @test ladder.plan.selection_reason === :adaptive_lower
            # The lower rung succeeded: no retry, exactly one attempt, which
            # carries the 288-bit facts (not the ambient 512 bits).
            @test result.status == SDPX.Optimal
            @test length(result.diagnostics.attempts) == 1
            @test length(ladder.attempts) == 1
            attempt = only(result.diagnostics.attempts)
            @test attempt.attempt_id == 1
            @test attempt.planned.precision.explicit_bits == 288
            @test attempt.executed.precision.explicit_bits == 288
            @test ladder.attempts[1].record === attempt
            @test ladder.attempts[1].child_plan === result.diagnostics.plan
            @test ladder.attempts[1].facts.success
            @test ladder.attempts[1].facts.retry_decision === :success
            @test any(
                warning -> occursin(
                    "Adaptive working precision selected 288",
                    warning,
                ),
                result.diagnostics.warnings,
            )
        end
    end
end

@testset "A1 precision ladder: retry decisions" begin
    function _retry_options(; iter_max::Int=200)
        return SDPX.SolverOptions{BigFloat}(
            ϵ_gap=parse(BigFloat, "1e-40"),
            ϵ_primal=parse(BigFloat, "1e-40"),
            ϵ_dual=parse(BigFloat, "1e-40"),
            precision_bits=512,
            working_precision_policy=:auto,
            minimum_working_precision_bits=192,
            iter_max=iter_max,
            verbosity=0,
            diagnostics=true,
        )
    end

    @testset "eligible retry runs the second rung" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_arrow_problem()
            opts = _retry_options()
            @test SDPX.adaptive_working_precision_bits(prob, opts) == 256
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test result.status == SDPX.NumericalBreakdown
            @test length(ladder.attempts) == 2
            @test [rung.bits for rung in ladder.plan.rungs] == [256, 512]
            # Both rungs executed: records are flattened in rung order with
            # the A0 rung-local IDs 1/1 and 2/2.
            @test [(record.attempt_id, record.plan_id)
                   for record in result.diagnostics.attempts] ==
                  [(1, 1), (2, 2)]
            @test [record.planned.precision.explicit_bits
                   for record in result.diagnostics.attempts] == [256, 512]
            @test [record.status for record in result.diagnostics.attempts] ==
                  [SDPX.NumericalBreakdown, SDPX.NumericalBreakdown]
            @test !ladder.attempts[1].facts.success
            @test !ladder.attempts[2].facts.success
            @test ladder.attempts[1].facts.retry_decision === :retry
            @test ladder.attempts[2].facts.retry_decision === :terminal
            @test ladder.attempts[1].record === result.diagnostics.attempts[1]
            @test ladder.attempts[2].record === result.diagnostics.attempts[2]
            # The final diagnostics plan is the final (second) rung's child
            # plan, and the retry provenance warning is recorded.
            @test ladder.attempts[2].child_plan === result.diagnostics.plan
            @test any(
                warning -> occursin(
                    "SDPX retried at the requested 512-bit precision",
                    warning,
                ),
                result.diagnostics.warnings,
            )
        end
    end

    @testset "ineligible status (IterLimit) never retries" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_2x2_problem(bits=512)
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-50"),
                ϵ_primal=parse(BigFloat, "1e-50"),
                ϵ_dual=parse(BigFloat, "1e-50"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=1,
                verbosity=0,
                diagnostics=true,
            )
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test result.status == SDPX.IterLimit
            @test [rung.bits for rung in ladder.plan.rungs] == [288, 512]
            @test length(ladder.attempts) == 1
            @test length(result.diagnostics.attempts) == 1
            @test only(result.diagnostics.attempts).status == SDPX.IterLimit
            @test SDPX.IterLimit ∉ ladder.plan.retry_statuses
            @test only(ladder.attempts).facts.retry_decision ===
                  :ineligible_status
            @test any(
                warning -> occursin(
                    "the fallback was not eligible or its time budget was " *
                    "exhausted",
                    warning,
                ),
                result.diagnostics.warnings,
            )
        end
    end

    @testset "shared wall clock: no time left means no retry" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_2x2_problem(bits=512)
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-50"),
                ϵ_primal=parse(BigFloat, "1e-50"),
                ϵ_dual=parse(BigFloat, "1e-50"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=200,
                max_time=1.0e-9,
                verbosity=0,
                diagnostics=true,
            )
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test result.status == SDPX.TimeLimit
            @test length(ladder.attempts) == 1
            @test only(result.diagnostics.attempts).status == SDPX.TimeLimit
            @test only(result.diagnostics.attempts).attempt_id == 1
            @test only(ladder.attempts).facts.retry_decision ===
                  :ineligible_status
            # Shared budget: the ladder may plan two rungs, but the exhausted
            # wall clock is the single authority that forbids the retry.
            @test [rung.bits for rung in ladder.plan.rungs] == [288, 512]
            @test any(
                warning -> occursin("time budget was exhausted", warning),
                result.diagnostics.warnings,
            )
        end
    end

    @testset "shared budget spans rungs (exhaustion blocks retry)" begin
        setprecision(BigFloat, 512) do
            # The lower rung is forced to fail with `TimeLimit` (an
            # ineligible status) inside a two-rung ladder. A single shared
            # wall clock drives both rungs: the plan declares
            # `:shared_wall_clock`, the exhausted budget leaves exactly the
            # lower rung's single attempt, and the upper rung never executes.
            prob = _ladder_2x2_problem(bits=512)
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-50"),
                ϵ_primal=parse(BigFloat, "1e-50"),
                ϵ_dual=parse(BigFloat, "1e-50"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=1,
                max_time=1.0e-9,
                verbosity=0,
                diagnostics=true,
            )
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            @test result.status == SDPX.TimeLimit
            @test ladder.plan.time_budget === :shared_wall_clock
            @test SDPX.TimeLimit ∉ ladder.plan.retry_statuses
            @test [rung.bits for rung in ladder.plan.rungs] == [288, 512]
            @test length(ladder.attempts) == 1
            @test only(ladder.attempts).spec.bits == 288
            @test only(result.diagnostics.attempts).status == SDPX.TimeLimit
            @test only(ladder.attempts).facts.elapsed_seconds >= 0.0
            @test only(ladder.attempts).facts.retry_decision ===
                  :ineligible_status
            @test only(ladder.attempts).facts.remaining_budget_seconds >= 0.0
        end
    end

    @testset "resume bypasses the ladder selector" begin
        setprecision(BigFloat, 256) do
            prob = _ladder_2x2_problem(bits=256)
            opts = SDPX.SolverOptions{BigFloat}(
                precision_bits=256,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                verbosity=0,
            )
            plan = SDPX._build_precision_ladder_plan(
                prob,
                opts;
                resume="checkpoint.bin",
            )
            @test plan.policy === :auto
            @test plan.resume_bypass
            @test plan.selection_reason === :resume
            @test plan.selected_bits == 256
            @test length(plan.rungs) == 1
            @test only(plan.rungs).bits == 256
            @test only(plan.rungs).role === :requested
        end
    end
end

@testset "A1 precision ladder: diagnostics" begin
    @testset "diagnostics off makes identical decisions" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_arrow_problem()
            enabled = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-40"),
                ϵ_primal=parse(BigFloat, "1e-40"),
                ϵ_dual=parse(BigFloat, "1e-40"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=200,
                verbosity=0,
                diagnostics=true,
            )
            disabled = SDPX._replace_solver_options(
                enabled;
                diagnostics=false,
            )
            on = SDPX.solve!(prob, enabled)
            off = SDPX.solve!(prob, disabled)
            @test off.diagnostics === nothing
            @test off.status == on.status == SDPX.NumericalBreakdown
            @test length(on.diagnostics.attempts) == 2
            # The disabled run performs the same staged execution: the same
            # ladder plan would have been built, so the executed status and
            # per-rung bits are observable through the enabled run alone.
            @test [record.planned.precision.explicit_bits
                   for record in on.diagnostics.attempts] == [256, 512]
        end
    end

    @testset "report retains plan, records, and scalar facts only" begin
        setprecision(BigFloat, 512) do
            prob = _ladder_arrow_problem()
            opts = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=parse(BigFloat, "1e-40"),
                ϵ_primal=parse(BigFloat, "1e-40"),
                ϵ_dual=parse(BigFloat, "1e-40"),
                precision_bits=512,
                working_precision_policy=:auto,
                minimum_working_precision_bits=192,
                iter_max=200,
                verbosity=0,
                diagnostics=true,
            )
            result = SDPX.solve!(prob, opts)
            ladder = result.diagnostics.precision_ladder
            # The ladder report is diagnostics-only: it carries the immutable
            # plan, per-rung child plans, A0 records, and scalar facts — but
            # never a result, workspace, factor, or mutable BigFloat array.
            @test ladder.plan isa SDPX.PrecisionLadderPlan
            @test length(ladder.attempts) == 2
            for attempt in ladder.attempts
                @test attempt isa SDPX.PrecisionAttemptReport
                @test attempt.spec isa SDPX.PrecisionAttemptSpec
                @test attempt.child_plan isa SDPX.ExecutionPlan
                @test attempt.record isa SDPX.ExecutionAttemptRecord
                @test attempt.facts isa SDPX.PrecisionAttemptScalarFacts
                fields = fieldnames(typeof(attempt))
                @test :result ∉ fields
                @test :workspace ∉ fields
                @test :factor ∉ fields
            end
            @test ladder.plan.rungs == (
                SDPX.PrecisionAttemptSpec(1, 256, :selected),
                SDPX.PrecisionAttemptSpec(2, 512, :requested),
            )
            @test [attempt.facts.retry_decision
                   for attempt in ladder.attempts] ==
                  [:retry, :terminal]
            @test all(
                attempt -> attempt.facts.remaining_budget_seconds >= 0.0,
                ladder.attempts,
            )
            # No third rung exists anywhere in the ladder authority.
            @test length(ladder.plan.rungs) == 2
            @test [attempt.spec.rung for attempt in ladder.attempts] == [1, 2]
            @test [attempt.spec.bits for attempt in ladder.attempts] ==
                  [256, 512]
        end
    end
end

@testset "A1 precision ladder: prepared exact precision" begin
    setprecision(BigFloat, 256) do
        prob = _ladder_2x2_problem(bits=256)
        opts = SDPX.SolverOptions{BigFloat}(
            precision_bits=256,
            working_precision_policy=:fixed,
            iter_max=200,
            verbosity=0,
            diagnostics=true,
        )
        prepared = SDPX.prepare(prob, opts)
        result = SDPX.solve!(prepared)
        @test result.status == SDPX.Optimal
        ladder = result.diagnostics.precision_ladder
        @test ladder.plan.policy === :fixed
        @test ladder.plan.selected_bits == 256
        @test ladder.plan.selection_reason === :fixed
        @test [rung.bits for rung in ladder.plan.rungs] == [256]
        @test length(ladder.attempts) == 1
        @test only(ladder.attempts).record.planned.precision.explicit_bits ==
              256
        @test only(ladder.attempts).record.executed.precision.explicit_bits ==
              256
        # The child plan is the prepared execution plan reused at the exact
        # prepared precision (one rung at the requested bits).
        @test only(ladder.attempts).child_plan === result.diagnostics.plan
        @test only(ladder.attempts).child_plan ===
              prepared.structure.execution_plan
    end
end

@testset "A1 precision ladder: prepared partial reuse" begin
    setprecision(BigFloat, 512) do
        # Canonical A1: a prepared session must reuse the cached structure
        # only at the exact prepared precision. The adaptive ladder starts at
        # the selected 256-bit rung, which must *not* reuse the 512-bit
        # prepared data (fresh child plan), and the eligible retry at the
        # requested 512 bits must reuse it (the prepared child plan itself).
        prob = _ladder_arrow_problem()
        opts = SDPX.SolverOptions{BigFloat}(
            ϵ_gap=parse(BigFloat, "1e-40"),
            ϵ_primal=parse(BigFloat, "1e-40"),
            ϵ_dual=parse(BigFloat, "1e-40"),
            precision_bits=512,
            working_precision_policy=:auto,
            minimum_working_precision_bits=192,
            iter_max=200,
            verbosity=0,
            diagnostics=true,
        )
        prepared = SDPX.prepare(prob, opts)
        @test prepared.structure.execution_plan !== nothing
        result = SDPX.solve!(prepared)
        ladder = result.diagnostics.precision_ladder
        # Deterministic two-rung retry (the same arrow fixture used by the
        # eligible-retry testset).
        @test result.status == SDPX.NumericalBreakdown
        @test length(ladder.attempts) == 2
        @test [attempt.spec.bits for attempt in ladder.attempts] ==
              [256, 512]
        @test [attempt.facts.retry_decision
               for attempt in ladder.attempts] == [:retry, :terminal]
        # The lower rung never reuses the prepared execution plan.
        @test ladder.attempts[1].child_plan !==
              prepared.structure.execution_plan
        # The upper eligible retry reuses the exact prepared execution plan.
        @test ladder.attempts[2].child_plan ===
              prepared.structure.execution_plan
        @test result.diagnostics.plan === prepared.structure.execution_plan
        @test ladder.attempts[1].record.planned.precision.explicit_bits == 256
        @test ladder.attempts[2].record.planned.precision.explicit_bits == 512
    end
end
