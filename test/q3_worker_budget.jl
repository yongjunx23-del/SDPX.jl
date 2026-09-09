# Q3 worker-budget, range-executor, and phase-timing regression (tests only).
#
# Covers the workspace-owned budget implementation:
#   * `_q3_worker_limit`: rejection of non-positive budgets, min(request, pool).
#   * `_q3_foreach`: empty/tiny-serial/large-parallel exactly-once visits,
#     returned task count bounded by the budget, nested invocation from a
#     spawned task, failure join (exception propagates, executor reusable),
#     concurrent executors with private buffers and deterministic results.
#   * Q3 equality workspaces with budgets 1/2 on the fixed-trace reference
#     fixture: own worker budgets, x4 provider thread-count config, no
#     mutual overwrite, bit-identical prepared results across all limbs.
#   * Phase-timing reset/snapshot coverage of `q3_worker_budget` and the
#     finer metric children; `_reset_q3_phase_timings!` zeros timers only.
using Test, SDPX, MultiFloats, MultiFloatLinearAlgebra, LinearAlgebra, SparseArrays

include(joinpath(@__DIR__, "..", "validation", "fixed_trace_q3_reference.jl"))

const _POOL = Threads.nthreads()

@testset "q3 worker budget" begin
@testset "_q3_worker_limit" begin
    @test_throws ArgumentError SDPX._q3_worker_limit(0)
    @test_throws ArgumentError SDPX._q3_worker_limit(-1)
    @test SDPX._q3_worker_limit(1) == 1
    @test SDPX._q3_worker_limit(2) == min(2, _POOL)
    @test SDPX._q3_worker_limit(10^9) == _POOL
    @test SDPX._q3_worker_limit(10^9) isa Int
end

@testset "_q3_foreach scheduling" begin
    # Empty range: no visits, zero tasks.
    calls = Ref(0)
    @test SDPX._q3_foreach(_ -> calls[] += 1, 1:0, 4) == 0
    @test calls[] == 0
    # Tiny range: serial, exactly once, in order.
    seen = Int[]
    @test SDPX._q3_foreach(1:10, 4; min_items=256) do i
        push!(seen, i)
    end == 1
    @test seen == collect(1:10)
    # Large range: partitioned across at most `budget` tasks, exactly once.
    n = 2000
    counts = [Threads.Atomic{Int}(0) for _ in 1:n]
    out = zeros(Int, n)
    @test SDPX._q3_foreach(1:n, 2) do i
        Threads.atomic_add!(counts[i], 1)
        out[i] = 2i
    end == min(2, _POOL)
    @test all(c -> c[] == 1, counts)
    @test out == collect(2:2:2n)
    # Budget 1 stays serial but exact.
    fill!(out, 0)
    @test SDPX._q3_foreach(1:n, 1) do i
        out[i] = 2i
    end == 1
    @test out == collect(2:2:2n)
    # Nested invocation from inside a spawned task.
    nested = fetch(Threads.@spawn begin
        a = SDPX._q3_foreach(1:10, 4; min_items=256) do i
            i
        end
        bcounts = [Threads.Atomic{Int}(0) for _ in 1:500]
        b = SDPX._q3_foreach(1:500, 2) do i
            Threads.atomic_add!(bcounts[i], 1)
        end
        (a, b, all(c -> c[] == 1, bcounts))
    end)
    @test nested == (1, min(2, _POOL), true)
    # Failure joins: the exception propagates and the executor is reusable.
    detonated = Threads.Atomic{Int}(0)
    exception_type = _POOL > 1 ? CompositeException : ErrorException
    @test_throws exception_type SDPX._q3_foreach(1:n, 4) do i
        Threads.atomic_add!(detonated, 1)
        i == 500 && error("boom-500")
    end
    @test detonated[] > 0
    redone = zeros(Int, n)
    @test SDPX._q3_foreach(1:n, 4) do i
        redone[i] = 2i
    end <= 4
    @test redone == collect(2:2:2n)
    # Concurrent separate executors with private buffers: deterministic.
    m = 1500
    job = function (budget)
        private = zeros(Int, m)
        ntasks = SDPX._q3_foreach(1:m, budget) do i
            private[i] = 3i + 1
        end
        return (ntasks, private)
    end
    @sync begin
        r1 = Threads.@spawn job(1)
        r2 = Threads.@spawn job(2)
        n1, p1 = fetch(r1)
        n2, p2 = fetch(r2)
        @test n1 == 1
        @test n2 <= 2
        @test p1 == p2 == collect(4:3:(3m + 1))
    end
end

@testset "equality workspace budgets across limbs" begin
    ext = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
    # x4 provider thread-count configuration is admitted per backend.
    @test ext._Provider(Float64x4; threads=1).config.thread_count == 1
    # The low-level provider retains its requested config; core admission
    # clamps the budget before constructing it, and kernels also cap workers.
    @test ext._Provider(Float64x4; threads=2).config.thread_count == 2
    for ST in (Float64, BigFloat, Float64x2, Float64x3, Float64x4)
        @testset "$ST" begin
            setprecision(BigFloat, 256) do
                problem = fixed_trace_problem(ST)
                reduction = SDPX._fixed_trace_q3_reduction(problem)
                @test reduction !== nothing
                blocks = size(reduction.active_ids, 2)
                variables = 2blocks + length(reduction.free_ids)
                @test sort(vcat(
                    vec(copy(reduction.active_ids)),
                    copy(reduction.free_ids))) == collect(1:variables)
                # One synthetic equality row; the panel carries no
                # structural contract beyond finiteness and width.
                panel = zeros(ST, 1, variables)
                panel[1] = one(ST)
                panel[end] = one(ST)
                ws1 = SDPX.FixedTraceQ3EqualitySchurWorkspace(
                    reduction, panel; workers=1)
                ws2 = SDPX.FixedTraceQ3EqualitySchurWorkspace(
                    reduction, panel; workers=2)
                @test ws1.worker_budget == 1
                @test ws2.worker_budget == min(2, _POOL)
                @test ws1.equality_work !== ws2.equality_work
                metric = ST.(Float64[4 9; 1 1.5; 5 7][1:3, 1:blocks])
                regularization = sqrt(eps(ST))
                @test SDPX.prepare_fixed_trace_q3_equality_schur!(
                    ws1, metric, regularization)
                schur1 = copy(ws1.schur)
                @test SDPX.prepare_fixed_trace_q3_equality_schur!(
                    ws2, metric, regularization)
                # Same fixed input under different budgets: bit-identical
                # prepared fields, privately owned buffers.
                @test ws2.schur == schur1
                @test ws2.transformed_panel == ws1.transformed_panel
                @test ws2.local_elimination.factors ==
                    ws1.local_elimination.factors
                @test ws2.local_elimination.inverse_pivots ==
                    ws1.local_elimination.inverse_pivots
                @test ws2.schur !== ws1.schur
                # No mutual overwrite: re-preparing ws2 leaves ws1 alone.
                @test SDPX.prepare_fixed_trace_q3_equality_schur!(
                    ws2, 2metric, regularization)
                @test ws1.schur == schur1
                # Children timers accumulate nonnegatively on real epochs.
                @test ws1.local_elimination_seconds >= 0.0
                @test ws1.panel_transform_seconds >= 0.0
                @test ws1.gram_seconds >= 0.0
            end
        end
    end
end

@testset "phase timing reset and snapshot" begin
    timings = SDPX.ProductHSDPhaseTimings()
    timings.q3_metric_seconds = 1.5
    timings.q3_workers = 3
    timings.q3_local_elimination_seconds = 0.5
    timings.q3_panel_transform_seconds = 0.25
    timings.q3_gram_seconds = 0.125
    SDPX.reset_phase_timings!(timings)
    @test timings.q3_metric_seconds == 0.0
    @test timings.q3_workers == 0
    @test timings.q3_local_elimination_seconds == 0.0
    @test timings.q3_panel_transform_seconds == 0.0
    @test timings.q3_gram_seconds == 0.0
    timings.q3_metric_seconds = 1.5
    timings.q3_workers = 3
    timings.q3_local_elimination_seconds = 0.5
    timings.q3_panel_transform_seconds = 0.25
    timings.q3_gram_seconds = 0.125
    snapshot = SDPX.phase_timings_snapshot(timings)
    @test snapshot.q3_worker_budget == 3
    @test snapshot.q3_workers == 3
    @test snapshot.q3_local_elimination_seconds == 0.5
    @test snapshot.q3_panel_transform_seconds == 0.25
    @test snapshot.q3_gram_seconds == 0.125
    children = snapshot.q3_local_elimination_seconds +
        snapshot.q3_panel_transform_seconds + snapshot.q3_gram_seconds
    @test children >= 0.0
end

@testset "solve-entry Q3 timer reset preserves factor metadata" begin
    problem = fixed_trace_problem(Float64)
    reduction = SDPX._fixed_trace_q3_reduction(problem)
    blocks = size(reduction.active_ids, 2)
    variables = 2blocks + length(reduction.free_ids)
    panel = zeros(Float64, 1, variables)
    panel[1] = 1.0
    panel[end] = 1.0
    equality = SDPX.FixedTraceQ3EqualitySchurWorkspace(reduction, panel; workers=2)
    metric = Float64[4 9; 1 1.5; 5 7][1:3, 1:blocks]
    @test SDPX.prepare_fixed_trace_q3_equality_schur!(
        equality, metric, sqrt(eps(Float64)))
    cache = SDPX.DenseSchurCholeskyCache{Float64}(1)
    residual = SDPX.NewtonResidual{Float64}(
        Float64[], Float64[], 0.0, Float64[], 0.0, Float64[])
    core = SDPX.FixedTraceQ3CoreWorkspace{Float64,Nothing,
        SDPX.DenseSchurCholeskyCache{Float64},
        typeof(equality),Nothing}(
        nothing, equality, nothing, cache,
        Array{Float64,3}(undef, 0, 0, 0), Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0),
        residual, 0.0, 0.0, 0.0, 0.0, :none,
        0, 0, 0, 0, 0, 0, 0, 0, 0, nothing, 0,
        Vector{Float64}(undef, 0), false, min(2, _POOL),
        SDPX.Q3EpochTimings(1.5, 2.5, 3.5, 7, 9))
    # Seed every solve-local timer with nonzero telemetry.
    equality.local_elimination_seconds = 4.5
    equality.panel_transform_seconds = 5.5
    equality.gram_seconds = 6.5
    factor_epoch = SDPX.factor_epoch(cache)
    matrix_epoch = SDPX.factor_matrix_epoch(cache)
    schur_before = copy(equality.schur)
    factors_before = copy(equality.local_elimination.factors)
    SDPX._reset_q3_phase_timings!(core)
    @test core.epoch_timing.metric_seconds == 0.0
    @test core.epoch_timing.factor_seconds == 0.0
    @test core.epoch_timing.homogeneous_seconds == 0.0
    @test core.epoch_timing.workers == 0
    @test core.epoch_timing.epochs == 0
    @test equality.local_elimination_seconds == 0.0
    @test equality.panel_transform_seconds == 0.0
    @test equality.gram_seconds == 0.0
    @test SDPX.factor_epoch(cache) == factor_epoch
    @test SDPX.factor_matrix_epoch(cache) == matrix_epoch
    @test equality.schur == schur_before
    @test equality.local_elimination.factors == factors_before
end

end
