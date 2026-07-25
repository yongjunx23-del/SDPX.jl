using LinearAlgebra
using SparseArrays
using MathOptInterface
using SDPX
using Test

const PIPELINE_MOI = MathOptInterface

function analytic_lp_problem(; duplicate_equality::Bool=false)
    c = [1.0, 1.0]
    A = [
        reshape([1.0, 0.0], 2, 1, 1),
        reshape([0.0, 1.0], 2, 1, 1),
        reshape([0.0, 2.0], 2, 1, 1),
    ]
    C = [fill(0.0, 1, 1), fill(2.0, 1, 1), fill(3.0, 1, 1)]
    B = duplicate_equality ? [1.0 2.0; 0.0 0.0] : reshape([1.0, 0.0], 2, 1)
    b = duplicate_equality ? [1.0, 2.0] : [1.0]
    return SDPX.ingest(c, A, C, B, b; verbosity=0)
end

@testset "SOCP classification through MOI" begin
    model = PIPELINE_MOI.Utilities.Model{Float64}()
    variables = PIPELINE_MOI.add_variables(model, 2)
    cone = PIPELINE_MOI.add_constraint(
        model,
        PIPELINE_MOI.VectorOfVariables(variables),
        PIPELINE_MOI.SecondOrderCone(2),
    )
    PIPELINE_MOI.add_constraint(
        model,
        variables[2],
        PIPELINE_MOI.EqualTo(1.0),
    )
    PIPELINE_MOI.set(
        model,
        PIPELINE_MOI.ObjectiveFunction{PIPELINE_MOI.VariableIndex}(),
        variables[1],
    )
    PIPELINE_MOI.set(
        model,
        PIPELINE_MOI.ObjectiveSense(),
        PIPELINE_MOI.MIN_SENSE,
    )
    optimizer = SDPX.Optimizer()
    PIPELINE_MOI.set(optimizer, PIPELINE_MOI.Silent(), true)
    index_map = PIPELINE_MOI.copy_to(optimizer, model)
    @test SDPX.classify_problem(optimizer.problem).cone == :socp
    PIPELINE_MOI.optimize!(optimizer)
    result = PIPELINE_MOI.get(optimizer, PIPELINE_MOI.RawSolver())
    @test result.status == SDPX.Optimal
    @test result.pObj ≈ 1.0 atol=1e-7
    @test result.diagnostics.plan.algorithm == :socp_psd_lift
    @test any(
        warning -> occursin("SOC-arrow PSD structure", warning),
        result.diagnostics.warnings,
    )
    @test PIPELINE_MOI.get(
        optimizer,
        PIPELINE_MOI.ConstraintPrimal(),
        index_map[cone],
    ) ≈ [1.0, 1.0] atol=1e-7
end

@testset "Automatic pipeline and dedicated LP path" begin
    problem = analytic_lp_problem(; duplicate_equality=true)
    classification = SDPX.classify_problem(problem)
    @test classification.cone == :lp
    @test SDPX.build_execution_plan(problem).algorithm == :lp_primal_dual

    fixed = SDPX.solve(
        problem;
        tolerance=1e-9,
        verbosity=0,
        parameter_strategy=:fixed,
    )
    adaptive = SDPX.solve(
        problem;
        tolerance=1e-9,
        verbosity=0,
        parameter_strategy=:adaptive,
    )
    for result in (fixed, adaptive)
        @test result.status == SDPX.Optimal
        @test result.pObj ≈ 3.0 atol=1e-7
        @test result.x ≈ [1.0, 2.0] atol=1e-7
        @test result.p_res <= 1e-7
        @test result.d_res <= 1e-7
        @test result.diagnostics.plan.algorithm == :lp_primal_dual
        @test result.diagnostics.presolve.removed_dependent_equalities == 1
        @test length(result.y) == 2
        @test length(result.parameter_history) == result.iterations
    end
    @test all(
        row -> 0.02 <= row.beta <= 0.50 && 0.65 <= row.gamma <= 0.95,
        adaptive.parameter_history,
    )

    controller = SDPX.AdaptiveIPMController(
        SDPX.SolverOptions{Float64}(
            β=0.1,
            γ=0.8,
            parameter_strategy=:adaptive,
        ),
    )
    for iteration in 6:7
        SDPX.record_and_update!(
            controller;
            iteration,
            predictor_quality=0.3,
            complementarity_before=100.0,
            complementarity_after=101.0,
            primal_residual=1.0,
            dual_residual=1.0,
            primal_step=1.0,
            dual_step=1e-3,
            backtracking_count=0,
        )
    end
    @test controller.fallback
    @test controller.beta == controller.default_beta
    @test controller.gamma == controller.default_gamma
    @test last(controller.history).fallback_reason == :stalled_progress

    setprecision(BigFloat, 256) do
        @test SDPX.estimate_backtracking_count(
            big"1e-1000",
            big"0.9",
            :backtrack,
        ) > 20_000
        near_one = one(BigFloat) - big"1e-70"
        @test SDPX.estimate_backtracking_count(
            big"0.5",
            near_one,
            :backtrack,
        ) == typemax(Int)
    end
    setprecision(BigFloat, 64) do
        diagnostic = SDPX._tolerance_precision_diagnostic(
            BigFloat,
            big"1e-100",
        )
        @test diagnostic.warn
        @test diagnostic.needed_bits >= 332
    end
    setprecision(BigFloat, 2048) do
        diagnostic = SDPX._tolerance_precision_diagnostic(
            BigFloat,
            big"1e-1000",
        )
        @test diagnostic.warn
        @test diagnostic.needed_bits >= 3_321
    end

    timed_out = SDPX.solve(
        problem;
        time_limit=0.0,
        verbosity=0,
    )
    @test timed_out.status == SDPX.TimeLimit
    @test timed_out.iterations == 0
    @test timed_out.diagnostics.termination.reason == :time_limit
    @test timed_out.diagnostics.timings.pipeline >= 0.0

    inconsistent = analytic_lp_problem(; duplicate_equality=true)
    inconsistent = SDPX.SDPProblem{Float64}(
        inconsistent.c,
        inconsistent.C,
        inconsistent.B,
        [1.0, 3.0],
        inconsistent.cons,
        inconsistent.dims,
        inconsistent.structure,
    )
    infeasible = SDPX.solve(inconsistent; verbosity=0)
    @test infeasible.status == SDPX.InfeasibleCert
    @test infeasible.iterations == 0

    mktempdir() do directory
        csv_path = joinpath(directory, "spectrum.csv")
        json_path = joinpath(directory, "spectrum.json")
        SDPX.export_spectrum(csv_path, adaptive)
        SDPX.export_spectrum(json_path, adaptive)
        @test startswith(read(csv_path, String), "source,block")
        @test startswith(read(json_path, String), "[")
    end
end

@testset "MOI scalar inequalities use the LP engine" begin
    model = PIPELINE_MOI.Utilities.Model{Float64}()
    variables = PIPELINE_MOI.add_variables(model, 2)
    lower_x = PIPELINE_MOI.add_constraint(
        model,
        variables[1],
        PIPELINE_MOI.GreaterThan(1.0),
    )
    PIPELINE_MOI.add_constraint(
        model,
        variables[2],
        PIPELINE_MOI.GreaterThan(2.0),
    )
    objective = PIPELINE_MOI.ScalarAffineFunction(
        [
            PIPELINE_MOI.ScalarAffineTerm(1.0, variables[1]),
            PIPELINE_MOI.ScalarAffineTerm(1.0, variables[2]),
        ],
        0.0,
    )
    PIPELINE_MOI.set(
        model,
        PIPELINE_MOI.ObjectiveFunction{typeof(objective)}(),
        objective,
    )
    PIPELINE_MOI.set(
        model,
        PIPELINE_MOI.ObjectiveSense(),
        PIPELINE_MOI.MIN_SENSE,
    )
    optimizer = SDPX.Optimizer()
    PIPELINE_MOI.set(optimizer, PIPELINE_MOI.Silent(), true)
    index_map = PIPELINE_MOI.copy_to(optimizer, model)
    PIPELINE_MOI.optimize!(optimizer)
    @test PIPELINE_MOI.get(optimizer, PIPELINE_MOI.TerminationStatus()) ==
          PIPELINE_MOI.OPTIMAL
    @test PIPELINE_MOI.get(optimizer, PIPELINE_MOI.ObjectiveValue()) ≈ 3.0 atol=1e-7
    @test PIPELINE_MOI.get(
        optimizer,
        PIPELINE_MOI.ConstraintPrimal(),
        index_map[lower_x],
    ) ≈ 1.0 atol=1e-7
    raw = PIPELINE_MOI.get(optimizer, PIPELINE_MOI.RawSolver())
    @test raw.diagnostics.plan.algorithm == :lp_primal_dual
end

"""A sparse arrow-structured model of 2x2 blocks — the shape
`recommended_parameters` has a dedicated profile for — with block constant terms
spanning four orders of magnitude so the initial-point rules have a real spread
to react to."""
function unbalanced_arrow_problem(; blocks::Int=4, shared::Int=2)
    m = shared + blocks                      # shared variables plus one local each
    coefficients = [Vector{SparseMatrixCSC{Float64,Int}}(undef, m) for _ in 1:blocks]
    for l in 1:blocks, i in 1:m
        # Shared variables act on every block; local variable `shared + l` acts
        # only on block `l`.
        active = i <= shared || i == shared + l
        coefficients[l][i] = active ?
                             sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2) :
                             spzeros(2, 2)
    end
    C = [Matrix{Float64}((2.0 * 10.0^(l - 1)) * I, 2, 2) for l in 1:blocks]
    prob = SDPX.ingest(ones(m), coefficients, C, zeros(m, 0), Float64[];
        sparse=:auto, verbosity=0)
    @assert prob.cons isa SDPX.SparseCons{Float64}
    return prob
end

@testset "SDP warm-start validation and end-to-end time limit" begin
    problem = unbalanced_arrow_problem(blocks=4)
    options = SDPX.SolverOptions{Float64}(
        algorithm=:sdp,
        parameter_policy=:fixed,
        max_time=0.0,
        verbosity=0,
    )
    native = warm -> SDPX.solve!(problem, options; warm...)
    simple = warm -> SDPX.solve(
        problem;
        algorithm=:sdp,
        time_limit=0.0,
        verbosity=0,
        warm_start=warm,
    )
    valid_blocks = [Matrix{Float64}(I, 2, 2) for _ in 1:problem.dims.L]

    for invoke in (native, simple)
        @test_throws ArgumentError invoke((; X0=valid_blocks))
        @test_throws ArgumentError invoke((; Y0=valid_blocks))
        @test_throws DimensionMismatch invoke((; x0=zeros(problem.dims.m + 1)))
        @test_throws DimensionMismatch invoke((; y0=[0.0]))
        @test_throws DimensionMismatch invoke(
            (; X0=valid_blocks[1:1], Y0=valid_blocks),
        )
        wrong_shape = [ones(1, 1), valid_blocks[2]]
        @test_throws DimensionMismatch invoke(
            (; X0=wrong_shape, Y0=valid_blocks),
        )
    end

    timed_out = native((;))
    @test timed_out.status == SDPX.TimeLimit
    @test timed_out.iterations == 0
    @test timed_out.diagnostics.termination.reason == :time_limit
    @test timed_out.diagnostics.timings.pipeline >= 0.0
end

@testset "initial-point scaling tracks the block data" begin
    prob = unbalanced_arrow_problem(blocks=4)

    stats = SDPX.block_norm_stats(prob)
    @test stats.maxnorm ≈ 2.0e3                      # 2·10^3 on the last block
    @test stats.spread ≈ 1.0e3
    @test stats.gmean ≈ 2.0 * 10.0^1.5

    # Ω is driven by the data rather than left at the old fixed default of 10,
    # which is what made the CSDR model fail to converge at all.
    rp = SDPX.recommended_parameters(prob, SDPX.SolverOptions{Float64}())
    @test rp.Ωp ≈ SDPX.OMEGA_DATA_MULTIPLIER * stats.maxnorm
    @test rp.Ωd == rp.Ωp
    # No profile turns the adaptive beta/gamma controller on by itself: the
    # measurement that once justified it was taken while the solve terminated
    # prematurely, and it reverses once that is fixed. The user's setting is
    # passed through untouched.
    @test rp.parameter_strategy == :fixed
    opted_in = SDPX.recommended_parameters(prob,
        SDPX.SolverOptions{Float64}(parameter_strategy=:adaptive))
    @test opted_in.parameter_strategy == :adaptive

    ones4 = ones(4)
    # `:auto` must agree with `:scalar`: per-block is opt-in, because it was
    # measured worse than a data-scaled scalar Ω on the CSDR model.
    @test SDPX.initial_block_scales(prob, SDPX.SolverOptions{Float64}(omega_scaling=:scalar)) == ones4
    @test SDPX.initial_block_scales(prob, SDPX.SolverOptions{Float64}(omega_scaling=:auto)) == ones4

    scales = SDPX.initial_block_scales(prob, SDPX.SolverOptions{Float64}(omega_scaling=:per_block))
    @test scales[4] / scales[1] ≈ 1.0e3          # ratio follows ‖C_l‖∞
    @test prod(scales) ≈ 1.0 atol = 1e-8         # normalised by the geometric mean

    @test_throws ArgumentError SDPX.initial_block_scales(
        prob, SDPX.SolverOptions{Float64}(omega_scaling=:nonsense))
end

@testset "adaptive refinement matches fixed refinement on the answer" begin
    prob = analytic_lp_problem()
    base = (ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10, verbosity=0, iter_max=200)
    r_fixed = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(; refine_policy=:fixed, base...))
    r_auto = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(; refine_policy=:auto, base...))
    @test r_fixed.status == SDPX.Optimal
    @test r_auto.status == SDPX.Optimal
    # The adaptive rule may take a different number of refinement passes, but
    # it must not change the solution it converges to.
    @test r_auto.pObj ≈ r_fixed.pObj rtol = 1e-8

    @test_throws ArgumentError SDPX.solve!(
        prob, SDPX.SolverOptions{Float64}(; refine_policy=:nonsense, base...))
end

@testset "stagnation detector" begin
    T = Float64

    @testset "slow but steady progress is not a stall" begin
        # The regression this exists to prevent: the old rule required a
        # `stall_tolerance` improvement on EVERY iteration and stopped after 15
        # consecutive misses, which killed the CSDR sparse solve at iteration 27
        # while it was still converging at ~0.05%/iteration. A detector must not
        # flag a sequence that is genuinely descending toward the tolerance.
        d = SDPX.StagnationDetector{T}(15)
        merit = 1e3
        stopped_at = 0
        for i in 1:120
            merit *= 0.94                      # steady ~6%/iteration
            if SDPX.observe!(d, merit, false, 400 - i)
                stopped_at = i
                break
            end
        end
        @test stopped_at == 0                  # never flagged
        @test merit < 1.0                      # and it did reach tolerance
    end

    @testset "a genuine plateau is flagged" begin
        d = SDPX.StagnationDetector{T}(10)
        stopped_at = 0
        for i in 1:60
            if SDPX.observe!(d, 55.0, false, 400 - i)   # no movement at all
                stopped_at = i
                break
            end
        end
        @test stopped_at > 0
        @test d.reason === :no_progress
    end

    @testset "a plateau at the precision floor is reported as such" begin
        d = SDPX.StagnationDetector{T}(10)
        stopped_at = 0
        for i in 1:60
            if SDPX.observe!(d, 55.0, true, 400 - i)
                stopped_at = i
                break
            end
        end
        @test stopped_at > 0
        @test d.reason === :precision_floor
        @test occursin("precision floor", SDPX.stagnation_message(d, 1e-12))
    end

    @testset "progress too slow for the remaining budget is flagged" begin
        # Descending, but far too slowly to reach tolerance in what is left.
        d = SDPX.StagnationDetector{T}(10)
        merit = 1e9
        stopped_at = 0
        for i in 1:60
            merit *= 0.999
            if SDPX.observe!(d, merit, false, 12 - i)
                stopped_at = i
                break
            end
        end
        @test stopped_at > 0
        @test d.reason === :too_slow
        @test d.rate > 0                       # it *was* improving
        @test isfinite(d.projected)
    end

    @testset "a marginal projection does not cut off a converging solve" begin
        # Regression: with `PROJECTION_SLACK = 1` the rule stopped as soon as
        # the projection merely exceeded the remaining budget. On the CSDR
        # sparse model that ended the solve at iteration 34 (projected ~373 vs
        # 366 remaining) at gap 7.1e-3, where letting it run reaches 8.4e-6.
        # A projection of the same order as the budget must not stop anything.
        d = SDPX.StagnationDetector{T}(15)
        merit = 1e5
        rate = 0.03                    # nats/iteration: ~384 iterations to go
        stopped_at = 0
        for i in 1:60
            merit *= exp(-rate)
            if SDPX.observe!(d, merit, false, 366 - i)
                stopped_at = i
                break
            end
        end
        @test stopped_at == 0
    end

    @testset "merit is normalised by the requested tolerances" begin
        d = SDPX.StagnationDetector{T}(15)
        tight = SDPX.SolverOptions{T}(ϵ_gap=1e-10, ϵ_primal=1e-10, ϵ_dual=1e-10)
        loose = SDPX.SolverOptions{T}(ϵ_gap=1e-4, ϵ_primal=1e-4, ϵ_dual=1e-4)
        args = (1e-8, 1e-8, 1e-8, 0.0, 1.0, 1.0, 1.0)
        m_tight = SDPX.stagnation_merit(d, tight, args...)
        m_loose = SDPX.stagnation_merit(d, loose, args...)
        # Same residuals: unconverged against the tight ask, converged against
        # the loose one. `merit <= 1` is exactly the convergence statement.
        @test m_tight > 1
        @test m_loose < 1
        @test m_tight ≈ m_loose * 1e6 rtol = 1e-8

        # Complementarity participates rather than being ignored.
        with_comp = SDPX.stagnation_merit(d, loose, 1e-8, 1e-8, 1e-8, 1e3, 1.0, 1.0, 1.0)
        @test with_comp > m_loose
    end

    @testset "window must fill before any verdict" begin
        d = SDPX.StagnationDetector{T}(15)
        for i in 1:15
            @test SDPX.observe!(d, 42.0, false, 100) == false
        end
        @test SDPX.observe!(d, 42.0, false, 100) == true
    end

    @testset "a zero window disables detection entirely" begin
        d = SDPX.StagnationDetector{T}(0)
        for _ in 1:50
            @test SDPX.observe!(d, 42.0, false, 100) == false
        end
        @test d.reason === :none
    end
end
