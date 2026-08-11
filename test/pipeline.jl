using LinearAlgebra
using SparseArrays
using MathOptInterface
using MultiFloats: Float64x4
using SDPX
using StableRNGs
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
    @test result.diagnostics.plan.algorithm == :socp_psd2
    @test any(
        warning -> occursin("Lorentz-compatible 2x2 structure", warning),
        result.diagnostics.warnings,
    )
    @test PIPELINE_MOI.get(
        optimizer,
        PIPELINE_MOI.ConstraintPrimal(),
        index_map[cone],
    ) ≈ [1.0, 1.0] atol=1e-7
end

@testset "Automatic pipeline and dedicated LP path" begin
    @test SDPX._large_lattice_dense_schur_profile(
        6_119,
        394,
        32,
        0.0010204,
        0.842601,
    )
    @test !SDPX._large_lattice_dense_schur_profile(
        3_999,
        394,
        32,
        0.0010204,
        0.842601,
    )
    @test !SDPX._large_lattice_dense_schur_profile(
        6_119,
        394,
        32,
        0.01,
        0.842601,
    )

    problem = analytic_lp_problem(; duplicate_equality=true)
    classification = SDPX.classify_problem(problem)
    @test classification.cone == :lp
    @test SDPX.build_execution_plan(problem).algorithm == :lp_primal_dual
    @test SDPX.lp_initial_scale_indicator(problem) ≈ 2.0

    fast_parameters =
        SDPX.recommended_parameters(problem, SDPX.SolverOptions{Float64}())
    @test fast_parameters.profile == :lp_mehrotra_fast_start
    @test fast_parameters.β == 0.02
    @test fast_parameters.γ == 0.99

    distant_problem = SDPX.ingest(
        [1.0],
        [reshape([1.0], 1, 1, 1)],
        [fill(2_000.0, 1, 1)],
        zeros(1, 0),
        Float64[];
        sparse=true,
        verbosity=0,
    )
    @test distant_problem.cons isa SDPX.SparseCons{Float64}
    @test SDPX.lp_initial_scale_indicator(distant_problem) == 2_000.0
    conservative_parameters = SDPX.recommended_parameters(
        distant_problem,
        SDPX.SolverOptions{Float64}(),
    )
    @test conservative_parameters.profile ==
          :lp_mehrotra_conservative_start
    @test conservative_parameters.β == 0.1
    @test conservative_parameters.γ == 0.9
    distant_result = SDPX.solve!(
        distant_problem,
        SDPX.SolverOptions{Float64}(
            algorithm=:lp,
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            iter_max=150,
            verbosity=0,
        ),
    )
    @test distant_result.status == SDPX.Optimal
    @test distant_result.diagnostics.plan.parameter_profile ==
          :lp_mehrotra_conservative_start
    @test first(distant_result.parameter_history).beta == 0.1
    @test first(distant_result.parameter_history).gamma == 0.9
    setprecision(BigFloat, 256) do
        distant_bigfloat = SDPX.ingest(
            BigFloat[1],
            [reshape(BigFloat[1], 1, 1, 1)],
            [fill(BigFloat(2_000), 1, 1)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            verbosity=0,
        )
        bigfloat_parameters = SDPX.recommended_parameters(
            distant_bigfloat,
            SDPX.SolverOptions{BigFloat}(),
        )
        @test bigfloat_parameters.profile ==
              :lp_mehrotra_conservative_start
        @test bigfloat_parameters.β == BigFloat(1) / BigFloat(10)
        @test bigfloat_parameters.γ == BigFloat(9) / BigFloat(10)
    end

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
        row -> 0.02 <= row.beta <= 0.50 && 0.65 <= row.gamma <= 0.99,
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
function unbalanced_arrow_problem(;
    blocks::Int=4,
    shared::Int=2,
    constant_scale::Float64=2.0,
    sparse_mode=:auto,
    fix_shared::Bool=false,
)
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
    C = [
        Matrix{Float64}(
            (constant_scale * 10.0^(l - 1)) * I,
            2,
            2,
        )
        for l in 1:blocks
    ]
    B = fix_shared ?
        [Matrix{Float64}(I, shared, shared); zeros(blocks, shared)] :
        zeros(m, 0)
    b = fix_shared ? zeros(shared) : Float64[]
    prob = SDPX.ingest(ones(m), coefficients, C, B, b;
        sparse=sparse_mode, verbosity=0)
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
    # The public API now uses the guarded adaptive policy by default. A
    # cold-start safeguard preserves the fixed controller until normalized
    # feasibility and complementarity enter a reliable range.
    @test rp.parameter_strategy == :adaptive
    opted_out = SDPX.recommended_parameters(prob,
        SDPX.SolverOptions{Float64}(parameter_strategy=:fixed))
    @test opted_out.parameter_strategy == :fixed

    wide = unbalanced_arrow_problem(
        blocks=2,
        shared=144,
        constant_scale=0.55,
        sparse_mode=true,
    )
    wide_parameters = SDPX.recommended_parameters(
        wide,
        SDPX.SolverOptions{Float64}(),
    )
    @test wide_parameters.profile == :wide_arrow_2x2
    @test wide_parameters.β == 0.1
    @test wide_parameters.γ == 0.85
    @test wide_parameters.Ωp == 25.0
    @test wide_parameters.Ωd == 25.0

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

@testset "precision-limited stalls report InsufficientPrecision" begin
    # A Float64 solve asked for a tolerance below what Float64 can deliver must
    # say so, rather than returning a bare `Stalled` that gives the user nothing
    # to act on. This is the distinction the stagnation detector's
    # `:precision_floor` verdict exists to make.
    # Without these equalities the fixture has the recession direction
    # d_shared=(1,0,0), d_local=(-1,...,-1), whose objective is negative.
    # Pinning the shared variables makes this a bounded precision-floor test
    # rather than an accidental dual-infeasibility test.
    prob = unbalanced_arrow_problem(
        blocks=6,
        shared=3,
        fix_shared=true,
    )
    impossible = 1e-25          # far below eps(Float64)
    r = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(
        ϵ_gap=impossible, ϵ_primal=impossible, ϵ_dual=impossible,
        iter_max=150, verbosity=0, diagnostics=true))

    @test r.status != SDPX.Optimal          # must not claim success
    if r.termination.reason === :precision_floor
        @test r.status == SDPX.InsufficientPrecision
        @test occursin("precision floor", r.message)
    else
        # Whatever it was, it still may not be dressed up as optimal.
        @test r.status in (SDPX.Stalled, SDPX.IterLimit, SDPX.MaxRestartsExceeded,
                           SDPX.InsufficientPrecision, SDPX.NumericalBreakdown)
    end
end

@testset "no uncertified status maps to MOI.OPTIMAL" begin
    # Plan §5.2: a success code must never be produced for an internal state
    # that was not certified. Check the mapping exhaustively so a newly added
    # status cannot silently inherit `MOI.OPTIMAL`.
    certified = (SDPX.Optimal, SDPX.FeasibleCert)
    for status in instances(SDPX.SolveStatus)
        optimizer = SDPX.Optimizer()
        prob = analytic_lp_problem()
        r = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(verbosity=0))
        # Rebuild the result carrying the status under test.
        forced = SDPX.SDPResult{Float64}(
            status, "forced", r.x, r.X, r.y, r.Y, r.pObj, r.dObj, r.gap_rel,
            r.p_res, r.d_res, r.iterations, r.restarts, r.regularizations,
            r.timings, r.parameter_history, r.diagnostics)
        optimizer.result = forced
        mapped = PIPELINE_MOI.get(optimizer, PIPELINE_MOI.TerminationStatus())
        if status in certified
            @test mapped == PIPELINE_MOI.OPTIMAL
        else
            @test mapped != PIPELINE_MOI.OPTIMAL
        end
    end
end

@testset "equality presolve decides rank in the solve arithmetic" begin
    # Plan §5.4: presolve must not decide rank in narrower arithmetic than the
    # solve. Two equality columns differing by `delta` are genuinely dependent
    # at Float64 when delta < eps(Float64), and genuinely independent at
    # Float64x4. A downcast to Float64 anywhere in the rank decision would make
    # the extended types wrongly drop a column — silently changing the feasible
    # set — and that is exactly what this pins.
    function near_dependent(::Type{T}, delta) where {T}
        m = 4
        B = zeros(T, m, 2)
        B[1, 1] = one(T); B[2, 1] = one(T)
        B[1, 2] = one(T); B[2, 2] = one(T) + T(delta)
        A = [zeros(T, m, 2, 2)]
        for i in 1:m
            A[1][i, :, :] = Matrix{T}(I, 2, 2) .* T(i)
        end
        return SDPX.ingest(ones(T, m), A, [Matrix{T}(T(0.5) * I, 2, 2)],
            B, T[1, 1]; verbosity=0)
    end

    # Comfortably resolvable at every width: both columns are kept.
    for T in (Float64, BigFloat)
        prob = near_dependent(T, 1e-8)
        @test length(SDPX._equality_rank_indices(prob.B, 0)) == 2
    end

    # Below Float64's resolution: Float64 must call it dependent...
    narrow = near_dependent(Float64, 1e-20)
    @test length(SDPX._equality_rank_indices(narrow.B, 0)) == 1

    # ...while BigFloat, whose eps is ~1e-77, must not.
    setprecision(BigFloat, 256) do
        wide = near_dependent(BigFloat, 1e-20)
        @test length(SDPX._equality_rank_indices(wide.B, 0)) == 2
    end
end

@testset "workspace memory estimate is an upper bound" begin
    # Plan P0 / §19.3: the estimate is used as a memory budget, so it must never
    # come in under the real allocation. An estimate that is too small promises
    # a large high-precision solve will fit and then it does not.
    #
    # Before this was pinned, the estimate ran 1.05x-1.38x low for Float64 and
    # Float64x4 and 1.49x-1.71x low for BigFloat (whose per-element cost was
    # assumed to be 88 bytes at 256-bit precision against a measured 120).
    function block_problem(::Type{T}, L, m, k) where {T}
        rng = StableRNG(4)
        A = [zeros(T, m, k, k) for _ in 1:L]
        for l in 1:L, i in 1:m
            M = randn(rng, k, k)
            A[l][i, :, :] = T.(M + M')
        end
        C = [Matrix{T}(one(T) * I, k, k) for _ in 1:L]
        return SDPX.ingest(ones(T, m), A, C, zeros(T, m, 0), zeros(T, 0); verbosity=0)
    end

    for T in (Float64, BigFloat), (L, m, k) in ((2, 20, 5), (3, 40, 8)), threads in (1, 4)
        prob = block_problem(T, L, m, k)
        estimate = SDPX.estimate_sdp_workspace_bytes(prob, threads)
        SDPX.Workspace(prob)                      # warm up, so @allocated excludes compilation
        GC.gc()
        measured = minimum(@allocated(SDPX.Workspace(prob)) for _ in 1:3)
        @test estimate >= measured
    end
end

@testset "reported residuals match independent recomputation" begin
    # Plan §5.3: no scaled or reduced-system residual may be exposed as the
    # final public certificate. Whatever `SDPResult` advertises must be
    # reproducible from the original problem data, and `gap_rel` must be
    # reconstructible from the objectives reported alongside it.
    prob = unbalanced_arrow_problem(blocks=5, shared=3)
    opts = SDPX.SolverOptions{Float64}(verbosity=0, iter_max=200)
    r = SDPX.solve!(prob, opts)
    certificate = SDPX.result_certificate(prob, r, opts)

    # Public residuals include cone violations as well as affine residuals.
    # Final slack reconstruction can move a discrepancy from the affine term
    # to the equivalent PSD term, so compare the complete certificate metric.
    @test r.p_res ≈ certificate.primal_residual rtol = 1e-10
    @test r.d_res ≈ certificate.dual_residual rtol = 1e-10
    expected_gap = abs(r.pObj - r.dObj) / max(1, (abs(r.pObj) + abs(r.dObj)) / 2)
    @test r.gap_rel ≈ expected_gap rtol = 1e-10
end

@testset "crossover calibration cache" begin
    EPB = SDPX.ExtendedPrecisionBLAS
    mktempdir() do dir
        withenv("SDPX_CALIBRATION_DIR" => dir) do
            for family in (:fixed_extended, :bigfloat)
                static = EPB.static_profile(family)
                @test static.source === :static

                # With no cache present the static defaults must be used, so an
                # uncalibrated machine behaves exactly as before calibration existed.
                @test EPB.load_profile(family).source === :static
                @test EPB.load_profile(family).minimum_columns == static.minimum_columns

                # Round-trip a measured profile.
                measured = EPB.CalibrationProfile(
                    minimum_columns=11, minimum_work=1234.0, minimum_speedup=1.5,
                    minimum_schur_density=0.3, minimum_nnz_ratio=0.5,
                    source=:calibrated)
                path = EPB.save_profile(family, measured)
                @test path !== nothing && isfile(path)
                loaded = EPB.load_profile(family)
                @test loaded.source === :calibrated
                @test loaded.minimum_columns == 11
                @test loaded.minimum_work ≈ 1234.0
                @test loaded.minimum_nnz_ratio ≈ 0.5

                # A corrupt cache must never break a solve — it falls back silently.
                write(path, "this is not a calibration file\n@@@@\n")
                @test EPB.load_profile(family).source === :static

                # A *parseable* cache holding nonsense is the dangerous case:
                # it loads cleanly and every field lowers the bar for enabling
                # the packed kernel. NaN is the worst of them, because every
                # comparison against a NaN threshold is false, so a NaN minimum
                # speedup enables the kernel unconditionally. Each of these was
                # accepted before validation existed.
                for corrupt in (
                    "minimum_columns = -5\nminimum_speedup = 1.5\n",
                    "minimum_columns = 1\nminimum_speedup = 1.5\n",
                    "minimum_speedup = NaN\n",
                    "minimum_speedup = Inf\n",
                    "minimum_speedup = 0.1\n",
                    "minimum_speedup = 1.5\nminimum_work = -1.0\n",
                    "minimum_speedup = 1.5\nminimum_schur_density = 7.5\n",
                    "minimum_speedup = 1.5\nminimum_nnz_ratio = -3.0\n",
                    "minimum_speedup = 1.5\nminimum_schur_density = NaN\n",
                )
                    write(path, corrupt)
                    @test EPB.load_profile(family).source === :static
                end

                # The validator itself, stated directly.
                @test EPB.valid_profile(measured)
                @test !EPB.valid_profile(EPB.CalibrationProfile(
                    minimum_columns=2, minimum_work=0.0, minimum_speedup=NaN,
                    minimum_schur_density=0.5, minimum_nnz_ratio=0.5))
                @test EPB.valid_profile(EPB.CalibrationProfile(
                    minimum_columns=2, minimum_work=0.0, minimum_speedup=1.0,
                    minimum_schur_density=0.0, minimum_nnz_ratio=1.0))

                # A valid profile must still round-trip after all that.
                EPB.save_profile(family, measured)
                @test EPB.load_profile(family).source === :calibrated
            end
        end
    end

    # The signature must be stable within a run, or the cache would never hit.
    @test EPB.hardware_signature() == EPB.hardware_signature()
    @test !isempty(EPB.hardware_signature())
end

"""A small SDP with 2x2 blocks and an equality row, so it takes the dense
Schur/Workspace path rather than the dedicated LP solver."""
function unbalanced_block_sdp()
    m = 3
    A = [zeros(m, 2, 2)]
    for i in 1:m
        A[1][i, :, :] = Matrix{Float64}(I, 2, 2) .* i
    end
    C = [Matrix{Float64}(0.5I, 2, 2)]
    B = reshape([1.0, 0.0, 0.0], m, 1)
    return SDPX.ingest(ones(m), A, C, B, [1.0]; verbosity=0)
end

@testset "KKT backend abstraction" begin
    # Plan §15.1: the KKT path must be named in one place rather than rederived
    # from `if ws.arrow !== nothing` chains at each call site, and §10 requires
    # the selected plan be visible in diagnostics. The predicted
    # `plan.kkt_backend` is computed from problem structure before a workspace
    # exists; `select_backend` is the actual runtime choice. They are separate
    # code paths and can drift apart, so pin that they agree.
    function backends_for(prob)
        ws = SDPX.Workspace(prob)
        actual = SDPX.select_backend(ws)
        plan = SDPX.build_execution_plan(prob, SDPX.SolverOptions{Float64}())
        return (actual=actual, ws=ws, planned=plan.kkt_backend)
    end

    # An SDP model goes through the Schur/Workspace path.
    dense = unbalanced_block_sdp()
    d = backends_for(dense)
    @test d.actual isa SDPX.DenseCholeskyBackend
    @test SDPX.backend_name(d.actual) === :dense_cholesky
    @test SDPX.supports_equalities(d.actual)
    @test string(d.planned) == string(SDPX.backend_name(d.actual))

    # An all-scalar-cone model is dispatched to the dedicated LP solver, which
    # factorizes its own dense `K` and never builds an SDP `Workspace`. Its
    # backend is therefore selected separately, and the plan reports that one.
    lp = analytic_lp_problem()
    lp_plan = SDPX.build_execution_plan(lp, SDPX.SolverOptions{Float64}())
    lp_backend = SDPX.select_lp_backend(lp.dims.n)
    @test string(lp_plan.kkt_backend) == string(SDPX.backend_name(lp_backend))
    @test SDPX.backend_name(SDPX.select_lp_backend(0)) === :positive_definite_cholesky
    @test SDPX.backend_name(SDPX.select_lp_backend(3)) === :dense_lu

    arrow = unbalanced_arrow_problem(blocks=4, shared=2)
    a = backends_for(arrow)
    @test a.actual isa SDPX.ArrowBackend
    @test SDPX.backend_name(a.actual) === :block_arrow
    # The backend also supports equalities for the exactly block-diagonal
    # all-local specialization; ArrowWorkspace rejects incompatible
    # shared-variable/equality structures before backend selection.
    @test SDPX.supports_equalities(a.actual)
    @test string(a.planned) == string(SDPX.backend_name(a.actual))

    # `analyze` reports structure that is fixed for the whole solve, and is the
    # hook §15.2 symbolic reuse will attach to.
    info = SDPX.analyze(a.actual, arrow)
    @test info.backend === :block_arrow
    @test info.equalities == 0
    @test info.arrow_exact
    @test info.symbolic_reuse == false      # no sparse factorization backend yet

    # Counters that do not apply must read `nothing`, so "not applicable" is
    # distinguishable from "zero".
    @test SDPX.statistics(d.actual, d.ws).arrow_blocks === nothing
    @test SDPX.statistics(a.actual, a.ws).arrow_blocks == 4
end

@testset "worker reporting distinguishes cores from threads (§18.4)" begin
    # §18.4: "Do not describe oversubscribed workers as core scaling." That
    # requires keeping requested workers, effective workers, and actual physical
    # cores apart — a distinction `Threads.nthreads()` cannot make.
    physical = SDPX.physical_core_count()
    @test physical >= 1
    @test physical == SDPX.physical_core_count()       # stable within a run

    # A request the machine can actually satisfy is not oversubscription. Sized
    # from the detected core count rather than hard-coded: CI runners report as
    # few as one physical core, where asking for two genuinely *is*
    # oversubscription and the previous fixed `worker_report(2, 2)` was
    # asserting the opposite of the correct answer.
    modest = SDPX.worker_report(physical, physical)
    @test modest.requested_workers == physical
    @test modest.effective_workers == physical
    @test modest.physical_cores == physical
    @test !modest.oversubscribed

    # Asking for more workers than the machine has must be reported as
    # oversubscription, not as scaling headroom.
    heavy = SDPX.worker_report(4 * physical, 4 * physical)
    @test heavy.oversubscribed
    @test heavy.requested_workers == 4 * physical

    # Requested and effective differ whenever the plan caps the request.
    capped = SDPX.worker_report(64, 1)
    @test capped.requested_workers == 64
    @test capped.effective_workers == 1
    @test !capped.oversubscribed          # one worker cannot oversubscribe

    # Julia's visible core count is reported under its own name, because it can
    # be narrower than the hardware (affinity, JULIA_CPU_THREADS, containers)
    # and labelling it "logical cores" would be wrong.
    @test modest.julia_visible_cores == Sys.CPU_THREADS
    @test modest.logical_cores >= modest.physical_cores
end

@testset "mixed-precision :auto is gated, not unreachable (§16.4)" begin
    # An `:auto` path that can never fire is indistinguishable from `:off`, so
    # pin the boundary. `:auto` requires m >= MIXED_KKT_MINIMUM_AUTO_DIMENSION;
    # below that it declines, at and above it engages. `:on` is not subject to
    # the dimension gate (it still respects the memory and backend gates).
    function extended_problem(m)
        side, blocks = 4, 2
        rng = StableRNG(31)
        A = [zeros(BigFloat, m, side, side) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            M = randn(rng, side, side)
            A[l][i, :, :] = BigFloat.(M + M')
        end
        C = [Matrix{BigFloat}(one(BigFloat) * I, side, side) for _ in 1:blocks]
        return SDPX.ingest(ones(BigFloat, m), A, C,
            zeros(BigFloat, m, 0), BigFloat[]; verbosity=0)
    end

    threshold = SDPX.MIXED_KKT_MINIMUM_AUTO_DIMENSION
    @test threshold > 0

    setprecision(BigFloat, 256) do
        below = extended_problem(threshold - 1)
        below_plan = SDPX.build_execution_plan(
            below,
            SDPX.SolverOptions{BigFloat}(mixed_precision_kkt=:auto),
        )
        @test below_plan.backend_config.mixed_precision_mode === :off
        @test SDPX.planned_backend_name(below_plan) === :dense_cholesky
        @test below_plan.parameters.generic_mixed_precision_decision.reason ===
              :below_auto_dimension
        @test SDPX.Workspace(below; mixed_precision_kkt=:auto).mixed_precision === nothing
        # `:on` is not gated on dimension, so it must still be offered here.
        @test SDPX.Workspace(below; mixed_precision_kkt=:on).mixed_precision !== nothing

        at = extended_problem(threshold)
        at_plan = SDPX.build_execution_plan(
            at,
            SDPX.SolverOptions{BigFloat}(mixed_precision_kkt=:auto),
        )
        @test at_plan.backend_config.mixed_precision_mode === :auto
        @test SDPX.planned_backend_name(at_plan) === :mixed_precision
        at_workspace = SDPX.Workspace(at; execution_plan=at_plan)
        @test at_workspace.mixed_precision !== nothing
        @test SDPX.select_backend(at_workspace) isa SDPX.MixedPrecisionBackend

        fixed_plan = SDPX.build_execution_plan(
            at,
            SDPX.SolverOptions{BigFloat}(
                mixed_precision_kkt=:on,
                refine_policy=:fixed,
            ),
        )
        @test fixed_plan.backend_config.mixed_precision_mode === :off
        @test SDPX.planned_backend_name(fixed_plan) === :dense_cholesky
        @test fixed_plan.parameters.generic_mixed_precision_decision.reason ===
              :fixed_refinement_policy
        @test SDPX.Workspace(at; execution_plan=fixed_plan).mixed_precision ===
              nothing

        # `:off` declines regardless of size.
        @test SDPX.Workspace(at; mixed_precision_kkt=:off).mixed_precision === nothing
    end
end

@testset "null-space formulation (§12.2)" begin
    # x = x_particular + Z*z with B'Z = 0 parameterises the feasible set, so any
    # z gives a point satisfying the equalities exactly.
    function fixture(m, n; rank_deficient=false, seed=17)
        rng = StableRNG(seed)
        B = randn(rng, m, n)
        rank_deficient && n >= 2 && (B[:, n] .= B[:, 1])   # duplicate a column
        x_true = randn(rng, m)
        return B, transpose(B) * x_true
    end

    @testset "basis is orthogonal to the constraints" begin
        for (m, n) in ((60, 40), (100, 90))
            B, b = fixture(m, n)
            basis = SDPX.build_nullspace_basis(B, b)
            @test basis.rank == n
            @test basis.reduced_dimension == m - n
            @test basis.consistent
            r = SDPX.nullspace_residual(basis, B, b)
            # A basis not orthogonal to B would silently violate the equalities
            # in every reduced solve built on it.
            @test r.orthogonality < 1e-10
            @test r.feasibility < 1e-10
        end
    end

    @testset "any reduced point satisfies the equalities" begin
        B, b = fixture(60, 40)
        basis = SDPX.build_nullspace_basis(B, b)
        rng = StableRNG(5)
        for _ in 1:3
            z = randn(rng, basis.reduced_dimension)
            x = SDPX.recover_full_solution(basis, z)
            @test transpose(B) * x ≈ b rtol = 1e-8
        end
        @test_throws DimensionMismatch SDPX.recover_full_solution(basis, zeros(1))
    end

    @testset "rank deficiency is detected, not ignored" begin
        B, b = fixture(50, 20; rank_deficient=true)
        basis = SDPX.build_nullspace_basis(B, b)
        @test basis.rank == 19                     # one duplicated column
        @test basis.reduced_dimension == 50 - 19
        r = SDPX.nullspace_residual(basis, B, b)
        @test r.orthogonality < 1e-10
    end

    @testset "no equalities gives the identity basis" begin
        B = zeros(8, 0)
        basis = SDPX.build_nullspace_basis(B, Float64[])
        @test basis.reduced_dimension == 8
        @test basis.Z == Matrix{Float64}(I, 8, 8)
        @test basis.consistent
    end

    @testset "selector declines when the reduction is not worth it" begin
        # Few equalities: Z is nearly a dense m x m matrix and removes almost
        # nothing, so the formulation costs more than it saves.
        @test !SDPX.should_use_nullspace(variables=50, equalities=5)
        @test SDPX.should_use_nullspace(variables=100, equalities=90)
        @test !SDPX.should_use_nullspace(variables=100, equalities=0)
        # Never claim a reduction when the equalities outnumber the variables.
        @test !SDPX.should_use_nullspace(variables=10, equalities=10)
        # Memory estimate is available before Z is built, and gates the choice.
        # Without the rank it must be the worst case `m x m`, not `m x (m - n)`:
        # the equality count is an upper bound on the rank, so using it as the
        # rank under-reports whenever the rows are dependent. This assertion
        # previously encoded the unsafe formula.
        @test SDPX.nullspace_memory_bytes(1000, 900, Float64) == 8 * 1000 * 1000
        # Given the rank, it is exact.
        @test SDPX.nullspace_memory_bytes(1000, 900, Float64; rank=900) ==
              8 * 1000 * 100
        @test !SDPX.should_use_nullspace(variables=1000, equalities=900,
            memory_budget_bytes=1000)
    end
end

@testset "allocation ceilings (§19.4)" begin
    # Ceilings are set from measurement with headroom, not guessed. Their job is
    # to catch a regression that reintroduces per-iteration allocation into a
    # hot path — the kind of change that does not fail any correctness test but
    # quietly costs throughput and GC pressure.
    function arrow_problem(blocks, shared)
        m = shared + blocks
        rng = StableRNG(3)
        coeff = [Vector{SparseMatrixCSC{Float64,Int}}(undef, m) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            coeff[l][i] = (i <= shared || i == shared + l) ?
                sparse([1, 2, 2], [1, 1, 2], randn(rng, 3), 2, 2) : spzeros(2, 2)
        end
        C = [Matrix{Float64}(2.0I, 2, 2) for _ in 1:blocks]
        return SDPX.ingest(ones(m), coeff, C, zeros(m, 0), Float64[];
            sparse=:auto, verbosity=0)
    end

    function dense_problem(m, side, blocks)
        rng = StableRNG(4)
        A = [zeros(m, side, side) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            M = randn(rng, side, side)
            A[l][i, :, :] = M + M'
        end
        # Make the allocation fixture strictly feasible on both sides. The old
        # `c=1, C=I` random model was generally unbounded, entered restart
        # rescue at iteration eight, and charged cold restart workspace to the
        # "per-iteration" allocation gate.
        dual = Matrix{Float64}(1.0I, side, side)
        C = [-copy(dual) for _ in 1:blocks]
        objective = zeros(m)
        for l in 1:blocks, i in 1:m
            objective[i] += dot(A[l][i, :, :], dual)
        end
        return SDPX.ingest(
            objective,
            A,
            C,
            zeros(m, 0),
            Float64[];
            verbosity=0,
        )
    end

    @testset "Schur assembly is allocation-free once warmed" begin
        # Measured steady state: 0 B on the arrow path, 192 B dense.
        for (prob, ceiling) in ((arrow_problem(40, 8), 256), (dense_problem(40, 6, 2), 2048))
            ws = SDPX.Workspace(prob)
            X = [Matrix{Float64}(1.0I, k, k) for k in prob.dims.k]
            Y = [Matrix{Float64}(1.0I, k, k) for k in prob.dims.k]
            @test SDPX.factor_blocks!(ws, X, Y)
            SDPX.schur_build!(ws, prob, prob.cons, X, Y)         # warm up
            GC.gc()
            allocated = minimum(
                @allocated(SDPX.schur_build!(ws, prob, prob.cons, X, Y)) for _ in 1:3)
            @test allocated <= ceiling
        end
    end

    @testset "per-iteration allocation stays bounded" begin
        # Measured ~70 KB/iteration (arrow) and ~129 KB/iteration (dense) on
        # Julia 1.12. Julia 1.10 runs about 15% heavier on the dense case --
        # 402 KB/iteration against a 350 KB ceiling -- so the ceilings are set
        # from the highest supported version rather than the lowest, and still
        # sit far below the order-of-magnitude regression this is guarding
        # against.
        for (prob, ceiling) in ((arrow_problem(40, 8), 250_000),
                                (dense_problem(40, 6, 2), 500_000))
            o = SDPX.SolverOptions{Float64}(verbosity=0, iter_max=12)
            SDPX.solve!(prob, o)                                  # warm up
            GC.gc()
            # Thread pools and task-local scheduler state may initialize on
            # the first measured call even after method compilation is warm.
            # Use the same steady-state minimum-of-three protocol as the Schur
            # gate above so the ceiling measures solver allocation rather than
            # one-time runtime setup.
            totals = Int[]
            results = SDPX.SDPResult{Float64}[]
            for _ in 1:3
                result = Ref{SDPX.SDPResult{Float64}}()
                total = @allocated result[] = SDPX.solve!(prob, o)
                push!(totals, total)
                push!(results, result[])
            end
            best = argmin(totals)
            total = totals[best]
            result = results[best]
            @test result.iterations > 0
            @test total / result.iterations <= ceiling
        end
    end
end

@testset "certificate carries solve provenance (§20.3)" begin
    # §20.3 requires the certificate to record how the answer was produced, not
    # only what it is: a residual means something different depending on the
    # regularization applied, whether the factorization ran at reduced
    # precision, and the width the validation itself used.
    prob = unbalanced_arrow_problem(blocks=4, shared=2)
    opts = SDPX.SolverOptions{Float64}(verbosity=0, iter_max=200, diagnostics=true)
    result = SDPX.solve!(prob, opts)
    certificate = SDPX.result_certificate(prob, result, opts)

    prov = certificate.provenance
    @test prov.iterations == result.iterations
    @test prov.regularizations == result.regularizations
    @test prov.restarts == result.restarts
    @test prov.mixed_precision_used isa Bool

    # The validation width bounds how small a residual the certificate can
    # meaningfully claim. A 1e-30 residual "validated" in Float64 is a
    # statement about round-off, not about the solution.
    @test prov.validation_precision_bits == 52          # Float64 mantissa

    setprecision(BigFloat, 256) do
        wide = SDPX.ingest(BigFloat[2, 3],
            [reshape(BigFloat[1, 0, 0, 0, 0, 0, 0, 1], 2, 2, 2)],
            [BigFloat[0 1; 1 0]], reshape(BigFloat[1, 0], 2, 1), BigFloat[1];
            verbosity=0)
        wide_opts = SDPX.SolverOptions{BigFloat}(verbosity=0, diagnostics=true)
        wide_result = SDPX.solve!(wide, wide_opts)
        wide_certificate = SDPX.result_certificate(wide, wide_result, wide_opts)
        @test wide_certificate.provenance.validation_precision_bits == 256
    end
end

@testset "BigFloat staged working precision is conservative and certified" begin
    setprecision(BigFloat, 256) do
        problem = SDPX.ingest(
            BigFloat[1],
            [reshape(BigFloat[1], 1, 1, 1)],
            [fill(BigFloat(2), 1, 1)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=true,
            verbosity=0,
        )
        automatic_options = SDPX.SolverOptions{BigFloat}(
            ϵ_gap=big"1e-10",
            ϵ_primal=big"1e-10",
            ϵ_dual=big"1e-10",
            precision_bits=256,
            working_precision_policy=:auto,
            minimum_working_precision_bits=192,
            verbosity=0,
            diagnostics=true,
        )
        @test SDPX.adaptive_working_precision_bits(
            problem,
            automatic_options,
        ) == 192

        automatic = SDPX.solve!(problem, automatic_options)
        fixed = SDPX.solve!(
            problem,
            SDPX._replace_solver_options(
                automatic_options;
                working_precision_policy=:fixed,
            ),
        )
        @test automatic.status == SDPX.Optimal
        @test fixed.status == SDPX.Optimal
        @test isapprox(
            automatic.pObj,
            fixed.pObj;
            rtol=big"1e-40",
            atol=big"1e-40",
        )
        @test any(
            warning -> occursin(
                "Adaptive working precision selected 192",
                warning,
            ),
            automatic.diagnostics.warnings,
        )

        exact_options = SDPX._replace_solver_options(
            automatic_options;
            ϵ_gap=zero(BigFloat),
        )
        @test SDPX.adaptive_working_precision_bits(
            problem,
            exact_options,
        ) == 256
        @test_throws ArgumentError SDPX.solve!(
            problem,
            SDPX._replace_solver_options(
                automatic_options;
                working_precision_policy=:invalid,
            ),
        )
    end
end

@testset "solve_summary exposes the §21.3 contract" begin
    # §21.3 says "the exact field types may be refined, but the information
    # contract should remain stable". Pin the field names so a refactor cannot
    # quietly drop one — every field below is something a caller needs and would
    # otherwise have to reach into `diagnostics` or a separate certificate call
    # to obtain.
    prob = unbalanced_arrow_problem(blocks=4, shared=2)
    opts = SDPX.SolverOptions{Float64}(verbosity=0, iter_max=200, diagnostics=true)
    result = SDPX.solve!(prob, opts)
    summary = SDPX.solve_summary(prob, result, opts)

    @test hasproperty(result.termination, :total_refinement_steps)
    @test result.termination.total_refinement_steps >= 0
    callback_states = NamedTuple[]
    callback_result = SDPX.solve!(
        prob,
        SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-7,
            ϵ_primal=1e-7,
            ϵ_dual=1e-7,
            verbosity=0,
            callback=state -> begin
                push!(callback_states, state)
                false
            end,
        ),
    )
    @test callback_result.status != SDPX.UserStopped
    @test !isempty(callback_states)
    @test all(
        hasproperty(last(callback_states), field) for
        field in (:gap_rel, :complementarity, :termination_merit)
    )

    for field in (:status, :objective_value, :dual_objective_value,
                  :primal_solution, :dual_solution, :primal_residual,
                  :dual_residual, :relative_gap, :complementarity,
                  :psd_shift_lower_bound, :minimum_psd_eigenvalue,
                  :iterations, :solve_time,
                  :peak_memory_bytes, :selected_algorithms, :parameter_history,
                  :timings, :warnings, :certificate,
                  :infeasibility_diagnosis)
        @test hasproperty(summary, field)
    end

    # The summary must agree with the result it summarises, not recompute
    # something subtly different.
    @test summary.status === result.status
    @test summary.objective_value == result.pObj
    @test summary.dual_objective_value == result.dObj
    @test summary.relative_gap == result.gap_rel
    @test summary.iterations == result.iterations
    @test summary.primal_solution.x === result.x
    @test summary.dual_solution.Y === result.Y

    # A converged solve has non-negative complementarity and no PSD violation
    # (reported as the negated required diagonal shift, so zero means feasible).
    @test summary.complementarity >= 0
    # A shift lower bound is nonpositive by construction (zero when every
    # block passes unshifted) -- which is precisely why the old name
    # `minimum_psd_eigenvalue` was wrong: a PD solution's true minimum
    # eigenvalue is strictly positive. The alias must carry the same value.
    @test summary.psd_shift_lower_bound <= 0
    @test summary.minimum_psd_eigenvalue == summary.psd_shift_lower_bound
    @test summary.certificate.valid isa Bool
    @test summary.warnings isa Vector{String}

    # Works without diagnostics too, reporting `nothing` rather than throwing.
    bare = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(verbosity=0, diagnostics=false))
    bare_summary = SDPX.solve_summary(prob, bare,
        SDPX.SolverOptions{Float64}(verbosity=0, diagnostics=false))
    @test bare_summary.peak_memory_bytes === nothing
    @test bare_summary.selected_algorithms === nothing
    @test bare_summary.warnings == String[]
end

@testset "spectrum reports extraction cost separately (§22)" begin
    # §22 requires extraction time and memory to be reported separately from the
    # solve. Spectrum reconstruction is an eigendecomposition per block and can
    # cost more than an interior-point iteration, so folding it into the solve
    # timings would misattribute it.
    rng = StableRNG(41)
    m, blocks, side = 6, 3, 8
    A = [zeros(m, side, side) for _ in 1:blocks]
    for l in 1:blocks, i in 1:m
        M = randn(rng, side, side)
        A[l][i, :, :] = M + M'
    end
    C = [Matrix{Float64}(1.0I, side, side) for _ in 1:blocks]
    prob = SDPX.ingest(ones(m), A, C, zeros(m, 0), Float64[]; verbosity=0)
    result = SDPX.solve!(prob, SDPX.SolverOptions{Float64}(verbosity=0, iter_max=100))

    spectrum = SDPX.reconstruct_spectrum(result; allow_uncertified=true)
    metadata = spectrum.metadata

    @test hasproperty(metadata, :extraction_seconds)
    @test hasproperty(metadata, :extraction_bytes)
    @test hasproperty(metadata, :eigenvalues_extracted)
    @test metadata.extraction_seconds >= 0
    @test metadata.extraction_bytes >= 0
    # One eigenvalue per row of every block.
    @test metadata.eigenvalues_extracted == blocks * side
    @test length(spectrum) == metadata.eigenvalues_extracted

    # Extraction cost must not be conflated with the solve's own timings.
    @test !hasproperty(metadata, :total)
    # And the metadata still records the precision context §22 asks for.
    @test metadata.result_arithmetic == "Float64"
    @test metadata.requested_precision === :native
end

@testset "chordal structure detection (§8.3)" begin
    # Chordal decomposition replaces one k x k PSD constraint with one per
    # maximal clique. Detection has to answer three things: is the aggregate
    # sparsity pattern chordal, what are the cliques, and would splitting
    # actually be cheaper.
    function banded_problem(k, m; bandwidth=1)
        coefficients = [Vector{SparseMatrixCSC{Float64,Int}}(undef, m)]
        for i in 1:m
            rows, columns, values = Int[], Int[], Float64[]
            for r in 1:k, c in max(1, r - bandwidth):min(k, r + bandwidth)
                push!(rows, r); push!(columns, c)
                push!(values, r == c ? 2.0 : 0.3)
            end
            coefficients[1][i] = sparse(rows, columns, values, k, k)
        end
        return SDPX.ingest(ones(m), coefficients, [Matrix{Float64}(1.0I, k, k)],
            zeros(m, 0), Float64[]; sparse=true, verbosity=0)
    end

    function dense_problem(k, m)
        rng = StableRNG(51)
        A = [zeros(m, k, k)]
        for i in 1:m
            M = randn(rng, k, k)
            A[1][i, :, :] = M + M'
        end
        return SDPX.ingest(ones(m), A, [Matrix{Float64}(1.0I, k, k)],
            zeros(m, 0), Float64[]; verbosity=0)
    end

    @testset "banded patterns decompose into small cliques" begin
        analysis = SDPX.analyze_chordal_structure(banded_problem(40, 3; bandwidth=1), 1)
        @test analysis.dimension == 40
        @test analysis.chordal
        @test analysis.largest_clique == 2          # tridiagonal: cliques are edges
        @test length(analysis.cliques) == 39
        @test analysis.predicted_cost_ratio < 0.01
        @test analysis.beneficial

        wider = SDPX.analyze_chordal_structure(banded_problem(40, 3; bandwidth=3), 1)
        @test wider.chordal
        @test wider.largest_clique == 4
        @test wider.beneficial
        # A wider band means larger cliques and therefore less saving.
        @test wider.predicted_cost_ratio > analysis.predicted_cost_ratio
    end

    @testset "a dense block is chordal but not worth decomposing" begin
        # The case the `beneficial` flag exists for: a complete graph is
        # trivially chordal, so chordality alone must not trigger a
        # decomposition that splits one k^3 factorization into one k^3 clique.
        analysis = SDPX.analyze_chordal_structure(dense_problem(20, 3), 1)
        @test analysis.chordal
        @test length(analysis.cliques) == 1
        @test analysis.largest_clique == 20
        @test analysis.predicted_cost_ratio ≈ 1.0
        @test !analysis.beneficial
    end

    @testset "cliques cover the block and respect the sparsity" begin
        prob = banded_problem(20, 2; bandwidth=2)
        analysis = SDPX.analyze_chordal_structure(prob, 1)
        covered = reduce(union, analysis.cliques; init=Int[])
        @test sort(covered) == collect(1:20)        # every row appears
        neighbours = SDPX.aggregate_sparsity(prob, 1)
        # Every clique must be a clique in the aggregate graph, or the
        # decomposition it implies would be invalid.
        for clique in analysis.cliques, u in clique, v in clique
            u == v && continue
            @test v in neighbours[u]
        end
    end

    @testset "summary covers every block" begin
        @test length(SDPX.chordal_summary(banded_problem(12, 2))) == 1
    end
end

@testset "Schur accumulator capping is visible (§18.4, §19.3)" begin
    # Per-worker Schur accumulators are full m x m matrices, so their total
    # scales as threads * m^2 and a memory cap silently reduces the bin count.
    # On Task_Low08 (m = 6119), each replica costs about 286 MiB. The generic
    # 15% cap therefore limits parallelism on memory-constrained hosts. A
    # narrowly calibrated large-Float64 rule may use 25% when at least 16 GiB
    # is explicitly available; extended precision retains 15%. Section 18.4
    # requires such a selection change be reported, not inferred from timing.
    # A stated budget, not the host's free memory. One 6119-bin costs 285.7 MB
    # and one 200-bin costs 0.3 MB, and the cap allows 15% of the budget, so
    # 4 GiB affords eight of the small ones and only two of the large ones on
    # every machine. Reading the ambient figure instead made this testset
    # assert one thing on a laptop and the opposite on a 256 GB compute node,
    # where the cap correctly does not bind and all three "large" assertions
    # failed.
    budget = 4 * 1024^3
    @test SDPX._schur_accumulator_memory_fraction(
        Float64,
        6119,
        32,
        32,
        28 * 1024^3,
    ) == 0.25
    @test SDPX._schur_accumulator_memory_fraction(
        Float64,
        6119,
        32,
        32,
        15 * 1024^3,
    ) == 0.15
    @test SDPX._schur_accumulator_memory_fraction(
        Float64x4,
        6119,
        32,
        32,
        28 * 1024^3,
    ) == 0.15
    @test SDPX._schur_accumulator_memory_fraction(
        Float64,
        4095,
        32,
        32,
        28 * 1024^3,
    ) == 0.15
    task_large_budget = SDPX.schur_bin_report(
        Float64,
        6119,
        32,
        32;
        free_memory_bytes=28 * 1024^3,
    )
    @test task_large_budget.selected_bins == 25
    @test task_large_budget.capped
    @test task_large_budget.memory_fraction == 0.25
    @test task_large_budget.memory_budget_bytes == 7 * 1024^3
    task_small_budget = SDPX.schur_bin_report(
        Float64,
        6119,
        32,
        32;
        free_memory_bytes=15 * 1024^3,
    )
    @test task_small_budget.selected_bins == 8
    @test task_small_budget.capped
    @test task_small_budget.memory_fraction == 0.15
    @test task_small_budget.memory_budget_bytes == floor(Int, 0.15 * 15 * 1024^3)
    small = SDPX.schur_bin_report(Float64, 200, 32, 8; free_memory_bytes=budget)
    @test small.requested_bins == 8
    @test small.selected_bins == 8
    @test !small.capped
    @test small.total_bytes == small.would_have_been_bytes

    # A large m must cap, and must report both the actual and the avoided cost.
    large = SDPX.schur_bin_report(Float64, 6119, 32, 8; free_memory_bytes=budget)
    @test large.requested_bins == 8
    @test large.selected_bins < 8
    @test large.capped
    @test large.total_bytes < large.would_have_been_bytes
    @test large.bytes_per_bin == 6119^2 * 8
    @test large.total_bytes == large.selected_bins * large.bytes_per_bin

    # Bins never exceed the block count: more accumulators than blocks would be
    # pure waste.
    @test SDPX.schur_bin_report(Float64, 100, 3, 8).requested_bins == 3

    # Single-threaded never reports capping — there is nothing to reduce.
    @test !SDPX.schur_bin_report(Float64, 6119, 32, 1).capped

    # Wider arithmetic makes each accumulator larger, so it caps at least as
    # aggressively as Float64 at the same size.
    wide = SDPX.schur_bin_report(BigFloat, 2000, 32, 8)
    narrow = SDPX.schur_bin_report(Float64, 2000, 32, 8)
    @test wide.bytes_per_bin >= narrow.bytes_per_bin
    @test wide.selected_bins <= narrow.selected_bins

    @testset "dense workspace floor is a cheap lower bound" begin
        # Used as a pre-flight memory check, so two properties matter: it must
        # be below the full estimate (a floor that over-predicts would warn on
        # models that fit), and it must not walk the coefficient data.
        m, side, blocks = 80, 5, 3
        coefficients = [zeros(m, side, side) for _ in 1:blocks]
        for l in 1:blocks, i in 1:m
            coefficients[l][i, 1, 1] = float(i + l)
        end
        problem = SDPX.ingest(
            ones(m),
            coefficients,
            [Matrix{Float64}(1.0I, side, side) for _ in 1:blocks],
            zeros(m, 0),
            Float64[];
            verbosity=0,
        )
        floor_bytes = SDPX.dense_workspace_floor_bytes(
            Float64,
            problem.dims.m,
            problem.dims.n,
            problem.dims.L,
            1,
        )
        @test floor_bytes > 0
        @test floor_bytes <= SDPX.estimate_sdp_workspace_bytes(problem, 1)

        # O(1) in the problem data: same dimensions, same answer, whatever the
        # coefficients contain.
        @test floor_bytes == SDPX.dense_workspace_floor_bytes(
            Float64,
            problem.dims.m,
            problem.dims.n,
            problem.dims.L,
            1,
        )
        # Grows with the thread count, which is what makes it the right check
        # for a thread-driven memory blowup.
        @test SDPX.dense_workspace_floor_bytes(Float64, 500, 0, 8, 8) >
              SDPX.dense_workspace_floor_bytes(Float64, 500, 0, 8, 1)
        # Wider arithmetic needs more, not the same.
        @test SDPX.dense_workspace_floor_bytes(Float64x4, 500, 0, 8, 1) >
              SDPX.dense_workspace_floor_bytes(Float64, 500, 0, 8, 1)

        # Overflow saturates instead of wrapping. A wrapped estimate is
        # negative, compares as smaller than every budget, and approves
        # exactly the allocation the pre-flight exists to refuse -- measured
        # before the fix, m = 4e9 returned -6763251095801167872. These
        # dimensions are never allocated; only the arithmetic is exercised.
        for (m, n) in ((4_000_000_000, 0), (2_000_000_000, 1_000_000))
            floor_bytes = SDPX.dense_workspace_floor_bytes(Float64, m, n, 32, 8)
            @test floor_bytes > 0
            @test floor_bytes == typemax(Int)
        end
        @test SDPX.saturating_sum_bytes(typemax(Int), 1) == typemax(Int)
        @test SDPX.saturating_sum_bytes(3, 4) == 7

    end

    @testset "memory estimates follow the KKT route, not a dense upper bound" begin
        # The block-arrow route never forms the dense m x m Schur complement,
        # so the dense floor does not describe it. On the CSDR 200/2/10/400
        # model (m = 40,453 with 53 shared variables) the dense figure is
        # 3,218 GiB for a solve that runs in about 5 GiB -- a warning wrong by
        # three orders of magnitude, which drives users off runs that fit.
        blocks, shared = 2000, 53
        variables = shared + blocks
        coefficients = Vector{Vector{SparseMatrixCSC{Float64,Int}}}(undef, blocks)
        for block in 1:blocks
            slots = [spzeros(2, 2) for _ in 1:variables]
            for s in 1:shared
                slots[s] = sparse([1, 2], [1, 2], [1.0 / (s + block), 0.5], 2, 2)
            end
            # One local variable per block: the arrow structure.
            slots[shared + block] = sparse([1], [1], [1.0], 2, 2)
            coefficients[block] = slots
        end
        constants = [Matrix(sparse([1, 2], [1, 2], [-1.0, -1.0], 2, 2))
                     for _ in 1:blocks]
        problem = SDPX.ingest(ones(variables), coefficients, constants,
            zeros(variables, 0), Float64[]; sparse=true, verbosity=0)

        arrow = SDPX.arrow_workspace_floor_bytes(Float64, problem, 8)
        dense = SDPX.dense_workspace_floor_bytes(Float64, variables, 0, blocks, 8)
        @test arrow > 0                      # the decomposition was recognised
        @test arrow < dense ÷ 50             # measured ratio here is ~168x
        # It scales with the shared dimension, not with m: doubling the block
        # count must not square the estimate the way the dense figure does.
        @test arrow < 16 * variables * shared * sizeof(Float64)

        # The all-local equality-arrow route stores an m-by-n transformed
        # panel and two n-by-n Gram buffers. It must return a real lower bound,
        # not the historical zero that made memory preflight vacuous.
        local_blocks = 4
        local_variables = 2local_blocks
        local_coefficients = Vector{Vector{SparseMatrixCSC{Float64,Int}}}(
            undef,
            local_blocks,
        )
        for block in 1:local_blocks
            slots = [spzeros(2, 2) for _ in 1:local_variables]
            first_variable = 2block - 1
            slots[first_variable] = sparse([1, 2], [1, 2], [1.0, -1.0], 2, 2)
            slots[first_variable + 1] = sparse([1, 2], [2, 1], [1.0, 1.0], 2, 2)
            local_coefficients[block] = slots
        end
        equality_problem = SDPX.ingest(
            ones(local_variables),
            local_coefficients,
            [Matrix{Float64}(-I, 2, 2) for _ in 1:local_blocks],
            ones(local_variables, 2),
            zeros(2);
            sparse=true,
            verbosity=0,
        )
        equality_arrow = SDPX.arrow_workspace_floor_bytes(
            Float64,
            equality_problem,
            4,
        )
        @test equality_arrow >= local_variables * 2 * sizeof(Float64)

        # A problem with no arrow structure gets no arrow estimate, and the
        # caller must read 0 as "no estimate" rather than "needs nothing".
        dense_problem = SDPX.ingest(
            ones(4),
            [zeros(4, 2, 2)],
            [Matrix{Float64}(1.0I, 2, 2)],
            zeros(4, 0), Float64[]; sparse=false, verbosity=0)
        @test SDPX.arrow_workspace_floor_bytes(Float64, dense_problem, 4) == 0
    end

end
