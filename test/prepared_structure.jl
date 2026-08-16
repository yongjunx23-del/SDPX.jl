using LinearAlgebra
using SparseArrays
using SDPX
using Test

function _prepared_lp(
    objective::AbstractVector{T};
    rhs::AbstractVector{T}=T[1],
    dependent::Bool=false,
) where {T}
    G = Matrix{T}(I, 2, 2)
    Aeq = dependent ? T[1 1; 2 2] : reshape(T[1, 1], 1, 2)
    beq = dependent ? T[rhs[1], rhs[1] + rhs[1]] : collect(rhs)
    return SDPX.linear_program(
        objective,
        G,
        zeros(T, 2);
        Aeq=Aeq,
        beq=beq,
        T=T,
        verbosity=0,
    )
end

function _problem_with(
    problem::SDPX.SDPProblem{T};
    c=problem.c,
    C=problem.C,
    B=problem.B,
    b=problem.b,
    cons=problem.cons,
    dims=problem.dims,
    structure=problem.structure,
) where {T}
    return SDPX.SDPProblem{T}(c, C, B, b, cons, dims, structure)
end

@testset "PreparedStructure and SolveState" begin
    options = SDPX.SolverOptions{Float64}(
        scaling=:none,
        presolve=true,
        verbosity=0,
    )

    @testset "objective and RHS reuse match cold solves" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = SDPX.prepare(problem, options)
        fingerprint = prepared.structure.fingerprint
        template = prepared.structure.preprocessed_template

        result = SDPX.solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        cold_problem = _prepared_lp([2.0, 1.0]; rhs=[2.0])
        cold = SDPX.solve!(cold_problem, options)

        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8
        @test result.dObj ≈ cold.dObj atol=1e-8
        @test result.x ≈ cold.x atol=1e-8
        @test prepared.structure.fingerprint === fingerprint
        @test prepared.structure.preprocessed_template === template
        @test prepared.state.structure_reuses == 1
        @test prepared.state.numeric_generation == 1
        @test prepared.state.last_reuse === :structure_reused_numeric_state_fresh
        @test prepared.state.last_reduced_objective == [2.0, 1.0]
        @test prepared.state.last_reduced_rhs == [2.0]
        @test result.diagnostics.presolve.elapsed == 0.0
        @test result.diagnostics.presolve.preprocessing.elapsed == 0.0
        @test !(:workspace in fieldnames(typeof(prepared.structure)))
        @test !(:workspace in fieldnames(typeof(prepared.state)))
    end

    @testset "cached execution plan reuse matches cold planning" begin
        reuse_options = SDPX.SolverOptions{Float64}(
            scaling=:none,
            presolve=true,
            parameter_policy=:fixed,
            timing=true,
            verbosity=0,
        )
        problem = _prepared_lp([1.0, 2.0])
        prepared = SDPX.prepare(problem, reuse_options)
        @test prepared.structure.execution_plan isa SDPX.ExecutionPlan

        result = SDPX.solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        cold_problem = _prepared_lp([2.0, 1.0]; rhs=[2.0])
        cold = SDPX.solve!(cold_problem, reuse_options)

        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8
        @test result.dObj ≈ cold.dObj atol=1e-8
        @test result.x ≈ cold.x atol=1e-8
        @test SDPX.result_certificate(cold_problem, result, reuse_options).valid
        @test SDPX.result_certificate(cold_problem, cold, reuse_options).valid

        prepared_plan = result.diagnostics.plan
        cold_plan = cold.diagnostics.plan
        @test prepared_plan !== prepared.structure.execution_plan
        @test prepared_plan.payload isa SDPX.LPRoutePlan
        @test prepared.structure.execution_plan.payload === nothing
        @test prepared_plan.algorithm == cold_plan.algorithm
        @test prepared_plan.scaling == cold_plan.scaling
        @test prepared_plan.kkt_backend == cold_plan.kkt_backend
        @test prepared_plan.formulation_plan == cold_plan.formulation_plan
        @test prepared_plan.la_config == cold_plan.la_config
        @test prepared_plan.gram_kernel == cold_plan.gram_kernel
        @test prepared_plan.schedule == cold_plan.schedule
        @test prepared_plan.threads == cold_plan.threads
        @test prepared_plan.parameter_profile == cold_plan.parameter_profile
        @test prepared_plan.parameters == cold_plan.parameters

        @test result.diagnostics.timings.presolve == 0.0
        @test result.diagnostics.timings.structural_analysis == 0.0
        @test result.diagnostics.timings.execution_planning == 0.0
        @test cold.diagnostics.timings.structural_analysis >= 0.0
        @test cold.diagnostics.timings.execution_planning >= 0.0
    end

    @testset "auto plan reuse covers objective/RHS sessions" begin
        auto_options = SDPX.SolverOptions{Float64}(
            scaling=:none,
            presolve=true,
            timing=true,
            verbosity=0,
        )
        prepared = SDPX.prepare(_prepared_lp([1.0, 2.0]), auto_options)
        @test prepared.structure.execution_plan isa SDPX.ExecutionPlan

        result = SDPX.solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        @test result.status == SDPX.Optimal
        @test result.diagnostics.plan !== prepared.structure.execution_plan
        @test result.diagnostics.plan.payload isa SDPX.LPRoutePlan
        @test prepared.structure.execution_plan.payload === nothing
        @test result.diagnostics.timings.structural_analysis == 0.0
        @test result.diagnostics.timings.execution_planning == 0.0
        @test result.diagnostics.plan.parameter_profile == :automatic_mehrotra
        # The automatic rule reads only the immutable cone constants, so the
        # cached plan equals a fresh auto plan for the replacement session.
        cold_plan = SDPX.build_execution_plan(
            _prepared_lp([2.0, 1.0]; rhs=[2.0]),
            auto_options,
        )
        @test result.diagnostics.plan.parameters == cold_plan.parameters
    end

    @testset "auto SDP plan resolves once in every prepared session" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            zeros(2, 0),
            Float64[];
            verbosity=0,
        )
        auto_sdp_options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            scaling=:equilibrate,
            diagnostics=true,
            verbosity=0,
        )
        prepared = SDPX.prepare(problem, auto_sdp_options)
        plan = prepared.structure.execution_plan
        @test plan isa SDPX.ExecutionPlan
        @test plan.parameter_profile === :automatic_mehrotra

        first_result = SDPX.solve!(
            prepared;
            objective=[2.0, 3.0],
            warm_start=nothing,
        )
        second_result = SDPX.solve!(
            prepared;
            objective=[3.0, 2.0],
            warm_start=nothing,
        )
        @test first_result.status == second_result.status == SDPX.Optimal
        for result in (first_result, second_result)
            selected = result.diagnostics.selected_algorithms
            @test result.diagnostics.plan === plan
            @test selected.parameter_profile === :post_scaling_mehrotra
            @test selected.parameter_source === :post_scaling_mehrotra
            @test selected.parameter_resolution_count == 1
            @test selected.stage === :post_scaling
        end
        @test prepared.state.structure_reuses == 2
        @test prepared.state.numeric_generation == 2
    end

    @testset "fixed-variable objective transform refreshes offset" begin
        G = [1.0 0.0; -1.0 0.0; 0.0 1.0]
        h = [2.0, -2.0, 1.0]
        problem = SDPX.linear_program([3.0, 1.0], G, h; verbosity=0)
        prepared = SDPX.prepare(problem, options)
        transform = prepared.structure.transform

        @test transform.fixed_variables == [1]
        transformed = SDPX.transform_objective(
            transform,
            [4.0, 2.0],
        )
        @test transformed.reduced == [2.0]
        @test transformed.objective_offset == 8.0

        result = SDPX.solve!(
            prepared;
            objective=[4.0, 2.0],
            warm_start=nothing,
        )
        cold = SDPX.solve!(
            SDPX.linear_program([4.0, 2.0], G, h; verbosity=0),
            options,
        )
        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8
        @test prepared.state.last_objective_offset == 8.0
        @test result.diagnostics.presolve.preprocessing.elapsed == 0.0
    end

    @testset "dependent RHS relations are revalidated" begin
        problem = _prepared_lp([1.0, 2.0]; dependent=true)
        prepared = SDPX.prepare(problem, options)
        @test length(prepared.structure.transform.equality_keep) < 2 ||
              length(prepared.structure.transform.dependent_equality_keep) < 2

        transformed = SDPX.transform_rhs(
            prepared.structure,
            [2.0, 4.0],
        )
        @test length(transformed.reduced) == 1
        result = SDPX.solve!(prepared; rhs=[2.0, 4.0], warm_start=nothing)
        cold = SDPX.solve!(
            _prepared_lp([1.0, 2.0]; rhs=[2.0], dependent=true),
            options,
        )
        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8

        error = try
            SDPX.solve!(prepared; rhs=[2.0, 5.0], warm_start=nothing)
            nothing
        catch exception
            exception
        end
        @test error isa SDPX.PreparedStructureMismatch
        @test error.reason === :rhs_relation_failed
        @test prepared.state.numeric_generation == 1
    end

    @testset "fixed substitution tolerates only roundoff-sized reassociation" begin
        G = [1.0 0.0; -1.0 0.0; 0.0 1.0]
        h = [0.1, -0.1, 0.0]
        Aeq = [1.0 1.0; 3.0 3.0]
        beq = [0.3, 0.9]
        problem = SDPX.linear_program(
            [1.0, 1.0],
            G,
            h;
            Aeq=Aeq,
            beq=beq,
            verbosity=0,
        )
        prepared = SDPX.prepare(problem, options)
        transformed = SDPX.transform_rhs(
            prepared.structure,
            [0.3, 0.9],
        )
        @test length(transformed.reduced) == 1
        @test_throws SDPX.PreparedStructureMismatch SDPX.transform_rhs(
            prepared.structure,
            [0.3, 0.9001],
        )
    end

    @testset "coefficient and equality-rank changes invalidate" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = SDPX.prepare(problem, options)

        changed_C = [copy(block) for block in problem.C]
        changed_C[1][1, 1] += 0.5
        coefficient_changed = _problem_with(problem; C=changed_C)
        @test_throws SDPX.PreparedStructureMismatch SDPX.solve!(
            prepared,
            coefficient_changed;
            warm_start=nothing,
        )

        changed_B = copy(problem.B)
        changed_B[:, 1] .= 0.0
        rank_changed = _problem_with(problem; B=changed_B)
        @test_throws SDPX.PreparedStructureMismatch SDPX.solve!(
            prepared,
            rank_changed;
            warm_start=nothing,
        )
        @test prepared.state.structure_invalidations == 2
        @test prepared.state.last_reuse === :invalidated
    end

    @testset "prepared input is owned and sparse storage stays compact" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = SDPX.prepare(problem, options)
        @test prepared.problem !== problem
        @test prepared.problem.c !== problem.c
        @test prepared.problem.B !== problem.B
        @test prepared.problem.cons isa SDPX.SparseCons{Float64}
        @test all(
            block -> block isa SDPX.CompactScalarCoefficientVector ||
                     block isa SDPX.ActiveSparseCoefficientVector,
            prepared.problem.cons.Asp,
        )

        problem.C[1][1, 1] += 100.0
        result = SDPX.solve!(prepared; warm_start=nothing)
        @test result.status == SDPX.Optimal

        setprecision(BigFloat, 128) do
            big = _prepared_lp(BigFloat[1, 2])
            big_options = SDPX.SolverOptions{BigFloat}(
                precision_bits=128,
                working_precision_policy=:fixed,
                scaling=:none,
                verbosity=0,
            )
            owned = SDPX.prepare(big, big_options)
            @test owned.problem.c[1] !== big.c[1]
            @test owned.problem.C[1][1, 1] !== big.C[1][1, 1]
            @test owned.problem.cons.Asp[1].coefficient[1, 1] !==
                  big.cons.Asp[1].coefficient[1, 1]
        end
    end

    @testset "advanced internal surface remains explicitly qualified" begin
        @test SDPX.PreparedStructure === SDPX.PreparedStructure
        @test SDPX.SolveState === SDPX.SolveState
        @test SDPX.PreprocessTransform === SDPX.PreprocessTransform
        @test SDPX.StructureFingerprint === SDPX.StructureFingerprint
        @test SDPX.PreparedStructureMismatch ===
              SDPX.PreparedStructureMismatch
    end

    @testset "reentrant access fails without corrupting outer busy state" begin
        prepared = SDPX.prepare(_prepared_lp([1.0, 2.0]), options)
        lock(prepared.state.lock)
        prepared.busy = true
        try
            @test_throws ArgumentError SDPX.solve!(prepared; warm_start=nothing)
            @test prepared.busy
            @test prepared.solve_count == 0
        finally
            prepared.busy = false
            unlock(prepared.state.lock)
        end
        result = SDPX.solve!(prepared; warm_start=nothing)
        @test result.status == SDPX.Optimal
        @test prepared.solve_count == 1
        @test !prepared.busy
    end
end

# A2 — one Prepared per-solve finalization: each solve carries its own
# finalized `LPRoutePlan` on the diagnostics plan, never the deferred
# pre-row `:not_applicable` placeholder.  Red until the A2 source lands.
@testset "A2 prepared LP per-solve finalized route" begin
    options = SDPX.SolverOptions{Float64}(
        scaling=:none,
        presolve=true,
        parameter_policy=:fixed,
        diagnostics=true,
        verbosity=0,
    )
    prepared = SDPX.prepare(_prepared_lp([1.0, 2.0]), options)
    result = SDPX.solve!(
        prepared;
        objective=[2.0, 1.0],
        rhs=[2.0],
        warm_start=nothing,
    )
    payload = result.diagnostics.plan.payload
    @test payload isa SDPX.LPRoutePlan
    @test payload.route === :diagonal_reduced_cholesky
    @test payload.storage === :dense
    @test payload.provider === :reduced_kernel
    record = only(result.diagnostics.attempts)
    @test payload.route === record.executed.formulation
    @test payload.storage === record.executed.storage
    @test payload.provider === record.executed.provider
    @test payload.route !== :not_applicable
    @test record.planned.formulation === :not_applicable
end
