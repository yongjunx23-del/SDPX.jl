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
    return linear_program(
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
    problem::SDPProblem{T};
    c=problem.c,
    C=problem.C,
    B=problem.B,
    b=problem.b,
    cons=problem.cons,
    dims=problem.dims,
    structure=problem.structure,
) where {T}
    return SDPProblem{T}(c, C, B, b, cons, dims, structure)
end

@testset "PreparedStructure and SolveState" begin
    options = SolverOptions{Float64}(
        scaling=:none,
        presolve=true,
        verbosity=0,
    )

    @testset "objective and RHS reuse match cold solves" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = prepare(problem, options)
        fingerprint = prepared.structure.fingerprint
        template = prepared.structure.preprocessed_template

        result = solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        cold_problem = _prepared_lp([2.0, 1.0]; rhs=[2.0])
        cold = solve!(cold_problem, options)

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
        reuse_options = SolverOptions{Float64}(
            scaling=:none,
            presolve=true,
            parameter_policy=:fixed,
            timing=true,
            verbosity=0,
        )
        problem = _prepared_lp([1.0, 2.0])
        prepared = prepare(problem, reuse_options)
        @test prepared.structure.execution_plan isa SDPX.ExecutionPlan

        result = solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        cold_problem = _prepared_lp([2.0, 1.0]; rhs=[2.0])
        cold = solve!(cold_problem, reuse_options)

        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8
        @test result.dObj ≈ cold.dObj atol=1e-8
        @test result.x ≈ cold.x atol=1e-8
        @test SDPX.result_certificate(cold_problem, result, reuse_options).valid
        @test SDPX.result_certificate(cold_problem, cold, reuse_options).valid

        prepared_plan = result.diagnostics.plan
        cold_plan = cold.diagnostics.plan
        @test prepared_plan === prepared.structure.execution_plan
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

    @testset "RHS-sensitive LP planning is not cached" begin
        auto_options = SolverOptions{Float64}(
            scaling=:none,
            presolve=true,
            timing=true,
            verbosity=0,
        )
        prepared = prepare(_prepared_lp([1.0, 2.0]), auto_options)
        @test prepared.structure.execution_plan === nothing

        result = solve!(
            prepared;
            objective=[2.0, 1.0],
            rhs=[2.0],
            warm_start=nothing,
        )
        @test result.status == SDPX.Optimal
        @test result.diagnostics.plan isa SDPX.ExecutionPlan
        @test result.diagnostics.timings.structural_analysis >= 0.0
        @test result.diagnostics.timings.execution_planning >= 0.0
    end

    @testset "fixed-variable objective transform refreshes offset" begin
        G = [1.0 0.0; -1.0 0.0; 0.0 1.0]
        h = [2.0, -2.0, 1.0]
        problem = linear_program([3.0, 1.0], G, h; verbosity=0)
        prepared = prepare(problem, options)
        transform = prepared.structure.transform

        @test transform.fixed_variables == [1]
        transformed = SDPX.Experimental.transform_objective(
            transform,
            [4.0, 2.0],
        )
        @test transformed.reduced == [2.0]
        @test transformed.objective_offset == 8.0

        result = solve!(
            prepared;
            objective=[4.0, 2.0],
            warm_start=nothing,
        )
        cold = solve!(
            linear_program([4.0, 2.0], G, h; verbosity=0),
            options,
        )
        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8
        @test prepared.state.last_objective_offset == 8.0
        @test result.diagnostics.presolve.preprocessing.elapsed == 0.0
    end

    @testset "dependent RHS relations are revalidated" begin
        problem = _prepared_lp([1.0, 2.0]; dependent=true)
        prepared = prepare(problem, options)
        @test length(prepared.structure.transform.equality_keep) < 2 ||
              length(prepared.structure.transform.dependent_equality_keep) < 2

        transformed = SDPX.Experimental.transform_rhs(
            prepared.structure,
            [2.0, 4.0],
        )
        @test length(transformed.reduced) == 1
        result = solve!(prepared; rhs=[2.0, 4.0], warm_start=nothing)
        cold = solve!(
            _prepared_lp([1.0, 2.0]; rhs=[2.0], dependent=true),
            options,
        )
        @test result.status == cold.status == SDPX.Optimal
        @test result.pObj ≈ cold.pObj atol=1e-8

        error = try
            solve!(prepared; rhs=[2.0, 5.0], warm_start=nothing)
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
        problem = linear_program(
            [1.0, 1.0],
            G,
            h;
            Aeq=Aeq,
            beq=beq,
            verbosity=0,
        )
        prepared = prepare(problem, options)
        transformed = SDPX.Experimental.transform_rhs(
            prepared.structure,
            [0.3, 0.9],
        )
        @test length(transformed.reduced) == 1
        @test_throws SDPX.PreparedStructureMismatch SDPX.Experimental.transform_rhs(
            prepared.structure,
            [0.3, 0.9001],
        )
    end

    @testset "coefficient and equality-rank changes invalidate" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = prepare(problem, options)

        changed_C = [copy(block) for block in problem.C]
        changed_C[1][1, 1] += 0.5
        coefficient_changed = _problem_with(problem; C=changed_C)
        @test_throws SDPX.PreparedStructureMismatch solve!(
            prepared,
            coefficient_changed;
            warm_start=nothing,
        )

        changed_B = copy(problem.B)
        changed_B[:, 1] .= 0.0
        rank_changed = _problem_with(problem; B=changed_B)
        @test_throws SDPX.PreparedStructureMismatch solve!(
            prepared,
            rank_changed;
            warm_start=nothing,
        )
        @test prepared.state.structure_invalidations == 2
        @test prepared.state.last_reuse === :invalidated
    end

    @testset "prepared input is owned and sparse storage stays compact" begin
        problem = _prepared_lp([1.0, 2.0])
        prepared = prepare(problem, options)
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
        result = solve!(prepared; warm_start=nothing)
        @test result.status == SDPX.Optimal

        setprecision(BigFloat, 128) do
            big = _prepared_lp(BigFloat[1, 2])
            big_options = SolverOptions{BigFloat}(
                precision_bits=128,
                working_precision_policy=:fixed,
                scaling=:none,
                verbosity=0,
            )
            owned = prepare(big, big_options)
            @test owned.problem.c[1] !== big.c[1]
            @test owned.problem.C[1][1, 1] !== big.C[1][1, 1]
            @test owned.problem.cons.Asp[1].coefficient[1, 1] !==
                  big.cons.Asp[1].coefficient[1, 1]
        end
    end

    @testset "Experimental namespace owns the advanced surface" begin
        @test SDPX.Experimental.PreparedStructure === SDPX.PreparedStructure
        @test SDPX.Experimental.SolveState === SDPX.SolveState
        @test SDPX.Experimental.PreprocessTransform === SDPX.PreprocessTransform
        @test SDPX.Experimental.StructureFingerprint === SDPX.StructureFingerprint
        @test SDPX.Experimental.PreparedStructureMismatch ===
              SDPX.PreparedStructureMismatch
    end

    @testset "reentrant access fails without corrupting outer busy state" begin
        prepared = prepare(_prepared_lp([1.0, 2.0]), options)
        lock(prepared.state.lock)
        prepared.busy = true
        try
            @test_throws ArgumentError solve!(prepared; warm_start=nothing)
            @test prepared.busy
            @test prepared.solve_count == 0
        finally
            prepared.busy = false
            unlock(prepared.state.lock)
        end
        result = solve!(prepared; warm_start=nothing)
        @test result.status == SDPX.Optimal
        @test prepared.solve_count == 1
        @test !prepared.busy
    end
end
