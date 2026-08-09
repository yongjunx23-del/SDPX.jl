using MultiFloats
using SparseArrays
using Test
import MathOptInterface as PRE_MOI

function bound_heavy_problem(::Type{T}) where {T}
    empty = spzeros(T, 1, 1)
    scalar(first, second) = [
        iszero(first) ? empty : sparse([1], [1], T[first], 1, 1),
        iszero(second) ? empty : sparse([1], [1], T[second], 1, 1),
    ]
    A = [
        scalar(1, 0),    # x1 >= 1
        scalar(2, 0),    # duplicate x1 >= 1
        scalar(1, 0),    # dominated x1 >= 0
        scalar(-1, 0),   # x1 <= 1
        scalar(0, 1),    # x2 >= 0
        scalar(0, -1),   # x2 <= 2
    ]
    C = [
        reshape(T[1], 1, 1),
        reshape(T[2], 1, 1),
        reshape(T[0], 1, 1),
        reshape(T[-1], 1, 1),
        reshape(T[0], 1, 1),
        reshape(T[-2], 1, 1),
    ]
    return SDPX.ingest(
        T[2, 1],
        A,
        C,
        zeros(T, 2, 0),
        T[];
        sparse=true,
        verbosity=0,
    )
end

function cleanup_problem(::Type{T}; inconsistent_zero=false) where {T}
    A = [[
        sparse([1], [1], T[1], 1, 1),
        sparse([1], [1], T[1], 1, 1),
    ]]
    C = [reshape(T[0], 1, 1)]
    B = T[
        1 1 2 0 1
        2 2 4 0 0
    ]
    b = T[3, 3, 6, inconsistent_zero ? 1 : 0, 1]
    return SDPX.ingest(
        T[1, 1],
        A,
        C,
        B,
        b;
        sparse=true,
        verbosity=0,
    )
end

@testset "conservative preprocessing regressions" begin
    @testset "typed bounds merge and exact fixed elimination" begin
        for T in (Float64, Float64x4)
            problem = bound_heavy_problem(T)
            original_constants = [copy(matrix) for matrix in problem.C]
            reduced = SDPX.preprocess(
                problem,
                SDPX.SolverOptions{T}(verbosity=0),
            )
            report = reduced.report
            @test report.extracted_lower_bounds == 4
            @test report.extracted_upper_bounds == 2
            @test report.merged_bound_constraints == 4
            @test report.fixed_variables_eliminated == 1
            @test report.input.variables == 2
            @test report.output.variables == 1
            @test report.input.psd_blocks == 6
            @test report.output.psd_blocks == 2
            @test reduced.plan isa SDPX.PreprocessPlan{T}
            @test eltype(reduced.problem) === T
            @test reduced.reconstruction.fixed_variables == [1]
            @test reduced.reconstruction.fixed_values == T[1]
            @test reduced.reconstruction.objective_offset == T(2)
            @test problem.C == original_constants
            checks = SDPX.validate(
                SDPX.FixedVariableEliminationStage(),
                problem,
                reduced.problem,
                reduced.reconstruction,
            )
            @test all(values(checks))
        end

        setprecision(BigFloat, 192) do
            problem = bound_heavy_problem(BigFloat)
            original_constants = [deepcopy(matrix) for matrix in problem.C]
            reduced = SDPX.preprocess(
                problem,
                SDPX.SolverOptions{BigFloat}(
                    verbosity=0,
                    precision_bits=192,
                ),
            )
            @test eltype(reduced.problem) === BigFloat
            @test reduced.report.precision_bits == 192
            @test reduced.reconstruction.fixed_values == BigFloat[1]
            @test problem.C == original_constants
            @test all(
                value -> precision(value) == 192,
                reduced.reconstruction.fixed_values,
            )
        end
    end

    @testset "reconstruction preserves the original certificate" begin
        problem = bound_heavy_problem(Float64)
        options = SDPX.SolverOptions{Float64}(
            verbosity=0,
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            iter_max=100,
        )
        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.Optimal
        @test result.x[1] == 1.0
        @test result.x[2] >= -1e-8
        @test result.pObj ≈ 2.0 atol=1e-7
        @test result.p_res <= 1e-8
        @test result.d_res <= 1e-8
        certificate = SDPX.result_certificate(problem, result, options)
        @test certificate.valid
        @test result.diagnostics.presolve.preprocessing.changed
    end

    @testset "zero, duplicate, and exact proportional equalities" begin
        for T in (Float64, Float64x4)
            problem = cleanup_problem(T)
            reduced = SDPX.preprocess(
                problem,
                SDPX.SolverOptions{T}(
                    verbosity=0,
                    presolve_fixed_variables=false,
                    chordal_decomposition=:off,
                ),
            )
            @test reduced.problem.dims.n == 2
            @test reduced.report.zero_equalities_removed == 1
            @test reduced.report.duplicate_equalities_removed == 1
            @test reduced.report.proportional_equalities_removed == 1
            @test !reduced.inconsistent
        end

        setprecision(BigFloat, 192) do
            problem = cleanup_problem(BigFloat)
            original_B = deepcopy(problem.B)
            original_b = deepcopy(problem.b)
            reduced = SDPX.preprocess(
                problem,
                SDPX.SolverOptions{BigFloat}(
                    verbosity=0,
                    precision_bits=192,
                    presolve_fixed_variables=false,
                    chordal_decomposition=:off,
                ),
            )
            @test reduced.problem.dims.n == 2
            @test reduced.report.zero_equalities_removed == 1
            @test reduced.report.duplicate_equalities_removed == 1
            @test reduced.report.proportional_equalities_removed == 1
            @test problem.B == original_B
            @test problem.b == original_b
            @test all(
                isassigned(
                    reduced.reconstruction.equality_multiplier_map,
                    index,
                )
                for index in eachindex(
                    reduced.reconstruction.equality_multiplier_map,
                )
            )
            multiplier_map =
                reduced.reconstruction.equality_multiplier_map
            @test length(unique(objectid.(multiplier_map))) ==
                  length(multiplier_map)
        end

        inconsistent = cleanup_problem(Float64; inconsistent_zero=true)
        diagnosis = SDPX.preprocess(
            inconsistent,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                presolve_fixed_variables=false,
                chordal_decomposition=:off,
            ),
        )
        @test diagnosis.inconsistent
    end

    @testset "near-duplicate equalities are diagnostic only" begin
        problem = cleanup_problem(Float64)
        problem.B[:, 2] .= problem.B[:, 1]
        problem.B[1, 2] += 1e-12
        problem.b[2] = problem.b[1]
        reduced = SDPX.preprocess(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                presolve_fixed_variables=false,
                chordal_decomposition=:off,
            ),
        )
        @test reduced.report.near_duplicate_equalities >= 1
        @test 2 in reduced.reconstruction.reduced_to_original_equalities
    end

    @testset "large near-duplicate diagnostics have bounded work" begin
        variables = 16
        equalities = 20
        coefficients = [[
            sparse([1], [1], [1.0], 1, 1)
            for _ in 1:variables
        ]]
        B = Matrix{Float64}(undef, variables, equalities)
        @inbounds for equality in 1:equalities, variable in 1:variables
            B[variable, equality] =
                variable + equality / (variable + equality + 1)
        end
        b = collect(Float64, 1:equalities)
        B[:, equalities - 1] .= B[:, 1]
        b[equalities - 1] = b[1]
        B[:, equalities] .= 2 .* B[:, 2]
        b[equalities] = 2 * b[2]
        problem = SDPX.ingest(
            ones(variables),
            coefficients,
            [zeros(1, 1)],
            B,
            b;
            sparse=true,
            verbosity=0,
        )
        reduced = SDPX.preprocess(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                presolve_fixed_variables=false,
                chordal_decomposition=:off,
            ),
        )
        @test reduced.problem.dims.n == equalities - 2
        @test reduced.report.duplicate_equalities_removed == 1
        @test reduced.report.proportional_equalities_removed == 1
        @test reduced.report.near_duplicate_equalities == 0
        @test any(
            warning -> occursin(
                "Near-proportional equality diagnostics were skipped",
                warning,
            ),
            reduced.report.warnings,
        )
    end

    @testset "disabled stages leave the problem unchanged" begin
        problem = bound_heavy_problem(Float64)
        reduced = SDPX.preprocess(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                presolve=:off,
            ),
        )
        @test reduced.problem === problem
        @test reduced.plan === nothing
        @test !reduced.report.enabled
        @test !reduced.report.changed
    end

    @testset "compact bounds survive scaling and precision preparation" begin
        compact = SDPX.CompactScalarCoefficientVector(
            Float64,
            2,
            2,
            3.0,
        )
        problem = SDPX.ingest(
            [1.0, 2.0],
            [compact],
            [reshape([6.0], 1, 1)],
            zeros(2, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        scaled, _ = SDPX.equilibrate(
            problem,
            problem.cons::SDPX.SparseCons{Float64},
        )
        @test scaled.cons.Asp[1] isa
              SDPX.CompactScalarCoefficientVector{Float64}
        @test scaled.cons.active == [[2]]

        setprecision(BigFloat, 192) do
            compact_big = SDPX.CompactScalarCoefficientVector(
                BigFloat,
                2,
                2,
                BigFloat(3),
            )
            problem_big = SDPX.ingest(
                BigFloat[1, 2],
                [compact_big],
                [reshape(BigFloat[6], 1, 1)],
                zeros(BigFloat, 2, 0),
                BigFloat[];
                sparse=true,
                verbosity=0,
            )
            rerounded = SDPX.reround(problem_big, 160)
            @test rerounded.cons.Asp[1] isa
                  SDPX.CompactScalarCoefficientVector{BigFloat}
            coefficient = rerounded.cons.Asp[1][2][1, 1]
            @test precision(coefficient) == 160
            @test coefficient == BigFloat(3; precision=160)
        end
    end

    @testset "sparse equilibration copies only active matrices" begin
        coefficients = [[
            sparse([1], [1], [2.0], 1, 1),
            spzeros(1, 1),
            spzeros(1, 1),
        ]]
        problem = SDPX.ingest(
            [1.0, 2.0, 3.0],
            coefficients,
            [reshape([1.0], 1, 1)],
            zeros(3, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        scaled, _ = SDPX.equilibrate(
            problem,
            problem.cons::SDPX.SparseCons{Float64},
        )
        scaled_cons = scaled.cons::SDPX.SparseCons{Float64}
        source_cons = problem.cons::SDPX.SparseCons{Float64}
        @test scaled_cons.Asp[1][1] !== source_cons.Asp[1][1]
        @test scaled_cons.Asp[1][2] === scaled_cons.Asp[1][3]
        @test nnz(scaled_cons.Asp[1][2]) == 0
        @test source_cons.Asp[1][1][1, 1] == 2.0
    end

    @testset "Ruiz equilibration selects a bounded pass count" begin
        coefficients = [zeros(Float64, 2, 2, 2)]
        coefficients[1][1, :, :] .= [1.0 0.0; 0.0 1.0e-8]
        coefficients[1][2, :, :] .= [0.0 1.0e4; 1.0e4 1.0]
        problem = SDPX.ingest(
            [1.0, -2.0],
            coefficients,
            [[1.0e-6 0.0; 0.0 1.0e6]],
            zeros(2, 0),
            Float64[];
            sparse=false,
            verbosity=0,
        )
        _, adaptive = SDPX.equilibrate(problem)
        _, one_pass = SDPX.equilibrate(problem; ruiz_iters=1)
        @test length(adaptive.ruiz_passes) == problem.dims.L
        @test all(pass -> 2 <= pass <= 8, adaptive.ruiz_passes)
        @test one_pass.ruiz_passes == ones(Int, problem.dims.L)
    end

    @testset "MOI Interval survives preprocessing and reconstruction" begin
        source = PRE_MOI.Utilities.Model{Float64}()
        variables = PRE_MOI.add_variables(source, 2)
        fixed = PRE_MOI.add_constraint(
            source,
            variables[1],
            PRE_MOI.Interval(1.0, 1.0),
        )
        PRE_MOI.add_constraint(
            source,
            variables[2],
            PRE_MOI.Interval(0.0, 2.0),
        )
        objective = PRE_MOI.ScalarAffineFunction(
            [
                PRE_MOI.ScalarAffineTerm(2.0, variables[1]),
                PRE_MOI.ScalarAffineTerm(1.0, variables[2]),
            ],
            0.0,
        )
        PRE_MOI.set(source, PRE_MOI.ObjectiveSense(), PRE_MOI.MIN_SENSE)
        PRE_MOI.set(
            source,
            PRE_MOI.ObjectiveFunction{typeof(objective)}(),
            objective,
        )

        optimizer = SDPX.Optimizer{Float64}(
            verbose=0,
            tolerance=1e-8,
            max_iter=100,
        )
        indices = PRE_MOI.copy_to(optimizer, source)
        sparse_cons = optimizer.problem.cons::SDPX.SparseCons{Float64}
        @test all(
            block -> block isa SDPX.CompactScalarCoefficientVector{Float64},
            sparse_cons.Asp,
        )
        PRE_MOI.optimize!(optimizer)
        @test PRE_MOI.get(optimizer, PRE_MOI.TerminationStatus()) ==
              PRE_MOI.OPTIMAL
        @test PRE_MOI.get(
            optimizer,
            PRE_MOI.VariablePrimal(),
            indices[variables[1]],
        ) == 1.0
        @test PRE_MOI.get(
            optimizer,
            PRE_MOI.ConstraintPrimal(),
            indices[fixed],
        ) == 1.0
        @test PRE_MOI.get(
            optimizer,
            PRE_MOI.ConstraintDual(),
            indices[fixed],
        ) ≈ 2.0 atol=1e-7
    end
end
