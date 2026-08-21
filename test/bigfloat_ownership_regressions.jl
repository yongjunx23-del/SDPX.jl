using LinearAlgebra
import MutableArithmetics as MA
using SDPX
using Test

function ownership_lp_problem(::Type{T}) where {T}
    rows = reshape(T[1, -1], 2, 1)
    rhs = T[0, -1]
    coefficients = [
        reshape(Vector{T}(view(rows, row, :)), 1, 1, 1)
        for row in axes(rows, 1)
    ]
    constants = [reshape(T[rhs[row]], 1, 1) for row in eachindex(rhs)]
    return SDPX.ingest(
        T[1],
        coefficients,
        constants,
        Matrix{T}(undef, 1, 0),
        T[];
        sparse=:auto,
        verbosity=0,
    )
end

@testset "BigFloat ownership regressions" begin
    setprecision(BigFloat, 192) do
        @testset "owned array conversion and identity storage" begin
            source = BigFloat[1, 2, 3]
            converted = SDPX._owned_array_copy(BigFloat, source)
            MA.operate!(+, converted[1], BigFloat(10))
            @test source == BigFloat[1, 2, 3]
            @test converted == BigFloat[11, 2, 3]

            identity_matrix =
                SDPX._scaled_identity(BigFloat, BigFloat(4), 3)
            MA.operate!(+, identity_matrix[1, 1], BigFloat(1))
            @test identity_matrix[1, 1] == BigFloat(5)
            @test identity_matrix[2, 2] == BigFloat(4)
            @test identity_matrix[3, 3] == BigFloat(4)
            @test identity_matrix[1, 2] == 0
        end

        @testset "LP extraction does not alias problem data" begin
            problem = ownership_lp_problem(BigFloat)
            first_coefficient =
                problem.cons isa SDPX.DenseCons ?
                problem.cons.Av[1][1, 1] :
                problem.cons.Asp[1][1][1, 1]
            first_constant = problem.C[1][1, 1]
            coefficient_value = BigFloat(first_coefficient)
            constant_value = BigFloat(first_constant)

            G, h = SDPX._extract_lp_rows(problem)
            MA.operate!(+, G[1, 1], BigFloat(7))
            MA.operate!(+, h[1], BigFloat(9))

            @test first_coefficient == coefficient_value
            @test first_constant == constant_value
        end

        @testset "equality reconstruction owns restored multipliers" begin
            reduced_y = SDPX.alloc_zeros(BigFloat, 1)
            MA.operate_to!(reduced_y[1], copy, BigFloat(3))
            result = SDPX.SDPResult{BigFloat}(
                SDPX.IterLimit,
                "test",
                SDPX.alloc_zeros(BigFloat, 0),
                Matrix{BigFloat}[],
                reduced_y,
                Matrix{BigFloat}[],
                BigFloat(0),
                BigFloat(0),
                BigFloat(0),
                BigFloat(0),
                BigFloat(0),
                0,
                0,
                0,
                nothing,
                NamedTuple[],
                nothing,
            )
            multiplier_map = SDPX.alloc_zeros(BigFloat, 1, 2)
            MA.operate_to!(multiplier_map[1, 2], copy, BigFloat(1))
            mapping =
                SDPX.EqualityPresolveMap{BigFloat}(2, [2], multiplier_map)
            restored = SDPX._restore_equalities(result, mapping)

            MA.operate!(+, restored.y[2], BigFloat(5))
            @test reduced_y[1] == BigFloat(3)
            @test restored.y == BigFloat[0, 8]
        end

        @testset "callbacks cannot mutate live LP scalars" begin
            problem = ownership_lp_problem(BigFloat)
            options = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=big"1e-24",
                ϵ_primal=big"1e-24",
                ϵ_dual=big"1e-24",
                iter_max=80,
                parameter_policy=:fixed,
                presolve=false,
                scaling=:none,
                verbosity=0,
                precision_bits=192,
            )
            reference = SDPX.solve!(problem, options)
            callback_calls = Ref(0)
            adversarial = state -> begin
                callback_calls[] += 1
                MA.operate!(zero, state.pObj)
                MA.operate!(zero, state.dObj)
                MA.operate!(zero, state.gap)
                MA.operate!(zero, state.p_res)
                MA.operate!(zero, state.d_res)
                MA.operate!(zero, state.μ)
                return false
            end
            callback_options =
                SDPX._replace_solver_options(options; callback=adversarial)
            observed = SDPX.solve!(problem, callback_options)

            @test callback_calls[] > 0
            @test observed.status == reference.status
            @test observed.x ≈ reference.x rtol=big"1e-40" atol=big"1e-40"
            @test observed.pObj ≈ reference.pObj rtol=big"1e-40" atol=big"1e-40"
            @test observed.dObj ≈ reference.dObj rtol=big"1e-40" atol=big"1e-40"
            @test observed.p_res ≈ reference.p_res rtol=big"1e-40" atol=big"1e-40"
            @test observed.d_res ≈ reference.d_res rtol=big"1e-40" atol=big"1e-40"
        end

        @testset "certification preserves problem data" begin
            problem = ownership_lp_problem(BigFloat)
            x = BigFloat[big"0.25"]
            X = [reshape(BigFloat[big"0.25"], 1, 1),
                 reshape(BigFloat[big"0.75"], 1, 1)]
            y = BigFloat[]
            Y = [reshape(BigFloat[big"0.5"], 1, 1),
                 reshape(BigFloat[big"0.5"], 1, 1)]
            c_before = BigFloat.(problem.c)
            C_before = [BigFloat.(block) for block in problem.C]

            SDPX.solution_residuals(problem, x, X, y, Y)

            @test problem.c == c_before
            @test problem.C == C_before
        end

        @testset "public KKT solve repairs aliased outputs" begin
            variable_count = 3
            coefficients = [
                begin
                    block = zeros(BigFloat, variable_count, 1, 1)
                    block[index, 1, 1] = one(BigFloat)
                    block
                end
                for index in 1:variable_count
            ]
            problem = SDPX.ingest(
                ones(BigFloat, variable_count),
                coefficients,
                [zeros(BigFloat, 1, 1) for _ in coefficients],
                zeros(BigFloat, variable_count, 0),
                BigFloat[];
                sparse=false,
                verbosity=0,
            )
            workspace = SDPX.Workspace(problem; thread_count=1)
            schur = BigFloat[
                4 1 0
                1 3 1 / 5
                0 1 / 5 2
            ]
            SDPX.copy_owned!(workspace.S, schur)
            @test SDPX.factorize!(
                SDPX.select_backend(workspace),
                workspace,
                problem,
                SDPX.SolverOptions{BigFloat}(verbosity=0),
            ).ok

            rhs = BigFloat[1 / 3, -2 / 5, 7 / 11]
            expected = schur \ rhs
            output = zeros(BigFloat, variable_count)
            @test output[1] === output[2]
            SDPX.solve_kkt!(
                workspace,
                0,
                rhs,
                BigFloat[],
                output,
                BigFloat[],
            )
            @test output ≈ expected rtol=big"1e-55" atol=big"1e-55"
            @test output[1] !== output[2]
            @test output[2] !== output[3]
        end

        @testset "owned sparse block build preserves structural-zero ownership" begin
            coefficients = [zeros(BigFloat, 1, 3, 3)]
            coefficients[1][1, 2, 2] = BigFloat(1)
            constants = [zeros(BigFloat, 3, 3)]
            constants[1][1, 1] = BigFloat(-1)
            problem = SDPX.ingest(
                BigFloat[0],
                coefficients,
                constants,
                zeros(BigFloat, 1, 0),
                BigFloat[];
                sparse=true,
                verbosity=0,
            )
            @test problem.cons isa SDPX.SparseCons{BigFloat}

            public_block = zeros(BigFloat, 3, 3)
            @test public_block[1] === public_block[2]
            SDPX.buildP!(public_block, problem.cons, 1, BigFloat[2])
            @test all(
                first == second || public_block[first] !== public_block[second]
                for first in eachindex(public_block),
                    second in eachindex(public_block)
            )

            block = SDPX.alloc_zeros(BigFloat, 3, 3)
            SDPX.buildP_owned!(block, problem.cons, 1, BigFloat[2])
            SDPX.kaxpby_owned!(
                -one(BigFloat),
                problem.C[1],
                one(BigFloat),
                block,
            )

            expected = zeros(BigFloat, 3, 3)
            expected[1, 1] = BigFloat(1)
            expected[2, 2] = BigFloat(2)
            @test block == expected
            @test all(
                first == second || block[first] !== block[second]
                for first in eachindex(block), second in eachindex(block)
            )
        end
    end

    @testset "LP path applies BigFloat precision consistency (review P2.7)" begin
        # The SDP core warns when BigFloat inputs carry fewer bits than the
        # requested working precision; the dedicated LP path used to bypass
        # that entirely, so a 128-bit-input LP inside a 256-bit solve
        # proceeded without a word. Both paths must protect equally.
        low_bits = 128
        objective = setprecision(() -> BigFloat.([1.0, 2.0]), low_bits)
        setprecision(BigFloat, 256) do
            variables = 2
            rows = BigFloat[1 0; 0 1; -1 0; 0 -1]
            blocks = [zeros(BigFloat, variables, 1, 1) for _ in 1:4]
            for row in 1:4, column in 1:variables
                blocks[row][column, 1, 1] = rows[row, column]
            end
            righthand = BigFloat[-1, -1, -3, -3]
            constants = [reshape([righthand[row]], 1, 1) for row in 1:4]
            problem = SDPX.ingest(objective, blocks, constants,
                zeros(BigFloat, variables, 0), BigFloat[]; verbosity=0)

            options = SDPX.SolverOptions{BigFloat}(
                ϵ_gap=BigFloat(1e-30), ϵ_primal=BigFloat(1e-30),
                ϵ_dual=BigFloat(1e-30), verbosity=1, precision_bits=256)
            result = @test_logs (:warn, r"128-bit BigFloat precision") match_mode = :any begin
                SDPX.solve!(problem, options)
            end
            @test result.status == SDPX.Optimal

            # The check reports; it must not mutate what the user handed in.
            @test precision(objective[1]) == low_bits
        end
    end

end
