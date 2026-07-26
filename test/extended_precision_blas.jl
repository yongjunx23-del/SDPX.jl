using LinearAlgebra
using MultiFloats: Float64x4
using SparseArrays
using Test

const EPBLAS = SDPX.ExtendedPrecisionBLAS

function _extended_dense_problem(::Type{T}; variables::Int=36, dimension::Int=5) where {T}
    coefficients = zeros(T, variables, dimension, dimension)
    for variable in 1:variables
        for column in 1:dimension, row in 1:column
            value = T(
                sin(0.13 * variable + 0.19 * row + 0.07 * column),
            )
            coefficients[variable, row, column] = value
            coefficients[variable, column, row] = value
        end
    end
    return SDPX.ingest(
        ones(T, variables),
        [coefficients],
        [zeros(T, dimension, dimension)],
        zeros(T, variables, 1),
        zeros(T, 1);
        sparse=false,
    )
end

function _extended_sparse_problem(::Type{T}; variables::Int=18, dimension::Int=5) where {T}
    matrices = Vector{SparseMatrixCSC{T,Int}}(undef, variables)
    for variable in 1:variables
        dense = zeros(T, dimension, dimension)
        for column in 1:dimension, row in 1:column
            if !iszero(mod(variable + 2row + 3column, 4))
                value = T(
                    cos(0.11 * variable + 0.17 * row + 0.05 * column),
                )
                dense[row, column] = value
                dense[column, row] = value
            end
        end
        matrices[variable] = sparse(dense)
    end
    return SDPX.ingest(
        ones(T, variables),
        [matrices],
        [zeros(T, dimension, dimension)],
        zeros(T, variables, 1),
        zeros(T, 1);
        sparse=true,
    )
end

@testset "extended-precision BLAS" begin
    @testset "singleton-arrow reduced panel" begin
        T = Float64x4
        block_count = 5
        variable_count = block_count + 2
        coefficients = [
            zeros(T, variable_count, 2, 2)
            for _ in 1:block_count
        ]
        for block in 1:block_count
            coefficients[block][1, 1, 2] = T(1)
            coefficients[block][1, 2, 1] = T(1)
            coefficients[block][1, 2, 2] = T(3) / T(10)
            coefficients[block][2, 1, 2] = -T(1) / T(5)
            coefficients[block][2, 2, 1] = -T(1) / T(5)
            coefficients[block][2, 2, 2] = T(11) / T(10)
            coefficients[block][block + 2, 1, 1] = one(T)
        end
        problem = SDPX.ingest(
            ones(T, variable_count),
            coefficients,
            [zeros(T, 2, 2) for _ in 1:block_count],
            zeros(T, variable_count, 0),
            T[];
            sparse=true,
            verbosity=0,
        )
        small_auto_options = SDPX.SolverOptions{T}(
            extended_precision_blas=:auto,
            threads=1,
            verbosity=0,
        )
        @test !SDPX._reduced_arrow_crossover(
            problem,
            T,
            :auto,
            0.10,
            1;
            available_memory_bytes=2^30,
        ).enabled
        @test SDPX.build_execution_plan(
            problem,
            small_auto_options,
        ).gram_kernel == :fused_arrow_2x2
        @test !SDPX.Workspace(
            problem;
            thread_count=1,
            extended_precision_blas=:auto,
        ).arrow.reduced_panel_enabled
        memory_rejected = SDPX._reduced_arrow_crossover(
            problem,
            T,
            :on,
            0.0,
            1;
            available_memory_bytes=2^30,
        )
        @test !memory_rejected.enabled
        @test memory_rejected.reason == :memory_budget
        X = [
            T[
                T(2) + T(block) / T(20) T(1) / T(13)
                T(1) / T(13) T(3) / T(2) + T(block) / T(30)
            ]
            for block in 1:block_count
        ]
        Y = [
            T[
                T(7) / T(5) + T(block) / T(40) T(1) / T(17)
                T(1) / T(17) T(19) / T(10) + T(block) / T(50)
            ]
            for block in 1:block_count
        ]

        reference = SDPX.Workspace(
            problem;
            thread_count=1,
            extended_precision_blas=:off,
        )
        reduced = SDPX.Workspace(
            problem;
            thread_count=min(4, Threads.nthreads()),
            extended_precision_blas=:on,
        )
        @test reduced.arrow.reduced_panel_enabled
        @test size(reduced.arrow.reduced_panel) == (2 * block_count, 2)
        @test SDPX.factor_blocks!(reference, X, Y)
        @test SDPX.factor_blocks!(reduced, X, Y)
        SDPX.schur_build!(reference, problem, problem.cons, X, Y)
        reference_factor = SDPX.factor_arrow_kkt!(
            reference,
            SDPX.SolverOptions{T}(verbosity=0),
        )
        @test reference_factor.ok
        SDPX.schur_build!(reduced, problem, problem.cons, X, Y)
        @test reduced.arrow.reduced_panel_ready
        scale = max(maximum(abs, reference.arrow.Sred), one(T))
        @test maximum(
            abs,
            LowerTriangular(reduced.arrow.Sred - reference.arrow.Sred),
        ) / scale < T(1e-55)
        reference_schur = zeros(T, variable_count, variable_count)
        reduced_schur = similar(reference_schur)
        SDPX.materialize_schur!(reference_schur, reference)
        SDPX.materialize_schur!(reduced_schur, reduced)
        @test maximum(abs, reduced_schur - reference_schur) /
              max(maximum(abs, reference_schur), one(T)) < T(1e-54)
        reduced_factor = SDPX.factor_arrow_kkt!(
            reduced,
            SDPX.SolverOptions{T}(verbosity=0),
        )
        @test reduced_factor.ok
        automatic_refinement =
            SDPX.SolverOptions{T}(verbosity=0, refine_policy=:auto)
        @test SDPX._skip_automatic_refinement(
            reduced,
            automatic_refinement,
            reduced_factor,
        )
        @test !SDPX._skip_automatic_refinement(
            reference,
            automatic_refinement,
            reference_factor,
        )
        @test !SDPX._skip_automatic_refinement(
            reduced,
            SDPX.SolverOptions{T}(
                verbosity=0,
                refine_policy=:adaptive,
            ),
            reduced_factor,
        )
        @test !SDPX._skip_automatic_refinement(
            reduced,
            SDPX.SolverOptions{T}(
                verbosity=0,
                refine_policy=:auto,
                ϵ_gap=T(1e-40),
                ϵ_primal=T(1e-40),
                ϵ_dual=T(1e-40),
            ),
            reduced_factor,
        )
        @test !SDPX._skip_automatic_refinement(
            reduced,
            automatic_refinement,
            merge(reduced_factor, (reg_attempts=1,)),
        )

        right_hand_side = T.(1:variable_count) ./ T(7)
        reference_product = zeros(T, variable_count)
        reduced_product = zeros(T, variable_count)
        SDPX.schur_mul!(
            reference_product,
            reference,
            right_hand_side,
            one(T),
            zero(T),
        )
        SDPX.schur_mul!(
            reduced_product,
            reduced,
            right_hand_side,
            one(T),
            zero(T),
        )
        @test maximum(abs, reduced_product - reference_product) /
              max(maximum(abs, reference_product), one(T)) < T(1e-54)
        fill!(reduced_product, one(T))
        SDPX.schur_mul!(
            reduced_product,
            reduced,
            right_hand_side,
            T(2),
            T(3),
        )
        @test maximum(
            abs,
            reduced_product .- (T(2) .* reference_product .+ T(3)),
        ) / max(maximum(abs, reference_product), one(T)) < T(1e-54)

        reference_solution = zeros(T, variable_count)
        reduced_solution = zeros(T, variable_count)
        SDPX.solve_arrow_kkt!(
            reference,
            right_hand_side,
            reference_solution,
        )
        SDPX.solve_arrow_kkt!(
            reduced,
            right_hand_side,
            reduced_solution,
        )
        @test maximum(abs, reduced_solution - reference_solution) /
              max(maximum(abs, reference_solution), one(T)) < T(1e-54)
    end

    @testset "native BigFloat automatic refinement skip" begin
        setprecision(BigFloat, 256) do
            T = BigFloat
            block_count = 4
            shared_count = 2
            variable_count = shared_count + block_count
            coefficients = [
                zeros(T, variable_count, 2, 2)
                for _ in 1:block_count
            ]
            for block in 1:block_count
                for variable in 1:shared_count
                    coefficients[block][variable, 1, 2] =
                        T(variable + block) / T(17)
                    coefficients[block][variable, 2, 1] =
                        coefficients[block][variable, 1, 2]
                    coefficients[block][variable, 2, 2] =
                        T(variable + 2block) / T(23)
                end
                coefficients[
                    block
                ][shared_count + block, 1, 1] = one(T)
            end
            problem = SDPX.ingest(
                ones(T, variable_count),
                coefficients,
                [zeros(T, 2, 2) for _ in 1:block_count],
                zeros(T, variable_count, 0),
                T[];
                sparse=true,
                verbosity=0,
            )
            workspace = SDPX.Workspace(
                problem;
                thread_count=1,
                extended_precision_blas=:off,
            )
            X = [Matrix{T}(I, 2, 2) for _ in 1:block_count]
            Y = [Matrix{T}(I, 2, 2) for _ in 1:block_count]
            @test SDPX.factor_blocks!(workspace, X, Y)
            SDPX.schur_build!(
                workspace,
                problem,
                problem.cons,
                X,
                Y,
            )
            factor = SDPX.factor_arrow_kkt!(
                workspace,
                SDPX.SolverOptions{T}(verbosity=0),
            )
            @test factor.ok
            automatic = SDPX.SolverOptions{T}(
                verbosity=0,
                refine_policy=:auto,
                ϵ_gap=T("1e-10"),
                ϵ_primal=T("1e-10"),
                ϵ_dual=T("1e-10"),
            )
            @test SDPX._skip_automatic_refinement(
                workspace,
                automatic,
                factor,
            )
            @test !SDPX._skip_automatic_refinement(
                workspace,
                SDPX.SolverOptions{T}(
                    verbosity=0,
                    refine_policy=:auto,
                    ϵ_gap=T("1e-50"),
                    ϵ_primal=T("1e-50"),
                    ϵ_dual=T("1e-50"),
                ),
                factor,
            )
            @test !SDPX._skip_automatic_refinement(
                workspace,
                automatic,
                merge(factor, (reg_attempts=1,)),
            )
        end
    end

    @testset "singleton-arrow automatic crossover" begin
        T = Float64x4
        @test SDPX.SolverOptions{T}().extended_precision_blas === :auto
        @test SDPX.SolverOptions{Float64}().extended_precision_blas === :off
        @test SDPX.SolverOptions{BigFloat}().extended_precision_blas === :auto
        block_count = 192
        shared_count = 32
        variable_count = shared_count + block_count
        coefficients = [
            zeros(T, variable_count, 2, 2)
            for _ in 1:block_count
        ]
        for block in 1:block_count
            for variable in 1:shared_count
                coefficients[block][variable, 1, 2] =
                    T(variable + block) / T(1000)
                coefficients[block][variable, 2, 1] =
                    coefficients[block][variable, 1, 2]
                coefficients[block][variable, 2, 2] =
                    T(variable + 2block) / T(1700)
            end
            coefficients[block][shared_count + block, 1, 1] = one(T)
        end
        problem = SDPX.ingest(
            ones(T, variable_count),
            coefficients,
            [zeros(T, 2, 2) for _ in 1:block_count],
            zeros(T, variable_count, 0),
            T[];
            sparse=true,
            verbosity=0,
        )
        threads = min(Threads.nthreads(), 4)
        decision = SDPX._reduced_arrow_crossover(
            problem,
            T,
            :auto,
            0.10,
            threads;
            available_memory_bytes=2^30,
        )
        @test decision.enabled
        @test decision.reason == :predicted_speedup
        @test decision.estimated_speedup >=
              EPBLAS.static_profile(:fixed_extended).minimum_speedup

        options = SDPX.SolverOptions{T}(
            extended_precision_blas=:auto,
            threads=threads,
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(problem, options)
        workspace = SDPX.Workspace(
            problem;
            thread_count=threads,
            extended_precision_blas=:auto,
        )
        @test plan.gram_kernel in (
            :reduced_arrow_multifloatvec4_syrk,
            :reduced_arrow_threaded_multifloatvec4_syrk,
        )
        @test workspace.arrow.reduced_panel_enabled
    end

    @testset "triangular syrk kernel" begin
        for T in (Float64x4, BigFloat)
            T === BigFloat && setprecision(BigFloat, 256)
            panel = T.(reshape(1:126, 9, 14)) ./ T(37)
            output = zeros(T, 14, 14)
            T === BigFloat && EPBLAS.prepare_storage!(output)
            config =
                EPBLAS.KernelConfig(row_tile=4, column_tile=4, micro_tile=2)
            EPBLAS.syrk!(
                output,
                panel,
                one(T),
                zero(T),
                config,
                min(Threads.nthreads(), 4),
            )
            reference = transpose(panel) * panel
            tolerance = T === BigFloat ? big"1e-65" : T(1e-55)
            @test maximum(abs, LowerTriangular(output - reference)) <
                  tolerance
            @test all(iszero, triu(output, 1))
            fill!(output, T(3))
            T === BigFloat && EPBLAS.prepare_storage!(output)
            EPBLAS.syrk!(
                output,
                panel,
                T(2),
                T(3),
                config,
                min(Threads.nthreads(), 4),
            )
            @test maximum(
                abs,
                LowerTriangular(output - (T(2) .* reference .+ T(9))),
            ) < tolerance
            @test all(
                output[row, column] == T(3)
                for column in axes(output, 2) for row in 1:(column - 1)
            )
        end
    end

    @testset "crossover policy" begin
        sparse_features = EPBLAS.CrossoverFeatures(
            rows=1024,
            columns=500,
            matrix_dimension=32,
            average_nnz=3.0,
            active_density=0.8,
            expected_schur_density=0.9,
            thread_count=4,
            memory_budget_bytes=2^30,
            sparse_input=true,
        )
        sparse_decision =
            EPBLAS.choose_crossover(Float64x4, sparse_features; mode=:auto)
        @test !sparse_decision.enabled
        @test sparse_decision.reason == :sparse_outer_product_cheaper

        small_block_features = EPBLAS.CrossoverFeatures(
            rows=4,
            columns=200,
            matrix_dimension=2,
            average_nnz=3.0,
            active_density=0.9,
            expected_schur_density=0.155,
            thread_count=8,
            memory_budget_bytes=2^30,
            sparse_input=true,
        )
        @test EPBLAS.choose_crossover(
            Float64x4,
            small_block_features;
            mode=:auto,
        ).reason == :specialized_small_block
        @test EPBLAS.choose_crossover(
            BigFloat,
            small_block_features;
            mode=:auto,
        ).enabled

        dense_features = EPBLAS.CrossoverFeatures(
            rows=64,
            columns=256,
            matrix_dimension=8,
            average_nnz=64.0,
            active_density=1.0,
            expected_schur_density=1.0,
            thread_count=4,
            memory_budget_bytes=2^30,
            sparse_input=false,
        )
        @test EPBLAS.choose_crossover(
            Float64x4,
            dense_features;
            mode=:auto,
        ).enabled
        @test !EPBLAS.choose_crossover(
            Float64,
            dense_features;
            mode=:on,
        ).enabled
    end

    @testset "dense and sparse Schur integration" begin
        for T in (Float64x4, BigFloat)
            T === BigFloat && setprecision(BigFloat, 256)
            for problem_builder in (
                _extended_dense_problem,
                _extended_sparse_problem,
            )
                problem = problem_builder(T)
                old_workspace = SDPX.Workspace(problem)
                new_workspace = SDPX.Workspace(
                    problem;
                    extended_precision_blas=:on,
                    extended_precision_memory_fraction=0.05,
                )
                dimensions = problem.dims.k
                X = [
                    Matrix{T}(I, dimension, dimension)
                    for dimension in dimensions
                ]
                Y = [
                    Matrix{T}(I, dimension, dimension)
                    for dimension in dimensions
                ]
                @test SDPX.factor_blocks!(old_workspace, X, Y)
                @test SDPX.factor_blocks!(new_workspace, X, Y)
                SDPX.schur_build!(
                    old_workspace,
                    problem,
                    problem.cons,
                    X,
                    Y,
                )
                SDPX.schur_build!(
                    new_workspace,
                    problem,
                    problem.cons,
                    X,
                    Y,
                )
                old_schur = zeros(T, problem.dims.m, problem.dims.m)
                new_schur = similar(old_schur)
                SDPX.materialize_schur!(old_schur, old_workspace)
                SDPX.materialize_schur!(new_schur, new_workspace)
                relative_error =
                    norm(new_schur - old_schur) /
                    max(norm(old_schur), one(T))
                tolerance = T === BigFloat ? big"1e-60" : T(1e-48)
                @test relative_error < tolerance
                @test new_workspace.extended_precision.lower_only
            end
        end
    end
end
