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
