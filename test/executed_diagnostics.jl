using LinearAlgebra
using Random
using SDPX
using SparseArrays
using Test

# Diagnostics must report the algorithms that ran, not the ones the plan
# chose before presolve. The LP path selects its sparse Newton system at
# runtime, after the plan is frozen, and the record previously copied the
# plan: a solve that executed sparse Cholesky with a sparse Gram product
# reported `:positive_definite_cholesky` and `:blas_syrk`. Every benchmark
# table built from diagnostics inherited that. (Maintainer review P2.4.)
@testset "diagnostics report executed algorithms" begin
    @testset "sparse LP reports its runtime backend" begin
        rng = MersenneTwister(11)
        variables, base_rows, per_row = 220, 900, 2
        G = spzeros(Float64, base_rows, variables)
        for row in 1:base_rows, column in randperm(rng, variables)[1:per_row]
            G[row, column] = randn(rng)
        end
        G = [G; sparse(1.0I, variables, variables); -sparse(1.0I, variables, variables)]
        rows = size(G, 1)
        interior = randn(rng, variables)
        h = G * interior .- 1.0
        multipliers = rand(rng, rows) .+ 0.5
        objective = vec(transpose(G) * multipliers)
        coefficients = [
            [sparse([1], [1], [G[row, column]], 1, 1) for column in 1:variables]
            for row in 1:rows
        ]
        constants = [reshape([h[row]], 1, 1) for row in 1:rows]
        problem = SDPX.ingest(objective, coefficients, constants,
            zeros(variables, 0), Float64[]; sparse=true, verbosity=0)

        result = SDPX.solve(problem; tolerance=1e-9, verbosity=0, diagnostics=true)
        selected = result.diagnostics.selected_algorithms
        @test result.status == SDPX.Optimal
        # The gate selects the sparse system for this problem; the record must
        # say so, and must not claim a Gram kernel that never assembled.
        @test selected.kkt === :sparse_normal
        @test selected.gram === :sparse_gram
        # The plan stays visible under its own name rather than silently
        # replaced, so a plan/executed divergence is observable, not hidden.
        @test selected.planned.kkt !== :sparse_normal
    end

    @testset "dense LP reports the dense backend it used" begin
        variables = 3
        rows = Matrix{Float64}([1.0I(variables); -1.0I(variables)])
        righthand = [-ones(variables); -3ones(variables)]
        blocks = [zeros(variables, 1, 1) for _ in 1:(2variables)]
        for row in 1:(2variables), column in 1:variables
            blocks[row][column, 1, 1] = rows[row, column]
        end
        constants = [reshape([righthand[row]], 1, 1) for row in 1:(2variables)]
        problem = SDPX.ingest(ones(variables), blocks, constants,
            zeros(variables, 0), Float64[]; sparse=false, verbosity=0)

        result = SDPX.solve(problem; tolerance=1e-9, verbosity=0, diagnostics=true)
        selected = result.diagnostics.selected_algorithms
        @test result.status == SDPX.Optimal
        @test selected.kkt === :positive_definite_cholesky
        @test selected.gram === :blas_syrk
        @test selected.planned == (kkt=selected.kkt, gram=selected.gram)
    end

    @testset "SDP core reports its executed KKT backend" begin
        coefficients = zeros(2, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[2, 2, 2] = 1.0
        problem = SDPX.ingest([2.0, 3.0], [coefficients], [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0), Float64[]; verbosity=0)
        result = SDPX.solve(problem; tolerance=1e-8, verbosity=0, diagnostics=true)
        @test result.status == SDPX.Optimal
        @test result.diagnostics.selected_algorithms.kkt === :dense_cholesky
    end
end
