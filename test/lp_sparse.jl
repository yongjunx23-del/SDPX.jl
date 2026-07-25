using LinearAlgebra
using Random
using SDPX
using SparseArrays
using Test

@testset "sparse LP linear system" begin
    @testset "sign convention matches the dense factorization" begin
        # The symmetric augmented form yields `-y` where the dense unsymmetric
        # form yields `y`, and the two also differ by O(δ) in the multiplier
        # because the regularization enters the equality block with opposite
        # sign. Neither difference fails loudly, so both are pinned here
        # against a direct dense solve of the convention `_lp_populate_kkt!`
        # implements.
        rng = MersenneTwister(1)
        variables, equalities, rows = 6, 2, 12
        G = randn(rng, rows, variables)
        H = transpose(G) * G
        B = randn(rng, variables, equalities)
        regularization = 1e-8
        r1 = randn(rng, variables)
        r2 = randn(rng, equalities)

        dense = [
            (H + regularization * I) (-B)
            transpose(B) (-regularization * I)
        ]
        expected = dense \ [r1; r2]

        weights = ones(Float64, rows)
        system = SDPX.LPSparseSystem{Float64}(
            sparse(G),
            sparse(B),
            spzeros(0, 0),
            SDPX.SparseLDLBackend(),
            :sparse_ldl,
            variables,
            equalities,
            false,
            nothing,
        )
        @test SDPX.lp_sparse_factor!(system, weights, regularization)
        actual = SDPX.lp_sparse_solve!([r1; r2], system)

        # Both blocks agree only to the O(δ) by which the two regularizations
        # differ -- measured 6.6e-08 on the primal and 4.4e-07 on the
        # multiplier at δ = 1e-8, against solution components of order 1 and 5
        # respectively. The tolerances are loose enough to admit that and far
        # too tight to admit a wrong answer.
        @test actual[1:variables] ≈ expected[1:variables] atol = 1e-6
        multiplier = actual[(variables + 1):(variables + equalities)]
        @test multiplier ≈ expected[(variables + 1):(variables + equalities)] atol =
            1e-5
        # The sign itself is the part that must be exactly right: a flipped
        # multiplier would still pass a loose tolerance against zero, so
        # compare against the *negated* solution too and require it to be far.
        @test norm(multiplier + expected[(variables + 1):(variables + equalities)]) >
              norm(multiplier) / 2
    end

    @testset "formulation gate follows measured fill, not G's sparsity" begin
        # The crossover was measured at ~13 nonzeros per row of the *assembled*
        # system. `GᵀDG` can be far denser than `G`, so the gate must be
        # evaluated after assembly.
        @test SDPX.select_lp_formulation(;
            dimension=800,
            nonzeros=800 * 6,
            equalities=0,
            arithmetic=Float64,
        ) === :sparse_normal
        @test SDPX.select_lp_formulation(;
            dimension=800,
            nonzeros=800 * 20,
            equalities=0,
            arithmetic=Float64,
        ) === :dense_lu
        # Equalities make the system indefinite, so Cholesky no longer applies.
        @test SDPX.select_lp_formulation(;
            dimension=800,
            nonzeros=800 * 6,
            equalities=5,
            arithmetic=Float64,
        ) === :sparse_ldl
        # Below the minimum dimension the dense path wins regardless.
        @test SDPX.select_lp_formulation(;
            dimension=50,
            nonzeros=50,
            equalities=0,
            arithmetic=Float64,
        ) === :dense_lu
    end

    @testset "a sparse LP solves to the same optimum as the dense path" begin
        # Built strictly feasible on both sides, so `Optimal` is the only
        # correct answer and a regression shows up as a status change rather
        # than as a slightly different number.
        rng = MersenneTwister(11)
        variables, base_rows, per_row = 220, 900, 2
        G = spzeros(Float64, base_rows, variables)
        for row in 1:base_rows,
            column in randperm(rng, variables)[1:per_row]

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

        problem = SDPX.ingest(
            objective,
            coefficients,
            constants,
            zeros(variables, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        # The gate must actually select the sparse path here, or the rest of
        # this testset silently exercises the dense one.
        system = SDPX._lp_sparse_system(
            problem,
            Matrix(G),
            zeros(Float64, variables, 0),
        )
        @test system isa SDPX.LPSparseSystem{Float64}
        @test system.formulation === :sparse_normal

        result = SDPX.solve(problem; tolerance=1e-9, verbosity=0)
        @test result.status == SDPX.Optimal
        @test abs(result.pObj - result.dObj) /
              max(1.0, abs(result.pObj)) < 1e-7
    end
end
