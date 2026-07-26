using LinearAlgebra
using Random
using SDPX
using SparseArrays
using Test

@testset "sparse LP linear system" begin
    @testset "sign convention matches the dense factorization" begin
        # The symmetric augmented form yields `-y` where the dense unsymmetric
        # form yields `y`. `lp_sparse_solve!` is the single place that converts,
        # and getting it wrong is silent, so it is pinned here against a direct
        # dense solve of the convention `_lp_populate_kkt!` implements.
        rng = MersenneTwister(1)
        variables, equalities, rows = 6, 2, 12
        G = randn(rng, rows, variables)
        H = transpose(G) * G
        B = randn(rng, variables, equalities)
        regularization = 1e-8
        r1 = randn(rng, variables)
        r2 = randn(rng, equalities)

        # The dense convention, written out: `+δ` on the equality block, which
        # is what makes it agree with the symmetric form below.
        dense = [
            (H + regularization * I) (-B)
            transpose(B) (regularization * I)
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
        )
        @test SDPX.lp_sparse_factor!(system, weights, regularization)
        actual = SDPX.lp_sparse_solve!([r1; r2], system)

        # The two now solve the same regularized system, so they agree to
        # conditioning rather than to O(δ). The previous version of this test
        # asserted the O(δ) gap as though it were expected; it was a sign
        # mismatch in the equality block.
        @test actual[1:variables] ≈ expected[1:variables] atol = 1e-6
        multiplier = actual[(variables + 1):(variables + equalities)]
        @test multiplier ≈ expected[(variables + 1):(variables + equalities)] atol =
            1e-6
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

    @testset "dense and sparse KKT solve the same regularized system" begin
        # The two backends must not disagree about what problem they are
        # solving. Before the equality block was aligned, the difference grew
        # linearly with the regularization -- 2.2e-07 at delta 1e-8 and 0.22 at
        # 1e-2 -- and the LP loop escalates delta by ten up to eight times on a
        # hard factorization, so the large end was reachable in ordinary use.
        rng = MersenneTwister(1)
        variables, equalities, rows = 8, 3, 16
        G = randn(rng, rows, variables)
        H = transpose(G) * G
        B = randn(rng, variables, equalities)
        primal_rhs = randn(rng, variables)
        equality_rhs = randn(rng, equalities)

        for regularization in (1e-8, 1e-6, 1e-4, 1e-2)
            K = zeros(variables + equalities, variables + equalities)
            SDPX._lp_populate_kkt!(K, H, B, regularization)
            dense = lu(K) \ [primal_rhs; equality_rhs]

            system = SDPX.LPSparseSystem{Float64}(
                sparse(G), sparse(B), spzeros(0, 0),
                SDPX.SparseLDLBackend(), :sparse_ldl,
                variables, equalities, false,
            )
            @test SDPX.lp_sparse_factor!(system, ones(rows), regularization)
            sparse_direction =
                SDPX.lp_sparse_solve!([primal_rhs; equality_rhs], system)

            # Agreement improves as the system becomes better conditioned,
            # which is the signature of a shared formulation. A sign mismatch
            # shows the opposite trend.
            @test norm(dense - sparse_direction) < 1e-6 * max(1.0, regularization)
        end

        # State the convention once, so a future edit has something to check
        # against rather than rediscovering it from a failing solve.
        K = zeros(variables + equalities, variables + equalities)
        SDPX._lp_populate_kkt!(K, H, B, 0.25)
        @test K[1:variables, 1:variables] ≈ H + 0.25I
        @test K[1:variables, (variables + 1):end] ≈ -B
        @test K[(variables + 1):end, 1:variables] ≈ transpose(B)
        # Positive, matching the symmetric quasi-definite form the sparse
        # backend factors -- not negative.
        @test K[(variables + 1):end, (variables + 1):end] ≈ 0.25I
    end


    @testset "the first factorization is not performed twice" begin
        # `factorize!` analyses on its own when the pattern is new, so an
        # explicit `analyze!` beforehand factored the same matrix twice --
        # a full extra numeric factorization on the first iteration of every
        # sparse LP solve, which showed as factorizations = 2 after one call.
        rng = MersenneTwister(2)
        variables, rows = 40, 90
        G = sprandn(rng, rows, variables, 0.05) + sparse(1.0I, rows, variables)
        system = SDPX.LPSparseSystem{Float64}(
            G, spzeros(variables, 0), spzeros(0, 0),
            SDPX.SparseCholeskyBackend(), :sparse_normal,
            variables, 0, false,
        )

        @test SDPX.lp_sparse_factor!(system, ones(rows), 1e-8)
        first = SDPX.statistics(system.backend)
        @test first.analyses == 1
        @test first.factorizations == 1

        # Later calls change only the values, so they must reuse the symbolic
        # analysis rather than repeating it.
        for call in 2:4
            @test SDPX.lp_sparse_factor!(system, ones(rows) .+ 0.1 * call, 1e-8)
        end
        later = SDPX.statistics(system.backend)
        @test later.analyses == 1
        @test later.factorizations == 4
        @test later.symbolic_reuse_ratio > 0.7
    end

end
