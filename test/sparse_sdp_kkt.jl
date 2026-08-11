using SDPX
using LinearAlgebra
using SparseArrays
using Test

@testset "sparse Schur SDP KKT" begin
    variables = 8
    block_size = 3
    block_count = 4
    coefficients = [
        zeros(Float64, variables, block_size, block_size)
        for _ in 1:block_count
    ]
    for block in 1:block_count, variable in 1:variables
        mod(variable + block, 3) == 0 && continue
        diagonal = mod1(variable + block, block_size)
        coefficients[block][variable, diagonal, diagonal] =
            0.5 + 0.1 * variable + 0.05 * block
        other = mod1(diagonal + 1, block_size)
        coefficients[block][variable, diagonal, other] =
            0.02 * (variable + block)
        coefficients[block][variable, other, diagonal] =
            coefficients[block][variable, diagonal, other]
    end
    equality = sparse(
        [1, 3, 5, 7, 2, 4, 6, 8],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1.0, -0.5, 0.75, 0.2, 0.3, 1.0, -0.4, 0.8],
        variables,
        2,
    )
    problem = ingest(
        ones(variables),
        coefficients,
        [zeros(block_size, block_size) for _ in 1:block_count],
        equality,
        zeros(2);
        sparse=true,
        verbosity=0,
    )
    primal = [
        Matrix(2.0I, block_size, block_size)
        for _ in 1:block_count
    ]
    dual = [
        Matrix(1.5I, block_size, block_size)
        for _ in 1:block_count
    ]

    dense = SDPX.Workspace(problem; thread_count=1)
    @test SDPX.factor_blocks!(dense, primal, dual)
    SDPX.threaded_schur_build!(
        dense,
        problem,
        problem.cons,
        primal,
        dual,
    )
    expected_schur = Matrix(Symmetric(dense.S, :L))

    sparse_workspace = SDPX.Workspace(problem; thread_count=1)
    sparse_workspace.sparse_kkt =
        SDPX._sparse_schur_sdp_workspace(problem, 1)
    sparse_workspace.backend_config = SDPX.BackendConfiguration(
        :sparse_schur_cholesky,
        :auto,
        false,
        false,
        :off,
        (),
        false,
    )
    sparse_workspace.backend = SDPX.SparseSchurBackend()
    sparse_workspace.dense_sparse_assembly = false
    for block in 1:block_count
        active_count = length(problem.cons.schur_order[block])
        sparse_workspace.blk[block] = SDPX.BlockWS{Float64}(
            block_size,
            0,
            active_count * (active_count + 1) ÷ 2,
        )
    end
    @test SDPX.factor_blocks!(sparse_workspace, primal, dual)
    for block in 1:block_count
        SDPX.sparse_schur_block!(
            sparse_workspace.blk[block],
            problem.cons,
            block,
            primal[block],
            dual[block],
        )
    end
    SDPX.reduce_sparse_schur_csc!(
        sparse_workspace,
        problem.cons,
    )
    storage =
        sparse_workspace.sparse_kkt::SDPX.SparseSchurSDPWorkspace
    actual_schur = Matrix(
        Symmetric(Matrix(storage.matrix), :L),
    )
    @test actual_schur == expected_schur

    options = SolverOptions{Float64}(verbosity=0)
    sparse_backend = SDPX.select_backend(sparse_workspace)
    @test sparse_backend isa SDPX.SparseSchurBackend
    factor = SDPX.factorize!(
        sparse_backend,
        sparse_workspace,
        problem,
        options,
    )
    @test factor.ok
    @test storage.regularization == sqrt(eps(Float64))
    @test storage.factorization_quality > 0.0
    @test all(
        isapprox(sparse_workspace.Q[index, index], 1.0; atol=1e-14)
        for index in axes(sparse_workspace.Q, 1)
    )

    primal_rhs = collect(range(0.2, 1.1; length=variables))
    equality_rhs = [0.3, -0.2]
    SDPX.solve!(
        sparse_backend,
        sparse_workspace,
        2,
        primal_rhs,
        equality_rhs,
        sparse_workspace.dx,
        sparse_workspace.dy,
    )
    sparse_workspace.p .= equality_rhs
    refinement = SDPX.refine!(
        sparse_backend,
        sparse_workspace,
        problem,
        SolverOptions{Float64}(
            verbosity=0,
            refine_policy=:adaptive,
            refine_max_steps=8,
        ),
        primal_rhs,
    )
    @test SDPX._automatic_refinement_relative_tolerance(
        sparse_workspace,
        SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
        ),
    ) == 1e-10
    residual = maximum(
        abs,
        vcat(
            primal_rhs -
            expected_schur * sparse_workspace.dx +
            Matrix(equality) * sparse_workspace.dy,
            equality_rhs -
            transpose(Matrix(equality)) * sparse_workspace.dx,
        ),
    )
    @test refinement[1] > 0
    @test refinement[1] < 8
    @test residual <= 1e-11

    # Numeric refactorization must reuse the analysis; hashing the 139-million
    # entry B3 pattern on every iteration would itself be a material cost.
    storage.equality_requires_pivoting = true
    second_factor = SDPX.factorize!(
        sparse_backend,
        sparse_workspace,
        problem,
        options,
    )
    @test second_factor.ok
    @test second_factor.q_pivoted
    @test !second_factor.q_rank_deficient
    @test sparse_workspace.Qchol isa LinearAlgebra.CholeskyPivoted
    statistics = SDPX.statistics(storage.backend)
    @test statistics.analyses == 1
    @test statistics.factorizations == 2
end
