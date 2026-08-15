using LinearAlgebra
using SparseArrays
using Test

@testset "frozen CSC sparse LP normal equations" begin
    G = sparse(
        [1, 2, 2, 3, 3, 4, 4],
        [1, 1, 2, 2, 3, 3, 4],
        [2.0, 1.0, 2.5, 1.0, 2.0, 1.0, 2.0],
        4,
        4,
    )
    B = spzeros(Float64, 4, 0)
    system = SDPX.lp_sparse_candidate(G, B, Float64; storage=:sparse)
    @test system isa SDPX.LPSparseSystem{Float64}
    system = system::SDPX.LPSparseSystem{Float64}
    @test system.backend isa SDPX.CHOLMODSparseCholeskyBackend
    @test system.storage isa SDPX.SparseKKTStorage{Float64}
    @test system.K isa SparseMatrixCSC

    sparse_workspace = SDPX.LPWorkspace(
        Float64,
        size(G, 1),
        size(G, 2),
        0;
        sparse_storage=true,
    )
    @test isempty(sparse_workspace.H)
    @test isempty(sparse_workspace.K)
    @test isempty(sparse_workspace.weighted_G)

    weights = ones(Float64, size(G, 1))
    regularization = 1e-8
    @test SDPX.lp_sparse_factor!(system, weights, regularization)
    rhs = [1.0, -2.0, 0.5, 3.0]
    actual = SDPX.lp_sparse_solve!(copy(rhs), system)
    expected = (Matrix(transpose(G) * G) + regularization * I) \ rhs
    @test actual ≈ expected atol=1e-8

    # Explicit sparse equality KKT has no sparse pivoted-indefinite provider;
    # selection is fail-closed before any factorization or dense fallback.
    @test_throws ArgumentError SDPX.select_lp_formulation(
        dimension=4,
        nonzeros=8,
        equalities=1,
        arithmetic=Float64,
        storage=:sparse,
    )

    # The pre-execution storage plan also rejects this combination, before LP
    # row extraction or workspace allocation can materialize a dense proxy.
    equality_problem = SDPX.linear_program(
        ones(4),
        G,
        fill(-1.0, 4);
        Aeq=sparse([1], [1], [1.0], 1, 4),
        beq=[0.0],
        sparse=true,
        verbosity=0,
    )
    @test_throws ArgumentError SDPX.build_execution_plan(
        equality_problem,
        SDPX.SolverOptions{Float64}(sparse=:sparse, verbosity=0),
    )

    # An authoritative auto-sparse plan uses the same direct CSC ingress.  A
    # cycle overlap keeps the Schur graph sparse without triggering the
    # diagonal/reduced LP special case.
    dimension = 240
    rows = [cld(position, 2) for position in 1:(2dimension)]
    columns = [
        isodd(position) ? cld(position, 2) :
        mod1(cld(position, 2) + 1, dimension)
        for position in 1:(2dimension)
    ]
    auto_G = sparse(rows, columns, ones(2dimension), dimension, dimension)
    auto_problem = SDPX.linear_program(
        ones(dimension),
        auto_G,
        fill(-1.0, dimension);
        sparse=true,
        verbosity=0,
    )
    auto_plan = SDPX.build_execution_plan(
        auto_problem,
        SDPX.SolverOptions{Float64}(sparse=:auto, verbosity=0),
    )
    @test auto_plan.storage_plan.storage === :sparse
    ingress_G, _ = SDPX._extract_lp_rows_sparse(auto_problem)
    @test ingress_G isa SparseMatrixCSC{Float64,Int}
    @test nnz(ingress_G) == nnz(auto_G)
end
