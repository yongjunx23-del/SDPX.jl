using LinearAlgebra
using Random
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

    # `_lp_workspace_bytes` must include the sparse solver-owned graph after
    # factorization, while excluding model ingress G/B.  The one snapshot
    # graph deduplicates K/storage.matrix and shared symbolic state.
    sparse_workspace = SDPX.LPWorkspace(
        Float64,
        size(G, 1),
        size(G, 2),
        0;
        sparse_storage=true,
    )
    empty_bytes = SDPX._lp_workspace_bytes(sparse_workspace)
    sparse_workspace.sparse_system = system
    populated_bytes = SDPX._lp_workspace_bytes(sparse_workspace)
    sparse_snapshot = (
        K=system.K,
        storage=system.storage,
        assembly_map=system.assembly_map,
        backend=system.backend,
    )
    @test populated_bytes > empty_bytes
    @test populated_bytes - empty_bytes == Base.summarysize(sparse_snapshot)

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

# A2 — LP final-route freeze.  Red by design until the A2 source lands.
@testset "A2 explicit sparse equality is fail-closed" begin
    @test_throws ArgumentError SDPX.select_lp_formulation(
        dimension=4,
        nonzeros=8,
        equalities=1,
        arithmetic=Float64,
        storage=:sparse,
    )
    G = sparse([1, 2, 3, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], 3, 2)
    equality_problem = SDPX.linear_program(
        [1.0, 2.0],
        G,
        [1.0, 1.0, 3.0];
        Aeq=sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2),
        beq=[3.0],
        sparse=true,
        verbosity=0,
    )
    @test_throws ArgumentError SDPX.build_execution_plan(
        equality_problem,
        SDPX.SolverOptions{Float64}(sparse=:sparse, verbosity=0),
    )
end

@testset "A2 auto sparse rejection freezes the dense route" begin
    # SparseCons input whose measured G'G pattern is dense: `sparse=:auto`
    # keeps coefficient storage sparse but the Schur structure plan dense, so
    # the measured pattern probe decides and the finalized route is dense.
    rng = MersenneTwister(7)
    variables, base_rows, per_row = 220, 900, 2
    G = spzeros(Float64, base_rows, variables)
    for row in 1:base_rows, column in randperm(rng, variables)[1:per_row]
        G[row, column] = randn(rng)
    end
    # A fully dense first row fills G'G completely.
    G = [
        sparse(ones(Float64, 1, variables));
        G;
        sparse(1.0I, variables, variables);
        -sparse(1.0I, variables, variables);
    ]
    rows = size(G, 1)
    interior = randn(rng, variables)
    h = G * interior .- 1.0
    multipliers = rand(rng, rows) .+ 0.5
    objective = vec(transpose(G) * multipliers)
    lp = SDPX.linear_program(objective, G, h; sparse=:auto, verbosity=0)
    @test lp.cons isa SDPX.SparseCons{Float64}
    @test lp.structure.schur_plan.storage === :dense

    result = SDPX.solve(lp; tolerance=1e-9, verbosity=0, diagnostics=true)
    payload = result.diagnostics.plan.payload
    @test payload isa SDPX.LPRoutePlan
    @test payload.route === :positive_definite_cholesky
    @test payload.storage === :dense
    @test payload.provider === :blas_lapack
    # The measured pattern probe runs exactly once and rejects.
    @test payload.sparse_probe_count == 1
    record = only(result.diagnostics.attempts)
    @test payload.route === record.executed.formulation
    @test payload.storage === record.executed.storage
    @test payload.provider === record.executed.provider
end
