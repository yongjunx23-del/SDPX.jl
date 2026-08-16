using SDPX
using LinearAlgebra
using SparseArrays
using Test

@testset "Round-7 frozen sparse Schur structure" begin
    active = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    pattern = SDPX.schur_pattern_csc(active, 6, Float64)
    storage = SDPX.freeze_sparse_csc(
        pattern;
        provider=SDPX.CHOLMODSparseProvider(),
    )
    map = SDPX.schur_assembly_map(active, storage)
    signature = storage.pattern_signature
    values = [fill(0.0, length(range)) for range in map.block_ranges]
    for block in values
        block .= [10.0, 0.0, 10.0]
    end
    SDPX.assemble_sparse_schur!(storage, map, values)
    @test storage.pattern_signature == signature
    @test length(map.position) == sum(length, map.block_ranges)
    @test nnz(storage.matrix) == 11
    @test storage.matrix[1, 1] == 10.0
    factor = SDPX.sparse_factor(
        storage.matrix;
        provider=SDPX.CHOLMODSparseProvider(),
        symbolic=storage.symbolic,
    )
    solution = zeros(6)
    SDPX.sparse_factor_solve!(solution, factor, ones(6))
    @test isapprox(sum(solution), 0.4; atol=1e-12)
    diagnostics = SDPX.schur_structure_diagnostics(storage, map)
    @test diagnostics.pattern_reused
    @test diagnostics.nnz == 11
    @test diagnostics.factor_nnz > 0
end

function _round7_chordal_fixture(edges; dimension=4)
    m = length(edges)
    A = [zeros(Float64, m, dimension, dimension)]
    for (variable, (row, column)) in pairs(edges)
        A[1][variable, row, column] = 1.0
        A[1][variable, column, row] = 1.0
    end
    return SDPX.ingest(
        ones(m),
        A,
        [zeros(Float64, dimension, dimension)],
        zeros(Float64, m, 0),
        Float64[];
        sparse=:dense,
        verbosity=0,
    )
end

@testset "Round-7 chordal analysis-only plans" begin
    path = _round7_chordal_fixture([(1, 2), (2, 3), (3, 4)])
    candidate = SDPX.chordal_plan(path, 1)
    @test candidate.is_chordal
    @test candidate.beneficial
    @test candidate.candidate
    @test !candidate.selected
    @test candidate.transformation === :none

    complete = _round7_chordal_fixture([
        (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
    ])
    not_beneficial = SDPX.chordal_plan(complete, 1)
    @test not_beneficial.is_chordal
    @test !not_beneficial.beneficial

    cycle = _round7_chordal_fixture([(1, 2), (2, 3), (3, 4), (4, 1)])
    non_chordal = SDPX.chordal_plan(cycle, 1)
    @test !non_chordal.is_chordal
    @test !non_chordal.selected
end

@testset "Round-7 generic sparse Schur primitive ownership" begin
    active = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    # BigFloat is the MPFR ownership-sensitive provider; fixed-width
    # MultiFloat provider contracts remain covered by the existing MFLA suite
    # while production sparse-Schur routing stays fail-closed for both.
    for T in (BigFloat,)
        storage, map = SDPX.freeze_schur_pattern(
            active,
            6,
            T;
            provider=SDPX.GenericSparseProvider(T),
        )
        values = [fill(zero(T), length(range)) for range in map.block_ranges]
        for block in values
            block .= T[10, 0, 10]
        end
        SDPX.assemble_sparse_schur!(storage, map, values)
        @test storage.pattern_signature == map.pattern_signature
        @test storage.matrix[1, 1] == T(10)
        @test all(isfinite, storage.matrix.nzval)
    end
end

function _round7_ab_problem(storage)
    # Duplicate the analytic 2x2 toy block so each variable belongs to both
    # blocks.  This deliberately disables the singleton block-arrow shortcut
    # and exercises the general sparse-Schur route without a large model.
    A = [zeros(Float64, 2, 2, 2) for _ in 1:2]
    C = [zeros(Float64, 2, 2) for _ in 1:2]
    for block in 1:2
        A[block][1, 1, 1] = 1.0
        A[block][2, 2, 2] = 1.0
        C[block][1, 2] = 1.0
        C[block][2, 1] = 1.0
    end
    return SDPX.ingest(
        [2.0, 3.0],
        A,
        C,
        zeros(Float64, 2, 0),
        Float64[];
        sparse=storage,
        verbosity=0,
    )
end

@testset "Round-7 equality-bearing sparse Schur fails closed" begin
    equality_free = _round7_ab_problem(:sparse)
    workspace = SDPX.Workspace(equality_free; thread_count=1)
    @test workspace.sparse_kkt isa SDPX.GenericSparseSchurSDPWorkspace{Float64}

    equality_problem = SDPX.SDPProblem{Float64}(
        equality_free.c,
        equality_free.C,
        reshape([1.0, 0.0], 2, 1),
        [0.0],
        equality_free.cons,
        (
            L=equality_free.dims.L,
            m=equality_free.dims.m,
            n=1,
            k=equality_free.dims.k,
        ),
        equality_free.structure,
    )
    options = SDPX.SolverOptions{Float64}(verbosity=0)
    @test_throws ArgumentError SDPX._factor_sparse_schur_sdp!(
        workspace, equality_problem, options,
    )
    @test_throws ArgumentError SDPX._solve_sparse_schur_kkt_owned!(
        workspace,
        1,
        zeros(2),
        zeros(1),
        zeros(2),
        zeros(1),
    )
end


@testset "Round-7 Float64 dense/sparse end-to-end A/B" begin
    dense = SDPX.solve(
        _round7_ab_problem(:dense);
        tolerance=1e-7,
        maximum_iterations=80,
        threads=1,
        verbosity=0,
        timing=false,
    )
    sparse = SDPX.solve(
        _round7_ab_problem(:sparse);
        tolerance=1e-7,
        maximum_iterations=80,
        threads=1,
        verbosity=0,
        timing=false,
    )
    @test dense.status === SDPX.Optimal
    @test sparse.status === SDPX.Optimal
    @test sparse.pObj ≈ dense.pObj atol=1e-7
    @test sparse.dObj ≈ dense.dObj atol=1e-7
    @test sparse.p_res <= 1e-7
    @test sparse.d_res <= 1e-7
    @test dense.diagnostics.selected_algorithms.certificate.valid
    @test sparse.diagnostics.selected_algorithms.certificate.valid
    @test sparse.termination.executed.executed_backend ===
          :sparse_schur_cholesky
    backend = sparse.termination.sparse_schur_backend
    @test backend.available
    @test backend.actual_nnz > 0
    @test backend.pattern_reuse > 0
    @test backend.psd_block_count == 2
    @test backend.overlap_edges == 1
    @test backend.chordal.analysis_only
end

function _round7_generic_ab_problem(::Type{T}, storage) where {T}
    A = [zeros(T, 2, 2, 2) for _ in 1:2]
    C = [zeros(T, 2, 2) for _ in 1:2]
    for block in 1:2
        A[block][1, 1, 1] = one(T)
        A[block][2, 2, 2] = one(T)
        C[block][1, 2] = one(T)
        C[block][2, 1] = one(T)
    end
    return SDPX.ingest(
        T[2, 3], A, C, zeros(T, 2, 0), T[];
        T=T,
        sparse=storage,
        verbosity=0,
    )
end

@testset "Round-7 generic sparse workspace estimate" begin
    problem = _round7_generic_ab_problem(BigFloat, :sparse)
    bytes = SDPX.estimate_sdp_workspace_bytes(problem, 1)
    @test bytes > 0
    @test bytes < typemax(Int)
end

# MPFR/MultiFloat IPM integration is opt-in because some Mac runners abort in
# native MPFR startup before Julia can report a test failure.  CI/cluster
# validation enables this gate to exercise the same dense-vs-frozen-sparse A/B
# contract as Float64.
if get(ENV, "SDPX_RUN_GENERIC_SPARSE_SCHUR_INTEGRATION", "0") == "1"
    @testset "Round-7 generic BigFloat/MultiFloat dense/sparse A/B" begin
        generic_types = Any[BigFloat]
        try
            @eval import MultiFloats
            push!(generic_types, MultiFloats.Float64x2)
        catch
            # BigFloat remains the required generic provider; fixed-width
            # MultiFloat is covered when the weak dependency is installed.
        end
        for T in generic_types
            @test SDPX.supports_sparse_generic(T)
            dense = SDPX.solve(
                _round7_generic_ab_problem(T, :dense);
                tolerance=1e-7,
                maximum_iterations=80,
                threads=1,
                verbosity=0,
                timing=false,
            )
            sparse = SDPX.solve(
                _round7_generic_ab_problem(T, :sparse);
                tolerance=1e-7,
                maximum_iterations=80,
                threads=1,
                verbosity=0,
                timing=false,
            )
            @test dense.status === SDPX.Optimal
            @test sparse.status === SDPX.Optimal
            @test sparse.pObj ≈ dense.pObj atol=1e-7
            @test sparse.dObj ≈ dense.dObj atol=1e-7
            @test sparse.p_res <= 1e-7
            @test sparse.d_res <= 1e-7
            @test dense.diagnostics.selected_algorithms.certificate.valid
            @test sparse.diagnostics.selected_algorithms.certificate.valid
            @test sparse.termination.executed.executed_backend ===
                  :sparse_schur_cholesky
            backend = sparse.termination.sparse_schur_backend
            @test backend.available
            @test backend.pattern_reuse > 0
            @test backend.actual_nnz > 0
            @test backend.psd_block_count == 2
            @test backend.overlap_edges == 1
        end
    end
end
