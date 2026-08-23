using SDPX
using LinearAlgebra
using Random
using SparseArrays
using StableRNGs
using Test

function sparse_block_data(::Type{T}=Float64) where {T}
    m, k, L = 8, 2, 3
    A = [zeros(T, m, k, k) for _ in 1:L]
    active = [[1, 2, 5], [1, 3, 6], [1, 4, 7, 8]]
    for l in 1:L
        for (p, i) in pairs(active[l])
            A[l][i, 1, 1] = T(1 + p)
            A[l][i, 2, 2] = T(2 + l + p)
            if isodd(i + l)
                A[l][i, 1, 2] = A[l][i, 2, 1] = T(0.1 * (i + l))
            end
        end
    end
    C = [zeros(T, k, k) for _ in 1:L]
    c = ones(T, m)
    B = zeros(T, m, 0)
    b = zeros(T, 0)
    return c, A, C, B, b, active
end

@testset "incidence-aware sparse path" begin
    @testset "automatic structure classification" begin
        c, A, C, B, b, _ = sparse_block_data()
        automatic = SDPX.ingest(c, A, C, B, b; sparse=:auto)
        @test automatic.cons isa SDPX.SparseCons{Float64}
        @test automatic.structure.selected_storage == :sparse
        @test automatic.structure.recommended_storage == :sparse
        @test automatic.structure.schur_backend == :block_arrow

        m, dimension, blocks = 20, 5, 12
        sparse_coefficients = Vector{Vector{SparseMatrixCSC{Float64,Int}}}(
            undef,
            blocks,
        )
        upper_coordinates = [
            (row, column)
            for column in 1:dimension
            for row in 1:column
        ]
        for block in 1:blocks
            sparse_coefficients[block] = Vector{
                SparseMatrixCSC{Float64,Int}
            }(undef, m)
            for variable in 1:m
                row, column = upper_coordinates[
                    mod1(variable + block - 1, length(upper_coordinates))
                ]
                rows = row == column ? [row] : [row, column]
                columns = row == column ? [column] : [column, row]
                sparse_coefficients[block][variable] = sparse(
                    rows,
                    columns,
                    ones(length(rows)),
                    dimension,
                    dimension,
                )
            end
        end
        problem = SDPX.ingest(
            ones(m),
            sparse_coefficients,
            [zeros(dimension, dimension) for _ in 1:blocks],
            zeros(m, 0),
            Float64[];
            sparse=:auto,
        )
        summary = SDPX.structure_summary(problem)
        @test problem.cons isa SDPX.SparseCons{Float64}
        @test summary.profile ==
              :sparse_coefficients_dense_psd_dense_schur
        @test summary.coefficient_density < 0.10
        @test summary.block_pattern_density > 0.90
        @test summary.schur_density == 1.0
        @test summary.schur_backend == :dense_cholesky
        # `dense_sparse_assembly` trades packed-pair storage against
        # `schur_nbins * m^2` of task-local accumulators, so the decision moves
        # with the thread count (and, at large `m`, with free memory). Pin the
        # thread count: the assertion is about the trade-off being made
        # correctly for a given budget, not about the ambient machine.
        workspace = SDPX.Workspace(problem; thread_count=1)
        @test workspace.dense_sparse_assembly
        @test all(isempty(block.Ppanel) for block in workspace.blk)
        @test all(isempty(block.Svals) for block in workspace.blk)

        # Serial Float64 consumes only the lower Schur triangle. This model
        # sits above that footprint but below m², so direct scatter must win.
        active = [
            collect(1:10), collect(11:20),
            [collect(1:5); collect(11:15)],
            [collect(6:10); collect(16:20)],
        ]
        coefficients = [
            [variable in ids ? sparse([1], [1], [1.0], 3, 3) :
             spzeros(3, 3) for variable in 1:20]
            for ids in active
        ]
        lower_only_problem = SDPX.ingest(
            ones(20), coefficients, [zeros(3, 3) for _ in active],
            ones(20, 1), [0.0]; sparse=:auto,
        )
        @test lower_only_problem.structure.schur_backend == :dense_cholesky
        lower_only_workspace = SDPX.Workspace(
            lower_only_problem; thread_count=1,
        )
        @test lower_only_workspace.schur_lower_only
        @test lower_only_workspace.dense_sparse_assembly
        @test all(isempty(block.Svals) for block in lower_only_workspace.blk)
    end

    @testset "active sets and compact workspace" begin
        c, A, C, B, b, active = sparse_block_data()
        prob = SDPX.ingest(c, A, C, B, b; sparse=true)
        @test prob.cons isa SDPX.SparseCons{Float64}
        @test prob.cons.active == active
        @test all(size(coeffs, 1) == 3 for coeffs in prob.cons.packed2)
        for block in eachindex(prob.cons.packed2)
            coefficients = prob.cons.packed2[block]
            masks = prob.cons.packed2_mask[block]
            @test length(masks) == size(coefficients, 2)
            @test masks == [
                UInt8(
                    (!iszero(coefficients[1, position]) ? 0x01 : 0x00) |
                    (!iszero(coefficients[2, position]) ? 0x02 : 0x00) |
                    (!iszero(coefficients[3, position]) ? 0x04 : 0x00),
                )
                for position in axes(coefficients, 2)
            ]
        end
        selected = SDPX.recommended_parameters(
            prob,
            SDPX.SolverOptions{Float64}(parameter_policy=:auto),
        )
        @test selected.profile == :post_scaling_mehrotra
        @test selected.β == 0.1
        @test selected.γ == 0.9
        # The arrow block constants are all zero, so the initial point keeps
        # the unit-scale floor.
        @test selected.Ωp == 1.0
        @test selected.Ωd == 1.0

        ws = SDPX.Workspace(prob)
        @test isempty(ws.S)
        @test isempty(ws.Spartial)
        @test ws.arrow !== nothing
        @test ws.arrow.global_ids == [1]
        @test SDPX.build_execution_plan(prob).gram_kernel ==
              :fused_arrow_2x2
        # This model is an exact-arrow 2x2 case, so the fused compute+scatter
        # path applies and the packed pair buffer is deliberately not allocated
        # (it is 9.08 GB on the 4100-block CSDR model).
        @test ws.fused_arrow
        for l in 1:prob.dims.L
            @test isempty(ws.blk[l].Ppanel)
            @test isempty(ws.blk[l].Svals)
        end
    end

    @testset "sparse Schur equals dense canonical Schur" begin
        c, A, C, B, b, _ = sparse_block_data()
        dense = SDPX.ingest(c, A, C, B, b; sparse=false)
        sparse = SDPX.ingest(c, A, C, B, b; sparse=true)
        X = [
            [2.0 0.2; 0.2 1.5],
            [1.8 -0.1; -0.1 2.2],
            [2.5 0.3; 0.3 1.7],
        ]
        Y = [
            [1.4 0.1; 0.1 1.9],
            [2.1 0.2; 0.2 1.3],
            [1.6 -0.2; -0.2 2.4],
        ]

        wd = SDPX.Workspace(dense)
        ws = SDPX.Workspace(sparse)
        @test SDPX.factor_blocks!(wd, X, Y)
        @test SDPX.factor_blocks!(ws, X, Y)
        SDPX.schur_build!(wd, dense, dense.cons, X, Y)
        Sd = similar(wd.S)
        SDPX.materialize_schur!(Sd, wd)
        SDPX.schur_build!(ws, sparse, sparse.cons, X, Y)
        Ss = zeros(size(Sd))
        SDPX.materialize_schur!(Ss, ws)
        @test Ss ≈ Sd rtol=1e-12 atol=1e-12

        options = SDPX.SolverOptions{Float64}(verbosity=0)
        @test SDPX.factorize!(SDPX.select_backend(wd), wd, dense, options).ok
        @test SDPX.factorize!(SDPX.select_backend(ws), ws, sparse, options).ok
        r = collect(range(0.2, 1.6; length=dense.dims.m))
        p = Float64[]
        dxd, dys = zeros(dense.dims.m), Float64[]
        dxs, dys2 = zeros(dense.dims.m), Float64[]
        SDPX.solve_kkt!(wd, 0, r, p, dxd, dys)
        SDPX.solve_kkt!(ws, 0, r, p, dxs, dys2)
        @test dxs ≈ dxd rtol=1e-11 atol=1e-11

        dense_product = Ss * dxs
        arrow_product = zeros(length(dxs))
        SDPX.schur_mul!(
            arrow_product,
            ws,
            dxs,
            one(Float64),
            zero(Float64),
        )
        @test arrow_product ≈ dense_product rtol=1e-12 atol=1e-12
    end

    @testset "general sparse outer-product Schur equals dense path" begin
        m, dimension, blocks = 7, 4, 2
        coefficients = [zeros(m, dimension, dimension) for _ in 1:blocks]
        for block in 1:blocks, variable in 1:m
            row = mod1(variable + block, dimension)
            column = mod1(2variable + block, dimension)
            coefficients[block][variable, row, column] += 0.2variable
            coefficients[block][variable, column, row] += 0.2variable
            coefficients[block][variable, row, row] += 1 + 0.1block
            if iseven(variable)
                coefficients[block][variable, column, column] += 0.3
            end
        end
        c = ones(m)
        C = [zeros(dimension, dimension) for _ in 1:blocks]
        B, b = zeros(m, 1), [0.0]
        dense = SDPX.ingest(c, coefficients, C, B, b; sparse=false)
        sparse_problem = SDPX.ingest(
            c,
            coefficients,
            C,
            B,
            b;
            sparse=true,
        )
        for block in 1:blocks
            # `schur_order` is ascending by variable id. This is load-bearing,
            # not cosmetic: `reduce_sparse_schur!` partitions the scatter by
            # output column and relies on the positions feeding a contiguous
            # column range being themselves contiguous, which binary search can
            # then locate. It previously sorted by descending nnz; that ordering
            # gave no measured benefit and blocked the parallel scatter.
            order = sparse_problem.cons.schur_order[block]
            @test issorted(order)
            @test order == sparse_problem.cons.active[block]
        end
        X = [
            Matrix(Symmetric(1.5I + 0.03 .* ones(dimension, dimension)))
            for _ in 1:blocks
        ]
        Y = [
            Matrix(Symmetric(1.2I + 0.02 .* ones(dimension, dimension)))
            for _ in 1:blocks
        ]
        dense_workspace = SDPX.Workspace(dense)
        sparse_workspace = SDPX.Workspace(sparse_problem)
        @test SDPX.factor_blocks!(dense_workspace, X, Y)
        @test SDPX.factor_blocks!(sparse_workspace, X, Y)
        expected = copy(
            SDPX.schur_build!(
                dense_workspace,
                dense,
                dense.cons,
                X,
                Y,
            ),
        )
        actual = copy(
            SDPX.schur_build!(
                sparse_workspace,
                sparse_problem,
                sparse_problem.cons,
                X,
                Y,
            ),
        )
        @test actual ≈ expected rtol=1e-12 atol=1e-12
    end

    @testset "specialized 2x2 kernels" begin
        A = [2.0 0.25; 0.25 1.5]
        B = [1.2 -0.4; 0.3 2.1]
        C = fill(0.5, 2, 2)
        expected = 1.3 .* (A * B) .- 0.2 .* C
        SDPX.kmul!(C, A, B, 1.3, -0.2)
        @test C ≈ expected rtol=1e-15 atol=1e-15

        factor = copy(A)
        @test SDPX.kchol!(factor)
        rhs_matrix = [1.0 2.0; -0.5 0.75]
        expected_matrix = A \ rhs_matrix
        SDPX.kcholsolve!(factor, rhs_matrix)
        @test rhs_matrix ≈ expected_matrix rtol=1e-14 atol=1e-14

        rhs_vector = [0.3, -1.1]
        expected_vector = A \ rhs_vector
        SDPX.kcholsolve!(factor, rhs_vector)
        @test rhs_vector ≈ expected_vector rtol=1e-14 atol=1e-14

        direction = [-0.5 0.1; 0.1 -0.2]
        scratch = zeros(2, 2)
        for t in (0.1, 1.0, 3.0)
            @test SDPX.trial_isposdef!(scratch, A, t, direction) ==
                  isposdef(Symmetric(A + t * direction))
        end
        boundary_direction = [-3.0 0.2; 0.2 -0.7]
        bound = SDPX.fraction_to_boundary_bound!(scratch, A, boundary_direction)
        @test 0.0 < bound < 1.0
        @test isposdef(Symmetric(A + (1 - 1e-10) * bound * boundary_direction))
        @test !isposdef(Symmetric(A + (1 + 1e-10) * bound * boundary_direction))
    end

    @testset "BigFloat sparse Schur and arrow solve" begin
        setprecision(256) do
            c, A, C, B, b, _ = sparse_block_data(BigFloat)
            dense = SDPX.ingest(c, A, C, B, b; sparse=false)
            sparse = SDPX.ingest(c, A, C, B, b; sparse=true)
            X = [
                BigFloat[2.0 0.2; 0.2 1.5],
                BigFloat[1.8 -0.1; -0.1 2.2],
                BigFloat[2.5 0.3; 0.3 1.7],
            ]
            Y = [
                BigFloat[1.4 0.1; 0.1 1.9],
                BigFloat[2.1 0.2; 0.2 1.3],
                BigFloat[1.6 -0.2; -0.2 2.4],
            ]
            wd, ws = SDPX.Workspace(dense), SDPX.Workspace(sparse)
            @test SDPX.factor_blocks!(wd, X, Y)
            @test SDPX.factor_blocks!(ws, X, Y)
            SDPX.schur_build!(wd, dense, dense.cons, X, Y)
            Sd = similar(wd.S)
            SDPX.materialize_schur!(Sd, wd)
            SDPX.schur_build!(ws, sparse, sparse.cons, X, Y)
            Ss = zeros(BigFloat, size(Sd))
            SDPX.materialize_schur!(Ss, ws)
            @test Ss ≈ Sd rtol=big"1e-60" atol=big"1e-60"

            options = SDPX.SolverOptions{BigFloat}(verbosity=0)
            @test SDPX.factorize!(SDPX.select_backend(wd), wd, dense, options).ok
            @test SDPX.factorize!(SDPX.select_backend(ws), ws, sparse, options).ok
            rhs = BigFloat.(collect(range(0.2, 1.6; length=dense.dims.m)))
            dxd, dxs = zeros(BigFloat, dense.dims.m), zeros(BigFloat, dense.dims.m)
            SDPX.solve_kkt!(wd, 0, rhs, BigFloat[], dxd, BigFloat[])
            SDPX.solve_kkt!(ws, 0, rhs, BigFloat[], dxs, BigFloat[])
            @test dxs ≈ dxd rtol=big"1e-60" atol=big"1e-60"
        end
    end

    @testset "sparse and dense solves agree on analytic problem" begin
        A, C = zeros(Float64, 2, 2, 2), zeros(Float64, 2, 2)
        A[1, 1, 1] = 1.0
        A[2, 2, 2] = 1.0
        C[1, 2], C[2, 1] = 1.0, 1.0
        c, B, b = Float64[2, 3], zeros(Float64, 2, 0), Float64[]
        A = [A]
        C = [C]
        dense_problem = SDPX.ingest(c, A, C, B, b; sparse=false, verbosity=0)
        sparse_problem = SDPX.ingest(c, A, C, B, b; sparse=true, verbosity=0)
        dense = SDPX.solve!(
            dense_problem,
            SDPX.SolverOptions{Float64}(sparse=false, verbosity=0),
        )
        sparse = SDPX.solve!(
            sparse_problem,
            SDPX.SolverOptions{Float64}(sparse=true, verbosity=0),
        )
        @test dense.status == SDPX.Optimal
        @test sparse.status == SDPX.Optimal
        @test sparse.pObj ≈ dense.pObj rtol=1e-9 atol=1e-9
        @test sparse.dObj ≈ dense.dObj rtol=1e-9 atol=1e-9
    end

    @testset "column-partitioned Schur scatter equals the serial scatter" begin
        # The parallel scatter must be *bitwise* identical to the serial one,
        # not merely close: each output entry receives exactly the same
        # contributions summed in the same block order, so any difference here
        # means a lost or duplicated contribution rather than reassociation.
        # Exercised with varied block sizes and inactive variables, which is
        # what makes the per-block position ranges non-trivial.
        if Threads.nthreads() == 1
            @test_skip "requires at least two Julia threads"
        else
            rng = StableRNG(90125)
            dense_comparisons = 0
            for trial in 1:12
                L = rand(rng, 1:4)
                m = rand(rng, 4:30)
                ks = [rand(rng, 3:6) for _ in 1:L]
                A = [zeros(Float64, m, ks[l], ks[l]) for l in 1:L]
                for l in 1:L
                    for i in 1:m
                        if rand(rng) < 0.6
                            M = sprand(rng, Float64, ks[l], ks[l], 0.35)
                            A[l][i, :, :] = Matrix(M + M')
                        end
                    end
                    if all(iszero, A[l])
                        M = sprand(rng, Float64, ks[l], ks[l], 0.6)
                        A[l][1, :, :] = Matrix(M + M')
                    end
                end
                C = [zeros(Float64, ks[l], ks[l]) for l in 1:L]
                prob = SDPX.ingest(
                    randn(rng, m), A, C, zeros(Float64, m, 0), zeros(Float64, 0);
                    sparse=true, verbosity=0,
                )
                prob.cons isa SDPX.SparseCons || continue
                ws = SDPX.Workspace(prob)
                X = [Matrix(3.0I, ks[l], ks[l]) for l in 1:L]
                Y = [Matrix(2.0I, ks[l], ks[l]) for l in 1:L]
                SDPX.factor_blocks!(ws, X, Y)
                for l in 1:L
                    SDPX.sparse_schur_block!(ws.blk[l], prob.cons, l, X[l], Y[l])
                end
                materialized = Matrix{Float64}(
                    undef,
                    m,
                    m,
                )
                SDPX.materialize_schur!(materialized, ws)
                @test issymmetric(materialized)
                if ws.sparse_kkt !== nothing
                    sparse_workspace = ws.sparse_kkt
                    @test sparse_workspace isa
                          SDPX.GenericSparseSchurSDPWorkspace{Float64}
                    parallel = copy(
                        SDPX.assemble_sparse_schur!(
                            sparse_workspace.storage,
                            sparse_workspace.assembly_map,
                            [ws.blk[l].Svals for l in eachindex(ws.blk)],
                        ),
                    )
                    serial = Matrix{Float64}(undef, m, m)
                    fill!(serial, 0.0)
                    for l in eachindex(ws.blk)
                        ids = prob.cons.schur_order[l]
                        Svals = ws.blk[l].Svals
                        q = 0
                        for p in eachindex(ids), r in p:length(ids)
                            q += 1
                            i, j = ids[p], ids[r]
                            serial[j, i] += Svals[q]
                        end
                    end
                    @test Matrix(parallel) == serial
                    @test parallel == Matrix(sparse_workspace.storage.matrix)
                    dense_comparisons += 1
                else
                    @test ws.arrow !== nothing
                end
            end
            @test dense_comparisons > 0
        end
    end

    @testset "arrow KKT matches dense KKT at scale" begin
        # The existing equivalence check above uses 3 blocks and 1 shared
        # variable — too small for the threaded Schur reduction or the
        # per-block local pivots to engage. The arrow path is exercised almost
        # exclusively by large `n == 0` models (Task_Low08 takes the standard
        # path), so a defect that only appears at scale would not show up in
        # any other benchmark. Check it directly.
        function arrow_model(blocks, shared; seed=11)
            rng = MersenneTwister(seed)
            m = shared + blocks
            coeff = [Vector{SparseMatrixCSC{Float64,Int}}(undef, m) for _ in 1:blocks]
            for l in 1:blocks, i in 1:m
                if i <= shared || i == shared + l
                    v = randn(rng, 3)
                    dense_block = [v[1] v[2]; v[2] v[3]]
                    coeff[l][i] = sparse(dense_block)
                else
                    coeff[l][i] = spzeros(Float64, 2, 2)
                end
            end
            C = [Matrix(Symmetric(randn(rng, 2, 2))) for _ in 1:blocks]
            return (c=randn(rng, m), A=coeff, C=C, B=zeros(m, 0), b=Float64[])
        end

        for (blocks, shared) in ((20, 4), (100, 12))
            d = arrow_model(blocks, shared)
            sp = SDPX.ingest(d.c, d.A, d.C, d.B, d.b; sparse=true, verbosity=0)
            dn = SDPX.ingest(d.c, d.A, d.C, d.B, d.b; sparse=false, verbosity=0)
            @test sp.cons isa SDPX.SparseCons{Float64}

            rng = MersenneTwister(7)
            X = [Matrix(Symmetric([2.0 0.2; 0.2 1.8] .+ 0.1 .* randn(rng, 2, 2))) for _ in 1:blocks]
            Y = [Matrix(Symmetric([1.7 0.1; 0.1 2.1] .+ 0.1 .* randn(rng, 2, 2))) for _ in 1:blocks]
            ws, wd = SDPX.Workspace(sp), SDPX.Workspace(dn)
            @test ws.arrow !== nothing
            @test SDPX.factor_blocks!(ws, X, Y)
            @test SDPX.factor_blocks!(wd, X, Y)

            SDPX.schur_build!(wd, dn, dn.cons, X, Y)
            Sd = similar(wd.S)
            SDPX.materialize_schur!(Sd, wd)
            SDPX.schur_build!(ws, sp, sp.cons, X, Y)
            Ss = zeros(Float64, size(Sd))
            SDPX.materialize_schur!(Ss, ws)
            @test Ss ≈ Sd rtol = 1e-10

            options = SDPX.SolverOptions{Float64}(verbosity=0)
            @test SDPX.factorize!(SDPX.select_backend(ws), ws, sp, options).ok
            @test SDPX.factorize!(SDPX.select_backend(wd), wd, dn, options).ok
            m = sp.dims.m
            rhs = collect(range(0.2, 1.6; length=m))
            dxs, dxd = zeros(m), zeros(m)
            SDPX.solve_kkt!(ws, 0, copy(rhs), Float64[], dxs, Float64[])
            SDPX.solve_kkt!(wd, 0, copy(rhs), Float64[], dxd, Float64[])
            @test dxs ≈ dxd rtol = 1e-8
            # Independent of the dense path: the arrow solution must satisfy
            # the Schur system it claims to solve.
            @test Sd * dxs ≈ rhs rtol = 1e-8
        end
    end

    @testset "fused arrow assembly equals two-phase assembly" begin
        # `fused_arrow_schur_block!` skips the packed pair buffer entirely, so
        # it must be checked against the buffer-based path it replaces rather
        # than only against a converged solve.
        c, A, C, B, b, _ = sparse_block_data()
        prob = SDPX.ingest(c, A, C, B, b; sparse=true)
        ws = SDPX.Workspace(prob)
        @test ws.arrow !== nothing
        @test ws.fused_arrow
        cons = prob.cons::SDPX.SparseCons{Float64}
        X = [[2.0 0.2; 0.2 1.5], [1.8 -0.1; -0.1 2.2], [2.5 0.3; 0.3 1.7]]
        Y = [[1.4 0.1; 0.1 1.9], [2.1 0.2; 0.2 1.3], [1.6 -0.2; -0.2 2.4]]
        @test SDPX.factor_blocks!(ws, X, Y)

        fused = copy(SDPX.schur_build!(ws, prob, cons, X, Y))
        fused_coupling = [copy(m) for m in ws.arrow.coupling]
        fused_local = [copy(m) for m in ws.arrow.Dsrc]

        # Two-phase reference: allocate the pair buffers this model normally
        # skips, then run the original compute + scatter.
        arrow = ws.arrow
        fill!(arrow.Sgg, 0.0)
        for l in eachindex(arrow.Dsrc)
            fill!(arrow.Dsrc[l], 0.0)
            fill!(arrow.coupling[l], 0.0)
        end
        for l in 1:prob.dims.L
            na = length(cons.schur_order[l])
            dimension = prob.dims.k[l]
            ws.blk[l].Ppanel =
                zeros(Float64, dimension, dimension * na)
            ws.blk[l].Svals = zeros(Float64, na * (na + 1) ÷ 2)
            SDPX.sparse_schur_block!(ws.blk[l], cons, l, X[l], Y[l])
            SDPX.scatter_arrow_schur_block!(arrow, ws.blk[l], cons, l, arrow.Sgg)
        end

        @test arrow.Sgg ≈ fused rtol=1e-14 atol=1e-14
        for l in eachindex(arrow.coupling)
            @test arrow.coupling[l] ≈ fused_coupling[l] rtol=1e-14 atol=1e-14
            @test arrow.Dsrc[l] ≈ fused_local[l] rtol=1e-14 atol=1e-14
        end
    end

end
