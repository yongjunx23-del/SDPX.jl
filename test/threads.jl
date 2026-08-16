#=====================================================================
    Thread-safety / no-shared-state checks (§3.3, §P8). The original
    had `T`/`mode`/`sparseMode` as mutable globals — a second
    concurrent solve could leak state into an unrelated solve. `solve!` now touches no
    global mutable state at all: `SDPProblem`/`SolverOptions`/
    `Workspace` are all per-call. This runs several *different*
    problems concurrently and checks each gets its own correct,
    independent answer.
=====================================================================#

using SDPX
using LinearAlgebra
using Test

function _threads_solve(c, A, C, B, b)
    T = SDPX.infer_eltype(c, A, C, B, b)
    problem = SDPX.ingest(c, A, C, B, b; T=T, verbosity=0)
    return SDPX.solve!(problem, SDPX.SolverOptions{T}(verbosity=0))
end

function _t1_like(scale::Float64)
    T = Float64
    A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
    A[1, 1, 1] = 1
    A[2, 2, 2] = 1
    C[1, 2], C[2, 1] = 1, 1
    c = T[2*scale, 3*scale]
    B = Matrix{T}(undef, 2, 0)
    b = Array{T}(undef, 0)
    return c, [A], [C], B, b, scale
end

@testset "thread safety" begin
    @testset "concurrent solve! on distinct problems, no cross-contamination" begin
        # The property under test is *independence*, not "every scale converges
        # to Optimal within iterMax" (that's a convergence-robustness question,
        # answered separately by the correctness suite) — so compare concurrent
        # results against the same problems solved sequentially, one at a time,
        # rather than asserting a fixed expected status/value.
        scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.25, 4.0]
        sequential = [(s=s, prob=_threads_solve(_t1_like(s)[1:5]...)) for s in scales]

        results = Vector{Any}(undef, length(scales))
        Threads.@threads for i in eachindex(scales)
            c, A, C, B, b, s = _t1_like(scales[i])
            results[i] = (s=s, prob=_threads_solve(c, A, C, B, b))
        end
        for (seq, r) in zip(sequential, results)
            @test r.s == seq.s
            @test r.prob["status"] == seq.prob["status"]
            @test isapprox(r.prob["pObj"], seq.prob["pObj"]; atol=1e-10)
        end
    end

    @testset "two concurrent solve! calls on the *same* problem, separate workspaces" begin
        c, A, C, B, b, _ = _t1_like(1.0)
        prob = SDPX.ingest(c, A, C, B, b)
        opts = SDPX.SolverOptions{Float64}(verbosity=0)
        r1 = Ref{Any}()
        r2 = Ref{Any}()
        t1 = Threads.@spawn r1[] = SDPX.solve!(prob, opts)
        t2 = Threads.@spawn r2[] = SDPX.solve!(prob, opts)
        wait(t1)
        wait(t2)
        @test r1[].status == SDPX.Optimal
        @test r2[].status == SDPX.Optimal
        @test isapprox(r1[].pObj, r2[].pObj; atol=1e-8)
    end

    @testset "block update reductions preserve serial arithmetic" begin
        T = Float64
        block_count = 300
        variable_count = 2
        A = [zeros(T, 2, 2, variable_count) for _ in 1:block_count]
        C = [Matrix{T}(I, 2, 2) for _ in 1:block_count]
        for block in 1:block_count
            A[block][1, 1, 1] = T(block) / block_count
            A[block][2, 2, 2] = T(block_count - block + 1) / block_count
        end
        c = T[1, 2]
        B = Matrix{T}(undef, variable_count, 0)
        b = T[]
        problem = SDPX.ingest(c, A, C, B, b)
        workspace = SDPX.Workspace(
            problem;
            thread_count=min(4, Threads.nthreads()),
        )
        X = [T[2 0.1; 0.1 3] for _ in 1:block_count]
        Y = [T[4 -0.2; -0.2 5] for _ in 1:block_count]
        for block in 1:block_count
            workspace.blk[block].dX .= T[0.01 0.002; 0.002 -0.01]
            workspace.blk[block].dY .= T[-0.02 0.003; 0.003 0.02]
        end
        primal_step = T(0.7)
        dual_step = T(0.6)
        expected_X = [
            X[block] + primal_step * workspace.blk[block].dX
            for block in 1:block_count
        ]
        expected_Y = [
            Y[block] + dual_step * workspace.blk[block].dY
            for block in 1:block_count
        ]
        expected_complementarity = sum(
            block -> dot(expected_X[block], expected_Y[block]),
            1:block_count;
            init=zero(T),
        )

        complementarity, finite = SDPX.threaded_update_blocks!(
            workspace,
            X,
            Y,
            primal_step,
            dual_step,
        )
        @test finite
        @test X == expected_X
        @test Y == expected_Y
        @test complementarity == expected_complementarity

        μ = zeros(T, block_count)
        SDPX.threaded_update_mu!(
            workspace,
            μ,
            T(0.1),
            problem.dims.k,
            complementarity,
            false,
        )
        @test μ == [
            T(0.1) * dot(X[block], Y[block]) / problem.dims.k[block]
            for block in 1:block_count
        ]

        expected_dual = SDPX.dual_objective(problem, T[], Y)
        @test SDPX.threaded_dual_objective(
            workspace,
            problem,
            T[],
            Y,
        ) == expected_dual

        best = SDPX.BestIterateWorkspace(T[1, 2], X, T[], Y)
        SDPX._store_best_iterate!(
            best,
            workspace,
            T[1, 2],
            X,
            T[],
            Y,
            T(3),
            T(2),
            T(1),
            T(0),
            T(0),
            4,
        )
        @test best.valid
        @test best.X == X
        @test best.Y == Y
        @test best.iter == 4
    end
end
