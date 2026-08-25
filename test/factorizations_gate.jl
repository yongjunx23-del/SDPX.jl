#=====================================================================
    Wave C gate: factorizations_per_iteration == 1.

    The SDP hot loop must call `factorize!` exactly once per `newton_step!`
    (src/step.jl). This test builds a small SDP, runs one Newton step via
    `_kkt_cold_start_initialization` + `newton_step!`, and asserts the
    `Workspace.factorizations` counter is 1. A second step must increment it
    to 2 (one per iteration). This is a measurement/assertion gate only; it
    does not change the factorization algorithm.
=====================================================================#

@testset "factorizations gate" begin
    @testset "factorizations_per_iteration == 1" begin
        T = Float64
        k = 3
        m = k * (k + 1) ÷ 2
        c = zeros(T, m)
        c[1] = -one(T)
        A = zeros(T, m, k, k)
        A[1, 1, 1] = one(T)
        A[2, 2, 2] = one(T)
        A[3, 3, 3] = one(T)
        A[4, 1, 2] = one(T); A[4, 2, 1] = one(T)
        A[5, 1, 3] = one(T); A[5, 3, 1] = one(T)
        A[6, 2, 3] = one(T); A[6, 3, 2] = one(T)
        B = zeros(T, m, 1)
        B[1, 1] = one(T); B[2, 1] = one(T); B[3, 1] = one(T)
        prob = SDPX.ingest(c, [A], [zeros(T, k, k)], B, T[3];
            T=T, sparse=false, verbosity=0)

        opts = SDPX.SolverOptions{T}(
            algorithm=:sdp, presolve=false, scaling=:none, verbosity=0,
            iter_max=200,
        )
        ws = SDPX.Workspace(prob; thread_count=1)
        init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
        @test init.ok
        x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ

        # Counter starts at zero.
        @test ws.factorizations == 0

        # One Newton step performs exactly one factorization.
        SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)
        @test ws.factorizations == 1

        # A second step increments it to 2 (one per iteration).
        SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2)
        @test ws.factorizations == 2
    end
end
