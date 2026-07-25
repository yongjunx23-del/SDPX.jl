#=====================================================================
    Thread-safety / no-shared-state checks (§3.3, §P8). The original
    had `T`/`mode`/`sparseMode` as mutable globals — a second
    concurrent `sdp()` call (or a throw partway through `findFeasible`)
    could leak state into an unrelated solve. `solve!` now touches no
    global mutable state at all: `SDPProblem`/`SolverOptions`/
    `Workspace` are all per-call. This runs several *different*
    problems concurrently and checks each gets its own correct,
    independent answer.
=====================================================================#

using SDPX
using LinearAlgebra
using Test

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
        sequential = [(s=s, prob=SDPX.sdp(_t1_like(s)[1:5]...; verbosity=0)) for s in scales]

        results = Vector{Any}(undef, length(scales))
        Threads.@threads for i in eachindex(scales)
            c, A, C, B, b, s = _t1_like(scales[i])
            results[i] = (s=s, prob=SDPX.sdp(c, A, C, B, b; verbosity=0))
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

    @testset "setArithmeticType does not affect calls that carry their own type" begin
        # the one remaining piece of global state (_LEGACY_T) must not leak into
        # a call whose inputs already pin a concrete type (BigFloat here) —
        # only the all-Int/Rational edge case consults it.
        SDPX.setArithmeticType(Float64)
        c, A, C, B, b, _ = _t1_like(1.0)
        Ab = BigFloat.(A[1])
        cb, Ab3, Cb, Bb, bb = BigFloat.(c), [Ab], [BigFloat.(C[1])], BigFloat.(B), BigFloat.(b)
        prob = SDPX.sdp(cb, Ab3, Cb, Bb, bb; verbosity=0)
        @test prob["pObj"] isa BigFloat
        SDPX.setArithmeticType(BigFloat)  # restore the documented default
    end
end
