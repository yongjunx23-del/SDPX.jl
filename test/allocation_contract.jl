# Phase 4 allocation-contract regression gate.
#
# One full predictor-corrector Newton iteration (SDP route, Float64) in
# steady state must stay below a documented allocation ceiling. This is a
# regression gate (fail on serious allocation blowups), not the final
# zero-allocation target: benchmark/allocation_profile.jl reports the actual
# per-iteration value across the full arithmetic family (currently Float64
# ~9 KB/iter). The semantic gate (Optimal + valid certificate) pins the
# objective/iteration/residual/gap/certificate side of the CI contract.
using SDPX
using Test
using LinearAlgebra

# Comfortable headroom over the current ~9.3 KB/iter Float64 steady-state
# measurement so minor GC/compiler noise cannot fail the gate, while a real
# hot-loop allocation regression is still caught.
const ALLOC_PER_ITER_CEILING = 64_000

function _gate_sdp_problem()
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
    return SDPX.ingest(c, [A], [zeros(T, k, k)], B, T[3];
        T=T, sparse=false, verbosity=0)
end

function _steady_state_iteration_alloc(T::Type)
    prob = _gate_sdp_problem()
    opts = SDPX.SolverOptions{T}(
        algorithm=:sdp, presolve=false, scaling=:none, verbosity=0,
        iter_max=200,
    )
    ws = SDPX.Workspace(prob; thread_count=1)
    init = SDPX._kkt_cold_start_initialization(ws, prob, opts)
    @test init.ok
    x, X, y, Y, mu = init.x, init.X, init.y, init.Y, init.μ
    SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=1)  # JIT warm-up
    return minimum(
        @allocated(SDPX.newton_step!(ws, prob, opts, x, X, y, Y, mu; iteration=2))
        for _ in 1:3
    )
end

@testset "allocation contract (Float64 SDP Newton iteration)" begin
    alloc = _steady_state_iteration_alloc(Float64)
    @test alloc < ALLOC_PER_ITER_CEILING
    @info "Float64 per-iteration Julia allocation" alloc_bytes=alloc ceiling=ALLOC_PER_ITER_CEILING
end