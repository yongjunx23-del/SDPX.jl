#!/usr/bin/env julia
# `step_rule` defaults to :backtrack, but :auto selects :fraction_to_boundary
# exactly when every block is <= 2x2 -- which is this model. Backtracking
# accepts the first gamma^k that is positive definite, so its effective
# fraction-to-boundary factor wanders anywhere in [gamma, 1]: it can land the
# iterate essentially ON the cone boundary, which collapses eigenvalues and
# makes every later step tiny. fraction_to_boundary computes the exact bound
# (closed form for 2x2) and applies a consistent safety factor.
using LinearAlgebra, MultiFloats, Printf, Serialization, SDPX
using MultiFloats: Float64x4
const T = Float64x4
data = open(deserialize, get(ENV, "SPARSE_INPUT", "results/sparse-80-4-40-100.bin"))
prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
@printf("L=%d m=%d threads=%d  all blocks 2x2: %s\n\n", prob.dims.L, prob.dims.m,
    Threads.nthreads(), all(<=(2), prob.dims.k))
tol = T(1e-7)
@printf("%-22s %-6s %-6s %9s %5s %-22s %-15s %-15s %-10s %-9s\n",
    "step_rule", "beta", "cent", "sec", "it", "status", "pObj", "dObj", "gap", "d_res")
for rule in (:backtrack, :fraction_to_boundary), b in (0.05, 0.1), cent in (0, 4)
    o = SDPX.SolverOptions{T}(β=T(b), γ=T(0.85), Ωp=T(100), Ωd=T(100),
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol, iter_max=400,
        verbosity=0, max_time=2400.0, parameter_policy=:fixed,
        step_rule=rule, max_centering=cent, threads=Threads.nthreads())
    t = @elapsed r = SDPX.solve!(prob, o)
    @printf("%-22s %-6.3g %-6d %9.1f %5d %-22s %-15.10g %-15.10g %-10.3e %-9.2e\n",
        rule, b, cent, t, r.iterations, r.status, Float64(r.pObj), Float64(r.dObj),
        Float64(r.gap_rel), Float64(r.d_res))
    flush(stdout)
end
println("\nClarabel: OPTIMAL 13.5808486042076 gap 8.4e-8 in 1114.86 s / 70 it")
