#!/usr/bin/env julia
# The solve was being terminated by the stall detector, not by step collapse.
# The old rule demanded a `stall_tolerance` improvement on EVERY iteration and
# stopped after 15 consecutive misses; the CSDR model converges at ~0.05% per
# iteration near the end, so it was killed at iteration 27 with the gap at 9e-4
# while Clarabel needs 70 iterations for the same problem. The rule is now
# cumulative over the window. Measure what that buys, and how far it goes if
# stall detection is disabled entirely.
using LinearAlgebra, MultiFloats, Printf, Serialization, SDPX
using MultiFloats: Float64x4
const T = Float64x4
data = open(deserialize, get(ENV, "SPARSE_INPUT", "results/sparse-80-4-40-100.bin"))
prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
@printf("L=%d m=%d threads=%d\n\n", prob.dims.L, prob.dims.m, Threads.nthreads())
tol = T(1e-7)
@printf("%-28s %9s %5s %-22s %-15s %-15s %-10s %-9s\n",
    "config", "sec", "it", "status", "pObj", "dObj", "gap", "d_res")
configs = [
  ("window stall (new)",        (stall_iterations=15,)),
  ("stall off, iter_max=400",   (stall_iterations=0,)),
  ("stall off + fraction_bnd",  (stall_iterations=0, step_rule=:fraction_to_boundary)),
  ("stall off + recentering",   (stall_iterations=0, max_centering=6)),
]
for (name, kw) in configs
    o = SDPX.SolverOptions{T}(β=T(0.05), γ=T(0.85), Ωp=T(100), Ωd=T(100),
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol, iter_max=400,
        verbosity=0, max_time=3000.0, parameter_policy=:fixed,
        threads=Threads.nthreads(); kw...)
    t = @elapsed r = SDPX.solve!(prob, o)
    @printf("%-28s %9.1f %5d %-22s %-15.10g %-15.10g %-10.3e %-9.2e\n",
        name, t, r.iterations, r.status, Float64(r.pObj), Float64(r.dObj),
        Float64(r.gap_rel), Float64(r.d_res))
    flush(stdout)
end
println("\nClarabel: OPTIMAL 13.5808486042076 gap 8.4e-8 in 1114.86 s / 70 it")
