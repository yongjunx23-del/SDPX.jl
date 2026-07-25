#!/usr/bin/env julia
# The Omega multiplier (3x max||C_l||inf) and the per-profile adaptive beta/gamma
# choice were both calibrated against runs that we now know were terminating
# prematurely. Re-validate them with the termination fixes in place, rather than
# shipping constants fitted to invalid data. Four configs, no sweeping.
using LinearAlgebra, MultiFloats, Printf, Serialization, SDPX
using MultiFloats: Float64x4
const T = Float64x4
data = open(deserialize, get(ENV, "SPARSE_INPUT", "results/sparse-80-4-40-100.bin"))
prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
st = SDPX.block_norm_stats(prob)
@printf("L=%d m=%d threads=%d  max||C||=%.5g\n\n", prob.dims.L, prob.dims.m,
    Threads.nthreads(), Float64(st.maxnorm))
tol = T(1e-7)
@printf("%-30s %9s %5s %-12s %-16s %-16s %-10s %-9s %-12s\n",
    "config", "sec", "it", "status", "pObj", "dObj", "gap", "d_res", "reason")
for mult in (1.0, 3.0), strat in (:fixed, :adaptive)
    W = T(mult) * st.maxnorm
    o = SDPX.SolverOptions{T}(β=T(0.05), γ=T(0.85), Ωp=W, Ωd=W,
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol, iter_max=400,
        verbosity=0, max_time=5400.0, parameter_policy=:fixed,
        parameter_strategy=strat, threads=Threads.nthreads())
    t = @elapsed r = SDPX.solve!(prob, o)
    @printf("Omega=%.0fx maxC, %-9s %9.1f %5d %-12s %-16.10g %-16.10g %-10.3e %-9.2e %-12s\n",
        mult, strat, t, r.iterations, r.status, Float64(r.pObj), Float64(r.dObj),
        Float64(r.gap_rel), Float64(r.d_res), r.termination.reason)
    flush(stdout)
end
println("\nClarabel: OPTIMAL pObj 13.5808486042076 gap 8.38e-8 in 1114.86 s / 70 it")
