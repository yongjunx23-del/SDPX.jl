#!/usr/bin/env julia
# Definitive comparison of the shipped defaults before and after this round of
# fixes, on the CSDR sparse model. Three defects compounded:
#   1. stall detector used a fixed per-iteration bar (now a rolling-window
#      stagnation detector normalised by the requested tolerances);
#   2. a collapsed step on an ALREADY-FEASIBLE side terminated the whole solve
#      (primal reaches 1e-47 by iteration 2 here, so tX collapses harmlessly);
#   3. step_rule defaulted to :backtrack instead of the exact 2x2
#      fraction-to-boundary rule.
using LinearAlgebra, MultiFloats, Printf, Serialization, SDPX
using MultiFloats: Float64x4
const T = Float64x4
data = open(deserialize, get(ENV, "SPARSE_INPUT", "results/sparse-80-4-40-100.bin"))
prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
@printf("L=%d m=%d threads=%d\n\n", prob.dims.L, prob.dims.m, Threads.nthreads())
tol = T(1e-7)
@printf("%-34s %9s %5s %-16s %-16s %-16s %-10s %-9s %-9s\n",
    "config", "sec", "it", "status", "pObj", "dObj", "gap", "d_res", "reason")
configs = Any[
 ("old: backtrack, β=.05, Ω=100",  (step_rule=:backtrack, β=T(0.05), γ=T(0.85),
                                    Ωp=T(100), Ωd=T(100), parameter_policy=:fixed)),
 ("new defaults (:auto profile)",  NamedTuple()),
 ("new defaults, iter_max=800",    (iter_max=800,)),
]
for (name, kw) in configs
    o = SDPX.SolverOptions{T}(ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol,
        iter_max=400, verbosity=0, max_time=5400.0,
        threads=Threads.nthreads(); kw...)
    t = @elapsed r = SDPX.solve!(prob, o)
    @printf("%-34s %9.1f %5d %-16s %-16.10g %-16.10g %-10.3e %-9.2e %-9s\n",
        name, t, r.iterations, r.status, Float64(r.pObj), Float64(r.dObj),
        Float64(r.gap_rel), Float64(r.d_res), r.termination.reason)
    flush(stdout)
end
println("\nClarabel: OPTIMAL pObj 13.5808486042076 dObj 13.5808497422048 gap 8.38e-8 in 1114.86 s / 70 it")
