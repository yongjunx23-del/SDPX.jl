#!/usr/bin/env julia

"""Parameter sweep for the CSDR sparse PSD dual.

The `:auto` profile picks `:large_arrow_2x2` (β=0.4, γ=0.7) for this model
because it keys off `max_active`, whose thresholds top out at 15 — but this
instance has 385 active variables per block, far outside the range that profile
was calibrated on. It does not converge. This sweeps the centering and
backtracking parameters to find what does, and to recalibrate the profile.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Serialization
using SDPX

const INPUT = get(ENV, "SPARSE_INPUT",
    joinpath(@__DIR__, "..", "results", "sparse-80-4-40-100.bin"))
const OUT = get(ENV, "SWEEP_OUT", joinpath(@__DIR__, "..", "results", "sweep-sparse.csv"))

function main()
    tol = parse(Float64, get(ENV, "SPARSE_TOL", "1e-7"))
    iter_max = parse(Int, get(ENV, "SWEEP_ITERS", "300"))
    max_time = parse(Float64, get(ENV, "SWEEP_MAXTIME", "1800"))
    T = Float64x4

    data = open(deserialize, INPUT)
    prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
    @printf("L=%d m=%d n=%d threads=%d\n",
        prob.dims.L, prob.dims.m, prob.dims.n, Threads.nthreads())

    betas = [parse(Float64, x) for x in split(get(ENV, "SWEEP_BETA", "0.01,0.05,0.1,0.2,0.4"), ',')]
    gammas = [parse(Float64, x) for x in split(get(ENV, "SWEEP_GAMMA", "0.7,0.85,0.9"), ',')]
    omegas = [parse(Float64, x) for x in split(get(ENV, "SWEEP_OMEGA", "1,10,100"), ',')]

    header = !isfile(OUT) || filesize(OUT) == 0
    open(OUT, "a") do io
        header && println(io, "beta,gamma,omega,seconds,iterations,status,pobj,gap_rel,p_res,d_res")
    end

    for b in betas, g in gammas, w in omegas
        opts = SDPX.SolverOptions{T}(
            β=T(b), γ=T(g), Ωp=T(w), Ωd=T(w),
            ϵ_gap=T(tol), ϵ_primal=T(tol), ϵ_dual=T(tol),
            iter_max=iter_max, verbosity=0, max_time=max_time,
            parameter_policy=:fixed, refine_steps=1,
            threads=Threads.nthreads(),
        )
        elapsed = @elapsed r = SDPX.solve!(prob, opts)
        @printf("beta=%-5.3g gamma=%-5.3g omega=%-6.4g  %8.1f s  %3d it  %-22s pObj=%.10g gap=%.2e\n",
            b, g, w, elapsed, r.iterations, r.status, Float64(r.pObj), Float64(r.gap_rel))
        flush(stdout)
        open(OUT, "a") do io
            @printf(io, "%g,%g,%g,%.3f,%d,%s,%.12g,%.4e,%.4e,%.4e\n",
                b, g, w, elapsed, r.iterations, r.status,
                Float64(r.pObj), Float64(r.gap_rel),
                Float64(r.p_res), Float64(r.d_res))
        end
    end
end

main()
