#!/usr/bin/env julia

"""Focused initial-scaling sweep for the CSDR sparse PSD dual.

The coarse (β, γ, Ω) sweep showed Ω is the deciding parameter: at Ω=100 the
primal residual reaches 8.8e-49 (exactly feasible) and the objective lands at
15.03 against Clarabel's 13.58, while Ω ∈ {0.01, 1} diverge to 1e17. What is
left is the *dual* residual, stuck at 8.5e-2 — so this sweeps Ωp and Ωd
independently, above the largest value the coarse grid reached.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Serialization
using SDPX

const INPUT = get(ENV, "SPARSE_INPUT",
    joinpath(@__DIR__, "..", "results", "sparse-80-4-40-100.bin"))
const OUT = get(ENV, "SWEEP_OUT", joinpath(@__DIR__, "..", "results", "sweep-omega.csv"))

function main()
    tol = parse(Float64, get(ENV, "SPARSE_TOL", "1e-7"))
    iter_max = parse(Int, get(ENV, "SWEEP_ITERS", "400"))
    max_time = parse(Float64, get(ENV, "SWEEP_MAXTIME", "2400"))
    T = Float64x4

    data = open(deserialize, INPUT)
    prob = SDPX.ingest(data.c, data.A, data.C, data.B, data.b; sparse=:auto, verbosity=0)
    @printf("L=%d m=%d n=%d threads=%d\n",
        prob.dims.L, prob.dims.m, prob.dims.n, Threads.nthreads())

    pairs = Tuple{Float64,Float64}[]
    for spec in split(get(ENV, "SWEEP_PAIRS",
        "100:100,300:300,1000:1000,3000:3000,10000:10000,100:1000,100:10000,1000:100"), ',')
        p, d = split(spec, ':')
        push!(pairs, (parse(Float64, p), parse(Float64, d)))
    end
    betas = [parse(Float64, x) for x in split(get(ENV, "SWEEP_BETA", "0.01,0.1"), ',')]

    header = !isfile(OUT) || filesize(OUT) == 0
    open(OUT, "a") do io
        header && println(io, "beta,omega_p,omega_d,seconds,iterations,status,pobj,dobj,gap_rel,p_res,d_res")
    end

    for b in betas, (wp, wd) in pairs
        opts = SDPX.SolverOptions{T}(
            β=T(b), γ=T(0.85), Ωp=T(wp), Ωd=T(wd),
            ϵ_gap=T(tol), ϵ_primal=T(tol), ϵ_dual=T(tol),
            iter_max=iter_max, verbosity=0, max_time=max_time,
            parameter_policy=:fixed, refine_steps=1,
            threads=Threads.nthreads(),
        )
        elapsed = @elapsed r = SDPX.solve!(prob, opts)
        @printf("b=%-5.3g Wp=%-7.5g Wd=%-7.5g  %8.1f s  %3d it  %-22s pObj=%.10g dObj=%.10g gap=%.2e d_res=%.2e\n",
            b, wp, wd, elapsed, r.iterations, r.status,
            Float64(r.pObj), Float64(r.dObj), Float64(r.gap_rel), Float64(r.d_res))
        flush(stdout)
        open(OUT, "a") do io
            @printf(io, "%g,%g,%g,%.3f,%d,%s,%.12g,%.12g,%.4e,%.4e,%.4e\n",
                b, wp, wd, elapsed, r.iterations, r.status,
                Float64(r.pObj), Float64(r.dObj), Float64(r.gap_rel),
                Float64(r.p_res), Float64(r.d_res))
        end
    end
end

main()
