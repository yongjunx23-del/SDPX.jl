#!/usr/bin/env julia

"""Sampling profile of a full Task_Low08 Float64 solve.

Answers "where does the time actually go" without hand-instrumenting every
phase: runs the real tuned configuration from the 2026-07-24 three-solver
report, then reports a flat self-time profile aggregated by source location.
"""

using LinearAlgebra
using Printf
using Profile
using SDPX
using SparseArrays

# The lattice driver ends in `main(ARGS)`; include only its definitions so
# that loading it here does not kick off a full benchmark run.
let path = joinpath(@__DIR__, "..", "lattice_bootstrap", "benchmark_sdpx_float64_solve.jl")
    source = read(path, String)
    include_string(@__MODULE__, replace(source, r"\nmain\(ARGS\)\s*$" => "\n"), path)
end

const INPUT = get(
    ENV,
    "LATTICE_INPUT",
    joinpath(@__DIR__, "..", "lattice_bootstrap", "results", "20260724-low08", "problem-float64.bin"),
)

function build()
    data = read_problem(INPUT)
    equality_ids, rank, _, _ = equality_basis(data.B, data.b)
    @printf("equality presolve: %d -> %d\n", size(data.B, 2), rank)
    problem = SDPX.ingest(
        data.c, data.A, data.C, data.B[:, equality_ids], data.b[equality_ids];
        sparse=:auto, validate=false, symmetrize=false, verbosity=0,
    )
    return data, problem
end

function options(tol, iter_max)
    return SDPX.SolverOptions{Float64}(
        β=0.1, γ=0.85, Ωp=100.0, Ωd=0.001,
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol,
        iter_max=iter_max, verbosity=1, timing=true,
        restart=true, max_restarts=5, sparse=:auto,
        parameter_policy=:fixed, predictor=:sdpb,
        step_rule=:backtrack, refine_steps=1,
    )
end

function main()
    iter_max = parse(Int, get(ENV, "PROFILE_ITERS", "6"))
    tol = parse(Float64, get(ENV, "PROFILE_TOL", "1e-6"))
    data, problem = build()
    println("dims: ", problem.dims)
    println("structure: ", SDPX.structure_summary(problem))
    println("threads: julia=", Threads.nthreads(), " blas=", BLAS.get_num_threads())

    # untimed warmup so JIT is out of the profile
    SDPX.solve!(problem, options(tol, 1))

    elapsed = @elapsed result = SDPX.solve!(problem, options(tol, iter_max))
    @printf("\n%d-iteration solve: %.3f s (%.3f s/iter), status=%s\n",
        result.iterations, elapsed, elapsed / max(result.iterations, 1), result.status)

    Profile.clear()
    Profile.init(; n=10_000_000, delay=0.001)
    @profile SDPX.solve!(problem, options(tol, iter_max))

    println("\n================ FLAT PROFILE (self time) ================")
    data_prof, lidict = Profile.retrieve()
    counts = Dict{String,Int}()
    for ip in data_prof
        frames = get(lidict, ip, nothing)
        frames === nothing && continue
        isempty(frames) && continue
        fr = frames[1]                       # leaf frame = self time
        fr.from_c && continue
        key = string(fr.file, ":", fr.line, " ", fr.func)
        counts[key] = get(counts, key, 0) + 1
    end
    total = sum(values(counts); init=0)
    ranked = sort!(collect(counts); by=last, rev=true)
    for (key, n) in first(ranked, 35)
        @printf("%6.2f%%  %7d  %s\n", 100n / max(total, 1), n, key)
    end
end

main()
