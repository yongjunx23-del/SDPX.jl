#!/usr/bin/env julia

"""Single-thread-count scaling point for the Task_Low08 lattice solve.

One process per thread count (Julia fixes its thread pool at startup), so the
PBS driver invokes this repeatedly. Emits one CSV row with runtime, phase
breakdown, allocations, peak RSS, and accuracy so a whole sweep can be
concatenated.
"""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

let path = joinpath(@__DIR__, "benchmark_sdpx_float64_solve.jl")
    source = read(path, String)
    include_string(@__MODULE__, replace(source, r"\nmain\(ARGS\)\s*$" => "\n"), path)
end

peak_rss_gb() = try
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    value = Sys.maxrss()
    value / (Sys.islinux() ? 1024^3 : 1024^3)
catch
    NaN
end

function main()
    input = ARGS[1]
    output = ARGS[2]
    blas_threads = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : Threads.nthreads()
    iters = parse(Int, get(ENV, "SCALE_ITERS", "8"))
    tol = parse(Float64, get(ENV, "SCALE_TOL", "1e-6"))
    reps = parse(Int, get(ENV, "SCALE_REPS", "3"))

    BLAS.set_num_threads(blas_threads)
    data = read_problem(input)
    ids, rank, _, _ = equality_basis(data.B, data.b)
    problem = SDPX.ingest(
        data.c, data.A, data.C, data.B[:, ids], data.b[ids];
        sparse=:auto, validate=false, symmetrize=false, verbosity=0,
    )

    opts(n) = SDPX.SolverOptions{Float64}(
        β=0.1, γ=0.85, Ωp=100.0, Ωd=0.001,
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol,
        iter_max=n, verbosity=0, timing=true,
        restart=true, max_restarts=5, sparse=:auto,
        parameter_policy=:fixed, predictor=:sdpb,
        step_rule=:backtrack, refine_steps=1,
    )

    SDPX.solve!(problem, opts(1))            # warmup / JIT

    best = Inf
    local chosen
    allocated = 0
    for _ in 1:reps
        GC.gc()
        stats = @timed SDPX.solve!(problem, opts(iters))
        if stats.time < best
            best = stats.time
            chosen = stats.value
            allocated = stats.bytes
        end
    end
    t = chosen.timings

    header = !isfile(output) || filesize(output) == 0
    open(output, "a") do io
        header && println(io, join([
                "julia_threads", "blas_threads", "iterations", "status",
                "best_seconds", "seconds_per_iter",
                "residual_block_factor", "schur_assembly", "kkt_factorization",
                "predictor", "corrector", "line_search", "update",
                "allocated_bytes", "peak_rss_gb",
                "pobj", "dobj", "gap_rel", "p_res", "d_res",
            ], ","))
        println(io, join([
                Threads.nthreads(), blas_threads, chosen.iterations, chosen.status,
                @sprintf("%.6f", best),
                @sprintf("%.6f", best / max(chosen.iterations, 1)),
                @sprintf("%.6f", t.residual_and_block_factor),
                @sprintf("%.6f", t.schur_assembly),
                @sprintf("%.6f", t.kkt_factorization),
                @sprintf("%.6f", t.predictor),
                @sprintf("%.6f", t.corrector),
                @sprintf("%.6f", t.line_search),
                @sprintf("%.6f", t.update),
                allocated,
                @sprintf("%.3f", peak_rss_gb()),
                @sprintf("%.15g", chosen.pObj),
                @sprintf("%.15g", chosen.dObj),
                @sprintf("%.6e", chosen.gap_rel),
                @sprintf("%.6e", chosen.p_res),
                @sprintf("%.6e", chosen.d_res),
            ], ","))
    end
    @printf("threads=%d blas=%d  %.3f s  (%.3f s/iter)  pObj=%.12g\n",
        Threads.nthreads(), blas_threads, best,
        best / max(chosen.iterations, 1), chosen.pObj)
end

main()
