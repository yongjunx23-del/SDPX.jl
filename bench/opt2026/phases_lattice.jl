#!/usr/bin/env julia

"""Phase-level timing breakdown for the Task_Low08 lattice solve."""

using LinearAlgebra
using Printf
using SDPX
using SparseArrays

let path = joinpath(@__DIR__, "..", "lattice_bootstrap", "benchmark_sdpx_float64_solve.jl")
    source = read(path, String)
    include_string(@__MODULE__, replace(source, r"\nmain\(ARGS\)\s*$" => "\n"), path)
end

const INPUT = get(ENV, "LATTICE_INPUT",
    joinpath(@__DIR__, "..", "lattice_bootstrap", "results", "20260724-low08", "problem-float64.bin"))

function main()
    iters = parse(Int, get(ENV, "PHASE_ITERS", "8"))
    tol = parse(Float64, get(ENV, "PHASE_TOL", "1e-6"))
    data = read_problem(INPUT)
    ids, rank, _, _ = equality_basis(data.B, data.b)
    ingest_time = @elapsed problem = SDPX.ingest(
        data.c, data.A, data.C, data.B[:, ids], data.b[ids];
        sparse=:auto, validate=false, symmetrize=false, verbosity=0,
    )
    @printf("ingest %.3f s   dims L=%d m=%d n=%d\n",
        ingest_time, problem.dims.L, problem.dims.m, problem.dims.n)
    println("threads: julia=", Threads.nthreads(), " blas=", BLAS.get_num_threads())

    opts(n) = SDPX.SolverOptions{Float64}(
        β=0.1, γ=0.85, Ωp=100.0, Ωd=0.001,
        ϵ_gap=tol, ϵ_primal=tol, ϵ_dual=tol,
        iter_max=n, verbosity=0, timing=true,
        restart=true, max_restarts=5, sparse=:auto,
        parameter_policy=:fixed, predictor=:sdpb,
        step_rule=:backtrack, refine_steps=1,
    )
    SDPX.solve!(problem, opts(1))                        # warmup
    result = SDPX.solve!(problem, opts(iters))
    t = result.timings
    @printf("\n%d iterations, %.3f s total (%.3f s/iter), status=%s\n",
        result.iterations, t.total, t.total / max(result.iterations, 1), result.status)
    println("---- phase breakdown ----")
    for name in (:residual_and_block_factor, :schur_assembly, :kkt_factorization,
        :predictor, :corrector, :line_search, :update)
        v = getfield(t, name)
        @printf("  %-26s %8.3f s  %5.1f%%   %7.3f s/iter\n",
            name, v, 100v / t.total, v / max(result.iterations, 1))
    end
    accounted = sum(getfield(t, n) for n in (:residual_and_block_factor, :schur_assembly,
        :kkt_factorization, :predictor, :corrector, :line_search, :update))
    @printf("  %-26s %8.3f s  %5.1f%%\n", "unaccounted", t.total - accounted,
        100(t.total - accounted) / t.total)
end

main()
