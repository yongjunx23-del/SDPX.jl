#!/usr/bin/env julia

"""BigFloat vs Float64x4 on the CSDR sparse PSD dual.

One arithmetic type per process (BigFloat precision is global and Julia's
thread pool is fixed at startup), so the PBS driver invokes this once per
configuration and appends a CSV row.

Note on fairness: `Float64x4` carries ~212 significand bits and `BigFloat` here
is set to 256, so BigFloat is doing *more* precise arithmetic — it is the
closest standard pairing, not an exact match. SDPX also keeps BigFloat on the
serial path because its mutable-scalar solver workspaces rely on strict
ownership and aliasing invariants. The honest single-number comparison is
therefore at one thread; extra threads only help Float64x4, which is itself a
real practical difference worth reporting.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Serialization
using SDPX

const INPUT = get(ENV, "SPARSE_INPUT",
    joinpath(@__DIR__, "..", "results", "sparse-80-4-40-100.bin"))

peak_rss_gb() = Sys.maxrss() / 1024^3

convert_problem(data, ::Type{Float64x4}) = data
function convert_problem(data, ::Type{BigFloat})
    return (
        c=BigFloat.(data.c),
        A=[BigFloat.(block) for block in data.A],
        C=[BigFloat.(block) for block in data.C],
        B=BigFloat.(data.B),
        b=BigFloat.(data.b),
    )
end

function run_case(::Type{T}, data, tol, reps, iter_max, output, label) where {T}
    convert_seconds = @elapsed converted = convert_problem(data, T)
    ingest_seconds = @elapsed problem = SDPX.ingest(
        converted.c, converted.A, converted.C, converted.B, converted.b;
        sparse=:auto, verbosity=0,
    )
    @printf("[%s] convert %.2f s, ingest %.2f s, dims L=%d m=%d n=%d\n",
        label, convert_seconds, ingest_seconds,
        problem.dims.L, problem.dims.m, problem.dims.n)
    println("[$label] structure: ", SDPX.structure_summary(problem))

    options(n) = SDPX.SolverOptions{T}(
        ϵ_gap=T(tol), ϵ_primal=T(tol), ϵ_dual=T(tol),
        iter_max=n, verbosity=0, max_time=7200.0,
        scaling=:none, refine_steps=1, parameter_policy=:auto,
    )

    SDPX.solve!(problem, options(2))          # warmup / JIT
    best = Inf
    local result
    for _ in 1:reps
        GC.gc()
        elapsed = @elapsed candidate = SDPX.solve!(problem, options(iter_max))
        if elapsed < best
            best = elapsed
            result = candidate
        end
    end
    t = result.timings

    @printf("[%s] solve %.3f s  status=%s iters=%d  %.4f s/iter\n",
        label, best, result.status, result.iterations,
        best / max(result.iterations, 1))
    @printf("[%s] pObj=%.15g gap_rel=%.3e p_res=%.3e d_res=%.3e peak=%.2f GB\n",
        label, Float64(result.pObj), Float64(result.gap_rel),
        Float64(result.p_res), Float64(result.d_res), peak_rss_gb())

    header = !isfile(output) || filesize(output) == 0
    open(output, "a") do io
        header && println(io, join([
                "arithmetic", "bits", "julia_threads", "blas_threads",
                "convert_seconds", "ingest_seconds", "best_seconds",
                "iterations", "seconds_per_iter", "status",
                "schur_assembly", "kkt_factorization",
                "peak_rss_gb", "pobj", "gap_rel", "p_res", "d_res",
            ], ","))
        println(io, join([
                label, SDPX.sig_bits(T), Threads.nthreads(), BLAS.get_num_threads(),
                @sprintf("%.3f", convert_seconds), @sprintf("%.3f", ingest_seconds),
                @sprintf("%.4f", best), result.iterations,
                @sprintf("%.5f", best / max(result.iterations, 1)),
                result.status,
                t === nothing ? "" : @sprintf("%.4f", t.schur_assembly),
                t === nothing ? "" : @sprintf("%.4f", t.kkt_factorization),
                @sprintf("%.2f", peak_rss_gb()),
                @sprintf("%.15g", Float64(result.pObj)),
                @sprintf("%.4e", Float64(result.gap_rel)),
                @sprintf("%.4e", Float64(result.p_res)),
                @sprintf("%.4e", Float64(result.d_res)),
            ], ","))
    end
    return best
end

function main()
    which = get(ENV, "ARITH", "Float64x4")
    tol = parse(Float64, get(ENV, "SPARSE_TOL", "1e-7"))
    reps = parse(Int, get(ENV, "SPARSE_REPS", "2"))
    iter_max = parse(Int, get(ENV, "SPARSE_ITERS", "1000"))
    bits = parse(Int, get(ENV, "SPARSE_BITS", "256"))
    output = get(ENV, "SPARSE_OUT", joinpath(@__DIR__, "..", "results", "arith-sparse.csv"))

    println("loading ", INPUT)
    load_seconds = @elapsed data = open(deserialize, INPUT)
    @printf("load %.2f s; julia threads=%d blas=%d\n",
        load_seconds, Threads.nthreads(), BLAS.get_num_threads())

    if which == "Float64x4"
        run_case(Float64x4, data, tol, reps, iter_max, output, "Float64x4")
    else
        setprecision(BigFloat, bits) do
            run_case(BigFloat, data, tol, reps, iter_max, output, "BigFloat$(bits)")
        end
    end
end

main()
