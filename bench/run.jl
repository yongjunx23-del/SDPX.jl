#=====================================================================
    Benchmark harness (§0.2). Per Appendix D's measurement protocol:
    one warmup solve (JIT), report the minimum of 3 timed runs,
    record wall-clock, allocations, GC%, iteration count, thread
    count, and commit.

    NOTE (scope): this historical harness records aggregate per-solve
    metrics (wall-clock/iteration, bytes allocated, GC%). The solver now
    exposes phase timings through `result.timings` and
    `result.diagnostics.timings`; newer benchmark drivers persist those
    fields. This small CI harness intentionally keeps its original compact
    output schema.
=====================================================================#

using SDPX
using Printf
include("generate.jl")

function run_once(::Type{T}, tier::Symbol; seed=1, iterMax=200, kwargs...) where {T}
    inst = tier_instance(tier, T; seed=seed)
    GC.gc()
    stats = @timed SDPX.sdp(inst.c, inst.A, inst.C, inst.B, inst.b;
        verbosity=0, iterMax=iterMax, kwargs...)
    r = stats.value
    return (status=r.message, iterations=r.iterations, wall_s=stats.time,
        gc_s=stats.gctime, bytes=stats.bytes,
        gc_pct=stats.time > 0 ? 100 * stats.gctime / stats.time : 0.0)
end

function bench_tier(::Type{T}, tier::Symbol; seed=1, reps=3, iterMax=200, kwargs...) where {T}
    # warmup: small tier, untimed, just to trigger JIT
    run_once(T, :small; seed=seed, iterMax=3)
    runs = [run_once(T, tier; seed=seed, iterMax=iterMax, kwargs...) for _ in 1:reps]
    return argmin(r -> r.wall_s, runs)
end

function append_results_row(path::AbstractString, row::NamedTuple; commit="", threads=Threads.nthreads())
    header_needed = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        if header_needed
            println(io, "| commit | tier | T | threads | status | iters | wall(s) | alloc(MB) | GC% |")
            println(io, "|---|---|---|---|---|---|---|---|---|")
        end
        @printf(io, "| %s | %s | %s | %d | %s | %d | %.3f | %.1f | %.1f |\n",
            commit, row.tier, row.T, threads, row.status, row.iterations, row.wall_s,
            row.bytes / 1e6, row.gc_pct)
    end
end

function main(; tiers=(:small,), types=(Float64,), reps=3, results_path=joinpath(@__DIR__, "RESULTS.md"))
    commit = try
        strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    for tier in tiers, T in types
        r = bench_tier(T, tier; reps=reps)
        append_results_row(results_path, (; tier=String(tier), T=string(T), r...); commit=commit)
        println("$tier / $T: status=$(r.status) iters=$(r.iterations) wall=$(round(r.wall_s,digits=3))s " *
                "alloc=$(round(r.bytes/1e6,digits=1))MB gc=$(round(r.gc_pct,digits=1))%")
    end
end
