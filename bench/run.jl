#=====================================================================
    Benchmark harness (plan §3, §6.4).

    Every emitted row is self-describing: the revision, machine, thread
    configuration, arithmetic, input fingerprint, and validation outcome
    travel with the timing. A number without those is not reproducible,
    and the plan's evidence policy requires published claims to come
    from the published revision.

    Compilation is measured separately from the solve, results are
    written as JSON and CSV as well as Markdown, every measured run is
    validated in the original coordinates, and thread oversubscription
    is detected rather than left to be inferred from odd timings.
=====================================================================#

using SDPX
using LinearAlgebra
using Printf
include("generate.jl")
include("environment.jl")
using .BenchEnvironment

"""
    solve_instance(T, tier; seed, iterMax, kwargs...)

One solve, returning the result plus the ingested problem so the caller can
validate the answer in the original coordinates.
"""
function solve_instance(::Type{T}, tier::Symbol; seed=1, iterMax=200, kwargs...) where {T}
    inst = tier === :small_closed_form ?
           closed_form_instance(T) :
           tier_instance(tier, T; seed=seed)
    problem = SDPX.ingest(inst.c, inst.A, inst.C, inst.B, inst.b; verbosity=0)
    options = SDPX.SolverOptions{T}(; verbosity=0, iter_max=iterMax, kwargs...)
    return (instance=inst, problem=problem, options=options,
            result=SDPX.solve!(problem, options))
end

"""
    validate(problem, result, options) -> (ok, detail)

Independent check of the returned solution. §6.4 requires a benchmark to fail
when its output does not validate — a fast wrong answer is not a result.
"""
function validate(problem, result, options)
    try
        certificate = SDPX.result_certificate(problem, result, options)
        ok = hasproperty(certificate, :acceptable) ? certificate.acceptable :
             hasproperty(certificate, :valid) ? certificate.valid : true
        return (ok=ok, detail=string(hasproperty(certificate, :status) ?
                                     certificate.status : "certificate computed"))
    catch err
        return (ok=false, detail="certificate failed: $(sprint(showerror, err))")
    end
end

"""
    bench_tier(T, tier; seed, reps, iterMax, kwargs...) -> NamedTuple

Warm up once (untimed), then take `reps` measured samples, validating the
solution each time and reporting the full dispersion rather than a single
number.
"""
function bench_tier(::Type{T}, tier::Symbol; seed=1, reps=3, iterMax=200, kwargs...) where {T}
    GC.gc()
    run() = solve_instance(T, tier; seed=seed, iterMax=iterMax, kwargs...)
    # Untimed warm-up on the same shape, so JIT cost is attributed to
    # `compile_seconds` and never to the samples.
    warmup_tier = tier === :small_closed_form ? :small_closed_form : :small
    compile_seconds = @elapsed solve_instance(T, warmup_tier; seed=seed, iterMax=3)

    samples = Float64[]
    allocated = 0
    gc_fraction = 0.0
    outcome = nothing
    validation = (ok=true, detail="")
    for _ in 1:max(reps, 1)
        GC.gc()
        stats = @timed run()
        outcome = stats.value
        push!(samples, stats.time)
        allocated = max(allocated, stats.bytes)
        gc_fraction = stats.time > 0 ? 100 * stats.gctime / stats.time : 0.0
        validation = validate(outcome.problem, outcome.result, outcome.options)
        validation.ok || break          # stop early: a wrong answer is not worth timing
    end

    timing = summarize_samples(samples)
    result = outcome.result
    fingerprint = problem_fingerprint((outcome.instance.c, outcome.instance.A,
                                       outcome.instance.C, outcome.instance.B,
                                       outcome.instance.b))
    dims = outcome.problem.dims
    return (;
        label=string(tier),
        arithmetic=string(T),
        precision_bits=T === BigFloat ? precision(BigFloat) : Int(round(-log2(Float64(eps(T))))),
        input_hash=fingerprint.hash[1:16],
        blocks=dims.L, variables=dims.m, equalities=dims.n,
        status=string(result.status),
        iterations=result.iterations,
        seconds_min=timing.minimum,
        seconds_median=timing.median,
        seconds_mean=timing.mean,
        seconds_stddev=timing.stddev,
        relative_spread=timing.relative_spread,
        reps=timing.reps,
        compile_seconds=compile_seconds,
        allocated_mb=allocated / 1e6,
        gc_percent=gc_fraction,
        peak_rss_gb=Sys.maxrss() / 2^30,
        objective=Float64(result.pObj),
        gap_rel=Float64(result.gap_rel),
        validated=validation.ok,
        validation_detail=validation.detail,
    )
end

function main(; tiers=(:small,), types=(Float64,), reps=3,
    results_path=joinpath(@__DIR__, "RESULTS.md"),
    json_path=nothing, csv_path=nothing)
    warning = oversubscription_warning()
    warning === nothing || @warn warning
    environment = environment_record(; oversubscription=something(warning, ""))

    records = NamedTuple[]
    failures = String[]
    for tier in tiers, T in types
        measurement = bench_tier(T, tier; reps=reps)
        push!(records, merge(environment, measurement))
        @printf("%-8s %-10s  %7.3f s (median %7.3f, spread %5.1f%%)  %3d it  %-14s %s\n",
            tier, T, measurement.seconds_min, measurement.seconds_median,
            100 * (isfinite(measurement.relative_spread) ? measurement.relative_spread : 0.0),
            measurement.iterations, measurement.status,
            measurement.validated ? "validated" : "VALIDATION FAILED")
        measurement.validated ||
            push!(failures, "$(tier)/$(T): $(measurement.validation_detail)")
        # The benchmark measures convergence quality as well as time: a row
        # that returns Stalled without an Optimal certificate is a failed
        # solve, not a timing sample.  `validate` checks the answer in the
        # original coordinates, and a non-Optimal result is never accepted
        # here, even when the certificate happens to be permissive.
        measurement.status == "Optimal" ||
            push!(failures, "$(tier)/$(T): status $(measurement.status), not Optimal")
    end

    base = endswith(results_path, ".md") ? results_path[1:end-3] : results_path
    write_records(records;
        markdown_path=results_path,
        json_path=something(json_path, base * ".json"),
        csv_path=something(csv_path, base * ".csv"),
        title="SDPX benchmark")

    # §6.4: the benchmark fails when its own output does not validate.
    isempty(failures) ||
        error("benchmark validation failed:\n  " * join(failures, "\n  "))
    return records
end
