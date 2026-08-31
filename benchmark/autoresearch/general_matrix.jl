#!/usr/bin/env julia
# Certified public-API autoresearch matrix: one representative for every
# general cone family.  It deliberately uses no private solver entry point.

using SDPX
include(joinpath(@__DIR__, "..", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

const CASE_IDS = (
    :lp_afiro_style,
    :socp_portfolio_small,
    :sdp_maxcut_k4,
    :exp_unit_small,
    :power_epigraph_small,
)

function result_for(id::Symbol)
    spec = only(filter(item -> item.id === id, inventory(; tier=:small)))
    result = run_one(spec, Float64)
    result.expectation_met || error(
        "correctness failure for $(id): status=$(result.status), " *
        "certificate=$(result.certificate_valid), objective=$(result.objective)",
    )
    return result
end

@inline function median_value(values)
    ordered = sort!(collect(values))
    return ordered[cld(length(ordered), 2)]
end

function main()
    samples = parse(Int, get(ENV, "SDPX_AUTORESEARCH_SAMPLES", "5"))
    isodd(samples) && samples >= 3 || error(
        "SDPX_AUTORESEARCH_SAMPLES must be an odd integer >= 3",
    )

    # One complete unmeasured warm-up compiles every family before timing.
    foreach(result_for, CASE_IDS)

    seconds = 0.0
    bytes = 0
    iterations = 0
    for id in CASE_IDS
        measured = [result_for(id) for _ in 1:samples]
        reference = first(measured)
        all(result -> result.status === reference.status &&
                      result.certificate_valid === reference.certificate_valid &&
                      result.objective == reference.objective &&
                      result.iterations == reference.iterations,
            measured) || error("nondeterministic certified result for $(id)")
        median_seconds = median_value(result.seconds for result in measured)
        median_bytes = median_value(result.bytes for result in measured)
        seconds += median_seconds
        bytes += median_bytes
        iterations += reference.iterations
        println(
            "CASE id=$(id) status=$(reference.status) " *
            "cert=$(reference.certificate_valid) median_time_s=$(median_seconds) " *
            "median_bytes=$(median_bytes) iter=$(reference.iterations) " *
            "objective=$(reference.objective) rp=$(reference.primal_residual) " *
            "rd=$(reference.dual_residual) gap=$(reference.relative_gap)",
        )
    end
    rss = try Int(Sys.maxrss()) catch; 0 end
    println("METRIC solver_seconds=$(seconds)")
    println("METRIC allocation_bytes=$(bytes)")
    println("METRIC iterations=$(iterations)")
    println("METRIC peak_rss_bytes=$(rss)")
end

main()
