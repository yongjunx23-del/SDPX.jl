include(joinpath(dirname(@__DIR__), "benchmark", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark
using SDPX
using Test
specs = filter(s -> s.id in (:lp_random_small, :lp_afiro_style, :socp_portfolio_small),
    GenericConicBenchmark.inventory(; tier=:small))
total = 0.0
for spec in specs
    t = @timed GenericConicBenchmark.run_one(spec, Float64)
    r = t.value
    r.status === spec.expected_status || error("$(spec.id) status=$(r.status)")
    r.certificate_valid || error("$(spec.id) invalid certificate")
    global total += t.time
end
println("METRIC lp_seconds=$total")
