#!/usr/bin/env julia

# Compatibility entry point for the analytic referee. All execution, sampling,
# certification, reporting and comparison live in the canonical runner.
include(joinpath(@__DIR__, "..", "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry

args = copy(ARGS)
if isempty(args) || startswith(first(args), "--")
    pushfirst!(args, "analytic_fast")
end
SDPXBenchmarkRegistry.main(args)
