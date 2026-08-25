#!/usr/bin/env julia
length(ARGS) in (2, 3) || error(
    "usage: julia --project=. benchmark/compare.jl BASELINE.toml CANDIDATE.toml [OUTPUT.tsv]",
)
include(joinpath(@__DIR__, "PhysicsBenchmarkHarness.jl"))
using .PhysicsBenchmarkHarness
output = length(ARGS) == 3 ? ARGS[3] : nothing
rows = PhysicsBenchmarkHarness.compare_result_files(ARGS[1], ARGS[2]; output=output)
println("compared ", length(rows), " rows")
