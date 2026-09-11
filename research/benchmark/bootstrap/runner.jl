#!/usr/bin/env julia
include(joinpath(@__DIR__, "PhysicsBenchmarkHarness.jl"))
include(joinpath(@__DIR__, "fixtures", "smoke_catalog.jl"))
using .PhysicsBenchmarkHarness

PhysicsBenchmarkHarness.main(
    ARGS; catalog=physics_benchmark_catalog(),
)
