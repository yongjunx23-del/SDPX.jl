#!/usr/bin/env julia
include(joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry
SDPXBenchmarkRegistry.main(ARGS)
