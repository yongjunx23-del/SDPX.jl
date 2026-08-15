#!/usr/bin/env julia
using Pkg
root = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=root))
Pkg.instantiate()
println("SDPX CLI ready. Example:")
println("  $(joinpath(@__DIR__, "sdpx")) --help")
