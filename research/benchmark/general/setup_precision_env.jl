#!/usr/bin/env julia
using Pkg
root=normpath(joinpath(@__DIR__,"..",".."))
Pkg.develop(path=root)
Pkg.instantiate()
Pkg.precompile()
