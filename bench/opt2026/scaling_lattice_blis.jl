#!/usr/bin/env julia

"""Run one Task_Low08 scaling point with BLIS and LAPACK through BLISBLAS.jl."""

Sys.islinux() && Sys.ARCH === :x86_64 ||
    error("the BLIS scaling driver requires Linux x86_64")

using BLISBLAS
using LinearAlgebra

println("dense_backend=blis")
println("lbt_config=", BLAS.lbt_get_config())

driver = get(
    ENV,
    "SDPX_SCALING_DRIVER",
    joinpath(@__DIR__, "scaling_lattice.jl"),
)
include(driver)
