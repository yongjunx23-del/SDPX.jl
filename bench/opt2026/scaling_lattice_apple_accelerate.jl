#!/usr/bin/env julia

"""Run one strict-single-thread Task_Low08 point with Apple Accelerate."""

Sys.isapple() ||
    error("Apple Accelerate is available only on macOS")

using AppleAccelerate
using SDPX

SDPX.blas_backend() === :apple_accelerate ||
    error("the SDPX AppleAccelerate extension did not activate")
SDPX.set_blas_threads!(1)
SDPX.blas_threads() == 1 ||
    error("Apple Accelerate did not enter single-threaded mode")

println(
    "dense_backend=", SDPX.blas_backend(),
    " blas_threads=", SDPX.blas_threads(),
)

include(joinpath(@__DIR__, "scaling_lattice.jl"))
