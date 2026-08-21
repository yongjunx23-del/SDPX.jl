#!/usr/bin/env julia

"""Preflight Task_Low08 workspace memory without allocating a solver workspace."""

using MultiFloats: Float64x4
using Printf
using SDPX

let root = get(
        ENV,
        "SDPX_BENCH_SOURCE",
        normpath(joinpath(@__DIR__, "..", "..")),
    ),
    path = joinpath(
        root,
        "bench",
        "extended_precision_blas",
        "benchmark_schur.jl",
    )
    source = read(path, String)
    include_string(
        @__MODULE__,
        replace(source, r"\nmain\(ARGS\)\s*$" => "\n"),
        path,
    )
end

function report_type(input::String, ::Type{T}) where {T}
    GC.gc()
    load = @timed read_lattice_problem(input, T)
    data = load.value
    ingest = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=:auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    problem = ingest.value
    @printf(
        "arithmetic=%s precision_bits=%d load_seconds=%.3f ingest_seconds=%.3f rss_gib=%.3f\n",
        T,
        T === BigFloat ? precision(BigFloat) : precision(T),
        load.time,
        ingest.time,
        Sys.maxrss() / 2.0^30,
    )
    @printf(
        "variables=%d equalities=%d blocks=%d mixed_kkt_gib=%.3f\n",
        problem.dims.m,
        problem.dims.n,
        problem.dims.L,
        SDPX._mixed_precision_storage_bytes(
            problem.dims.m,
            problem.dims.n,
        ) / 2.0^30,
    )
    for threads in (1, 2, 4, 8, 16, 32, 64, 128)
        floor_bytes = SDPX.dense_workspace_floor_bytes(
            T,
            problem.dims.m,
            problem.dims.n,
            problem.dims.L,
            threads,
        )
        estimate_bytes =
            SDPX.estimate_sdp_workspace_bytes(problem, threads)
        bins = SDPX.schur_bin_report(
            T,
            problem.dims.m,
            problem.dims.L,
            threads,
        )
        @printf(
            "threads=%3d floor_gib=%8.3f estimate_gib=%8.3f schur_bins=%d/%d schur_bin_gib=%.3f\n",
            threads,
            floor_bytes / 2.0^30,
            estimate_bytes / 2.0^30,
            bins.selected_bins,
            bins.requested_bins,
            bins.total_bytes / 2.0^30,
        )
    end
    return nothing
end

function main(arguments)
    length(arguments) == 1 ||
        error("usage: estimate_task_low08_extended_memory.jl INPUT")
    input = abspath(arguments[1])
    report_type(input, Float64x4)
    setprecision(BigFloat, 256) do
        report_type(input, BigFloat)
    end
end

main(ARGS)
