#!/usr/bin/env julia

"""
Benchmark the high-precision LP Hessian kernels introduced for SDPX.

Float64x4 compares the legacy serial weighted outer product with the optional
blocked triangular SYRK path. BigFloat compares the former allocating scalar
loop with the allocation-free MPFR weighted outer product. BigFloat is always
run with one compute thread.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Random
using SDPX
using Statistics

BLAS.set_num_threads(1)

function legacy_weighted_outer_product!(H, G, weights)
    variables = size(G, 2)
    fill!(H, zero(eltype(H)))
    @inbounds for row in axes(G, 1)
        weight = weights[row]
        for column in 1:variables
            scaled = weight * G[row, column]
            for inner in 1:column
                H[inner, column] += G[row, inner] * scaled
            end
        end
    end
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            H[row, column] = H[column, row]
        end
    end
    return H
end

function measure(operation, repetitions)
    operation()
    samples = NamedTuple[]
    for _ in 1:repetitions
        GC.gc()
        sample = @timed operation()
        push!(
            samples,
            (
                seconds=sample.time,
                allocated_bytes=sample.bytes,
                gc_seconds=sample.gctime,
            ),
        )
    end
    return (
        median_seconds=median(getfield.(samples, :seconds)),
        minimum_seconds=minimum(getfield.(samples, :seconds)),
        minimum_allocated_bytes=minimum(
            getfield.(samples, :allocated_bytes),
        ),
        median_gc_seconds=median(getfield.(samples, :gc_seconds)),
    )
end

function benchmark_float64x4(rows, variables, repetitions, threads)
    rng = MersenneTwister(702)
    G = Float64x4.(randn(rng, rows, variables))
    weights = Float64x4.(rand(rng, rows) .+ 0.25)
    serial = SDPX.LPWorkspace(
        Float64x4,
        rows,
        variables,
        0;
        packed_hessian=true,
    )
    packed = SDPX.LPWorkspace(
        Float64x4,
        rows,
        variables,
        0;
        packed_hessian=true,
    )
    serial.weights .= weights
    packed.weights .= weights
    old = measure(
        () -> SDPX._lp_assemble_hessian!(
            serial,
            G,
            1,
            :serial_weighted_outer_product,
        ),
        repetitions,
    )
    new = measure(
        () -> SDPX._lp_assemble_hessian!(
            packed,
            G,
            threads,
            threads > 1 ? :threaded_blocked_syrk : :blocked_syrk,
        ),
        repetitions,
    )
    relative_error =
        maximum(abs, packed.H - serial.H) /
        maximum(abs, serial.H)
    return (
        arithmetic="Float64x4",
        precision_bits=precision(Float64x4),
        rows,
        variables,
        threads,
        old_seconds=old.median_seconds,
        new_seconds=new.median_seconds,
        speedup=old.median_seconds / new.median_seconds,
        old_allocated_bytes=old.minimum_allocated_bytes,
        new_allocated_bytes=new.minimum_allocated_bytes,
        old_workspace_bytes=SDPX._lp_workspace_bytes(serial),
        new_workspace_bytes=SDPX._lp_workspace_bytes(packed),
        default_serial_workspace_bytes=SDPX._lp_workspace_bytes(
            SDPX.LPWorkspace(
                Float64x4,
                rows,
                variables,
                0;
                packed_hessian=false,
            ),
        ),
        relative_error=Float64(relative_error),
        process_peak_rss_bytes=SDPX._process_peak_rss_bytes(),
    )
end

function benchmark_bigfloat(rows, variables, repetitions, precision_bits)
    Threads.nthreads() == 1 ||
        @warn "BigFloat kernel is serial; extra Julia threads are unused"
    return setprecision(BigFloat, precision_bits) do
        rng = MersenneTwister(703)
        G = BigFloat.(randn(rng, rows, variables))
        weights = BigFloat.(rand(rng, rows) .+ 0.25)
        legacy_workspace = SDPX.LPWorkspace(
            BigFloat,
            rows,
            variables,
            0;
            packed_hessian=true,
        )
        workspace = SDPX.LPWorkspace(
            BigFloat,
            rows,
            variables,
            0;
            packed_hessian=false,
        )
        legacy_workspace.weights .= weights
        workspace.weights .= weights
        old = measure(
            () -> legacy_weighted_outer_product!(
                legacy_workspace.H,
                G,
                legacy_workspace.weights,
            ),
            repetitions,
        )
        new = measure(
            () -> SDPX._lp_assemble_hessian!(
                workspace,
                G,
                1,
                :serial_mpfr_weighted_outer_product,
            ),
            repetitions,
        )
        relative_error =
            maximum(abs, workspace.H - legacy_workspace.H) /
            maximum(abs, legacy_workspace.H)
        return (
            arithmetic="BigFloat",
            precision_bits,
            rows,
            variables,
            threads=1,
            old_seconds=old.median_seconds,
            new_seconds=new.median_seconds,
            speedup=old.median_seconds / new.median_seconds,
            old_allocated_bytes=old.minimum_allocated_bytes,
            new_allocated_bytes=new.minimum_allocated_bytes,
            old_workspace_bytes=SDPX._lp_workspace_bytes(legacy_workspace),
            new_workspace_bytes=SDPX._lp_workspace_bytes(workspace),
            default_serial_workspace_bytes=SDPX._lp_workspace_bytes(workspace),
            relative_error=Float64(relative_error),
            process_peak_rss_bytes=SDPX._process_peak_rss_bytes(),
        )
    end
end

function csv_field(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path, rows)
    mkpath(dirname(path))
    fields = propertynames(first(rows))
    open(path, "w") do output
        println(output, join(fields, ','))
        for row in rows
            println(
                output,
                join(
                    (csv_field(getfield(row, field)) for field in fields),
                    ',',
                ),
            )
        end
    end
end

function main(arguments)
    output = isempty(arguments) ?
             joinpath(@__DIR__, "results", "high-precision-lp.csv") :
             abspath(arguments[1])
    rows = length(arguments) >= 2 ? parse(Int, arguments[2]) : 384
    variables = length(arguments) >= 3 ? parse(Int, arguments[3]) : 96
    repetitions = length(arguments) >= 4 ? parse(Int, arguments[4]) : 5
    precision_bits = length(arguments) >= 5 ? parse(Int, arguments[5]) : 256
    requested_threads = length(arguments) >= 6 ?
                        parse(Int, arguments[6]) :
                        Threads.nthreads()
    arithmetic = length(arguments) >= 7 ?
                 lowercase(arguments[7]) : "both"
    arithmetic in ("both", "float64x4", "bigfloat") ||
        throw(ArgumentError(
            "arithmetic must be both, Float64x4, or BigFloat",
        ))
    threads = min(max(requested_threads, 1), Threads.nthreads())
    results = NamedTuple[]
    arithmetic in ("both", "float64x4") && push!(
        results,
        benchmark_float64x4(rows, variables, repetitions, threads),
    )
    arithmetic in ("both", "bigfloat") && push!(
        results,
        benchmark_bigfloat(
            min(rows, 192),
            min(variables, 72),
            repetitions,
            precision_bits,
        ),
    )
    write_csv(output, results)
    foreach(println, results)
    println("Wrote $(abspath(output))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
