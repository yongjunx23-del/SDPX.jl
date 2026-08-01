#!/usr/bin/env julia

"""Benchmark ownership-safe BigFloat triangular SYRK tile widths."""

using SDPX

const EPBLAS = SDPX.ExtendedPrecisionBLAS

function parse_cli(arguments)
    values = Dict{String,String}()
    for argument in arguments
        startswith(argument, "--") || error("unknown argument: $argument")
        key, value = split(argument[3:end], '='; limit=2)
        values[key] = value
    end
    return (
        rows=parse(Int, get(values, "rows", "3400")),
        columns=parse(Int, get(values, "columns", "144")),
        repetitions=parse(Int, get(values, "repetitions", "3")),
        precision_bits=parse(Int, get(values, "precision-bits", "256")),
        tiles=parse.(Int, split(get(values, "tiles", "4,6,8,12,16,24"), ',')),
        kernel_threads=parse.(
            Int,
            split(
                get(
                    values,
                    "kernel-threads",
                    string(Threads.nthreads()),
                ),
                ',',
            ),
        ),
    )
end

function build_panel(rows::Int, columns::Int)
    panel = Matrix{BigFloat}(undef, rows, columns)
    @inbounds for column in 1:columns, row in 1:rows
        numerator = mod(37row + 19column, 1009) - 504
        panel[row, column] = BigFloat(numerator) / BigFloat(1009)
    end
    return panel
end

function main(arguments)
    cli = parse_cli(arguments)
    setprecision(BigFloat, cli.precision_bits) do
        panel = build_panel(cli.rows, cli.columns)
        output = SDPX.alloc_zeros(
            BigFloat,
            cli.columns,
            cli.columns,
        )
        for requested_threads in cli.kernel_threads
            kernel_threads = min(
                max(requested_threads, 1),
                Threads.nthreads(),
            )
            for tile in cli.tiles
                config = EPBLAS.KernelConfig(
                    row_tile=24,
                    column_tile=tile,
                    micro_tile=1,
                )
                selected_workers =
                    EPBLAS._syrk_bigfloat_selected_workers(
                        panel,
                        config,
                        kernel_threads,
                    )
                EPBLAS.zero_triangle!(output)
                EPBLAS.syrk!(
                    output,
                    panel,
                    one(BigFloat),
                    zero(BigFloat),
                    config,
                    kernel_threads,
                )
                elapsed = Float64[]
                allocated = Int[]
                for _ in 1:cli.repetitions
                    EPBLAS.zero_triangle!(output)
                    measurement = @timed EPBLAS.syrk!(
                        output,
                        panel,
                        one(BigFloat),
                        zero(BigFloat),
                        config,
                        kernel_threads,
                    )
                    push!(elapsed, measurement.time)
                    push!(allocated, measurement.bytes)
                end
                println(
                    "tile=$tile requested_threads=$requested_threads " *
                    "selected_workers=$selected_workers " *
                    "julia_threads=$(Threads.nthreads()) " *
                    "best_seconds=$(minimum(elapsed)) " *
                    "median_seconds=$(sort(elapsed)[cld(length(elapsed), 2)]) " *
                    "minimum_allocated_bytes=$(minimum(allocated))",
                )
            end
        end
    end
end

main(ARGS)
