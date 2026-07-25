#!/usr/bin/env julia

using LinearAlgebra
using MultiFloats: Float64x4
using SDPX

const EPBLAS = SDPX.ExtendedPrecisionBLAS

function append_row(path, row)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(names, ','))
        println(
            output,
            join((getproperty(row, name) for name in names), ','),
        )
    end
end

function benchmark(::Type{T}, rows::Int, columns::Int, repetitions::Int, output) where {T}
    panel = Matrix{T}(undef, rows, columns)
    @inbounds for column in 1:columns, row in 1:rows
        panel[row, column] =
            T(sin(0.013 * row + 0.007 * column))
    end
    old_output = zeros(T, columns, columns)
    new_output = zeros(T, columns, columns)
    T === BigFloat && EPBLAS.prepare_storage!(new_output)
    features = EPBLAS.CrossoverFeatures(
        rows=rows,
        columns=columns,
        matrix_dimension=round(Int, sqrt(rows)),
        average_nnz=rows,
        active_density=1.0,
        expected_schur_density=1.0,
        thread_count=T === BigFloat ? 1 : Threads.nthreads(),
        memory_budget_bytes=typemax(Int),
        sparse_input=false,
    )
    config =
        EPBLAS.choose_crossover(T, features; mode=:on).config
    SDPX._dense_gram_add!(old_output, panel)
    EPBLAS.syrk!(
        new_output,
        panel,
        one(T),
        zero(T),
        config,
        T === BigFloat ? 1 : Threads.nthreads(),
    )

    old_times = Float64[]
    new_times = Float64[]
    old_allocations = Int[]
    new_allocations = Int[]
    for _ in 1:repetitions
        fill!(old_output, zero(T))
        old_measurement =
            @timed SDPX._dense_gram_add!(old_output, panel)
        EPBLAS.zero_triangle!(new_output)
        new_measurement = @timed EPBLAS.syrk!(
            new_output,
            panel,
            one(T),
            zero(T),
            config,
            T === BigFloat ? 1 : Threads.nthreads(),
        )
        push!(old_times, old_measurement.time)
        push!(new_times, new_measurement.time)
        push!(old_allocations, old_measurement.bytes)
        push!(new_allocations, new_measurement.bytes)
    end
    relative_error =
        maximum(abs, LowerTriangular(old_output - new_output)) /
        max(maximum(abs, LowerTriangular(old_output)), one(T))
    old_time = minimum(old_times)
    new_time = minimum(new_times)
    row = (
        arithmetic=string(T),
        precision_bits=T === BigFloat ? precision(BigFloat) : precision(T),
        threads=T === BigFloat ? 1 : Threads.nthreads(),
        panel_rows=rows,
        panel_columns=columns,
        old_seconds=old_time,
        new_seconds=new_time,
        speedup=old_time / new_time,
        old_allocated_bytes=minimum(old_allocations),
        new_allocated_bytes=minimum(new_allocations),
        peak_rss_megabytes=Sys.maxrss() / 1.0e6,
        relative_error=Float64(relative_error),
    )
    append_row(output, row)
    println(row)
end

function main(arguments)
    length(arguments) == 5 ||
        error(
            "usage: benchmark_dense_kernel.jl " *
            "float64x4|bigfloat ROWS COLUMNS REPETITIONS OUTPUT.csv",
        )
    arithmetic = Symbol(arguments[1])
    rows = parse(Int, arguments[2])
    columns = parse(Int, arguments[3])
    repetitions = parse(Int, arguments[4])
    output = abspath(arguments[5])
    if arithmetic === :float64x4
        benchmark(Float64x4, rows, columns, repetitions, output)
    elseif arithmetic === :bigfloat
        setprecision(BigFloat, 256) do
            benchmark(BigFloat, rows, columns, repetitions, output)
        end
    else
        error("arithmetic must be float64x4 or bigfloat")
    end
end

main(ARGS)
