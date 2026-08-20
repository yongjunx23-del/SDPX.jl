#!/usr/bin/env julia

module COOContractionBenchmark

using SDPX
using Statistics
using TOML

const SCHEMA_VERSION = 1
const DEFAULT_REPETITIONS = 100_000
const DEFAULT_SAMPLES = 15

export run_benchmark, compare_benchmarks, main

function _fixture()
    matrix = reshape(Float64.(1:64), 8, 8)
    linear_indices = Int32[2, 13, 24, 37, 50, 63]
    rows = Int32[((Int(index) - 1) % 8) + 1 for index in linear_indices]
    columns = Int32[((Int(index) - 1) ÷ 8) + 1 for index in linear_indices]
    values = Float64[1.25, -2.0, 0.5, 3.0, -1.5, 0.75]
    coo = SDPX.SparseBlockCOO{Float64}(
        Int32[1, length(linear_indices) + 1],
        linear_indices,
        rows,
        columns,
        values,
    )
    expected = sum(
        matrix[Int(rows[index]), Int(columns[index])] * values[index]
        for index in eachindex(values)
    )
    return matrix, coo, expected
end

Base.@noinline function _repeated_contraction(matrix, coo, repetitions::Int)
    accumulator = 0.0
    @inbounds for _ in 1:repetitions
        accumulator += SDPX._dot_dense_coo(matrix, coo, 1)
    end
    return accumulator
end

function _median_absolute_deviation(values)
    center = median(values)
    return median(abs.(values .- center))
end

function _git_value(arguments...)
    repository = normpath(joinpath(@__DIR__, "..", ".."))
    try
        return readchomp(`git -C $repository $(arguments...)`)
    catch
        return "unknown"
    end
end

function _write_document(path::AbstractString, document)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return abspath(path)
end

"""Measure the sparse-Schur COO inner contraction after a full untimed warmup."""
function run_benchmark(
    output::AbstractString;
    repetitions::Integer=DEFAULT_REPETITIONS,
    samples::Integer=DEFAULT_SAMPLES,
)
    repetitions >= 1 || throw(ArgumentError("repetitions must be positive"))
    samples >= 3 || throw(ArgumentError("samples must be at least three"))
    repetitions = Int(repetitions)
    samples = Int(samples)

    matrix, coo, expected = _fixture()
    single = SDPX._dot_dense_coo(matrix, coo, 1)
    single == expected || error(
        "COO contraction mismatch: measured=$single expected=$expected",
    )

    # Compile the exact production call and execute one full untimed sample so
    # compilation and first-use initialization do not enter the measurements.
    _repeated_contraction(matrix, coo, 1)
    expected_checksum = _repeated_contraction(matrix, coo, repetitions)

    elapsed = Float64[]
    allocated = Int[]
    checksums = Float64[]
    for _ in 1:samples
        GC.gc(false)
        measurement = @timed _repeated_contraction(matrix, coo, repetitions)
        push!(elapsed, measurement.time)
        push!(allocated, measurement.bytes)
        push!(checksums, measurement.value)
    end
    all(isequal(expected_checksum), checksums) || error(
        "COO contraction checksum changed between samples",
    )

    median_seconds = median(elapsed)
    median_bytes = median(allocated)
    document = Dict{String,Any}(
        "schema_version" => SCHEMA_VERSION,
        "benchmark" => Dict{String,Any}(
            "name" => "sparse_coo_contraction",
            "source_commit" => _git_value("rev-parse", "HEAD"),
            "source_status" => _git_value("status", "--porcelain=v1"),
            "julia_version" => string(VERSION),
            "julia_threads" => Threads.nthreads(),
            "repetitions" => repetitions,
            "samples" => samples,
        ),
        "result" => Dict{String,Any}(
            "single_value" => single,
            "checksum" => expected_checksum,
            "median_seconds" => median_seconds,
            "minimum_seconds" => minimum(elapsed),
            "maximum_seconds" => maximum(elapsed),
            "mad_seconds" => _median_absolute_deviation(elapsed),
            "median_nanoseconds_per_call" =>
                median_seconds * 1.0e9 / repetitions,
            "median_allocated_bytes" => Float64(median_bytes),
            "minimum_allocated_bytes" => minimum(allocated),
            "maximum_allocated_bytes" => maximum(allocated),
            "median_bytes_per_call" => Float64(median_bytes) / repetitions,
            "sample_seconds" => elapsed,
            "sample_allocated_bytes" => allocated,
        ),
    )
    path = _write_document(output, document)
    println(
        "coo_contraction median=", median_seconds, "s allocation=",
        median_bytes, " bytes ns/call=",
        document["result"]["median_nanoseconds_per_call"],
        " output=", path,
    )
    return document
end

function _result(document)
    get(document, "schema_version", 0) == SCHEMA_VERSION ||
        throw(ArgumentError("unsupported COO benchmark schema"))
    return document["result"]
end

"""Compare a candidate with its exact parent and classify retain/neutral/reject."""
function compare_benchmarks(
    baseline_path::AbstractString,
    candidate_path::AbstractString,
    output::AbstractString,
)
    baseline = _result(TOML.parsefile(baseline_path))
    candidate = _result(TOML.parsefile(candidate_path))

    baseline_checksum = Float64(baseline["checksum"])
    candidate_checksum = Float64(candidate["checksum"])
    correctness_valid = isequal(baseline_checksum, candidate_checksum)

    baseline_seconds = Float64(baseline["median_seconds"])
    candidate_seconds = Float64(candidate["median_seconds"])
    baseline_bytes = Float64(baseline["median_allocated_bytes"])
    candidate_bytes = Float64(candidate["median_allocated_bytes"])
    runtime_ratio = candidate_seconds / baseline_seconds
    allocation_ratio = baseline_bytes == 0 ?
        (candidate_bytes == 0 ? 1.0 : Inf) : candidate_bytes / baseline_bytes

    runtime_regression = runtime_ratio > 1.05
    allocation_regression = candidate_bytes > baseline_bytes * 1.05 + 256
    material_runtime_gain = runtime_ratio <= 0.95
    material_allocation_gain =
        baseline_bytes - candidate_bytes >= 4096 &&
        candidate_bytes <= max(512.0, baseline_bytes * 0.10)

    decision = if !correctness_valid || runtime_regression || allocation_regression
        "reject"
    elseif material_runtime_gain || material_allocation_gain
        "retain"
    else
        "neutral"
    end

    document = Dict{String,Any}(
        "schema_version" => SCHEMA_VERSION,
        "comparison" => Dict{String,Any}(
            "benchmark" => "sparse_coo_contraction",
            "decision" => decision,
            "correctness_valid" => correctness_valid,
            "runtime_ratio" => runtime_ratio,
            "runtime_change_percent" => (runtime_ratio - 1.0) * 100,
            "allocation_ratio" => allocation_ratio,
            "allocation_change_percent" => (allocation_ratio - 1.0) * 100,
            "baseline_seconds" => baseline_seconds,
            "candidate_seconds" => candidate_seconds,
            "baseline_allocated_bytes" => baseline_bytes,
            "candidate_allocated_bytes" => candidate_bytes,
            "runtime_regression" => runtime_regression,
            "allocation_regression" => allocation_regression,
            "material_runtime_gain" => material_runtime_gain,
            "material_allocation_gain" => material_allocation_gain,
        ),
    )
    path = _write_document(output, document)
    println(
        "coo_comparison decision=", decision,
        " runtime_ratio=", runtime_ratio,
        " allocation_ratio=", allocation_ratio,
        " output=", path,
    )
    decision == "reject" && error(
        "sparse COO candidate rejected by correctness/performance gate",
    )
    return document
end

function _option(args, prefix, default)
    for argument in args
        startswith(argument, prefix) || continue
        return split(argument, "="; limit=2)[2]
    end
    return default
end

function main(args=ARGS)
    isempty(args) && throw(ArgumentError(
        "usage: coo_contraction.jl run OUTPUT [--repetitions=N] [--samples=N] " *
        "or coo_contraction.jl compare BASELINE CANDIDATE OUTPUT",
    ))
    mode = first(args)
    if mode == "run"
        length(args) >= 2 || throw(ArgumentError("run requires an output path"))
        repetitions = parse(Int, _option(args[3:end], "--repetitions=", string(DEFAULT_REPETITIONS)))
        samples = parse(Int, _option(args[3:end], "--samples=", string(DEFAULT_SAMPLES)))
        return run_benchmark(args[2]; repetitions, samples)
    elseif mode == "compare"
        length(args) == 4 || throw(ArgumentError(
            "compare requires baseline, candidate, and output paths",
        ))
        return compare_benchmarks(args[2], args[3], args[4])
    end
    throw(ArgumentError("unknown mode $mode"))
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    COOContractionBenchmark.main(ARGS)
end
