#!/usr/bin/env julia

"""
Benchmark old and opt-in extended-precision Schur assembly on either the
serialized CSDR PSD dual or the Task_Low08 lattice binary.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using Serialization
using SparseArrays

const LATTICE_MAGIC = collect(codeunits("LATSDP01"))

function parse_cli(arguments)
    values = Dict{String,String}()
    for argument in arguments
        startswith(argument, "--") ||
            error("unknown argument: $argument")
        key, value = split(argument[3:end], "=", limit=2)
        values[key] = value
    end
    for required in ("problem", "input", "output", "arithmetic", "mode")
        haskey(values, required) || error("missing --$required")
    end
    problem = Symbol(values["problem"])
    problem in (:sparse, :lattice) ||
        error("--problem must be sparse or lattice")
    arithmetic = Symbol(values["arithmetic"])
    arithmetic in (:float64, :float64x4, :bigfloat) ||
        error("--arithmetic must be float64, float64x4, or bigfloat")
    mode = Symbol(values["mode"])
    mode in (:off, :auto, :on) ||
        error("--mode must be off, auto, or on")
    return (
        problem=problem,
        input=abspath(values["input"]),
        output=abspath(values["output"]),
        arithmetic=arithmetic,
        mode=mode,
        precision_bits=parse(Int, get(values, "precision-bits", "256")),
        repetitions=parse(Int, get(values, "repetitions", "2")),
        validation_samples=parse(Int, get(values, "validation-samples", "64")),
        memory_fraction=parse(
            Float64,
            get(values, "memory-fraction", "0.10"),
        ),
    )
end

function arithmetic_type(symbol::Symbol)
    symbol === :float64 && return Float64
    symbol === :float64x4 && return Float64x4
    return BigFloat
end

function read_vector(input::IO, ::Type{T}, count::Int) where {T}
    values = Vector{T}(undef, count)
    read!(input, values)
    return values
end

function read_lattice_problem(path::String, ::Type{T}) where {T}
    open(path, "r") do input
        read_vector(input, UInt8, length(LATTICE_MAGIC)) == LATTICE_MAGIC ||
            error("unexpected lattice benchmark file format")
        variables = Int(read(input, Int64))
        equalities = Int(read(input, Int64))
        block_count = Int(read(input, Int64))
        tolerance = T(read(input, Float64))
        coefficients =
            Vector{Vector{SparseMatrixCSC{T,Int}}}(undef, block_count)
        block_constants = Vector{Matrix{T}}(undef, block_count)
        dimensions = Vector{Int}(undef, block_count)
        for block in 1:block_count
            dimension = Int(read(input, Int32))
            dimensions[block] = dimension
            entry_count = Int(read(input, Int64))
            rows_by_slot = Dict{Int,Vector{Int}}()
            columns_by_slot = Dict{Int,Vector{Int}}()
            values_by_slot = Dict{Int,Vector{T}}()
            constant = zeros(T, dimension, dimension)
            for _ in 1:entry_count
                row = Int(read(input, Int32)) + 1
                column = Int(read(input, Int32)) + 1
                slot = Int(read(input, Int32))
                value = T(read(input, Float64))
                if slot == 0
                    constant[row, column] += value
                    row != column &&
                        (constant[column, row] += value)
                else
                    rows = get!(rows_by_slot, slot, Int[])
                    columns = get!(columns_by_slot, slot, Int[])
                    entries = get!(values_by_slot, slot, T[])
                    push!(rows, row)
                    push!(columns, column)
                    push!(entries, value)
                    if row != column
                        push!(rows, column)
                        push!(columns, row)
                        push!(entries, value)
                    end
                end
            end
            coefficients[block] = [
                haskey(rows_by_slot, variable) ?
                sparse(
                    rows_by_slot[variable],
                    columns_by_slot[variable],
                    values_by_slot[variable],
                    dimension,
                    dimension,
                ) :
                spzeros(T, dimension, dimension)
                for variable in 1:variables
            ]
            block_constants[block] = -constant
        end
        equality_nnz = Int(read(input, Int64))
        rows = Vector{Int}(undef, equality_nnz)
        columns = Vector{Int}(undef, equality_nnz)
        entries = Vector{T}(undef, equality_nnz)
        for index in 1:equality_nnz
            rows[index] = Int(read(input, Int32)) + 1
            columns[index] = Int(read(input, Int32)) + 1
            entries[index] = T(read(input, Float64))
        end
        equality_constants =
            T.(read_vector(input, Float64, equalities))
        equality_matrix =
            sparse(rows, columns, entries, equalities, variables)
        objective = zeros(T, variables)
        objective[1] = one(T)
        return (
            c=objective,
            A=coefficients,
            C=block_constants,
            B=Matrix(transpose(equality_matrix)),
            b=-equality_constants,
            tolerance=tolerance,
            dimensions=dimensions,
        )
    end
end

function convert_csdr(source, ::Type{Float64x4})
    return source
end

function convert_csdr(source, ::Type{T}) where {T}
    return (
        c=T.(source.c),
        A=[T.(block) for block in source.A],
        C=[T.(block) for block in source.C],
        B=T.(source.B),
        b=T.(source.b),
        dimensions=fill(2, source.ncell),
    )
end

function load_problem(cli, ::Type{T}) where {T}
    if cli.problem === :lattice
        return read_lattice_problem(cli.input, T)
    end
    source = open(deserialize, cli.input)
    data = convert_csdr(source, T)
    return (
        c=data.c,
        A=data.A,
        C=data.C,
        B=data.B,
        b=data.b,
        dimensions=hasproperty(data, :dimensions) ?
                   data.dimensions : fill(2, data.ncell),
    )
end

function sampled_schur_error(
    problem::SDPX.SDPProblem{T},
    workspace::SDPX.Workspace{T},
    samples::Int,
) where {T}
    materialized = zeros(T, problem.dims.m, problem.dims.m)
    SDPX.materialize_schur!(materialized, workspace)
    state = UInt64(0x9e3779b97f4a7c15)
    maximum_relative_error = 0.0
    maximum_absolute_error = 0.0
    cons = problem.cons
    for _ in 1:samples
        state =
            state * UInt64(6364136223846793005) +
            UInt64(1442695040888963407)
        first = Int(rem(state, UInt64(problem.dims.m))) + 1
        state =
            state * UInt64(6364136223846793005) +
            UInt64(1442695040888963407)
        second = Int(rem(state, UInt64(problem.dims.m))) + 1
        expected = zero(T)
        if cons isa SDPX.SparseCons{T}
            sparse_cons = cons::SDPX.SparseCons{T}
            for block in sparse_cons.Asp
                expected += dot(block[first], block[second])
            end
        else
            dense_cons = cons::SDPX.DenseCons{T}
            for panel in dense_cons.Av
                expected += dot(
                    view(panel, :, first),
                    view(panel, :, second),
                )
            end
        end
        actual = materialized[first, second]
        absolute_error = Float64(abs(actual - expected))
        relative_error =
            Float64(abs(actual - expected) / max(abs(expected), one(T)))
        maximum_absolute_error =
            max(maximum_absolute_error, absolute_error)
        maximum_relative_error =
            max(maximum_relative_error, relative_error)
    end
    return maximum_absolute_error, maximum_relative_error
end

function decision_summary(workspace)
    selected = 0
    reasons = Dict{Symbol,Int}()
    predicted = Float64[]
    for plan in workspace.extended_precision.block_plans
        decision = plan.decision
        decision.enabled && (selected += 1)
        reasons[decision.reason] = get(reasons, decision.reason, 0) + 1
        push!(predicted, decision.estimated_speedup)
    end
    reason_string = join(
        (
            "$(key):$(value)"
            for (key, value) in sort!(collect(reasons); by=first)
        ),
        ';',
    )
    return (
        selected=selected,
        reasons=reason_string,
        maximum_predicted_speedup=maximum(predicted; init=1.0),
    )
end

function append_csv(path::String, row)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(names, ','))
        values = (
            replace(string(getproperty(row, name)), ',' => ';')
            for name in names
        )
        println(output, join(values, ','))
    end
end

function benchmark(cli, ::Type{T}) where {T}
    load_timing = @timed load_problem(cli, T)
    data = load_timing.value
    ingest_timing = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=cli.problem === :sparse ? true : :auto,
        validate=false,
        symmetrize=false,
        verbosity=0,
    )
    problem = ingest_timing.value
    workspace_timing = @timed SDPX.Workspace(
        problem;
        extended_precision_blas=cli.mode,
        extended_precision_memory_fraction=cli.memory_fraction,
    )
    workspace = workspace_timing.value
    X = [
        Matrix{T}(I, dimension, dimension)
        for dimension in data.dimensions
    ]
    Y = [
        Matrix{T}(I, dimension, dimension)
        for dimension in data.dimensions
    ]
    SDPX.factor_blocks!(workspace, X, Y) ||
        error("identity factorization failed")
    SDPX.threaded_schur_build!(
        workspace,
        problem,
        problem.cons,
        X,
        Y,
    )

    timings = Float64[]
    allocations = Int[]
    gc_times = Float64[]
    for _ in 1:cli.repetitions
        GC.gc()
        measurement = @timed SDPX.threaded_schur_build!(
            workspace,
            problem,
            problem.cons,
            X,
            Y,
        )
        push!(timings, measurement.time)
        push!(allocations, measurement.bytes)
        push!(gc_times, measurement.gctime)
    end
    absolute_error, relative_error = sampled_schur_error(
        problem,
        workspace,
        cli.validation_samples,
    )
    decisions = decision_summary(workspace)
    row = (
        problem=cli.problem,
        arithmetic=cli.arithmetic,
        precision_bits=T === BigFloat ? precision(BigFloat) : precision(T),
        mode=cli.mode,
        julia_threads=Threads.nthreads(),
        variables=problem.dims.m,
        equalities=problem.dims.n,
        psd_blocks=problem.dims.L,
        coefficient_density=problem.structure.coefficient_density,
        active_density=problem.structure.active_density,
        schur_density=problem.structure.schur_density,
        selected_blocks=decisions.selected,
        decision_reasons=decisions.reasons,
        maximum_predicted_speedup=decisions.maximum_predicted_speedup,
        packing_megabytes=
            workspace.extended_precision.packing_bytes / 1.0e6,
        lower_triangle_only=workspace.extended_precision.lower_only,
        input_seconds=load_timing.time,
        ingest_seconds=ingest_timing.time,
        workspace_seconds=workspace_timing.time,
        runtime_min_seconds=minimum(timings),
        runtime_median_seconds=sort(timings)[cld(length(timings), 2)],
        allocated_bytes_min=minimum(allocations),
        gc_seconds_min=minimum(gc_times),
        # Recursive `summarysize` walks every mutable MPFR object and can take
        # minutes on the 2,100-block benchmark. RSS is the authoritative peak
        # measurement for BigFloat, so avoid perturbing the run with that walk.
        workspace_megabytes=T === BigFloat ?
                            NaN : Base.summarysize(workspace) / 1.0e6,
        peak_rss_megabytes=Sys.maxrss() / 1.0e6,
        sampled_schur_max_absolute_error=absolute_error,
        sampled_schur_max_relative_error=relative_error,
    )
    append_csv(cli.output, row)
    for name in propertynames(row)
        @printf("%-36s %s\n", name, getproperty(row, name))
    end
    return row
end

function main(arguments)
    cli = parse_cli(arguments)
    cli.repetitions >= 1 || error("--repetitions must be positive")
    T = arithmetic_type(cli.arithmetic)
    if T === BigFloat
        setprecision(BigFloat, cli.precision_bits) do
            benchmark(cli, T)
        end
    else
        benchmark(cli, T)
    end
end

main(ARGS)
