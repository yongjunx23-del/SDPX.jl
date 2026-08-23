#!/usr/bin/env julia

"""Benchmark sparse Ruiz equilibration on the canonical medium CSDR model."""

using MultiFloats: Float64x4
using Printf
using SDPX
using SparseArrays

function scaled_checksum(problem)
    value = sum(Float64, problem.c)
    for constant in problem.C
        value += sum(Float64, constant)
    end
    sparse_cons = problem.cons::SDPX.SparseCons{Float64x4}
    for block in eachindex(sparse_cons.Asp)
        for variable in sparse_cons.active[block]
            value += sum(Float64, nonzeros(sparse_cons.Asp[block][variable]))
        end
    end
    return value
end

function append_csv(path::String, row::NamedTuple)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(string.(names), ','))
        println(
            output,
            join((string(getproperty(row, name)) for name in names), ','),
        )
    end
end

function main(arguments)
    length(arguments) == 5 || error(
        "usage: benchmark_sparse_equilibration.jl MODEL_IO MODEL OUTPUT REPS LABEL",
    )
    include(abspath(arguments[1]))
    loader = Base.invokelatest(getfield, Main, :CanonicalModelIO)
    model_path = abspath(arguments[2])
    output = abspath(arguments[3])
    repetitions = parse(Int, arguments[4])
    label = arguments[5]

    # The loader is included at run time so Julia 1.12 requires an explicit
    # latest-world dispatch from this precompiled top-level method.
    data = Base.invokelatest(
        loader.load_canonical_model,
        Float64x4,
        model_path,
    )
    problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=true,
        verbosity=0,
    )
    sparse_cons = problem.cons::SDPX.SparseCons{Float64x4}
    SDPX.equilibrate(problem, sparse_cons)

    for repetition in 1:repetitions
        GC.gc()
        measurement = @timed SDPX.equilibrate(problem, sparse_cons)
        scaled, _ = measurement.value
        row = (
            label,
            repetition,
            elapsed_seconds=measurement.time,
            allocated_bytes=measurement.bytes,
            gc_seconds=measurement.gctime,
            checksum=scaled_checksum(scaled),
            active_incidences=problem.structure.active_incidences,
            full_block_variable_slots=problem.dims.L * problem.dims.m,
        )
        append_csv(output, row)
        @printf(
            "label=%s rep=%d elapsed=%.6f allocated=%d checksum=%.17g\n",
            label,
            repetition,
            measurement.time,
            measurement.bytes,
            row.checksum,
        )
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
