#!/usr/bin/env julia

"""Benchmark arithmetic-aware equality presolve on Task_Low08."""

using Printf
using SDPX

include(
    joinpath(
        @__DIR__,
        "..",
        "lattice_bootstrap",
        "benchmark_sdpx_float64_solve.jl",
    ),
)

function main(arguments)
    length(arguments) == 3 || error(
        "usage: benchmark_equality_presolve.jl INPUT REPS LABEL",
    )
    input = abspath(arguments[1])
    repetitions = parse(Int, arguments[2])
    label = arguments[3]

    data = read_problem(input)
    problem = SDPX.ingest(
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
    options = SDPX.SolverOptions{Float64}(verbosity=0)
    SDPX.presolve_equalities(problem, options)

    for repetition in 1:repetitions
        GC.gc()
        measurement = @timed SDPX.presolve_equalities(problem, options)
        reduced, _, report = measurement.value
        @printf(
            "label=%s repetition=%d elapsed_seconds=%.9f allocated_bytes=%d equalities_before=%d equalities_after=%d inconsistent=%s\n",
            label,
            repetition,
            measurement.time,
            measurement.bytes,
            problem.dims.n,
            reduced.dims.n,
            report.inconsistent,
        )
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
