#!/usr/bin/env julia

"""
Benchmark the conservative frontend without entering the numerical solve.

The Task_Low08 path reports the cost of ingestion, exact preprocessing, and
arithmetic-aware equality presolve separately. The synthetic path exercises a
large collection of single-variable interval constraints through MOI and
verifies that their coefficient storage is linear rather than an `L × m`
reference grid.
"""

using MathOptInterface
using Printf
using SDPX

const MOI = MathOptInterface

include(
    joinpath(
        @__DIR__,
        "..",
        "lattice_bootstrap",
        "benchmark_sdpx_float64_solve.jl",
    ),
)

function benchmark_task(path::String)
    loaded = @timed read_problem(abspath(path))
    data = loaded.value
    ingested = @timed SDPX.ingest(
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
    problem = ingested.value
    options = SDPX.SolverOptions{Float64}(
        verbosity=0,
        presolve=:auto,
        scaling=:none,
        chordal_decomposition=:auto,
    )

    # Warm compilation on the same immutable problem. Preprocessing never
    # mutates caller-owned data.
    SDPX.preprocess(problem, options)
    GC.gc()
    prepared = @timed SDPX.preprocess(problem, options)
    preprocessed = prepared.value

    SDPX.presolve_equalities(preprocessed.problem, options)
    GC.gc()
    equality = @timed SDPX.presolve_equalities(
        preprocessed.problem,
        options,
    )
    reduced, _, equality_report = equality.value
    report = preprocessed.report
    @printf(
        "case=Task_Low08 variables=%d equalities=%d blocks=%d coefficient_nnz=%d block_pattern_density=%.8f\n",
        problem.dims.m,
        problem.dims.n,
        problem.dims.L,
        problem.structure.coefficient_nnz,
        problem.structure.block_pattern_density,
    )
    @printf(
        "input_seconds=%.6f input_bytes=%d ingest_seconds=%.6f ingest_bytes=%d\n",
        loaded.time,
        loaded.bytes,
        ingested.time,
        ingested.bytes,
    )
    @printf(
        "preprocess_seconds=%.6f preprocess_bytes=%d changed=%s bounds_lower=%d bounds_upper=%d fixed=%d exact_equalities_removed=%d chordal_analyzed=%s chordal_reason=%s\n",
        prepared.time,
        prepared.bytes,
        report.changed,
        report.extracted_lower_bounds,
        report.extracted_upper_bounds,
        report.fixed_variables_eliminated,
        report.zero_equalities_removed +
        report.duplicate_equalities_removed +
        report.proportional_equalities_removed,
        report.chordal.analyzed,
        replace(report.chordal.rejection_reason, ' ' => '_'),
    )
    @printf(
        "equality_presolve_seconds=%.6f equality_presolve_bytes=%d equalities_before=%d equalities_after=%d dependent_removed=%d inconsistent=%s\n",
        equality.time,
        equality.bytes,
        preprocessed.problem.dims.n,
        reduced.dims.n,
        equality_report.removed_dependent_equalities,
        equality_report.inconsistent,
    )
end

function interval_model(variables::Int)
    source = MOI.Utilities.Model{Float64}()
    x = MOI.add_variables(source, variables)
    for variable in x
        MOI.add_constraint(source, variable, MOI.Interval(-1.0, 1.0))
    end
    objective = MOI.ScalarAffineFunction(
        [
            MOI.ScalarAffineTerm(1.0 / variables, variable)
            for variable in x
        ],
        0.0,
    )
    MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        source,
        MOI.ObjectiveFunction{typeof(objective)}(),
        objective,
    )
    return source
end

function benchmark_intervals(variables::Int)
    source = interval_model(variables)
    optimizer = SDPX.Optimizer{Float64}(verbose=0)
    MOI.copy_to(optimizer, source)
    MOI.empty!(optimizer)
    GC.gc()
    copied = @timed MOI.copy_to(optimizer, source)
    problem = optimizer.problem
    sparse_cons = problem.cons::SDPX.SparseCons{Float64}
    compact_blocks = count(
        block -> block isa SDPX.CompactScalarCoefficientVector{Float64},
        sparse_cons.Asp,
    )

    options = SDPX.SolverOptions{Float64}(
        verbosity=0,
        scaling=:none,
        chordal_decomposition=:off,
    )
    SDPX.preprocess(problem, options)
    GC.gc()
    prepared = @timed SDPX.preprocess(problem, options)
    report = prepared.value.report
    historical_reference_bytes =
        2 * variables * variables * sizeof(Ptr{Nothing})
    @printf(
        "case=synthetic_intervals variables=%d blocks=%d compact_blocks=%d\n",
        variables,
        problem.dims.L,
        compact_blocks,
    )
    @printf(
        "copy_seconds=%.6f copy_bytes=%d preprocess_seconds=%.6f preprocess_bytes=%d lower_bounds=%d upper_bounds=%d changed=%s\n",
        copied.time,
        copied.bytes,
        prepared.time,
        prepared.bytes,
        report.extracted_lower_bounds,
        report.extracted_upper_bounds,
        report.changed,
    )
    @printf(
        "historical_reference_grid_floor_bytes=%d compact_coefficient_nonzeros=%d\n",
        historical_reference_bytes,
        problem.structure.coefficient_nnz,
    )
end

function main(arguments)
    isempty(arguments) && error(
        "usage: benchmark_frontend.jl task INPUT | intervals VARIABLE_COUNT",
    )
    mode = Symbol(arguments[1])
    if mode === :task
        length(arguments) == 2 ||
            error("task mode requires the Task_Low08 binary path")
        benchmark_task(arguments[2])
    elseif mode === :intervals
        length(arguments) == 2 ||
            error("intervals mode requires a variable count")
        benchmark_intervals(parse(Int, arguments[2]))
    else
        error("mode must be task or intervals")
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
