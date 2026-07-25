#!/usr/bin/env julia

"""
Benchmark the allocation-free BigFloat sparse `2x2` block-arrow kernels.

The default input is the generated CSDR s15 artifact. Override it with
`SPARSE_INPUT`, select MPFR precision with `BIGFLOAT_BITS`, and control the
number of samples with `BENCH_SAMPLES`. Results are written to `BENCH_OUTPUT`.
Run with one Julia thread; the solver deliberately keeps BigFloat serial.
"""

using LinearAlgebra
using MultiFloats
using Printf
using SDPX
using Serialization
using Statistics

const DEFAULT_INPUT = normpath(joinpath(
    @__DIR__,
    "..",
    "csdr_psd_dual",
    "results",
    "20260724-cluster-scale",
    "problem-s15.bin",
))

function convert_problem(data)
    return (
        c=BigFloat.(data.c),
        A=[BigFloat.(block) for block in data.A],
        C=[BigFloat.(block) for block in data.C],
        B=BigFloat.(data.B),
        b=BigFloat.(data.b),
    )
end

function synthetic_problem(active_variables::Int)
    active_variables > 0 ||
        throw(ArgumentError("SYNTHETIC_ACTIVE must be positive"))
    coefficients = [zeros(BigFloat, active_variables, 2, 2)]
    for variable in 1:active_variables
        diagonal =
            one(BigFloat) + BigFloat(variable) / BigFloat(active_variables)
        off_diagonal =
            BigFloat(mod(variable, 17) - 8) / BigFloat(101)
        coefficients[1][variable, 1, 1] = diagonal
        coefficients[1][variable, 1, 2] = off_diagonal
        coefficients[1][variable, 2, 1] = off_diagonal
        coefficients[1][variable, 2, 2] =
            diagonal + BigFloat(1) / BigFloat(7)
    end
    return SDPX.ingest(
        ones(BigFloat, active_variables),
        coefficients,
        [zeros(BigFloat, 2, 2)],
        zeros(BigFloat, active_variables, 0),
        BigFloat[];
        sparse=true,
        verbosity=0,
    )
end

function allocation_count(gcstats)
    return gcstats.malloc +
           gcstats.realloc +
           gcstats.poolalloc +
           gcstats.bigalloc
end

function measure(operation; samples::Int)
    operation()
    times = Vector{Float64}(undef, samples)
    bytes = Vector{Int}(undef, samples)
    allocations = Vector{Int}(undef, samples)
    for sample in 1:samples
        GC.gc()
        measurement = @timed operation()
        times[sample] = measurement.time
        bytes[sample] = measurement.bytes
        allocations[sample] = allocation_count(measurement.gcstats)
    end
    return (
        seconds=median(times),
        bytes=Int(median(bytes)),
        allocations=Int(median(allocations)),
    )
end

function reference_schur!(workspace, problem, X, Y)
    arrow = workspace.arrow
    SDPX._zero_arrow_schur!(arrow)
    for block in 1:problem.dims.L
        SDPX._fused_arrow_schur_block_generic!(
            arrow,
            workspace.blk[block],
            problem.cons,
            block,
            X[block],
            Y[block],
            arrow.Sgg,
        )
    end
    return arrow.Sgg
end

function optimized_schur!(workspace, problem, X, Y)
    return SDPX.schur_build!(
        workspace,
        problem,
        problem.cons,
        X,
        Y,
    )
end

function reference_buildP!(workspace, problem, x)
    for block in 1:problem.dims.L
        SDPX._buildP_sparse_generic!(
            workspace.blk[block].P,
            problem.cons,
            block,
            x,
        )
    end
    return workspace
end

function optimized_buildP!(workspace, problem, x)
    for block in 1:problem.dims.L
        SDPX.buildP!(
            workspace.blk[block].P,
            problem.cons,
            block,
            x,
        )
    end
    return workspace
end

function reference_accumulate!(workspace, problem, matrices)
    SDPX.zero_owned!(workspace.v)
    for block in 1:problem.dims.L
        SDPX._accumulate_v_sparse_generic!(
            workspace.v,
            problem.cons,
            block,
            matrices[block],
            -one(BigFloat),
        )
    end
    return workspace.v
end

function optimized_accumulate!(workspace, problem, matrices)
    SDPX.zero_owned!(workspace.v)
    for block in 1:problem.dims.L
        SDPX.accumulate_v_owned!(
            workspace.v,
            problem.cons,
            block,
            matrices[block],
            -one(BigFloat),
        )
    end
    return workspace.v
end

function materialized_schur(workspace, problem)
    result = SDPX.alloc_zeros(
        BigFloat,
        problem.dims.m,
        problem.dims.m,
    )
    SDPX.materialize_schur!(result, workspace)
    return deepcopy(result)
end

function report_row(io, stage, reference, optimized, relative_error)
    speedup = reference.seconds / optimized.seconds
    @printf(
        "%-20s %10.4f %10.4f %8.2f %14d %14d %12d %12d %.3e\n",
        stage,
        1_000reference.seconds,
        1_000optimized.seconds,
        speedup,
        reference.bytes,
        optimized.bytes,
        reference.allocations,
        optimized.allocations,
        relative_error,
    )
    println(io, join((
        stage,
        @sprintf("%.9f", reference.seconds),
        @sprintf("%.9f", optimized.seconds),
        @sprintf("%.4f", speedup),
        reference.bytes,
        optimized.bytes,
        reference.allocations,
        optimized.allocations,
        @sprintf("%.6e", relative_error),
    ), ","))
end

function main()
    Threads.nthreads() == 1 ||
        error("BigFloat sparse benchmark must run with one Julia thread")
    input = abspath(get(ENV, "SPARSE_INPUT", DEFAULT_INPUT))
    bits = parse(Int, get(ENV, "BIGFLOAT_BITS", "256"))
    samples = parse(Int, get(ENV, "BENCH_SAMPLES", "7"))
    synthetic_active =
        parse(Int, get(ENV, "SYNTHETIC_ACTIVE", "0"))
    synthetic_active == 0 && !isfile(input) &&
        error(
            "CSDR input not found at $input. Generate it with " *
            "bench/csdr_psd_dual/prepare_problem.jl, set SPARSE_INPUT, " *
            "or set SYNTHETIC_ACTIVE to run the self-contained inner-kernel " *
            "benchmark.",
        )
    output = abspath(get(
        ENV,
        "BENCH_OUTPUT",
        joinpath(@__DIR__, "results", "bigfloat-sparse-schur.csv"),
    ))

    data = synthetic_active == 0 ? open(deserialize, input) : nothing
    mkpath(dirname(output))
    setprecision(BigFloat, bits) do
        problem = if synthetic_active > 0
            synthetic_problem(synthetic_active)
        else
            converted = convert_problem(data)
            SDPX.ingest(
                converted.c,
                converted.A,
                converted.C,
                converted.B,
                converted.b;
                sparse=true,
                verbosity=0,
            )
        end
        workspace = SDPX.Workspace(problem; thread_count=1)
        workspace.arrow === nothing &&
            error("input does not have exact block-arrow structure")
        workspace.fused_arrow ||
            error("input is not eligible for the fused 2x2 kernel")

        X = [
            Matrix{BigFloat}(I, dimension, dimension)
            for dimension in problem.dims.k
        ]
        Y = [
            Matrix{BigFloat}(I, dimension, dimension)
            for dimension in problem.dims.k
        ]
        SDPX.factor_blocks!(workspace, X, Y) ||
            error("benchmark starting point is not positive definite")
        x = ones(BigFloat, problem.dims.m)

        reference_schur!(workspace, problem, X, Y)
        reference_matrix = materialized_schur(workspace, problem)
        optimized_schur!(workspace, problem, X, Y)
        optimized_matrix = materialized_schur(workspace, problem)
        schur_scale =
            max(maximum(abs, reference_matrix), one(BigFloat))
        schur_error = Float64(
            maximum(abs, optimized_matrix - reference_matrix) /
            schur_scale,
        )

        reference_accumulate!(workspace, problem, Y)
        reference_vector = deepcopy(workspace.v)
        optimized_accumulate!(workspace, problem, Y)
        contraction_scale =
            max(maximum(abs, reference_vector), one(BigFloat))
        contraction_error = Float64(
            maximum(abs, workspace.v - reference_vector) /
            contraction_scale,
        )

        reference_build =
            measure(
                () -> reference_buildP!(workspace, problem, x);
                samples=samples,
            )
        optimized_build =
            measure(
                () -> optimized_buildP!(workspace, problem, x);
                samples=samples,
            )
        reference_accumulate =
            measure(
                () -> reference_accumulate!(workspace, problem, Y);
                samples=samples,
            )
        optimized_accumulate =
            measure(
                () -> optimized_accumulate!(workspace, problem, Y);
                samples=samples,
            )
        reference_schur =
            measure(
                () -> reference_schur!(workspace, problem, X, Y);
                samples=samples,
            )
        optimized_schur =
            measure(
                () -> optimized_schur!(workspace, problem, X, Y);
                samples=samples,
            )

        println(
            "BigFloat bits=$bits, blocks=$(problem.dims.L), " *
            "variables=$(problem.dims.m), active range=" *
            "$(extrema(length.(problem.cons.active)))",
        )
        println(
            "stage                 old_ms     new_ms  speedup      old_bytes" *
            "      new_bytes   old_allocs   new_allocs relative_error",
        )
        open(output, "w") do io
            println(
                io,
                "stage,reference_seconds,optimized_seconds,speedup," *
                "reference_bytes,optimized_bytes,reference_allocations," *
                "optimized_allocations,relative_error",
            )
            report_row(
                io,
                "packed_buildP",
                reference_build,
                optimized_build,
                0.0,
            )
            report_row(
                io,
                "packed_accumulate",
                reference_accumulate,
                optimized_accumulate,
                contraction_error,
            )
            report_row(
                io,
                "fused_arrow_schur",
                reference_schur,
                optimized_schur,
                schur_error,
            )
        end
        println("CSV: $output")
    end
end

main()
