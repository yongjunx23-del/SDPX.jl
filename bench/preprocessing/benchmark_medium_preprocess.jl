#!/usr/bin/env julia

"""
Compare preprocessing disabled and automatic on the canonical medium CSDR
model. Both variants use the same ingested problem, arithmetic, parameters,
thread counts, and validation path. The script depends on the benchmark
directory's `model_io.jl` loader but does not modify that benchmark.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX

function append_csv(path::String, row::NamedTuple)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(string.(names), ','))
        println(
            output,
            join(
                (
                    replace(string(getproperty(row, name)), ',' => ';')
                    for name in names
                ),
                ',',
            ),
        )
    end
end

function options(presolve)
    T = Float64x4
    return SDPX.SolverOptions{T}(
        β=T(0.1),
        γ=T(0.85),
        Ωp=T(25),
        Ωd=T(25),
        ϵ_gap=T(1e-10),
        ϵ_primal=T(1e-10),
        ϵ_dual=T(1e-10),
        iter_max=500,
        max_time=7200.0,
        verbosity=0,
        timing=true,
        predictor=:sdpb,
        step_rule=:fraction_to_boundary,
        refine_steps=1,
        refine_policy=:auto,
        parameter_policy=:fixed,
        parameter_strategy=:fixed,
        extended_precision_blas=:auto,
        mixed_precision_kkt=:off,
        presolve=presolve,
        scaling=:none,
        threads=Threads.nthreads(),
        diagnostics=true,
        expert_mode=true,
    )
end

function timing_value(timings, name)
    timings === nothing && return 0.0
    return get(timings, name, 0.0)
end

function main(arguments)
    length(arguments) == 4 || error(
        "usage: benchmark_medium_preprocess.jl MODEL_IO MODEL OUTPUT REPS",
    )
    model_io = abspath(arguments[1])
    model_path = abspath(arguments[2])
    output = abspath(arguments[3])
    repetitions = parse(Int, arguments[4])
    include(model_io)
    loader = Base.invokelatest(getfield, Main, :CanonicalModelIO)

    # The loader is included at run time so Julia 1.12 requires an explicit
    # latest-world dispatch from this precompiled top-level method.
    data_timing = @timed Base.invokelatest(
        loader.load_canonical_model,
        Float64x4,
        model_path,
    )
    data = data_timing.value
    ingest_timing = @timed SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=true,
        verbosity=0,
    )
    problem = ingest_timing.value
    @printf(
        "model_load_seconds=%.6f ingest_seconds=%.6f variables=%d blocks=%d threads=%d\n",
        data_timing.time,
        ingest_timing.time,
        problem.dims.m,
        problem.dims.L,
        Threads.nthreads(),
    )

    # Compile both branches before any retained measurement.
    for strategy in (:off, :auto)
        warm = SDPX.solve!(problem, options(strategy))
        warm.status == SDPX.Optimal ||
            error("warmup failed for presolve=$strategy: $(warm.status)")
    end

    for repetition in 1:repetitions
        for strategy in (:off, :auto)
            GC.gc()
            measurement = @timed SDPX.solve!(
                problem,
                options(strategy),
            )
            result = measurement.value
            result.status == SDPX.Optimal ||
                error(
                    "solve failed for presolve=$strategy repetition=$repetition: " *
                    "$(result.status)",
                )
            diagnostics = result.diagnostics
            preprocessing = diagnostics.presolve.preprocessing
            certificate =
                diagnostics.selected_algorithms.certificate
            row = (
                repetition,
                presolve=strategy,
                julia_threads=Threads.nthreads(),
                blas_threads=SDPX.blas_threads(),
                solve_seconds=measurement.time,
                allocated_bytes=measurement.bytes,
                gc_seconds=measurement.gctime,
                peak_rss_bytes=diagnostics.memory.process_peak_rss_bytes,
                preprocessing_seconds=
                    preprocessing === nothing ? 0.0 : preprocessing.elapsed,
                preprocessing_bytes=
                    preprocessing === nothing ? 0 : preprocessing.allocated_bytes,
                preprocessing_changed=
                    preprocessing === nothing ? false : preprocessing.changed,
                iterations=result.iterations,
                status=result.status,
                primal_objective=result.pObj,
                dual_objective=result.dObj,
                relative_gap=result.gap_rel,
                primal_residual=result.p_res,
                dual_residual=result.d_res,
                minimum_primal_psd_eigenvalue=
                    -certificate.primal_cone_violation,
                minimum_dual_psd_eigenvalue=
                    -certificate.dual_cone_violation,
                certificate_valid=certificate.valid,
                schur_seconds=timing_value(
                    result.timings,
                    :schur_assembly,
                ),
                kkt_seconds=timing_value(
                    result.timings,
                    :kkt_factorization,
                ),
            )
            append_csv(output, row)
            @printf(
                "rep=%d presolve=%s solve=%.6f preprocess=%.6f iterations=%d gap=%.3e certificate=%s\n",
                repetition,
                strategy,
                measurement.time,
                row.preprocessing_seconds,
                result.iterations,
                Float64(result.gap_rel),
                certificate.valid,
            )
            flush(stdout)
        end
    end
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
