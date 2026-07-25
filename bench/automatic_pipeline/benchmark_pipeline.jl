#!/usr/bin/env julia

"""Benchmark the automatic pipeline, dedicated LP path, and adaptive IPM."""

using LinearAlgebra
using MultiFloats
using Printf
using Random
using SDPX
using Serialization
using Statistics

BLAS.set_num_threads(1)

function dense_lp_problem(
    variables::Int,
    inequalities::Int;
    seed::Int=42,
)
    inequalities >= variables ||
        throw(ArgumentError("inequalities must be at least variables"))
    rng = MersenneTwister(seed)
    G = randn(rng, inequalities, variables) / sqrt(variables)
    G[1:variables, :] .= Matrix{Float64}(I, variables, variables)
    optimum = randn(rng, variables)
    h = G * optimum
    h[(variables + 1):end] .-= 0.5 .+
        rand(rng, inequalities - variables)
    dual = zeros(inequalities)
    dual[1:variables] .= 0.5 .+ rand(rng, variables)
    c = transpose(G) * dual
    A = [
        reshape(copy(G[row, :]), variables, 1, 1)
        for row in 1:inequalities
    ]
    C = [fill(h[row], 1, 1) for row in 1:inequalities]
    problem = SDPX.ingest(
        c,
        A,
        C,
        zeros(variables, 0),
        Float64[];
        sparse=:auto,
        verbosity=0,
    )
    return problem, dot(c, optimum)
end

function load_csdr_problem(path::String)
    source = deserialize(path)
    return SDPX.ingest(
        Float64.(source.c),
        [Float64.(block) for block in source.A],
        [Float64.(block) for block in source.C],
        Float64.(source.B),
        Float64.(source.b);
        sparse=true,
        verbosity=0,
    )
end

function minimum_psd_eigenvalue(result)
    return minimum(
        matrix -> eigmin(Symmetric(matrix)),
        result.X;
        init=Inf,
    )
end

function measure(
    name::String,
    problem,
    options;
    repetitions::Int=5,
    expected_objective::Union{Nothing,Float64}=nothing,
)
    SDPX.solve!(problem, options)
    measurements = NamedTuple[]
    for repetition in 1:repetitions
        GC.gc()
        sample = @timed SDPX.solve!(problem, options)
        push!(
            measurements,
            (
                repetition,
                seconds=sample.time,
                allocated_bytes=sample.bytes,
                gc_seconds=sample.gctime,
                result=sample.value,
            ),
        )
    end
    representative = measurements[argmin(getfield.(measurements, :seconds))]
    result = representative.result
    return (
        name,
        arithmetic=string(eltype(problem)),
        julia_threads=Threads.nthreads(),
        requested_threads=options.threads,
        blas_threads=BLAS.get_num_threads(),
        strategy=string(options.parameter_strategy),
        algorithm=string(result.diagnostics.plan.algorithm),
        gram_kernel=string(result.diagnostics.plan.gram_kernel),
        parameter_profile=string(
            result.diagnostics.plan.parameter_profile,
        ),
        selected_beta=Float64(
            result.diagnostics.plan.parameters.beta,
        ),
        selected_gamma=Float64(
            result.diagnostics.plan.parameters.gamma,
        ),
        median_seconds=median(getfield.(measurements, :seconds)),
        minimum_seconds=minimum(getfield.(measurements, :seconds)),
        allocated_bytes=representative.allocated_bytes,
        gc_seconds=representative.gc_seconds,
        workspace_bytes=result.diagnostics.memory.workspace_bytes,
        process_peak_rss_bytes=result.diagnostics.memory.process_peak_rss_bytes,
        status=string(result.status),
        iterations=result.iterations,
        objective=Float64(result.pObj),
        objective_error=expected_objective === nothing ? NaN :
                        abs(Float64(result.pObj) - expected_objective),
        relative_gap=Float64(result.gap_rel),
        primal_residual=Float64(result.p_res),
        dual_residual=Float64(result.d_res),
        minimum_psd_eigenvalue=minimum_psd_eigenvalue(result),
        presolve_equalities_removed=
            result.diagnostics.presolve.removed_dependent_equalities,
        redundant_constraints_removed=
            result.diagnostics.presolve.removed_redundant_constraints,
    )
end

function write_csv(path::String, rows)
    mkpath(dirname(path))
    fields = propertynames(first(rows))
    open(path, "w") do output
        println(output, join(fields, ','))
        for row in rows
            println(output, join((getfield(row, field) for field in fields), ','))
        end
    end
end

function main(arguments)
    output = isempty(arguments) ?
             joinpath(@__DIR__, "results", "pipeline.csv") :
             abspath(arguments[1])
    csdr_path = length(arguments) >= 2 ? abspath(arguments[2]) :
        normpath(
            @__DIR__,
            "..",
            "csdr_psd_dual",
            "results",
            "20260724-cluster-scale",
            "problem-s15.bin",
        )

    medium_lp, medium_optimum = dense_lp_problem(80, 400; seed=42)
    large_lp, large_optimum = dense_lp_problem(256, 4_000; seed=43)
    csdr = load_csdr_problem(csdr_path)

    lp_base = (
        verbosity=0,
        timing=true,
        ϵ_gap=1e-8,
        ϵ_primal=1e-8,
        ϵ_dual=1e-8,
        iter_max=100,
    )
    sdp_base = (
        β=0.1,
        γ=0.8,
        Ωp=10.0,
        Ωd=10.0,
        ϵ_gap=1e-7,
        ϵ_primal=1e-7,
        ϵ_dual=1e-7,
        iter_max=100,
        verbosity=0,
        timing=true,
        predictor=:sdpb,
        refine_steps=0,
        max_restarts=10,
        # This benchmark isolates fixed/adaptive iteration policy and Ω.
        # Automatic structural profiles intentionally override those fields.
        parameter_policy=:fixed,
    )

    rows = NamedTuple[]
    push!(
        rows,
        measure(
            "medium_lp_general_conic",
            medium_lp,
            SDPX.SolverOptions{Float64}(;
                lp_base...,
                algorithm=:sdp,
                parameter_strategy=:fixed,
                threads=1,
            );
            repetitions=5,
            expected_objective=medium_optimum,
        ),
    )
    push!(
        rows,
        measure(
            "medium_lp_dedicated_legacy_parameters",
            medium_lp,
            SDPX.SolverOptions{Float64}(;
                lp_base...,
                β=0.1,
                γ=0.9,
                algorithm=:lp,
                parameter_policy=:fixed,
                parameter_strategy=:fixed,
                threads=1,
            );
            repetitions=5,
            expected_objective=medium_optimum,
        ),
    )
    for strategy in (:fixed, :adaptive)
        push!(
            rows,
            measure(
                "medium_lp_dedicated_$strategy",
                medium_lp,
                SDPX.SolverOptions{Float64}(;
                    lp_base...,
                    algorithm=:lp,
                    parameter_strategy=strategy,
                    threads=1,
                );
                repetitions=5,
                expected_objective=medium_optimum,
            ),
        )
    end
    push!(
        rows,
        measure(
            "large_lp_legacy_1_thread",
            large_lp,
            SDPX.SolverOptions{Float64}(;
                lp_base...,
                β=0.1,
                γ=0.9,
                algorithm=:lp,
                parameter_policy=:fixed,
                parameter_strategy=:fixed,
                threads=1,
            );
            repetitions=5,
            expected_objective=large_optimum,
        ),
    )
    for threads in (1, 2, 4, 8)
        threads <= Threads.nthreads() || continue
        push!(
            rows,
            measure(
                "large_lp_$(threads)_threads",
                large_lp,
                SDPX.SolverOptions{Float64}(;
                    lp_base...,
                    algorithm=:lp,
                    parameter_strategy=:fixed,
                    threads,
                );
                repetitions=5,
                expected_objective=large_optimum,
            ),
        )
    end
    for strategy in (:fixed, :adaptive)
        push!(
            rows,
            measure(
                "csdr_s15_$strategy",
                csdr,
                SDPX.SolverOptions{Float64}(;
                    sdp_base...,
                    parameter_strategy=strategy,
                    threads=min(Threads.nthreads(), 8),
                );
                repetitions=5,
            ),
        )
    end
    push!(
        rows,
        measure(
            "csdr_s15_omega_1",
            csdr,
            SDPX.SolverOptions{Float64}(;
                sdp_base...,
                Ωp=1.0,
                Ωd=1.0,
                max_time=30.0,
                parameter_strategy=:fixed,
                threads=min(Threads.nthreads(), 8),
            );
            repetitions=1,
        ),
    )

    write_csv(output, rows)
    foreach(println, rows)
    println("wrote $output")
end

main(ARGS)
