#!/usr/bin/env julia

"""
Run the ownership-safe native BigFloat equality-arrow path on one immutable
CSDR model.

The serialized campaign models were assembled at BigFloat1024 and rounded
once to Float64x4.  This driver performs the established low-energy
elimination in Float64x4, then converts the reduced problem to independently
owned BigFloat values.  It therefore measures the solver path on the exact
same rounded mathematical model as the production baseline.  It does not
claim that Float64x4 coefficients have recovered the discarded precomputation
digits; the original-grid and off-grid physical checks remain separate gates.
"""

using LinearAlgebra
using Serialization
using SparseArrays
using SDPX
using TOML

const CSDR_SOURCE = get(ENV, "CSDR_SOURCE", "")
isempty(CSDR_SOURCE) && error(
    "set CSDR_SOURCE to the immutable CSDRBootstrap source checkout",
)
include(joinpath(CSDR_SOURCE, "src", "CSDRBootstrap.jl"))
using .CSDRBootstrap

function parse_cli(arguments)
    values = Dict{String,String}()
    for argument in arguments
        startswith(argument, "--") || error("unknown argument: $argument")
        key, value = split(argument[3:end], '='; limit=2)
        values[key] = value
    end
    haskey(values, "model") || error("--model is required")
    haskey(values, "output") || error("--output is required")
    return (
        model=abspath(values["model"]),
        output=abspath(values["output"]),
        precision_bits=parse(Int, get(values, "precision-bits", "512")),
        threads=parse(Int, get(values, "threads", string(Threads.nthreads()))),
        maximum_iterations=parse(Int, get(values, "maximum-iterations", "220")),
        time_limit=parse(Float64, get(values, "time-limit", "18000")),
    )
end

function convert_sparse_matrix(matrix::SparseMatrixCSC)
    rows, columns, values = findnz(matrix)
    return sparse(
        rows,
        columns,
        BigFloat.(values),
        size(matrix, 1),
        size(matrix, 2),
    )
end

function convert_reduced_problem(problem)
    cons = problem.cons
    cons isa SDPX.SparseCons || error("CSDR conversion requires sparse constraints")
    blocks = Vector{
        SDPX.ActiveSparseCoefficientVector{BigFloat}
    }(undef, problem.dims.L)
    for block in 1:problem.dims.L
        variables = copy(cons.active[block])
        coefficients = [
            convert_sparse_matrix(cons.Asp[block][variable])
            for variable in variables
        ]
        blocks[block] = SDPX.ActiveSparseCoefficientVector(
            BigFloat,
            problem.dims.m,
            variables,
            coefficients,
            problem.dims.k[block],
        )
    end
    constants = [BigFloat.(matrix) for matrix in problem.C]
    return SDPX.ingest(
        BigFloat.(problem.c),
        blocks,
        constants,
        convert_sparse_matrix(problem.B),
        BigFloat.(problem.b);
        sparse=true,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
end

function physical_objective(payload, full_x)
    label_index = Dict(
        label => index
        for (index, label) in enumerate(payload.coefficient_labels)
    )
    objective = zero(BigFloat)
    for (label, value_string) in payload.config.objective
        coefficient = parse(BigFloat, value_string)
        value = if haskey(label_index, label)
            full_x[label_index[label]]
        else
            parse(BigFloat, payload.config.fixed_coefficients[label])
        end
        objective += coefficient * value
    end
    return objective
end

function write_summary(path, entries)
    open(path, "w") do io
        TOML.print(io, entries; sorted=true)
    end
end

function main(arguments)
    cli = parse_cli(arguments)
    mkpath(cli.output)
    setprecision(BigFloat, cli.precision_bits) do
        payload = open(deserialize, cli.model)
        tolerance = parse(BigFloat, payload.config.tolerance)

        elimination = nothing
        elimination_seconds = @elapsed begin
            elimination = CSDRBootstrap._eliminate_low_energy_variables(payload)
        end
        problem = nothing
        conversion_seconds = @elapsed begin
            problem = convert_reduced_problem(elimination.problem)
        end
        SDPX._supports_owned_bigfloat_arrow_equalities(problem) ||
            error("converted problem did not select the owned BigFloat arrow path")

        options = SDPX.SolverOptions{BigFloat}(;
            ϵ_gap=tolerance,
            ϵ_primal=tolerance,
            ϵ_dual=tolerance,
            iter_max=cli.maximum_iterations,
            max_time=cli.time_limit,
            precision_bits=cli.precision_bits,
            working_precision_policy=:fixed,
            sparse=true,
            threads=min(max(cli.threads, 1), Threads.nthreads()),
            predictor=:sdpb,
            step_rule=:auto,
            max_restarts=10,
            verbosity=1,
            diagnostics=true,
            timing=true,
            parameter_policy=:auto,
            parameter_strategy=:adaptive,
            refine_policy=:auto,
            equilibrate=true,
            scaling=:auto,
            stall_iterations=0,
            extended_precision_blas=:on,
            equality_solver=:auto,
            mixed_precision_kkt=:off,
        )

        result = nothing
        solver_seconds = @elapsed begin
            result = SDPX.solve!(problem, options)
        end

        coefficient_constant = BigFloat.(elimination.coefficient_constant)
        coefficient_from_spectrum = BigFloat.(
            elimination.coefficient_from_spectrum,
        )
        coefficients =
            coefficient_constant + coefficient_from_spectrum * result.x
        full_x = vcat(coefficients, result.x)
        status = string(result.status)
        objective = physical_objective(payload, full_x)
        solution = (
            status=status,
            x=full_x,
            solver_profile="native_bigfloat_owned_arrow_adaptive_ruiz",
            physical_objective=objective,
        )
        solution_path = joinpath(cli.output, "solution.bin")
        open(solution_path, "w") do io
            serialize(io, solution)
        end

        validation_dir = joinpath(cli.output, "validation")
        validation_seconds = @elapsed begin
            CSDRBootstrap.run_validation(
                cli.model,
                solution_path,
                validation_dir,
            )
        end
        validation = TOML.parsefile(joinpath(validation_dir, "validation.toml"))

        timings = result.timings === nothing ? NamedTuple() : result.timings
        summary = Dict{String,Any}(
            "status" => status,
            "message" => result.message,
            "precision_bits" => cli.precision_bits,
            "julia_threads" => Threads.nthreads(),
            "solver_threads" => options.threads,
            "variables" => problem.dims.m,
            "equalities" => problem.dims.n,
            "blocks" => problem.dims.L,
            "iterations" => result.iterations,
            "restarts" => result.restarts,
            "regularizations" => result.regularizations,
            "total_refinement_steps" =>
                result.termination.total_refinement_steps,
            "solver_seconds" => solver_seconds,
            "elimination_seconds" => elimination_seconds,
            "conversion_seconds" => conversion_seconds,
            "validation_seconds" => validation_seconds,
            "primal_objective" => string(result.pObj),
            "dual_objective" => string(result.dObj),
            "physical_objective" => string(objective),
            "relative_gap" => string(result.gap_rel),
            "primal_residual" => string(result.p_res),
            "dual_residual" => string(result.d_res),
            "peak_rss_bytes" => Int(Sys.maxrss()),
            "owned_bigfloat_arrow" => true,
            "coefficient_source" => "BigFloat1024 precompute rounded once to Float64x4",
            "validation" => validation,
        )
        for name in propertynames(timings)
            summary["timing_$(name)_seconds"] = getproperty(timings, name)
        end
        if result.diagnostics !== nothing
            plan = result.diagnostics.plan
            summary["plan_kkt_backend"] = string(plan.kkt_backend)
            summary["plan_gram_kernel"] = string(plan.gram_kernel)
            summary["plan_schedule"] = string(plan.schedule)
            if hasproperty(result.termination, :executed)
                summary["executed_kkt"] =
                    string(result.termination.executed.kkt)
                summary["executed_equality"] =
                    string(result.termination.executed.equality)
            end
        end
        write_summary(joinpath(cli.output, "summary.toml"), summary)
        println("result_root=$(cli.output)")
        println("status=$status")
        println("physical_objective=$objective")
        println("relative_gap=$(result.gap_rel)")
        validation_accepted = validation["accepted"]
        println("validation_accepted=$validation_accepted")
    end
end

main(ARGS)
