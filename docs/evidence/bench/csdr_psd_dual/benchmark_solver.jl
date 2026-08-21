#!/usr/bin/env julia

# Core-solve benchmark for the exact same serialized Float64x4 PSD dual.
# Model ingestion/build and Julia compilation are measured separately from
# the steady-state solve. One full warmup precedes all timed repetitions.

using Clarabel
using GenericLinearAlgebra
using JuMP
using MultiFloats: Float64x4
using Printf
using SDPX
using Serialization
using SHA

const REFERENCE_ROOT = normpath(get(
    ENV,
    "CLARABEL_CSDR_ROOT",
    get(ENV, "CLARABEL_CSDR_ROOT", joinpath(@__DIR__, "reference-data")),
))
include(joinpath(
    REFERENCE_ROOT,
    "julia",
    "src",
    "clarabel_double64_psd_support.jl",
))

function parse_cli(args)
    values = Dict{String, String}()
    for arg in args
        startswith(arg, "--") || error("unknown argument: $arg")
        key, value = split(arg[3:end], "=", limit = 2)
        values[key] = value
    end
    for required in ("solver", "input", "output")
        haskey(values, required) || error("missing --$required")
    end
    solver = Symbol(values["solver"])
    solver in (:clarabel, :sdpx) || error("solver must be clarabel or sdpx")
    return (;
        solver,
        input = abspath(values["input"]),
        output = abspath(values["output"]),
        reps = parse(Int, get(values, "reps", "3")),
        tol = get(values, "tol", "1e-7"),
        equilibrate = parse(Int, get(values, "equilibrate", "1")) == 1,
        sparse = parse(Int, get(values, "sparse", "0")) == 1,
        beta = get(values, "beta", "0.01"),
        gamma = get(values, "gamma", "0.9"),
        omega_p = get(values, "omega-p", "10"),
        omega_d = get(values, "omega-d", "10"),
        predictor = Symbol(get(values, "predictor", "sdpb")),
        step_rule = Symbol(get(values, "step-rule", "backtrack")),
        max_restarts = parse(Int, get(values, "max-restarts", "10")),
        refine_steps = parse(Int, get(values, "refine-steps", "1")),
        parameter_policy = Symbol(get(values, "parameter-policy", "fixed")),
    )
end

empty_affine(::Type{T}, variable) where {T} = zero(T) * variable

function clarabel_factory(::Type{T}, tolerance::T) where {T <: AbstractFloat}
    return () -> Clarabel.Optimizer{T}(;
        verbose = false,
        max_iter = 1000,
        time_limit = 600.0,
        tol_gap_abs = tolerance,
        tol_gap_rel = tolerance,
        tol_feas = tolerance,
        tol_infeas_abs = tolerance,
        tol_infeas_rel = tolerance,
        tol_ktratio = tolerance,
        reduced_tol_gap_abs = tolerance,
        reduced_tol_gap_rel = tolerance,
        reduced_tol_feas = tolerance,
        reduced_tol_infeas_abs = tolerance,
        reduced_tol_infeas_rel = tolerance,
        reduced_tol_ktratio = tolerance,
        static_regularization_enable = false,
        dynamic_regularization_enable = false,
        direct_solve_method = :qdldl,
        max_threads = 1,
        chordal_decomposition_enable = false,
        presolve_enable = false,
    )
end

function build_clarabel_model(data, tolerance)
    T = eltype(data.c)
    model = GenericModel{T}(clarabel_factory(T, tolerance))
    @variable(model, x[1:data.m])

    for block in 1:data.ncell
        function entry(row, column)
            expression = empty_affine(T, x[1])
            constant = -data.C[block][row, column]
            iszero(constant) || add_to_expression!(expression, constant)
            for index in 1:data.m
                coefficient = data.A[block][index, row, column]
                iszero(coefficient) ||
                    add_to_expression!(expression, coefficient, x[index])
            end
            return expression
        end
        diagonal_1 = entry(1, 1)
        offdiagonal = entry(1, 2)
        diagonal_2 = entry(2, 2)
        @constraint(
            model,
            Symmetric([
                diagonal_1 offdiagonal
                offdiagonal diagonal_2
            ]) in PSDCone(),
        )
    end

    objective = empty_affine(T, x[1])
    for index in 1:data.m
        coefficient = data.c[index]
        iszero(coefficient) ||
            add_to_expression!(objective, coefficient, x[index])
    end
    @objective(model, Min, objective)
    return (; model, x)
end

function minimum_block_slack(data, x)
    T = eltype(data.c)
    minimum_slack = nothing
    for block in 1:data.ncell
        a = -data.C[block][1, 1]
        offdiag = -data.C[block][1, 2]
        d = -data.C[block][2, 2]
        for index in 1:data.m
            xi = x[index]
            a += xi * data.A[block][index, 1, 1]
            offdiag += xi * data.A[block][index, 1, 2]
            d += xi * data.A[block][index, 2, 2]
        end
        eigenvalue_min =
            (a + d) / T(2) - hypot((a - d) / T(2), offdiag)
        minimum_slack = isnothing(minimum_slack) ?
            eigenvalue_min : min(minimum_slack, eigenvalue_min)
    end
    return minimum_slack
end

function run_clarabel(data, tolerance)
    local built
    build_seconds = @elapsed built = build_clarabel_model(data, tolerance)
    solve_seconds = @elapsed optimize!(built.model)
    backend = unsafe_backend(built.model)
    info = backend.solver_info
    x = value.(built.x)
    pobj = objective_value(built.model)
    dobj = dual_objective_value(built.model)
    gap_rel = abs(pobj - dobj) /
        max(one(pobj), (abs(pobj) + abs(dobj)) / eltype(data.c)(2))
    return (;
        build_seconds,
        solve_seconds,
        status = string(termination_status(built.model)),
        raw_status = string(raw_status(built.model)),
        iterations = Int(info.iterations),
        pobj,
        dobj,
        gap_rel,
        primal_residual = info.res_primal,
        dual_residual = info.res_dual,
        minimum_slack = minimum_block_slack(data, x),
    )
end

function run_sdpx(data, tolerance, cli)
    local problem
    build_seconds = @elapsed problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse = cli.sparse,
        verbosity = 0,
    )
    options = SDPX.SolverOptions{eltype(data.c)}(
        β = parse(eltype(data.c), cli.beta),
        γ = parse(eltype(data.c), cli.gamma),
        Ωp = parse(eltype(data.c), cli.omega_p),
        Ωd = parse(eltype(data.c), cli.omega_d),
        ϵ_gap = tolerance,
        ϵ_primal = tolerance,
        ϵ_dual = tolerance,
        iter_max = 1000,
        verbosity = 0,
        max_time = 600.0,
        scaling = cli.equilibrate ? :equilibrate : :none,
        predictor = cli.predictor,
        step_rule = cli.step_rule,
        max_restarts = cli.max_restarts,
        refine_steps = cli.refine_steps,
        parameter_policy = cli.parameter_policy,
    )
    selected = cli.parameter_policy === :auto ?
        SDPX.recommended_parameters(problem, options) :
        (; β=options.β, γ=options.γ, profile=:fixed)
    local result
    solve_seconds = @elapsed result = SDPX.solve!(problem, options)
    return (;
        build_seconds,
        solve_seconds,
        status = string(result.status),
        raw_status = result.message,
        iterations = result.iterations,
        pobj = result.pObj,
        dobj = result.dObj,
        gap_rel = result.gap_rel,
        primal_residual = result.p_res,
        dual_residual = result.d_res,
        minimum_slack = minimum_block_slack(data, result.x),
        effective_beta = selected.β,
        effective_gamma = selected.γ,
        parameter_profile = selected.profile,
    )
end

function append_row(path, sha, solver, rep, result, cli)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        new_file && println(
            io,
            "problem_sha256,solver,threads,rep,build_seconds,solve_seconds," *
            "sparse,equilibrate,predictor,step_rule,parameter_policy,parameter_profile," *
            "beta,gamma,omega_p,omega_d," *
            "max_restarts,refine_steps," *
            "status,raw_status,iterations,pobj,dobj,gap_rel," *
            "primal_residual,dual_residual,minimum_slack",
        )
        println(
            io,
            join((
                sha,
                solver,
                Threads.nthreads(),
                rep,
                @sprintf("%.9f", result.build_seconds),
                @sprintf("%.9f", result.solve_seconds),
                solver == :sdpx ? Int(cli.sparse) : "NA",
                solver == :sdpx ? Int(cli.equilibrate) : "NA",
                solver == :sdpx ? cli.predictor : "NA",
                solver == :sdpx ? cli.step_rule : "NA",
                solver == :sdpx ? cli.parameter_policy : "NA",
                solver == :sdpx ? result.parameter_profile : "NA",
                solver == :sdpx ? result.effective_beta : "NA",
                solver == :sdpx ? result.effective_gamma : "NA",
                solver == :sdpx ? cli.omega_p : "NA",
                solver == :sdpx ? cli.omega_d : "NA",
                solver == :sdpx ? cli.max_restarts : "NA",
                solver == :sdpx ? cli.refine_steps : "NA",
                result.status,
                replace(result.raw_status, ',' => ';'),
                result.iterations,
                result.pobj,
                result.dobj,
                result.gap_rel,
                result.primal_residual,
                result.dual_residual,
                result.minimum_slack,
            ), ','),
        )
    end
end

function main(args)
    cli = parse_cli(args)
    cli.reps >= 1 || error("reps must be positive")
    bytes = read(cli.input)
    sha = bytes2hex(sha256(bytes))
    data = deserialize(IOBuffer(bytes))
    eltype(data.c) == Float64x4 ||
        error("benchmark input must use Float64x4")
    tolerance = parse(Float64x4, cli.tol)
    runner = cli.solver == :clarabel ?
        ((data, tolerance) -> run_clarabel(data, tolerance)) :
        ((data, tolerance) -> run_sdpx(data, tolerance, cli))

    println(
        "warmup solver=$(cli.solver) threads=$(Threads.nthreads()) " *
        "sha256=$sha blocks=$(data.ncell) variables=$(data.m)",
    )
    warmup = runner(data, tolerance)
    println(
        "warmup status=$(warmup.status) iterations=$(warmup.iterations) " *
        "solve=$(round(warmup.solve_seconds; digits=3))s",
    )

    for rep in 1:cli.reps
        GC.gc()
        result = runner(data, tolerance)
        append_row(cli.output, sha, cli.solver, rep, result, cli)
        println(
            "rep=$rep status=$(result.status) iterations=$(result.iterations) " *
            "solve=$(round(result.solve_seconds; digits=6))s " *
            "objective=$(Float64(result.pobj)) gap=$(Float64(result.gap_rel))",
        )
    end
end

main(ARGS)
