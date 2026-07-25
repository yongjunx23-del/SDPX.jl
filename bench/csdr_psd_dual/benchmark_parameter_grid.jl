#!/usr/bin/env julia

# Core-solve grid search for SDPX centering and line-search parameters.
# Ingestion and compilation are excluded; one warmup solve precedes the grid.

using MultiFloats: Float64x4
using Printf
using SDPX
using Serialization
using SHA

function parse_cli(args)
    values = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || error("unknown argument: $arg")
        key, value = split(arg[3:end], "=", limit=2)
        values[key] = value
    end
    for required in ("input", "output")
        haskey(values, required) || error("missing --$required")
    end
    split_values(key, default) =
        [strip(value) for value in split(get(values, key, default), ',')]
    return (;
        input=abspath(values["input"]),
        output=abspath(values["output"]),
        betas=split_values("betas", "0.03,0.05,0.1,0.15,0.2,0.3"),
        gammas=split_values("gammas", "0.7,0.75,0.8,0.85,0.9,0.95"),
        reps=parse(Int, get(values, "reps", "1")),
        tol=get(values, "tol", "1e-7"),
        omega_p=get(values, "omega-p", "10"),
        omega_d=get(values, "omega-d", "10"),
        refine_steps=parse(Int, get(values, "refine-steps", "1")),
        step_rule=Symbol(get(values, "step-rule", "backtrack")),
    )
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
        eigenvalue_min = (a + d) / T(2) - hypot((a - d) / T(2), offdiag)
        minimum_slack =
            isnothing(minimum_slack) ? eigenvalue_min : min(minimum_slack, eigenvalue_min)
    end
    return minimum_slack
end

function options(::Type{T}, cli, beta, gamma) where {T}
    tolerance = parse(T, cli.tol)
    return SDPX.SolverOptions{T}(;
        β=parse(T, beta),
        γ=parse(T, gamma),
        Ωp=parse(T, cli.omega_p),
        Ωd=parse(T, cli.omega_d),
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        iter_max=1000,
        verbosity=0,
        max_time=600.0,
        predictor=:sdpb,
        max_restarts=10,
        refine_steps=cli.refine_steps,
        step_rule=cli.step_rule,
    )
end

function append_row(path, sha, data, beta, gamma, rep, seconds, result, slack)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        new_file && println(
            io,
            "problem_sha256,blocks,variables,threads,beta,gamma,rep,solve_seconds," *
            "status,iterations,pobj,dobj,gap_rel,primal_residual,dual_residual,minimum_slack",
        )
        println(io, join((
            sha,
            data.ncell,
            data.m,
            Threads.nthreads(),
            beta,
            gamma,
            rep,
            @sprintf("%.9f", seconds),
            result.status,
            result.iterations,
            result.pObj,
            result.dObj,
            result.gap_rel,
            result.p_res,
            result.d_res,
            slack,
        ), ','))
    end
end

function main(args)
    cli = parse_cli(args)
    cli.reps >= 1 || error("reps must be positive")
    bytes = read(cli.input)
    sha = bytes2hex(sha256(bytes))
    data = deserialize(IOBuffer(bytes))
    eltype(data.c) == Float64x4 || error("input must use Float64x4")
    problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=true,
        verbosity=0,
    )

    warm_options = options(eltype(data.c), cli, first(cli.betas), first(cli.gammas))
    warm = SDPX.solve!(problem, warm_options)
    println(
        "warmup status=$(warm.status) iterations=$(warm.iterations) " *
        "blocks=$(data.ncell) variables=$(data.m) threads=$(Threads.nthreads())",
    )

    for beta in cli.betas, gamma in cli.gammas
        opts = options(eltype(data.c), cli, beta, gamma)
        for rep in 1:cli.reps
            GC.gc()
            local result
            seconds = @elapsed result = SDPX.solve!(problem, opts)
            slack = minimum_block_slack(data, result.x)
            append_row(cli.output, sha, data, beta, gamma, rep, seconds, result, slack)
            println(
                "beta=$beta gamma=$gamma rep=$rep status=$(result.status) " *
                "iterations=$(result.iterations) solve=$(round(seconds; digits=6))s " *
                "gap=$(Float64(result.gap_rel))",
            )
        end
    end
end

main(ARGS)
