#!/usr/bin/env julia

# SDPX precision benchmark on one serialized Float64x4 CSDR problem.
# BigFloat runs use the same coefficient values rounded to the requested
# working precision. They measure arithmetic cost, not extra source accuracy.

using MultiFloats: Float64x4
using Printf
using SDPX
using Serialization
using SHA

function parse_cli(args)
    values = Dict{String,String}()
    for argument in args
        startswith(argument, "--") || error("unknown argument: $argument")
        key, value = split(argument[3:end], "=", limit=2)
        values[key] = value
    end
    for required in ("input", "output", "arithmetic")
        haskey(values, required) || error("missing --$required")
    end
    arithmetic = Symbol(values["arithmetic"])
    arithmetic in (:float64x4, :bigfloat) ||
        error("--arithmetic must be float64x4 or bigfloat")
    return (;
        input=abspath(values["input"]),
        output=abspath(values["output"]),
        arithmetic,
        precision_bits=parse(Int, get(values, "precision-bits", "256")),
        reps=parse(Int, get(values, "reps", "3")),
        tolerance=get(values, "tol", "1e-7"),
        beta=get(values, "beta", "0.1"),
        gamma=get(values, "gamma", "0.85"),
        omega_p=get(values, "omega-p", "10"),
        omega_d=get(values, "omega-d", "10"),
        predictor=Symbol(get(values, "predictor", "sdpb")),
        max_restarts=parse(Int, get(values, "max-restarts", "10")),
        refine_steps=parse(Int, get(values, "refine-steps", "0")),
        extended_precision_blas=Symbol(
            get(values, "extended-precision-blas", "off"),
        ),
        skip_warmup=parse(Int, get(values, "skip-warmup", "0")) == 1,
    )
end

function convert_data(data, ::Type{Float64x4}, precision_bits)
    return data
end

function convert_data(data, ::Type{BigFloat}, precision_bits)
    return setprecision(precision_bits) do
        (
            c=BigFloat.(data.c),
            A=[BigFloat.(block) for block in data.A],
            C=[BigFloat.(block) for block in data.C],
            B=BigFloat.(data.B),
            b=BigFloat.(data.b),
            m=data.m,
            ncell=data.ncell,
        )
    end
end

function minimum_block_slack(data, x)
    T = eltype(data.c)
    minimum_slack = nothing
    for block in 1:data.ncell
        a = -data.C[block][1, 1]
        offdiagonal = -data.C[block][1, 2]
        d = -data.C[block][2, 2]
        for index in 1:data.m
            xi = x[index]
            a += xi * data.A[block][index, 1, 1]
            offdiagonal += xi * data.A[block][index, 1, 2]
            d += xi * data.A[block][index, 2, 2]
        end
        eigenvalue_min =
            (a + d) / T(2) - sqrt(((a - d) / T(2))^2 + offdiagonal^2)
        minimum_slack = isnothing(minimum_slack) ?
            eigenvalue_min : min(minimum_slack, eigenvalue_min)
    end
    return minimum_slack
end

function run_sdpx(data, cli)
    T = eltype(data.c)
    local problem
    build_seconds = @elapsed problem = SDPX.ingest(
        data.c,
        data.A,
        data.C,
        data.B,
        data.b;
        sparse=true,
        verbosity=0,
    )
    options = SDPX.SolverOptions{T}(
        β=parse(T, cli.beta),
        γ=parse(T, cli.gamma),
        Ωp=parse(T, cli.omega_p),
        Ωd=parse(T, cli.omega_d),
        ϵ_gap=parse(T, cli.tolerance),
        ϵ_primal=parse(T, cli.tolerance),
        ϵ_dual=parse(T, cli.tolerance),
        iter_max=1000,
        verbosity=0,
        max_time=600.0,
        predictor=cli.predictor,
        max_restarts=cli.max_restarts,
        refine_steps=cli.refine_steps,
        extended_precision_blas=cli.extended_precision_blas,
    )
    local result
    solve_measurement = @timed SDPX.solve!(problem, options)
    result = solve_measurement.value
    return (;
        build_seconds,
        solve_seconds=solve_measurement.time,
        solve_allocated_bytes=solve_measurement.bytes,
        solve_gc_seconds=solve_measurement.gctime,
        result,
        minimum_slack=minimum_block_slack(data, result.x),
    )
end

function append_row(file, sha, cli, run, rep)
    mkpath(dirname(file))
    new_file = !isfile(file) || filesize(file) == 0
    open(file, "a") do io
        new_file && println(
            io,
            "problem_sha256,arithmetic,precision_bits,threads,rep," *
            "build_seconds,solve_seconds,solve_allocated_bytes,solve_gc_seconds," *
            "peak_rss_megabytes,extended_precision_blas,beta,gamma,refine_steps,status," *
            "iterations,pobj,dobj,gap_rel,primal_residual,dual_residual," *
            "minimum_slack",
        )
        result = run.result
        println(
            io,
            join((
                sha,
                cli.arithmetic,
                cli.arithmetic == :bigfloat ? cli.precision_bits :
                    precision(Float64x4),
                Threads.nthreads(),
                rep,
                @sprintf("%.9f", run.build_seconds),
                @sprintf("%.9f", run.solve_seconds),
                run.solve_allocated_bytes,
                @sprintf("%.9f", run.solve_gc_seconds),
                Sys.maxrss() / 1.0e6,
                cli.extended_precision_blas,
                cli.beta,
                cli.gamma,
                cli.refine_steps,
                result.status,
                result.iterations,
                result.pObj,
                result.dObj,
                result.gap_rel,
                result.p_res,
                result.d_res,
                run.minimum_slack,
            ), ','),
        )
    end
end

function main(args)
    cli = parse_cli(args)
    cli.reps > 0 || error("--reps must be positive")
    cli.precision_bits >= 64 || error("--precision-bits must be at least 64")
    cli.extended_precision_blas in (:off, :auto, :on) ||
        error("--extended-precision-blas must be off, auto, or on")
    bytes = read(cli.input)
    sha = bytes2hex(sha256(bytes))
    source = deserialize(IOBuffer(bytes))
    eltype(source.c) == Float64x4 ||
        error("serialized source must use Float64x4")
    T = cli.arithmetic == :bigfloat ? BigFloat : Float64x4
    data = convert_data(source, T, cli.precision_bits)

    if !cli.skip_warmup
        println(
            "warmup arithmetic=$(cli.arithmetic) precision=$(cli.precision_bits) " *
            "threads=$(Threads.nthreads()) sha256=$sha",
        )
        warmup = setprecision(cli.precision_bits) do
            run_sdpx(data, cli)
        end
        println(
            "warmup status=$(warmup.result.status) " *
            "iterations=$(warmup.result.iterations) " *
            "solve=$(round(warmup.solve_seconds; digits=6))s",
        )
    end

    for rep in 1:cli.reps
        GC.gc()
        run = setprecision(cli.precision_bits) do
            run_sdpx(data, cli)
        end
        append_row(cli.output, sha, cli, run, rep)
        println(
            "rep=$rep status=$(run.result.status) " *
            "iterations=$(run.result.iterations) " *
            "solve=$(round(run.solve_seconds; digits=6))s " *
            "gap=$(Float64(run.result.gap_rel))",
        )
    end
end

main(ARGS)
