#!/usr/bin/env julia
using Dates
using JuMP
using SDPX
using LinearAlgebra

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "generators", "SDPXPathologicalBenchmarks.jl"))
using .SDPXPathologicalBenchmarks

function parse_cli(args)
    values = Dict{String,String}()
    for a in args
        startswith(a, "--") || continue
        k, v = split(a[3:end], "="; limit=2)
        values[k] = v
    end
    return values
end

function csv_escape(x)
    s = string(x)
    return occursin(r"[,\n\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function run_one(::Type{T}, case::Symbol, kwargs; tol, threads) where {T}
    model, meta = build_case(case, T; kwargs..., tol=tol, threads=threads)
    GC.gc()
    wall = @elapsed optimize!(model)
    status = termination_status(model)
    objective = try objective_value(model) catch; missing end
    solve_seconds = try solve_time(model) catch; wall end
    expected = get(meta, :expected_objective, nothing)
    abs_error = if objective === missing || expected === nothing
        missing
    else
        abs(objective - expected)
    end
    return (
        timestamp_utc=Dates.now(Dates.UTC),
        arithmetic=string(T),
        precision_bits=precision(T),
        case=case,
        status=status,
        objective=objective,
        expected_objective=expected,
        absolute_objective_error=abs_error,
        solve_seconds=solve_seconds,
        wall_seconds=wall,
        metadata=repr(meta),
    )
end

function default_cases(::Type{T}, tol, threads) where {T}
    rows = NamedTuple[]
    for eps in ("1e-8", "1e-16", "1e-32")
        push!(rows, run_one(T, :lp_near_dependent, (; n=16, epsilon=eps); tol, threads))
        push!(rows, run_one(T, :socp_near_tangent, (; epsilon=eps); tol, threads))
        push!(rows, run_one(T, :sdp_weak_infeasible_2x2, (; delta=eps); tol, threads))
        push!(rows, run_one(T, :sdp_small_eigenvalue, (; n=8, epsilon=eps); tol, threads))
    end
    push!(rows, run_one(T, :sdp_weak_infeasible_2x2, (; delta="0"); tol, threads))
    return rows
end

function write_csv(path, rows)
    mkpath(dirname(path))
    fields = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(fields, ","))
        for row in rows
            println(io, join((csv_escape(getfield(row, f)) for f in fields), ","))
        end
    end
end

function main(args)
    cfg = parse_cli(args)
    arithmetic = lowercase(get(cfg, "precision", "float64"))
    threads = parse(Int, get(cfg, "threads", "1"))
    output = get(cfg, "output", joinpath(ROOT, "data", "generated", "pathological-$arithmetic.csv"))

    if arithmetic == "float64"
        rows = default_cases(Float64, 1e-8, threads)
    elseif arithmetic == "float64x4"
        @eval using MultiFloats: Float64x4
        rows = default_cases(Float64x4, Float64x4(1e-24), threads)
    elseif startswith(arithmetic, "bigfloat")
        bits = parse(Int, replace(arithmetic, "bigfloat" => ""))
        rows = setprecision(BigFloat, bits) do
            default_cases(BigFloat, parse(BigFloat, "1e-40"), 1)
        end
    else
        error("precision must be float64, float64x4, or bigfloat<bits>")
    end
    write_csv(output, rows)
    println("Wrote $output ($(length(rows)) cases)")
end

main(ARGS)
