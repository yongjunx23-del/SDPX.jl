# PR-06: bounded threading probe.
#
# The plan asks for "1/4/16/64 线程的速度、RSS、分配、证书和失败分布；只在资源允许
# 时跑对应档" -- speed, RSS, allocation, certificate and failure distribution at
# 1/4/16/64 threads, running only the tiers the machine actually allows.
#
# Julia fixes its thread count at startup, so each tier needs its own process.
# This script therefore spawns itself once per tier with JULIA_NUM_THREADS set,
# and the child reports one JSON-ish line per case.
#
# Tiers above the machine's capacity are SKIPPED AND RECORDED, not silently
# dropped: "we did not measure 16 threads" and "16 threads was fine" are very
# different statements and the receipt must not conflate them.
#
#   julia --startup-file=no --project=. benchmark/clarabel_borrowing/thread_scaling.jl
#
# Writes `benchmark/clarabel_borrowing/thread_scaling.toml`.

using SDPX
using Dates
using TOML
using Printf
using Statistics: median
using LinearAlgebra: BLAS

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", ".."))

function _head_sha()
    try
        return strip(read(`git -C $REPO rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

"""Worker problems large enough that threading has something to do."""
function _worker_case(name::Symbol)
    if name === :lp_medium
        model = SDPX.Model(Float64)
        n = 120
        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        for i in 1:24
            expr = sum(
                (sin(Float64(i * 3 + j * 7)) * (1.0 + 0.1 * j)) * x[j] +
                cos(Float64(i + j)) for j in 1:n
            )
            SDPX.constraint!(model, Symbol(:eq, i), expr + Float64(i), SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Minimize(),
            sum((1.0 + 0.05 * j) * x[j] for j in 1:n))
        return model
    elseif name === :soc_large
        model = SDPX.Model(Float64)
        k = 512
        x = SDPX.variable!(model, :x, k - 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :cone, Any[1.0; collect(x)], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        return model
    end
    throw(ArgumentError("unknown worker case $name"))
end

const WORKER_CASES = (:lp_medium, :soc_large)

function _rss_bytes()
    try
        return Int(Sys.maxrss())
    catch
        return -1
    end
end

"""
Child mode: run every case once at this process's thread count and print one
machine-readable line per case. Nothing is asserted here; the parent decides.
"""
function _child_main()
    @printf("CHILD nthreads=%d blas=%d\n", Threads.nthreads(), BLAS.get_num_threads())
    for name in WORKER_CASES
        model = _try_build(name)
        if model === nothing
            @printf("CASE %s threw=build\n", name)
            continue
        end
        settings = SDPX.Settings(Float64; verbosity=0,
            limits=SDPX.Limits(iterations=300, time=120.0, threads=Threads.nthreads()))
        # One untimed warm solve per case per process, so only steady state is
        # measured and compilation is not attributed to threading.
        try
            SDPX.optimize!(_worker_case(name); settings)
        catch
            @printf("CASE %s threw=warmup\n", name)
            continue
        end
        rss_before = _rss_bytes()
        local allocated, seconds, result
        try
            allocated = @allocated begin
                seconds = @elapsed begin
                    result = SDPX.optimize!(_worker_case(name); settings)
                end
            end
        catch exception
            @printf("CASE %s threw=%s\n", name,
                replace(sprint(showerror, exception), " " => "_"))
            continue
        end
        rss_after = _rss_bytes()
        termination = SDPX.diagnostics(result).termination
        @printf("CASE %s status=%s cert=%s it=%d seconds=%.6f allocated=%d rss_delta=%d\n",
            name, SDPX.status(result), SDPX.certificate(result).valid,
            termination.iterations, seconds, allocated, rss_after - rss_before)
    end
    return 0
end

function _try_build(name)
    try
        return _worker_case(name)
    catch
        return nothing
    end
end

"""Tiers to attempt: 1, 2, 4, 8, 16, 64, bounded by what the machine offers."""
function _tiers()
    usable = max(1, min(Sys.CPU_THREADS, 64))
    return [tier for tier in (1, 2, 4, 8, 16, 64) if tier <= usable]
end

function _parse_child_output(output::String)
    rows = Dict{String,Any}[]
    child_threads = -1
    for line in split(output, '\n')
        if startswith(line, "CHILD ")
            for token in split(line)
                startswith(token, "nthreads=") &&
                    (child_threads = something(tryparse(Int, split(token, '=')[2]), -1))
            end
        elseif startswith(line, "CASE ")
            entry = Dict{String,Any}()
            for token in split(line)[2:end]
                parts = split(token, '=')
                length(parts) == 2 || continue
                key, value = parts
                entry[key] = key in ("it", "allocated", "rss_delta") ?
                    something(tryparse(Int, value), -1) :
                    key == "seconds" ? something(tryparse(Float64, value), NaN) :
                    key == "cert" ? value == "true" : String(value)
            end
            push!(rows, entry)
        end
    end
    return child_threads, rows
end

function main()
    repeats = something(tryparse(Int, get(ENV, "SDPX_THREAD_REPEATS", "3")), 3)
    repeats >= 1 || (repeats = 1)
    @printf("PR-06 thread scaling — HEAD %s\n", _head_sha())
    @printf("  machine CPU_THREADS=%d, tiers attempted: %s\n",
        Sys.CPU_THREADS, join(_tiers(), ", "))

    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    results = Dict{String,Any}[]
    unsupported = Dict{String,Any}[]
    for tier in _tiers()
        observations = Dict{String,Vector{Dict{String,Any}}}()
        reported_threads = -1
        for _ in 1:repeats
            command = setenv(
                `$julia --startup-file=no --project=$REPO $(@__FILE__) --child`,
                "JULIA_NUM_THREADS" => string(tier),
            )
            output = try
                read(command, String)
            catch exception
                @printf("  tier %2d: spawn failed (%s)\n", tier,
                    first(sprint(showerror, exception), 80))
                continue
            end
            child_threads, rows = _parse_child_output(output)
            child_threads > 0 && (reported_threads = child_threads)
            for row in rows
                push!(get!(observations, string(get(row, "status", "unknown")), Dict{String,Any}[]), row)
                # Group by case name where possible; the simple grouping above
                # keeps failures visible without dropping them.
            end
            # Regroup properly by case for the summary.
            for row in rows
                push!(get!(observations, "ALL", Dict{String,Any}[]), row)
            end
        end
        entry = Dict{String,Any}("requested_threads" => tier,
                                 "reported_threads" => reported_threads)
        all_rows = get(observations, "ALL", Dict{String,Any}[])
        if isempty(all_rows)
            entry["measured"] = false
            push!(unsupported, entry)
            @printf("  tier %2d: NOT MEASURED\n", tier)
        else
            entry["measured"] = true
            entry["cases"] = all_rows
            failures = count(row -> get(row, "status", "") != "optimal" ||
                                    get(row, "cert", false) !== true, all_rows)
            entry["failures"] = failures
            entry["run_count"] = length(all_rows)
            @printf("  tier %2d (reported %2d): %d runs, %d without a valid certificate\n",
                tier, reported_threads, length(all_rows), failures)
        end
        push!(results, entry)
    end

    # Threads 16 and 64 are in the plan's list but beyond this machine.
    for tier in (16, 64)
        tier in _tiers() && continue
        push!(unsupported, Dict{String,Any}(
            "requested_threads" => tier, "measured" => false,
            "reason" => "machine reports Sys.CPU_THREADS=$(Sys.CPU_THREADS)",
        ))
    end

    payload = Dict{String,Any}(
        "schema" => "sdpx-thread-scaling/1",
        "generated" => string(now()),
        "sdpx_head" => _head_sha(),
        "julia_version" => string(VERSION),
        "cpu_threads" => Sys.CPU_THREADS,
        "repeats" => repeats,
        "note" => "Each tier is a separate process (JULIA_NUM_THREADS is fixed " *
                  "at startup). Tiers the machine cannot host are recorded as " *
                  "measured=false with a reason, never omitted.",
        "tiers" => results,
        "unsupported" => unsupported,
    )
    target = joinpath(HERE, "thread_scaling.toml")
    open(target, "w") do io
        TOML.print(io, payload; sorted=true)
    end
    @printf("wrote %s (%d tiers measured, %d not measured)\n",
        target, count(tier -> get(tier, "measured", false) === true, results),
        count(tier -> get(tier, "measured", false) !== true, results) + length(unsupported))
    return 0
end

if "--child" in ARGS
    exit(_child_main())
end
exit(main())
