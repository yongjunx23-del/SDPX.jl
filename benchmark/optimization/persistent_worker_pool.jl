#!/usr/bin/env julia
# Persistent warmed one-thread process worker pool for independent solves,
# versus the current fresh-process-per-item launcher (Astra rank 1,
# docs/design/PARALLEL_EXECUTION_DESIGN.md section 6).
#
# Workload: the existing generic LP target `lp_random_large`
# (GenericConicBenchmark, kind=:planted, base seed=0x004c5003, m=400,
# n=1200). A queue of N independent items is built by varying ONLY the
# planted seed deterministically:
#
#     seed_i = 0x004c5003 + UInt32(i)   for i in 0..N-1
#
# The model structure is identical for every item (n=1200 box LP, same
# constraint pattern); only the objective coefficients and RHS bounds
# differ, and each item carries its own independently recomputed known
# objective via the same `_planted_lp_objective` formula. No solver
# settings or tolerances are changed: every solve goes through
# `GenericConicBenchmark.run_one(spec, Float64; threads=1)`, which uses
# `certification=true`.
#
# Each item is REBUILT per solve (a fresh `Model` per item, in both modes).
# Rationale: rebuild cost is milliseconds against multi-second solves, and
# a fresh model per item matches fresh-process semantics exactly, so the
# comparison isolates startup/JIT amortization and any residual state
# leakage is attributable to process reuse rather than session reuse.
#
# Modes (selected by flags):
#   --solve-one --item-index=I --result-dir=R
#       Fresh-process child: load SDPX once, solve exactly one item, write
#       one atomic TOML receipt, exit. Fails closed (exit 1) on any
#       non-optimal/invalid result.
#   --worker --worker-id=K --n-items=N --queue-dir=Q --result-dir=R
#       Persistent child: load SDPX once, run one excluded warmup solve,
#       then claim items from the shared queue directory (atomic mkdir
#       locks) until every item has a result receipt. Writes one receipt
#       per item plus a worker summary (peak/retained RSS, failures).
#       Fails closed (exit 1) if any claimed item is non-optimal/invalid.
#   --mode=fresh --items=N --workers=W --outdir=D
#       Parent: run the SAME item set with one fresh process per item, at
#       most W concurrent (same reserved cores as the pool).
#   --mode=persistent --items=N --workers=W --outdir=D
#       Parent: spawn W persistent workers against one shared queue.
#   --mode=compare --items=N --workers=W --outdir=D --reps=R
#       Parent: interleave fresh/persistent whole-batch runs R times each
#       (fresh, persistent, fresh, persistent, ...) and write a comparison
#       summary with the >=2% certified-throughput gate verdict inputs.
#
# Execution management only: this file contains no solver numerics.
module PersistentWorkerPool

using TOML
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SCRIPT = abspath(@__FILE__)

if !isdefined(Main, :GenericConicBenchmark)
    Base.include(Main, joinpath(ROOT, "benchmark", "general", "GenericConicBenchmark.jl"))
end
const G = Main.GenericConicBenchmark

# --- Workload identity -------------------------------------------------------
const BASE_SEED = UInt32(0x004c5003)
const WORKLOAD_M = 400
const WORKLOAD_N = 1200
const WARMUP_SALT = UInt32(0x00FFFFFF)
const PROTOCOL_VERSION = 1

# --- Small utilities ----------------------------------------------------------
function _arg(name::String, default=nothing)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function _source_porcelain()::String
    return readchomp(`git -C $ROOT status --porcelain`)
end

"""Fail closed on any *tracked* modification (solver/source drift would
invalidate the comparison). Untracked files (e.g. this not-yet-committed
harness) are tolerated but recorded verbatim in batch summaries for audit."""
function _require_clean_source(stage::AbstractString)
    tracked = readchomp(`git -C $ROOT diff --name-only`) *
              readchomp(`git -C $ROOT diff --cached --name-only`)
    isempty(strip(tracked)) || throw(ArgumentError(
        "tracked source modifications at $stage: $tracked"))
    return true
end

function _atomic_toml(path::AbstractString, value::Dict{String,Any})
    mkpath(dirname(abspath(path)))
    temporary, io = mktemp(dirname(abspath(path)); cleanup=false)
    try
        TOML.print(io, value; sorted=true)
        flush(io)
        close(io)
        mv(temporary, abspath(path); force=false)
    catch
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return abspath(path)
end

"""Current-process RSS in bytes. Prefers /proc (Linux), falls back to ps."""
function current_rss_bytes()::Int
    try
        if isfile("/proc/self/status")
            for line in eachline("/proc/self/status")
                if startswith(line, "VmRSS:")
                    parts = split(strip(line))
                    return parse(Int, parts[2]) * 1024
                end
            end
        end
    catch
    end
    try
        out = readchomp(`ps -o rss= -p $(Libc.getpid())`)
        return parse(Int, strip(out)) * 1024
    catch
    end
    return 0
end

# --- Item definition ------------------------------------------------------------
function item_seed(index::Integer)::UInt32
    0 <= index || throw(ArgumentError("item index must be non-negative"))
    return BASE_SEED + UInt32(index)
end

function item_spec(index::Integer)
    base_specs = Base.invokelatest(getproperty(G, :inventory); tier=:large, family=:lp)
    matches = filter(s -> s.id === :lp_random_large, base_specs)
    length(matches) == 1 || throw(ArgumentError("lp_random_large is not unique"))
    base = only(matches)
    seed = item_seed(index)
    params = merge(base.params, (seed=seed,))
    known = Base.invokelatest(getproperty(G, :_planted_lp_objective), seed, WORKLOAD_M, WORKLOAD_N)
    return G.BenchmarkSpec(Symbol("lp_random_large_item$(index)"), :lp, :large,
        base.problem, params, :optimal, known, base.objective_tolerance, base.source)
end

# --- Core solve --------------------------------------------------------------------
"""Build and solve one item; returns a TOML-serializable receipt Dict."""
function solve_item(index::Integer; label::AbstractString="item")
    spec = item_spec(index)
    wall_start = time_ns()
    result = Base.invokelatest(getproperty(G, :run_one), spec, Float64; threads=1)
    wall_seconds = (time_ns() - wall_start) * 1e-9
    return Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "label" => String(label),
        "item_index" => Int(index),
        "seed" => Int(item_seed(index)),
        "workload" => "lp_random_large",
        "model_structure" => "planted_box_lp_m$(WORKLOAD_M)_n$(WORKLOAD_N)_rebuilt_per_item",
        "status" => String(result.status),
        "objective" => Float64(result.objective),
        "dual_objective" => Float64(result.dual_objective),
        "primal_residual" => Float64(result.primal_residual),
        "dual_residual" => Float64(result.dual_residual),
        "relative_gap" => Float64(result.relative_gap),
        "certificate_valid" => result.certificate_valid,
        "expectation_met" => result.expectation_met,
        "known_objective" => Float64(spec.known_objective),
        "objective_tolerance" => Float64(spec.objective_tolerance),
        "iterations" => Int(result.iterations),
        "solve_seconds" => Float64(result.seconds),
        "item_wall_seconds" => Float64(wall_seconds),
        "pid" => Libc.getpid(),
        "maxrss_bytes" => Int(Sys.maxrss()),
        "rss_bytes" => current_rss_bytes(),
        "certified" => (result.status === :optimal && result.certificate_valid &&
                        result.expectation_met),
    )
end

function result_path(result_dir::AbstractString, index::Integer)
    return joinpath(result_dir, "result_$(lpad(index, 4, '0')).toml")
end

# --- Fresh-process child ---------------------------------------------------------------
function solve_one_main()
    result_dir = _arg("result-dir")
    index_text = _arg("item-index")
    result_dir === nothing && throw(ArgumentError("--result-dir=DIR is required"))
    index_text === nothing && throw(ArgumentError("--item-index=I is required"))
    _require_clean_source("solve_one_start")
    index = parse(Int, index_text)
    receipt = solve_item(index)
    _atomic_toml(result_path(result_dir, index), receipt)
    _require_clean_source("solve_one_after_solve")
    println("SOLVE_ONE_WRITTEN item=$index pid=$(receipt["pid"]) status=$(receipt["status"]) " *
            "cert=$(receipt["certificate_valid"]) obj=$(receipt["objective"]) " *
            "solve_s=$(receipt["solve_seconds"]) wall_s=$(receipt["item_wall_seconds"])")
    if !receipt["certified"]
        println("SOLVE_ONE_FAILURE item=$index")
        exit(1)
    end
end

# --- Persistent worker child --------------------------------------------------------------
function _try_claim(queue_dir::AbstractString, index::Integer, worker_id::Integer)::Bool
    lockpath = joinpath(queue_dir, "item_$(lpad(index, 4, '0')).lock")
    try
        mkdir(lockpath)
        open(joinpath(lockpath, "owner"), "w") do io
            println(io, "worker=$worker_id pid=$(Libc.getpid())")
        end
        return true
    catch
        return false
    end
end

function _release_claim(queue_dir::AbstractString, index::Integer)
    rm(joinpath(queue_dir, "item_$(lpad(index, 4, '0')).lock"); recursive=true, force=true)
end

function worker_main()
    worker_id = parse(Int, _arg("worker-id", "0"))
    n_items = parse(Int, _arg("n-items", "1"))
    queue_dir = _arg("queue-dir")
    result_dir = _arg("result-dir")
    (queue_dir === nothing || result_dir === nothing) &&
        throw(ArgumentError("--queue-dir and --result-dir are required"))
    mkpath(queue_dir)
    _require_clean_source("worker$(worker_id)_start")
    # Excluded warmup: same structure, out-of-range salt seed, discarded.
    warmup_wall = time_ns()
    warmup_spec_seed = BASE_SEED + WARMUP_SALT + UInt32(worker_id)
    base_specs = Base.invokelatest(getproperty(G, :inventory); tier=:large, family=:lp)
    base = only(filter(s -> s.id === :lp_random_large, base_specs))
    warmup_params = merge(base.params, (seed=warmup_spec_seed,))
    warmup_known = Base.invokelatest(getproperty(G, :_planted_lp_objective),
        warmup_spec_seed, WORKLOAD_M, WORKLOAD_N)
    warmup_spec = G.BenchmarkSpec(Symbol("lp_random_large_warmup$(worker_id)"), :lp,
        :large, base.problem, warmup_params, :optimal, warmup_known,
        base.objective_tolerance, base.source)
    warmup_result = Base.invokelatest(getproperty(G, :run_one), warmup_spec, Float64; threads=1)
    warmup_seconds = (time_ns() - warmup_wall) * 1e-9
    warmup_ok = warmup_result.status === :optimal && warmup_result.certificate_valid
    _atomic_toml(joinpath(result_dir, "warmup_$(worker_id).toml"), Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "worker_id" => worker_id,
        "pid" => Libc.getpid(),
        "seed" => Int(warmup_spec_seed),
        "status" => String(warmup_result.status),
        "certificate_valid" => warmup_result.certificate_valid,
        "objective" => Float64(warmup_result.objective),
        "iterations" => Int(warmup_result.iterations),
        "warmup_seconds" => Float64(warmup_seconds),
        "solve_seconds" => Float64(warmup_result.seconds),
        "maxrss_bytes" => Int(Sys.maxrss()),
        "excluded_from_throughput" => true,
    ))
    println("WORKER $worker_id WARMUP_DONE ok=$warmup_ok solve_s=$(warmup_result.seconds) " *
            "wall_s=$(round(warmup_seconds; digits=2))")
    if !warmup_ok
        println("WORKER $worker_id WARMUP_FAILURE")
        exit(1)
    end
    claimed = Int[]
    failures = Int[]
    peak_rss = Int(Sys.maxrss())
    deadline = time() + parse(Float64, get(ENV, "SDPX_POOL_ITEM_TIMEOUT", "1700"))
    while time() < deadline
        remaining = [i for i in 0:(n_items-1)
                     if !isfile(result_path(result_dir, i))]
        isempty(remaining) && break
        progress = false
        for i in remaining
            isfile(result_path(result_dir, i)) && continue
            _try_claim(queue_dir, i, worker_id) || continue
            try
                receipt = solve_item(i)
                receipt["worker_id"] = worker_id
                _atomic_toml(result_path(result_dir, i), receipt)
                push!(claimed, i)
                receipt["certified"] || push!(failures, i)
                peak_rss = max(peak_rss, Int(Sys.maxrss()))
                progress = true
                println("WORKER $worker_id ITEM $i status=$(receipt["status"]) " *
                        "cert=$(receipt["certificate_valid"]) obj=$(receipt["objective"]) " *
                        "solve_s=$(round(receipt["solve_seconds"]; digits=2))")
            finally
                _release_claim(queue_dir, i)
            end
        end
        # If no item could be claimed and results are still missing, another
        # live worker owns them; back off briefly instead of spinning.
        progress || sleep(1.0)
        # Re-check: all results present?
        all(isfile(result_path(result_dir, i)) for i in 0:(n_items-1)) && break
    end
    missing = [i for i in 0:(n_items-1) if !isfile(result_path(result_dir, i))]
    summary = Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "worker_id" => worker_id,
        "pid" => Libc.getpid(),
        "claimed_items" => claimed,
        "failed_items" => failures,
        "missing_items" => missing,
        "warmup_seconds" => Float64(warmup_seconds),
        "peak_rss_bytes" => peak_rss,
        "retained_maxrss_bytes" => Int(Sys.maxrss()),
        "retained_rss_bytes" => current_rss_bytes(),
    )
    _atomic_toml(joinpath(result_dir, "worker_$(worker_id)_summary.toml"), summary)
    _require_clean_source("worker$(worker_id)_end")
    println("WORKER $worker_id DONE claimed=$(length(claimed)) failures=$(length(failures)) " *
            "missing=$(length(missing)) peak_rss=$peak_rss retained_rss=$(summary["retained_rss_bytes"])")
    if !isempty(failures) || !isempty(missing)
        exit(1)
    end
end

# --- Parent launch helpers ------------------------------------------------------------------
function _child_env()
    return Dict(
        "JULIA_NUM_THREADS" => "1",
        "JULIA_NUM_GC_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "OMP_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
        "JULIA_PKG_PRECOMPILE_AUTO" => "0",
        "JULIA_PKG_OFFLINE" => "true",
    )
end

function _child_cmd(extra_args::Vector{String})
    julia = Base.julia_cmd()
    project = Base.active_project()
    project === nothing && throw(ArgumentError("parent must run under an active project"))
    return `$julia --startup-file=no --threads=1 --gcthreads=1 --heap-size-hint=2G --project=$project $SCRIPT $extra_args`
end

function _run_throttled(labeled::Vector{Tuple{String,Cmd}}, max_concurrent::Integer;
                        deadline_seconds::Real=1700)
    env = _child_env()
    failures = String[]
    # Launch lazily, keeping at most max_concurrent live.
    pending = copy(labeled)
    live = Tuple{Base.Process,String}[]
    deadline = time() + deadline_seconds
    while !isempty(pending) || !isempty(live)
        time() > deadline && throw(ArgumentError("deadline exceeded waiting for child processes"))
        while !isempty(pending) && length(live) < max_concurrent
            (label, cmd) = popfirst!(pending)
            push!(live, (run(setenv(cmd, env...); wait=false), label))
        end
        # Poll for any completion.
        progressed = false
        for (k, (proc, label)) in enumerate(live)
            if !process_running(proc)
                wait(proc)
                proc.exitcode != 0 && push!(failures, label)
                deleteat!(live, k)
                progressed = true
                break
            end
        end
        progressed || sleep(1.0)
    end
    return failures
end

function _labeled(cmd::Cmd, label::String)
    return (label, cmd)
end

function run_fresh_batch(n_items::Integer, n_workers::Integer, batch_dir::AbstractString)
    result_dir = joinpath(batch_dir, "results")
    mkpath(result_dir)
    labeled = [_labeled(_child_cmd(["--solve-one", "--item-index=$(i)",
                                    "--result-dir=$(result_dir)"]),
                       "fresh_item_$i") for i in 0:(n_items-1)]
    wall_start = time_ns()
    # Child exit codes flag launch/crash failures; per-item certification
    # failures are additionally re-derived from receipts (fail-closed).
    failures = _run_throttled(labeled, n_workers)
    wall_seconds = (time_ns() - wall_start) * 1e-9
    return (; mode="fresh", batch_dir=String(batch_dir), result_dir,
            wall_seconds=Float64(wall_seconds), child_failures=failures)
end

function run_persistent_batch(n_items::Integer, n_workers::Integer, batch_dir::AbstractString)
    result_dir = joinpath(batch_dir, "results")
    queue_dir = joinpath(batch_dir, "queue")
    mkpath(result_dir)
    mkpath(queue_dir)
    labeled = [_labeled(_child_cmd(["--worker", "--worker-id=$(k)", "--n-items=$(n_items)",
                            "--queue-dir=$(queue_dir)", "--result-dir=$(result_dir)"]),
                       "persistent_worker_$k")
                for k in 0:(n_workers-1)]
    wall_start = time_ns()
    failures = _run_throttled(labeled, n_workers)
    wall_seconds = (time_ns() - wall_start) * 1e-9
    return (; mode="persistent", batch_dir=String(batch_dir), result_dir,
            wall_seconds=Float64(wall_seconds), child_failures=failures)
end

# --- Aggregation ------------------------------------------------------------------------------
function aggregate_batch(mode::AbstractString, batch_dir::AbstractString,
                         result_dir::AbstractString, wall_seconds::Float64,
                         child_failures, n_items::Integer)
    receipts = Dict{Int,Dict{String,Any}}()
    for i in 0:(n_items-1)
        path = result_path(result_dir, i)
        isfile(path) || continue
        receipts[i] = TOML.parsefile(path)
    end
    certified = sort!([i for (i, r) in receipts
                       if get(r, "certified", false) === true])
    failed = sort!([i for (i, r) in receipts
                    if get(r, "certified", false) !== true])
    missing = [i for i in 0:(n_items-1) if !haskey(receipts, i)]
    solve_times = [Float64(receipts[i]["solve_seconds"]) for i in certified]
    item_walls = [Float64(receipts[i]["item_wall_seconds"]) for i in certified]
    throughput = isempty(certified) ? 0.0 : length(certified) / (wall_seconds / 3600)
    warmups = Dict{Int,Dict{String,Any}}()
    for name in readdir(result_dir)
        m = match(r"^warmup_(\d+)\.toml$", name)
        m === nothing && continue
        warmups[parse(Int, m.captures[1])] = TOML.parsefile(joinpath(result_dir, name))
    end
    worker_summaries = Dict{Int,Dict{String,Any}}()
    for name in readdir(result_dir)
        m = match(r"^worker_(\d+)_summary\.toml$", name)
        m === nothing && continue
        worker_summaries[parse(Int, m.captures[1])] = TOML.parsefile(joinpath(result_dir, name))
    end
    peak_rss = isempty(receipts) ? 0 :
        maximum(Int(r["maxrss_bytes"]) for r in values(receipts))
    retained_rss = isempty(worker_summaries) ? 0 :
        maximum(Int(s["retained_rss_bytes"]) for s in values(worker_summaries))
    summary = Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "mode" => String(mode),
        "batch_dir" => String(batch_dir),
        "n_items" => Int(n_items),
        "n_received" => length(receipts),
        "n_certified" => length(certified),
        "n_failed" => length(failed),
        "failed_items" => failed,
        "missing_items" => missing,
        "child_process_failures" => child_failures,
        "wall_seconds" => Float64(wall_seconds),
        "certified_solves_per_hour" => Float64(throughput),
        "median_solve_seconds" => isempty(solve_times) ? 0.0 : Float64(median(solve_times)),
        "median_item_wall_seconds" => isempty(item_walls) ? 0.0 : Float64(median(item_walls)),
        "solve_seconds" => solve_times,
        "peak_rss_bytes" => peak_rss,
        "retained_rss_bytes" => retained_rss,
        "objectives" => Dict(string(i) => string(receipts[i]["objective"]) for i in keys(receipts)),
        "iterations" => Dict(string(i) => Int(receipts[i]["iterations"]) for i in keys(receipts)),
        "warmup_seconds" => Dict(string(k) => Float64(v["warmup_seconds"]) for (k, v) in warmups),
        "source_commit" => readchomp(`git -C $ROOT rev-parse HEAD`),
        "source_porcelain" => _source_porcelain(),
    )
    _atomic_toml(joinpath(batch_dir, "batch_summary.toml"), summary)
    return summary
end

function parent_single(mode::AbstractString, n_items::Integer, n_workers::Integer,
                       outdir::AbstractString)
    _require_clean_source("parent_start")
    batch_dir = joinpath(abspath(outdir), "batch_$(mode)_$(n_items)items_$(n_workers)w")
    ispath(batch_dir) && rm(batch_dir; recursive=true, force=true)
    mkpath(batch_dir)
    batch = mode == "fresh" ? run_fresh_batch(n_items, n_workers, batch_dir) :
        mode == "persistent" ? run_persistent_batch(n_items, n_workers, batch_dir) :
        throw(ArgumentError("unknown mode $mode"))
    summary = aggregate_batch(batch.mode, batch.batch_dir, batch.result_dir,
        batch.wall_seconds, batch.child_failures, n_items)
    _require_clean_source("parent_end")
    @printf("%-10s certified=%d/%d failed=%d missing=%d wall=%.1fs throughput=%.2f solves/h median_solve=%.2fs peak_rss=%.2fGiB\n",
        mode, summary["n_certified"], summary["n_items"], summary["n_failed"],
        length(summary["missing_items"]), summary["wall_seconds"],
        summary["certified_solves_per_hour"], summary["median_solve_seconds"],
        summary["peak_rss_bytes"] / 2.0^30)
    return summary
end

function parent_compare(n_items::Integer, n_workers::Integer, outdir::AbstractString, reps::Integer)
    _require_clean_source("compare_start")
    root = abspath(outdir)
    mkpath(root)
    order = String[]
    for rep in 1:reps
        push!(order, "fresh", "persistent")
    end
    summaries = Dict{String,Any}[]
    for (seq, mode) in enumerate(order)
        rep = div(seq - 1, 2) + 1
        batch_dir = joinpath(root, "rep$(rep)_$(mode)")
        ispath(batch_dir) && rm(batch_dir; recursive=true, force=true)
        mkpath(batch_dir)
        println("COMPARE [$seq/$(length(order))] mode=$mode rep=$rep")
        batch = mode == "fresh" ? run_fresh_batch(n_items, n_workers, batch_dir) :
            run_persistent_batch(n_items, n_workers, batch_dir)
        summary = aggregate_batch(batch.mode, batch.batch_dir, batch.result_dir,
            batch.wall_seconds, batch.child_failures, n_items)
        summary["rep"] = rep
        summary["sequence"] = seq
        # aggregate_batch already persisted batch_summary.toml in batch_dir;
        # rep/sequence live in memory and in comparison_summary.toml.
        push!(summaries, summary)
        @printf("  -> certified=%d/%d wall=%.1fs throughput=%.2f solves/h median_solve=%.2fs\n",
            summary["n_certified"], summary["n_items"], summary["wall_seconds"],
            summary["certified_solves_per_hour"], summary["median_solve_seconds"])
    end
    # State-isolation cross-check: per-item objective strings must agree
    # across ALL batches (fresh and persistent solve identical items).
    by_item = Dict{Int,Set{String}}()
    by_iter = Dict{Int,Set{Int}}()
    for summary in summaries, (item_text, obj_text) in summary["objectives"]
        item = parse(Int, item_text)
        push!(get!(by_item, item, Set{String}()), String(obj_text))
        push!(get!(by_iter, item, Set{Int}()), Int(summary["iterations"][item_text]))
    end
    isolation_ok = all(length(v) == 1 for v in values(by_item)) &&
                   all(length(v) == 1 for v in values(by_iter))
    fresh_tp = [s["certified_solves_per_hour"] for s in summaries if s["mode"] == "fresh"]
    persist_tp = [s["certified_solves_per_hour"] for s in summaries if s["mode"] == "persistent"]
    gate_ratio = median(persist_tp) / median(fresh_tp)
    comparison = Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "items" => Int(n_items),
        "workers" => Int(n_workers),
        "reps" => Int(reps),
        "interleaved_order" => order,
        "fresh_throughput" => fresh_tp,
        "persistent_throughput" => persist_tp,
        "median_fresh_throughput" => Float64(median(fresh_tp)),
        "median_persistent_throughput" => Float64(median(persist_tp)),
        "gate_ratio_persistent_over_fresh" => Float64(gate_ratio),
        "gate_threshold" => 1.02,
        "gate_pass" => Bool(gate_ratio >= 1.02),
        "state_isolation_ok" => isolation_ok,
        "source_commit" => readchomp(`git -C $ROOT rev-parse HEAD`),
    )
    _atomic_toml(joinpath(root, "comparison_summary.toml"), comparison)
    _require_clean_source("compare_end")
    println("COMPARE gate_ratio=$(round(gate_ratio; digits=4)) threshold=1.02 " *
            "pass=$(comparison["gate_pass"]) isolation_ok=$isolation_ok")
    return comparison
end

function main()
    if "--solve-one" in ARGS
        solve_one_main()
    elseif "--worker" in ARGS
        worker_main()
    elseif "--mode" in ARGS || any(a -> startswith(a, "--mode="), ARGS)
        mode = _arg("mode", "compare")
        n_items = parse(Int, _arg("items", "8"))
        n_workers = parse(Int, _arg("workers", "2"))
        outdir = _arg("outdir", joinpath(tempdir(), "sdpx_persistent_pool"))
        reps = parse(Int, _arg("reps", "2"))
        if mode == "compare"
            parent_compare(n_items, n_workers, outdir, reps)
        else
            parent_single(mode, n_items, n_workers, outdir)
        end
    else
        throw(ArgumentError("specify --solve-one, --worker, or --mode=fresh|persistent|compare"))
    end
end

end # module PersistentWorkerPool

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    PersistentWorkerPool.main()
end
