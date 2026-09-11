#!/usr/bin/env julia
# Solver-free contract tests for the persistent worker pool harness.
#
# STANDALONE ONLY. Do NOT include this file in test/runtests.jl: it defines
# a stub Main.GenericConicBenchmark (fast synthetic solves, no SDPX solver
# invocation) before including the production harness, while the repo
# regression suite loads the real benchmark module. The real SDPX package
# IS loaded (for genuine loaded-source identity answers); no solve is run.
#
# What runs here is production code in
# benchmark/optimization/persistent_worker_pool.jl, reached through two
# test-only seams that never touch production files:
#   1. A stub GenericConicBenchmark (inventory / _planted_lp_objective /
#      BenchmarkSpec / run_one) supplying deterministic synthetic solves.
#   2. In-test rebinding of the batch LAUNCHERS (run_fresh_batch /
#      run_persistent_batch) to synthetic receipt writers for
#      comparison-gate tests. Validation, aggregation, isolation, memory
#      and throughput gating, publication, freshness and CLI dispatch all
#      remain production code. Lifecycle tests use the real _run_throttled
#      with synthetic shell commands; worker tests use the real worker_main.
#
# Groups (run each as its own process under an external <=170 s watchdog
# with kill/reap; the file supports --group=<name> and --probe=<name>):
#   lifecycle   timeout / nonzero exit / signal death / spawn error through
#               the real _run_throttled; owned children reaped, pending work
#               never launched.
#   exclusivity existing output roots refused without touching sentinel
#               evidence (parent_single and parent_compare).
#   race        concurrent receipt publication during the real claim loop:
#               no duplicate solve, publisher bytes preserved, locks released.
#   receipts    table-driven receipt mutations incl. protocol version and
#               negative iterations through production valid_receipt.
#   workers     exact worker-ID sets, foreign identities, warmup
#               objective/expectation, claim coverage through production
#               valid_worker_artifacts.
#   memory      per-worker RSS validity (one unavailable sample among valid
#               samples fails) and zero/Inf batch walls through production
#               aggregate_batch.
#   gate        comparison-gate rejection (missing/failed artifacts, child
#               failure, fingerprint mismatch, bad/over-limit RSS,
#               zero/Inf duration) plus counterbalanced ordering, through
#               production parent_compare.
#   cli         CLI exit codes in guarded subprocesses (valid compare -> 0,
#               invalid batch -> 1, bad mode / non-positive items -> nonzero).
#
# No throughput, memory-bound, or qualification claims are made here.

using Test
using TOML
using Printf
using Statistics
using SHA

const POOL_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const POOL_SCRIPT = joinpath(POOL_ROOT, "benchmark", "optimization",
    "persistent_worker_pool.jl")

# Real SDPX: genuine pkgdir/source answers, zero solves.
using SDPX

# --- Stub solver-facing benchmark (test-harness mock) --------------------------
const STUB_CALLS = Any[]
const STUB_SLEEP = Dict{Int,Float64}()

module GenericConicBenchmark
import Main: SDPX
const BASE_SEED = UInt32(0x004c5003)
struct BenchmarkSpec
    id::Symbol
    family::Symbol
    tier::Symbol
    problem::Any
    params::Any
    expected_status::Symbol
    known_objective::Float64
    objective_tolerance::Float64
    source::String
end
function _planted_lp_objective(seed, m, n)
    Float64(800) + Float64(mod(Int(seed), 256)) +
        Float64(mod(Int(seed), 997)) / 1000
end
function inventory(; tier=nothing, family=nothing)
    params = (kind=:planted, name=:random_large, seed=BASE_SEED, m=400,
              n=1200)
    return [BenchmarkSpec(:lp_random_large, :lp, :large, :stub_problem,
        params, :optimal, _planted_lp_objective(BASE_SEED, 400, 1200),
        2e-5, "stub-harness")]
end
function run_one(spec, ::Type{T}=Float64; time_limit::Real=Inf,
                 threads::Integer=1) where {T<:AbstractFloat}
    push!(Main.STUB_CALLS, (spec=String(spec.id), t=time()))
    matched = match(r"item(\d+)$", String(spec.id))
    index = matched === nothing ? -1 : parse(Int, matched.captures[1])
    pause = get(Main.STUB_SLEEP, index, 0.0)
    pause > 0 && sleep(pause)
    return (; status=:optimal, certificate_valid=true,
            expectation_met=true, objective=spec.known_objective,
            dual_objective=spec.known_objective, primal_residual=1e-12,
            dual_residual=1e-12, relative_gap=1e-13, iterations=10,
            seconds=0.01)
end
end # module GenericConicBenchmark

# Pin harness source identity before it loads.
ENV["SDPX_POOL_SOURCE_COMMIT"] = readchomp(`git -C $POOL_ROOT rev-parse HEAD`)
ENV["SDPX_POOL_SCRIPT_SHA"] = bytes2hex(SHA.sha256(read(POOL_SCRIPT)))

include(POOL_SCRIPT)
using .PersistentWorkerPool
const P = PersistentWorkerPool
const STUB = GenericConicBenchmark

# --- Small test utilities -------------------------------------------------------
function with_args(f::Function, args::Vector{String})
    saved = copy(ARGS)
    try
        empty!(ARGS)
        append!(ARGS, args)
        return f()
    finally
        empty!(ARGS)
        append!(ARGS, saved)
    end
end

scratchdir(name::AbstractString) = mktempdir(tempdir(); prefix=name * "_")

function pids_with_marker(marker::AbstractString)::Vector{Int}
    try
        out = readchomp(pipeline(`pgrep -f $marker`; stderr=devnull))
        isempty(strip(out)) ? Int[] : parse.(Int, split(strip(out)))
    catch
        Int[]
    end
end

"""Run cmd with a hard deadline; SIGKILL + reap on expiry. Returns status."""
function run_guarded(cmd::Cmd, deadline_s::Real; logpath::AbstractString=tempname() * ".log")
    proc = run(pipeline(cmd; stdout=logpath, stderr=logpath); wait=false)
    started = time()
    timed_out = false
    while process_running(proc)
        if time() - started > deadline_s
            timed_out = true
            try
                kill(proc, Base.SIGKILL)
            catch
            end
            break
        end
        sleep(0.05)
    end
    wait(proc)
    return (; timed_out, exitcode=proc.exitcode,
            termsignal=proc.termsignal, logpath)
end

# --- Test-only batch-launcher substitution --------------------------------------
const SCENARIO = Ref{Symbol}(:all_valid)

# Test-process-only replacement of the two exact method signatures. Never
# rebind a function constant or delete unrelated methods. Calls through the
# latest world below see these substitutions; every group uses a fresh process.
const LAUNCHERS_INSTALLED = Ref(false)
function use_stub_launchers!()
    LAUNCHERS_INSTALLED[] && return
    Core.eval(P, :(run_fresh_batch(n_items::Integer, n_workers::Integer,
                                 batch_dir::AbstractString) =
        Main.__stub_fresh_batch(n_items, n_workers, batch_dir)))
    Core.eval(P, :(run_persistent_batch(n_items::Integer, n_workers::Integer,
                                      batch_dir::AbstractString) =
        Main.__stub_persistent_batch(n_items, n_workers, batch_dir)))
    LAUNCHERS_INSTALLED[] = true
    return
end

function restore_launchers!()
    return nothing
end

const SMALL_RSS = Dict("maxrss" => 1_000_000, "rss" => 500_000,
    "peak" => 1_100_000, "post" => 800_000, "warmmax" => 900_000)

"""Production-generated receipts, small deterministic RSS, scenario faults."""
function __write_scenario_batches!(result_dir::AbstractString,
                                   mode::AbstractString,
                                   n_items::Integer, n_workers::Integer)
    scenario = SCENARIO[]
    wall = mode == "fresh" ? 12.0 : 4.0
    failures = String[]
    skip_last = scenario === :missing
    for i in 0:(n_items - 1)
        (skip_last && i == n_items - 1) && continue
        receipt = P.solve_item(i)
        receipt["maxrss_bytes"] = SMALL_RSS["maxrss"] + i
        receipt["rss_bytes"] = SMALL_RSS["rss"] + i
        if scenario === :failed_receipt && i == 0
            receipt["status"] = "primal_infeasible"
            receipt["certificate_valid"] = false
        end
        if scenario === :fingerprint_mismatch && mode == "persistent" && i == 0
            receipt["objective"] = Float64(receipt["objective"]) + 1.0
        end
        if scenario === :over_limit
            receipt["maxrss_bytes"] = 2^40
            receipt["rss_bytes"] = 2^40
        end
        if mode == "persistent"
            receipt["worker_id"] = mod(i, n_workers)
            receipt["pid"] = 1000 + mod(i, n_workers)
        end
        P._atomic_toml(P.result_path(result_dir, i), receipt)
    end
    if mode == "persistent"
        for k in 0:(n_workers - 1)
            seed = P.BASE_SEED + P.WARMUP_SALT + UInt32(k)
            known = STUB._planted_lp_objective(seed, P.WORKLOAD_M,
                P.WORKLOAD_N)
            post = SMALL_RSS["post"] + k
            (scenario === :bad_rss && k == 0) && (post = 0)
            scenario === :over_limit && (post = 2^40)
            P._atomic_toml(joinpath(result_dir, "warmup_$(k).toml"),
                Dict{String,Any}(
                    "protocol_version" => P.PROTOCOL_VERSION,
                    "worker_id" => k,
                    "run_id" => P.RUN_ID,
                    "source_commit" => P.SOURCE_COMMIT,
                    "script_sha256" => P.SCRIPT_SHA,
                    "pid" => 1000 + k,
                    "seed" => Int(seed),
                    "expectation_met" => true,
                    "status" => "optimal",
                    "certificate_valid" => true,
                    "objective" => Float64(known),
                    "iterations" => 10,
                    "warmup_seconds" => 1.0,
                    "solve_seconds" => 0.5,
                    "maxrss_bytes" => scenario === :over_limit ? 2^40 :
                        SMALL_RSS["warmmax"] + k,
                    "excluded_from_throughput" => true))
            claimed = [i for i in 0:(n_items - 1)
                       if !(skip_last && i == n_items - 1) &&
                       mod(i, n_workers) == k]
            P._atomic_toml(joinpath(result_dir, "worker_$(k)_summary.toml"),
                Dict{String,Any}(
                    "protocol_version" => P.PROTOCOL_VERSION,
                    "worker_id" => k,
                    "run_id" => P.RUN_ID,
                    "source_commit" => P.SOURCE_COMMIT,
                    "script_sha256" => P.SCRIPT_SHA,
                    "pid" => 1000 + k,
                    "claimed_items" => claimed,
                    "failed_items" => [],
                    "missing_items" => [],
                    "warmup_seconds" => 1.0,
                    "peak_rss_bytes" => scenario === :over_limit ? 2^40 :
                        SMALL_RSS["peak"] + k,
                    "post_batch_rss_bytes" => post))
        end
    end
    scenario === :child_failure && (failures = ["injected_failure"])
    scenario === :zero_wall && mode == "persistent" && (wall = 0.0)
    scenario === :inf_wall && mode == "persistent" && (wall = Inf)
    return wall, failures
end

function __stub_fresh_batch(n_items, n_workers, batch_dir)
    result_dir = joinpath(String(batch_dir), "results")
    mkpath(result_dir)
    wall, failures = __write_scenario_batches!(result_dir, "fresh",
        Int(n_items), Int(n_workers))
    return (; mode="fresh", batch_dir=String(batch_dir), result_dir,
            wall_seconds=Float64(wall), child_failures=failures)
end

function __stub_persistent_batch(n_items, n_workers, batch_dir)
    result_dir = joinpath(String(batch_dir), "results")
    queue_dir = joinpath(String(batch_dir), "queue")
    mkpath(result_dir)
    mkpath(queue_dir)
    wall, failures = __write_scenario_batches!(result_dir, "persistent",
        Int(n_items), Int(n_workers))
    return (; mode="persistent", batch_dir=String(batch_dir), result_dir,
            wall_seconds=Float64(wall), child_failures=failures)
end

# --- Groups ----------------------------------------------------------------------
function group_lifecycle()
    @testset "lifecycle: owned children reaped, pending work never launched" begin
        # Timeout: 60 s sleeper, 1 s parent deadline, pending writer queued.
        marker = "POOLPROBE_$(getpid())_TIMEOUT"
        started_file = joinpath(scratchdir("lifecycle"), "started")
        pending_file = joinpath(dirname(started_file), "pending")
        sleeper = `bash -c "echo started >> $(started_file); exec -a $(marker) sleep 60"`
        pending = `bash -c "echo launched >> $(pending_file)"`
        labeled = [("sleeper", sleeper), ("pending", pending)]
        wall = time()
        threw = false
        try
            P._run_throttled(labeled, 1; deadline_seconds=1.0)
        catch error
            threw = true
            @test error isa ArgumentError
        end
        @test threw
        @test time() - wall < 30.0
        @test isfile(started_file) # sleeper really launched: reap check below is not vacuous
        @test isempty(pids_with_marker(marker)) # killed and reaped
        @test !isfile(pending_file) # pending work never launched

        # Nonzero exit stops the queue: exit 7, then a pending writer.
        pending2 = joinpath(scratchdir("lifecycle"), "pending2")
        labeled2 = [("boom", `bash -c 'exit 7'`),
            ("pending2", `bash -c "echo launched >> $(pending2)"`)]
        failures = P._run_throttled(labeled2, 1; deadline_seconds=30.0)
        @test failures == ["boom"]
        @test !isfile(pending2)

        # Signal death is recorded as a child failure. The direct run first
        # proves the command really dies by SIGTERM (not bad usage).
        sigcmd = `bash -c "kill -TERM \$\$"`
        direct = run(sigcmd; wait=false)
        wait(direct)
        @test direct.termsignal == Base.SIGTERM
        failures = P._run_throttled([("term", sigcmd)],
            1; deadline_seconds=30.0)
        @test failures == ["term"]

        # Spawn error propagates and the live peer does not remain alive.
        marker2 = "POOLPROBE_$(getpid())_SPAWN"
        started2 = joinpath(scratchdir("lifecycle"), "started2")
        peer = `bash -c "echo started >> $(started2); exec -a $(marker2) sleep 60"`
        threw = false
        try
            P._run_throttled([("peer", peer),
                    ("bad", Cmd(["/nonexistent-pool-probe-binary-xyz"]))],
                2; deadline_seconds=30.0)
        catch
            threw = true
        end
        @test threw
        @test isfile(started2)
        @test isempty(pids_with_marker(marker2))

        # Degenerate inputs fail closed without launching anything.
        @test P._run_throttled(Tuple{String,Cmd}[], 1) == String[]
        @test_throws ArgumentError P._run_throttled([("x", `true`)], 0)
        @test_throws ArgumentError P._run_throttled([("x", `true`)], 1; deadline_seconds=Inf)
        @test_throws ArgumentError P._run_throttled([("x", `true`)], 1; deadline_seconds=0.0)
    end
end

function group_exclusivity()
    @testset "exclusivity: existing roots refused, sentinel bytes preserved" begin
        root = joinpath(scratchdir("excl"), "single_root")
        mkpath(root)
        sentinel = joinpath(root, "sentinel.toml")
        write(sentinel, "SENTINEL_SINGLE")
        @test_throws Exception P.parent_single("fresh", 2, 1, root)
        @test read(sentinel, String) == "SENTINEL_SINGLE"
        @test readdir(root) == ["sentinel.toml"]

        croot = joinpath(scratchdir("excl"), "compare_root")
        mkpath(croot)
        csentinel = joinpath(croot, "comparison_summary.toml")
        write(csentinel, "SENTINEL_COMPARE")
        @test_throws Exception P.parent_compare(2, 1, croot, 1)
        @test read(csentinel, String) == "SENTINEL_COMPARE"
        @test readdir(croot) == ["comparison_summary.toml"]
    end
end

function group_race()
    @testset "race: concurrent publication means no duplicate solve" begin
        # Publisher wins while the worker is inside the item-0 solve.
        base = scratchdir("race")
        queue_dir = joinpath(base, "queue")
        result_dir = joinpath(base, "results")
        mkpath(queue_dir)
        mkpath(result_dir)
        published = P.solve_item(1)
        empty!(STUB_CALLS)
        STUB_SLEEP[0] = 3.0
        try
            worker = @async with_args(["--worker", "--worker-id=0",
                    "--n-items=2", "--queue-dir=$(queue_dir)",
                    "--result-dir=$(result_dir)"]) do
                P.worker_main()
            end
            published_at = 0.0
            deadline = time() + 15.0
            while time() < deadline
                if any(call -> occursin("item1", String(call.spec)), STUB_CALLS)
                    error("worker solved the concurrently published item")
                end
                if any(call -> occursin("item0", String(call.spec)), STUB_CALLS)
                    P._atomic_toml(P.result_path(result_dir, 1), published)
                    published_at = time()
                    break
                end
                sleep(0.05)
            end
            @test published_at > 0.0
            wait(worker)
            @test istaskdone(worker)
            @test !istaskfailed(worker)
            after = TOML.parsefile(P.result_path(result_dir, 1))
            @test after == published # published receipt preserved: no duplicate solve/overwrite
            window = [call for call in STUB_CALLS]
            @test count(call -> occursin("item0", String(call.spec)), window) == 1
            @test count(call -> occursin("item1", String(call.spec)), window) == 0
            @test isempty(readdir(queue_dir)) # every claim released
            summary = TOML.parsefile(joinpath(result_dir,
                "worker_0_summary.toml"))
            @test summary["claimed_items"] == [0]
            @test isempty(summary["failed_items"])
            @test isempty(summary["missing_items"])
        finally
            delete!(STUB_SLEEP, 0)
            empty!(STUB_CALLS)
        end

        # Fully pre-published queue: worker solves nothing, leaks no locks.
        base2 = scratchdir("racepre")
        queue2 = joinpath(base2, "queue")
        results2 = joinpath(base2, "results")
        mkpath(queue2)
        mkpath(results2)
        for i in 0:1
            P._atomic_toml(P.result_path(results2, i), P.solve_item(i))
        end
        empty!(STUB_CALLS)
        with_args(["--worker", "--worker-id=0", "--n-items=2",
                "--queue-dir=$(queue2)", "--result-dir=$(results2)"]) do
            P.worker_main()
        end
        item_solves = filter(call -> occursin(r"item\d+$", String(call.spec)), STUB_CALLS)
        @test isempty(item_solves) # warmup-only solves are legitimate here
        @test isempty(readdir(queue2))
        summary = TOML.parsefile(joinpath(results2, "worker_0_summary.toml"))
        @test summary["claimed_items"] == []
        @test isempty(summary["missing_items"])
        empty!(STUB_CALLS)
    end
end

function receipt_mutations()
    base = P.solve_item(0)
    empty!(STUB_CALLS)
    @test P.valid_receipt(base, 0) === true
    # Convenience certified=false with otherwise valid facts stays accepted.
    accepted = deepcopy(base)
    accepted["certified"] = false
    @test P.valid_receipt(accepted, 0) === true
    # Missing RSS samples pass receipt validation; the batch memory gate
    # (not the receipt) enforces positivity.
    norSS = deepcopy(base)
    norSS["maxrss_bytes"] = 0
    norSS["rss_bytes"] = -5
    @test P.valid_receipt(norSS, 0) === true
    cases = [
        ("protocol_version", r -> r["protocol_version"] = 999),
        ("legacy_protocol", r -> r["protocol_version"] = 1),
        ("run_id", r -> r["run_id"] = "foreign-run"),
        ("loaded_sdpx", r -> r["loaded_sdpx"] = "/nonexistent"),
        ("item_index", r -> r["item_index"] = 99),
        ("seed", r -> r["seed"] = 1),
        ("status", r -> r["status"] = "primal_infeasible"),
        ("certificate_valid", r -> r["certificate_valid"] = false),
        ("expectation_met", r -> r["expectation_met"] = false),
        ("objective_shift", r -> r["objective"] = r["objective"] + 1.0),
        ("objective_nan", r -> r["objective"] = NaN),
        ("objective_inf", r -> r["objective"] = Inf),
        ("dual_nan", r -> r["dual_objective"] = NaN),
        ("primal_residual_negative", r -> r["primal_residual"] = -1.0),
        ("dual_residual_negative", r -> r["dual_residual"] = -1.0),
        ("gap_negative", r -> r["relative_gap"] = -1.0),
        ("solve_inf", r -> r["solve_seconds"] = Inf),
        ("solve_negative", r -> r["solve_seconds"] = -1.0),
        ("solve_nan", r -> r["solve_seconds"] = NaN),
        ("wall_inf", r -> r["item_wall_seconds"] = Inf),
        ("wall_negative", r -> r["item_wall_seconds"] = -0.5),
        ("iterations_negative", r -> r["iterations"] = -1),
        ("iterations_float", r -> r["iterations"] = 1.5),
        ("iterations_string", r -> r["iterations"] = "10"),
        ("workload", r -> r["workload"] = "lp_other"),
        ("model_structure", r -> r["model_structure"] = "other"),
        ("julia_version", r -> r["julia_version"] = "0.0.0"),
        ("source_commit", r -> r["source_commit"] = "0"^40),
        ("script_sha", r -> r["script_sha256"] = "0"^64),
        ("missing_key", r -> delete!(r, "objective")),
    ]
    for (name, mutate) in cases
        mutated = deepcopy(base)
        try
            mutate(mutated)
        catch
            # A mutation that cannot even apply is itself a rejection.
        end
        @test P.valid_receipt(mutated, 0) === false
    end
    empty!(STUB_CALLS)
end

function group_receipts()
    @testset "receipts: mutation table through production valid_receipt" begin
        receipt_mutations()
    end
end

function good_worker_dicts(n_items::Integer, n_workers::Integer)
    receipts = Dict{Int,Dict{String,Any}}()
    for i in 0:(n_items - 1)
        receipt = P.solve_item(i)
        receipt["worker_id"] = mod(i, n_workers)
        receipt["pid"] = 1000 + mod(i, n_workers)
        receipts[i] = receipt
    end
    empty!(STUB_CALLS)
    warmups = Dict{Int,Dict{String,Any}}()
    summaries = Dict{Int,Dict{String,Any}}()
    for k in 0:(n_workers - 1)
        seed = P.BASE_SEED + P.WARMUP_SALT + UInt32(k)
        known = STUB._planted_lp_objective(seed, P.WORKLOAD_M, P.WORKLOAD_N)
        warmups[k] = Dict{String,Any}(
            "protocol_version" => P.PROTOCOL_VERSION,
            "worker_id" => k,
            "run_id" => P.RUN_ID,
            "source_commit" => P.SOURCE_COMMIT,
            "script_sha256" => P.SCRIPT_SHA,
            "pid" => 1000 + k,
            "seed" => Int(seed),
            "expectation_met" => true,
            "status" => "optimal",
            "certificate_valid" => true,
            "objective" => Float64(known),
            "iterations" => 10,
            "warmup_seconds" => 1.0,
            "solve_seconds" => 0.5,
            "maxrss_bytes" => 900_000,
            "excluded_from_throughput" => true)
        summaries[k] = Dict{String,Any}(
            "protocol_version" => P.PROTOCOL_VERSION,
            "worker_id" => k,
            "run_id" => P.RUN_ID,
            "source_commit" => P.SOURCE_COMMIT,
            "script_sha256" => P.SCRIPT_SHA,
            "pid" => 1000 + k,
            "claimed_items" => [i for i in 0:(n_items - 1) if mod(i, n_workers) == k],
            "failed_items" => [],
            "missing_items" => [],
            "warmup_seconds" => 1.0,
            "peak_rss_bytes" => 1_100_000,
            "post_batch_rss_bytes" => 800_000)
    end
    return receipts, warmups, summaries
end

function group_workers()
    @testset "workers: exact identities, warmup facts, claim coverage" begin
        receipts, warmups, summaries = good_worker_dicts(2, 2)
        @test P.valid_worker_artifacts(warmups, summaries, receipts, 2, 2) === true

        # Unknown IDs in place of an expected worker are rejected.
        foreign_w = Dict(99 => deepcopy(warmups[1]))
        foreign_s = Dict(99 => deepcopy(summaries[1]))
        @test P.valid_worker_artifacts(foreign_w, foreign_s, receipts, 2, 2) === false

        # Filename key vs embedded worker ID mismatch is rejected.
        mismatch_w = deepcopy(warmups)
        mismatch_w[0]["worker_id"] = 99
        @test P.valid_worker_artifacts(mismatch_w, summaries, receipts, 2, 2) === false

        foreign_run = deepcopy(warmups)
        foreign_run[0]["run_id"] = "foreign-run"
        @test P.valid_worker_artifacts(foreign_run, summaries, receipts, 2, 2) === false

        bad_protocol = deepcopy(summaries)
        bad_protocol[1]["protocol_version"] = 999
        @test P.valid_worker_artifacts(warmups, bad_protocol, receipts, 2, 2) === false

        inf_objective = deepcopy(warmups)
        inf_objective[0]["objective"] = Inf
        @test P.valid_worker_artifacts(inf_objective, summaries, receipts, 2, 2) === false

        bad_expectation = deepcopy(warmups)
        bad_expectation[1]["expectation_met"] = false
        @test P.valid_worker_artifacts(bad_expectation, summaries, receipts, 2, 2) === false

        wrong_seed = deepcopy(warmups)
        wrong_seed[0]["seed"] = 1
        @test P.valid_worker_artifacts(wrong_seed, summaries, receipts, 2, 2) === false

        bad_iters = deepcopy(warmups)
        bad_iters[0]["iterations"] = -1
        @test P.valid_worker_artifacts(bad_iters, summaries, receipts, 2, 2) === false

        failed = deepcopy(summaries)
        failed[0]["failed_items"] = [0]
        @test P.valid_worker_artifacts(warmups, failed, receipts, 2, 2) === false

        missing = deepcopy(summaries)
        missing[1]["missing_items"] = [1]
        @test P.valid_worker_artifacts(warmups, missing, receipts, 2, 2) === false

        pid_mismatch = deepcopy(summaries)
        pid_mismatch[0]["pid"] = 4242
        @test P.valid_worker_artifacts(warmups, pid_mismatch, receipts, 2, 2) === false

        wrong_owner = deepcopy(receipts)
        wrong_owner[0]["worker_id"] = 1
        @test P.valid_worker_artifacts(warmups, summaries, wrong_owner, 2, 2) === false

        partial = deepcopy(summaries)
        partial[1]["claimed_items"] = []
        @test P.valid_worker_artifacts(warmups, partial, receipts, 2, 2) === false

        ghost = deepcopy(summaries)
        ghost[0]["claimed_items"] = [0, 5]
        @test P.valid_worker_artifacts(warmups, ghost, receipts, 2, 2) === false

        # Empty claims stay permissible when workers outnumber items.
        receipts1, warmups1, summaries1 = good_worker_dicts(1, 2)
        @test summaries1[1]["claimed_items"] == []
        @test P.valid_worker_artifacts(warmups1, summaries1, receipts1, 1, 2) === true
        empty!(STUB_CALLS)
    end
end

function write_memory_batch(scenario::Symbol)
    base = scratchdir("memory")
    batch_dir = joinpath(base, "batch")
    result_dir = joinpath(batch_dir, "results")
    mkpath(result_dir)
    receipts, warmups, summaries = good_worker_dicts(2, 1)
    for (i, receipt) in receipts
        P._atomic_toml(P.result_path(result_dir, i), receipt)
    end
    for (k, warmup) in warmups
        P._atomic_toml(joinpath(result_dir, "warmup_$(k).toml"), warmup)
    end
    for (k, summary) in summaries
        if scenario === :zero_post
            summary["post_batch_rss_bytes"] = 0
        elseif scenario === :zero_receipt_rss
            r = TOML.parsefile(P.result_path(result_dir, 0))
            r["rss_bytes"] = 0
            rm(P.result_path(result_dir, 0))
            P._atomic_toml(P.result_path(result_dir, 0), r)
        end
        P._atomic_toml(joinpath(result_dir, "worker_$(k)_summary.toml"),
            summary)
    end
    return batch_dir, result_dir
end

function group_memory()
    @testset "memory and duration gates through production aggregate_batch" begin
        batch_dir, result_dir = write_memory_batch(:good)
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            10.0, String[], 2, 1)
        @test summary["batch_valid"] === true
        @test summary["rss_samples_valid"] === true
        @test summary["certified_solves_per_hour"] ≈ 2 / (10.0 / 3600)

        # One unavailable per-worker sample among valid samples fails.
        batch_dir, result_dir = write_memory_batch(:zero_post)
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            10.0, String[], 2, 1)
        @test summary["rss_samples_valid"] === false
        @test summary["batch_valid"] === true # memory enforced at comparison, not here

        batch_dir, result_dir = write_memory_batch(:zero_receipt_rss)
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            10.0, String[], 2, 1)
        @test summary["rss_samples_valid"] === false

        # Zero and infinite batch walls fail closed with zero throughput.
        batch_dir, result_dir = write_memory_batch(:good)
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            0.0, String[], 2, 1)
        @test summary["batch_valid"] === false
        @test summary["certified_solves_per_hour"] == 0.0

        batch_dir, result_dir = write_memory_batch(:good)
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            Inf, String[], 2, 1)
        @test summary["batch_valid"] === false
        @test summary["certified_solves_per_hour"] == 0.0

        # A missing receipt fails the batch.
        batch_dir, result_dir = write_memory_batch(:good)
        rm(P.result_path(result_dir, 1))
        summary = P.aggregate_batch("persistent", batch_dir, result_dir,
            10.0, String[], 2, 1)
        @test summary["batch_valid"] === false

        # Fresh mode needs no worker artifacts.
        fresh_base = scratchdir("memoryfresh")
        fresh_results = joinpath(fresh_base, "results")
        mkpath(fresh_results)
        for i in 0:1
            P._atomic_toml(P.result_path(fresh_results, i), P.solve_item(i))
        end
        empty!(STUB_CALLS)
        summary = P.aggregate_batch("fresh", fresh_base, fresh_results,
            10.0, String[], 2, 1)
        @test summary["batch_valid"] === true
    end
end

function run_compare_case(scenario::Symbol; reps::Integer=1,
                          extra_args::Vector{String}=String[])
    SCENARIO[] = scenario
    use_stub_launchers!()
    try
        outdir = joinpath(scratchdir("gate"), "run")
        return with_args(extra_args) do
            Base.invokelatest(P.parent_compare, 2, 1, outdir, reps)
        end
    finally
        restore_launchers!()
        empty!(STUB_CALLS)
    end
end

function group_gate()
    @testset "comparison gate: rejection cases and counterbalanced order" begin
        comparison = run_compare_case(:all_valid)
        @test comparison["gate_pass"] === true
        @test comparison["all_batches_valid"] === true
        @test comparison["state_isolation_ok"] === true
        @test comparison["memory_samples_within_limit"] === true
        @test comparison["interleaved_order"] == ["fresh", "persistent"]

        comparison = run_compare_case(:all_valid; reps=2)
        @test comparison["gate_pass"] === true
        @test comparison["interleaved_order"] ==
              ["fresh", "persistent", "persistent", "fresh"]

        comparison = run_compare_case(:missing)
        @test comparison["gate_pass"] === false
        @test comparison["all_batches_valid"] === false

        comparison = run_compare_case(:failed_receipt)
        @test comparison["gate_pass"] === false
        @test comparison["all_batches_valid"] === false

        comparison = run_compare_case(:child_failure)
        @test comparison["gate_pass"] === false
        @test comparison["all_batches_valid"] === false

        comparison = run_compare_case(:fingerprint_mismatch)
        @test comparison["gate_pass"] === false
        @test comparison["state_isolation_ok"] === false

        comparison = run_compare_case(:zero_wall)
        @test comparison["gate_pass"] === false
        @test comparison["all_batches_valid"] === false

        comparison = run_compare_case(:inf_wall)
        @test comparison["gate_pass"] === false
        @test comparison["all_batches_valid"] === false

        comparison = run_compare_case(:bad_rss)
        @test comparison["gate_pass"] === false
        @test comparison["memory_samples_within_limit"] === false

        comparison = run_compare_case(:over_limit;
            extra_args=["--rss-limit-bytes=4096"])
        @test comparison["gate_pass"] === false
        @test comparison["memory_samples_within_limit"] === false
    end
end

const JULIA_BIN = Base.julia_cmd().exec[1]
const BOUNDED_FLAGS = ["--startup-file=no", "--threads=1", "--gcthreads=1",
    "--heap-size-hint=2G", "--project=$(Base.active_project())"]
const CHILD_ENV = Dict{String,String}(
    "JULIA_NUM_THREADS" => "1",
    "JULIA_NUM_GC_THREADS" => "1",
    "OPENBLAS_NUM_THREADS" => "1",
    "OMP_NUM_THREADS" => "1",
    "MKL_NUM_THREADS" => "1",
    "JULIA_PKG_PRECOMPILE_AUTO" => "0",
    "JULIA_PKG_OFFLINE" => "true")

function run_probe(probe::AbstractString, scenario::AbstractString)
    script = abspath(@__FILE__)
    outdir = joinpath(scratchdir("cliprobe"), "run")
    args = probe == "badmode" ? ["--mode=bogus"] :
        probe == "zeroitems" ? ["--mode=compare", "--items=0"] :
        ["--mode=compare", "--items=2", "--workers=1",
         "--outdir=$(outdir)", "--reps=1"]
    cmd = `$JULIA_BIN $BOUNDED_FLAGS $script --probe=$(probe) --probe-scenario=$(scenario) $args`
    result = run_guarded(addenv(cmd, CHILD_ENV...), 150.0;
        logpath=joinpath(scratchdir("clilog"), probe * ".log"))
    @test result.timed_out === false
    return result
end

function group_cli()
    @testset "cli: exit codes from guarded subprocess probes" begin
        valid = run_probe("compare", "all_valid")
        @test valid.exitcode == 0

        invalid = run_probe("compare", "failed_receipt")
        @test invalid.exitcode != 0

        badmode = run_probe("badmode", "all_valid")
        @test badmode.exitcode != 0

        zeroitems = run_probe("zeroitems", "all_valid")
        @test zeroitems.exitcode != 0
    end
end

# --- Probe mode (single scenario in a subprocess for exit-code checks) ---------
function probe_main()
    value(option::AbstractString) = begin
        prefix = "--" * option * "="
        for arg in ARGS
            startswith(arg, prefix) && return arg[length(prefix) + 1:end]
        end
        return nothing
    end
    scenario = Symbol(something(value("probe-scenario"), "all_valid"))
    SCENARIO[] = scenario
    use_stub_launchers!()
    try
        Base.invokelatest(P.main)
    finally
        restore_launchers!()
    end
end

function test_main()
    value(option::AbstractString, default=nothing) = begin
        prefix = "--" * option * "="
        for arg in ARGS
            startswith(arg, prefix) && return arg[length(prefix) + 1:end]
        end
        return default
    end
    group = something(value("group"), "all")
    groups = Dict("lifecycle" => group_lifecycle,
        "exclusivity" => group_exclusivity,
        "race" => group_race,
        "receipts" => group_receipts,
        "workers" => group_workers,
        "memory" => group_memory,
        "gate" => group_gate,
        "cli" => group_cli)
    if group == "all"
        @testset "persistent worker pool contract (solver-free)" begin
            for name in ("lifecycle", "exclusivity", "race", "receipts",
                         "workers", "memory", "gate")
                groups[name]()
            end
        end
        println("NOTE: cli group runs separately (spawns guarded subprocesses).")
    elseif haskey(groups, group)
        groups[group]()
    else
        throw(ArgumentError("unknown group $group"))
    end
end

if any(arg -> startswith(arg, "--probe="), ARGS)
    probe_main()
elseif abspath(PROGRAM_FILE) == abspath(@__FILE__)
    test_main()
end
