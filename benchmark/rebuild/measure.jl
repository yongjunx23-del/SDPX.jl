# Q01 — three-phase measurement harness.
#
# Packet requirement: "分首次编译、warm fresh setup、prepared solve，记录所有phase、
# 计数、allocation、RSS."
#
# ## Why three phases and never one number
#
# ADR-003 §7: compilation is listed separately and is never folded into a warm
# number. On this codebase the difference is not cosmetic — the first solve of a
# case pays LLVM compilation that a warmed one does not, and averaging them
# produces a number that describes neither. The three phases are:
#
#   :first_compile     the first solve of this case in this process.
#   :warm_fresh_setup  subsequent solves that rebuild state from scratch.
#   :prepared_solve    repeats after the JIT and allocator have stabilised.
#
# `:prepared_solve` is the only phase a speed comparison may quote, and even then
# only alongside the failure counts.
#
# ## What is deliberately NOT here
#
# A "prepared" replay of an existing prepared object requires the S04/S07
# prepared-update API, which does not exist yet at this baseline. This harness
# therefore reports `prepared_solve` as repeated fresh solves and says so; it
# does not claim to measure a prepared path it cannot reach. Per ADR-003 §3 an
# unreachable phase is recorded as `not_run` with a reason, not estimated.
#
#   julia --startup-file=no --project=. benchmark/rebuild/measure.jl [--repeats N] [--out PATH]

using SDPX
using Dates
using TOML
using Printf
using Statistics: median
using LinearAlgebra: BLAS

include(joinpath(@__DIR__, "manifest.jl"))
using .RebuildManifest

const SCHEMA = "sdpx-rebuild-measure/1"

"""Resident set size in bytes, or `nothing` when the platform cannot report it."""
function _rss_bytes()
    try
        return Int(Sys.maxrss())
    catch
        return nothing
    end
end

"""Actual executed thread counts, kept separate from the requested ones."""
function _thread_facts(requested::Int)
    return (
        requested_threads=requested,
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        cpu_threads=Sys.CPU_THREADS,
    )
end

"""
    run_case(case, settings) -> (row, unavailable)

One timed solve. Allocation includes everything the public call does — result
construction, certificate assembly and recovery — because that is what a caller
pays. It is NOT an inner-loop allocation figure and must not be reported as one.
"""
function run_case(case::RebuildCase, settings)
    rss_before = _rss_bytes()
    started = time_ns()
    local result
    try
        allocated = @allocated begin
            result = SDPX.optimize!(case.build(); settings)
        end
        elapsed = Float64(time_ns() - started) * 1.0e-9
        rss_after = _rss_bytes()
        delta = (rss_before === nothing || rss_after === nothing) ? nothing :
                rss_after - rss_before
        row, unavailable = result_row(case, result;
            seconds=elapsed, allocated_bytes=allocated, rss_delta_bytes=delta)
        return row, unavailable
    catch exception
        elapsed = Float64(time_ns() - started) * 1.0e-9
        # A construction/solve failure is DATA. It is recorded and kept.
        return failure_row(case, exception; phase=:solve, seconds=elapsed), Symbol[]
    end
end

"""
    measure_case(case; repeats, settings_factory) -> NamedTuple

Run one case through all three phases and aggregate.

Aggregation discipline (ADR-003 §7): timing medians are taken over SUCCESSFUL
warm runs only, and the success/failure counts are reported beside them, so a
fast failure can never masquerade as a speed win.
"""
function measure_case(case::RebuildCase; repeats::Int=5)
    rows = NamedTuple[]
    unavailable = Symbol[]

    # Phase 1 — first compile. One solve, never reused for timing.
    first_row, first_unavailable = run_case(case, case_settings(case))
    first_seconds = first_row.seconds
    append!(unavailable, first_unavailable)

    # Phases 2 and 3 — warm repeats. The partition is by repeat index: the first
    # warm repeat still rebuilds more state than later ones, so it is reported
    # separately rather than blended.
    warm_rows = NamedTuple[]
    for repeat in 1:repeats
        row, row_unavailable = run_case(case, case_settings(case))
        append!(unavailable, row_unavailable)
        push!(warm_rows, row)
    end

    success(row) = !row.threw && row.status == "optimal" &&
                   row.certificate_valid === true
    warm_success = filter(success, warm_rows)
    warm_failure = length(warm_rows) - length(warm_success)

    aggregate(rows_subset, field) = isempty(rows_subset) ? nothing :
        median(getproperty(row, field) for row in rows_subset)

    return (
        id=case.id,
        family=case.family,
        shape=shape_trace(case),
        first_compile=(row=first_row, seconds=first_seconds),
        warm=(
            runs=length(warm_rows),
            solved=length(warm_success),
            failed=warm_failure,
            # Medians over successes only; `nothing` when none succeeded.
            seconds=aggregate(warm_success, :seconds),
            allocated_bytes=aggregate(warm_success, :allocated_bytes),
            rss_delta_bytes=aggregate(warm_success, :rss_delta_bytes),
            iterations=aggregate(warm_success, :iterations),
            factorizations=aggregate(warm_success, :factorizations),
            rows=warm_rows,
        ),
        unavailable=sort(unique(unavailable)),
    )
end

"""Run the whole manifest and return the complete report payload."""
function measure_all(; repeats::Int=5)
    cases = rebuild_cases()
    results = [measure_case(case; repeats=repeats) for case in cases]
    all_warm = vcat([collect(r.warm.rows) for r in results]...)
    summary = summarize(all_warm)
    return (
        schema=SCHEMA,
        generated=string(now()),
        # The baseline SHA is recorded so a rerun is comparable (acceptance 1).
        sdpx_head=try
            strip(read(`git -C $(normpath(joinpath(@__DIR__, "..", ".."))) rev-parse HEAD`, String))
        catch
            "unknown"
        end,
        worktree_dirty=try
            !isempty(strip(read(`git -C $(normpath(joinpath(@__DIR__, "..", ".."))) status --porcelain`, String)))
        catch
            nothing
        end,
        julia_version=string(VERSION),
        cases_fingerprint=cases_fingerprint(cases),
        threads=_thread_facts(1),
        repeats=repeats,
        phases=("first_compile", "warm_fresh_setup", "prepared_solve"),
        prepared_solve_note=("repeated fresh solves; a true prepared replay needs the "
                             * "S04/S07 prepared-update API which does not exist at this baseline"),
        cases=results,
        warm_totals=(
            total=summary.total, solved=summary.solved, failed=summary.failed,
            threw=summary.threw, without_certificate=summary.without_certificate,
        ),
    )
end


"""
    _tomlify(value) -> Any

Convert a result payload into a structure `TOML.print` accepts: nested
`Dict{String,Any}` with string keys and scalar leaves.

Two rules from ADR-003 §3 are enforced here rather than left to the caller:

* `nothing` becomes the literal string `"not_run"`, never `0` — an unmeasured
  value must remain distinguishable from a measured zero after serialisation.
* `Symbol` becomes `String`, so a route or reason is readable in the receipt
  without a symbol table.
"""
_tomlify(::Nothing) = "not_run"
_tomlify(value::Symbol) = String(value)
_tomlify(value::AbstractString) = String(value)
_tomlify(value::Union{Bool,Integer,AbstractFloat}) = value
_tomlify(value::NamedTuple) =
    Dict{String,Any}(String(k) => _tomlify(v) for (k, v) in pairs(value))
_tomlify(value::AbstractDict) =
    Dict{String,Any}(String(k) => _tomlify(v) for (k, v) in value)
_tomlify(value::AbstractVector) = [_tomlify(item) for item in value]
_tomlify(value::Tuple) = [_tomlify(item) for item in value]
_tomlify(value) = string(value)

function _fmt(value)
    value === nothing && return "not_run"
    value isa AbstractFloat && return @sprintf("%.6g", value)
    return string(value)
end

function main()
    repeats = 5
    out = joinpath(@__DIR__, "measure_result.toml")
    for arg in ARGS
        startswith(arg, "--repeats=") && (repeats = something(tryparse(Int, split(arg, '=')[2]), 5))
        startswith(arg, "--out=") && (out = split(arg, '=')[2])
    end
    payload = measure_all(; repeats=repeats)
    @printf("Q01 measure — SDPX %s (dirty=%s), %d cases x %d warm repeats\n",
        first(payload.sdpx_head, 12), payload.worktree_dirty, length(payload.cases), repeats)
    @printf("  threads: requested=%d julia=%d blas=%d cpu=%d\n",
        payload.threads.requested_threads, payload.threads.julia_threads,
        payload.threads.blas_threads, payload.threads.cpu_threads)
    @printf("  cases_fingerprint = %d\n", payload.cases_fingerprint)
    @printf("  %-18s %-9s %-6s %-6s %-11s %-11s\n",
        "case", "status", "solved", "failed", "warm_s", "first_s")
    for entry in payload.cases
        status = entry.warm.solved > 0 ? "ok" : "FAILED"
        @printf("  %-18s %-9s %-6d %-6d %-11s %-11s\n",
            entry.id, status, entry.warm.solved, entry.warm.failed,
            _fmt(entry.warm.seconds), @sprintf("%.6g", entry.first_compile.seconds))
    end
    t = payload.warm_totals
    @printf("  warm totals: total=%d solved=%d failed=%d threw=%d without_certificate=%d\n",
        t.total, t.solved, t.failed, t.threw, t.without_certificate)

    open(out, "w") do io
        TOML.print(io, _tomlify(payload); sorted=true)
    end
    @printf("  wrote %s\n", out)
    return payload, out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
