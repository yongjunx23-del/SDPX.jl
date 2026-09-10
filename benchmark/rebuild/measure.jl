# Q01 — three-phase measurement harness with an arithmetic axis.
#
# Packet requirement: "分首次编译、warm fresh setup、prepared solve，记录所有phase、
# 计数、allocation、RSS."  ADR-003 §7 adds: compilation listed separately and never
# folded into a warm number; BigFloat allocation must distinguish Julia heap /
# cell identity / native allocator / RSS; timing medians over SUCCESSFUL runs
# only, with the counts beside them.
#
# ## The three phases, and exactly what each one is
#
#   :first_compile     the first solve of this case in this process, with the
#                      cross-solve structure cache cleared. Includes LLVM
#                      compilation. Reported for diagnosis, never quoted as a
#                      speed.
#   :warm_fresh_setup  later solves that rebuild the structure from scratch:
#                      `clear_structure_cache!()` before every repeat. Compiled
#                      code is warm; the symbolic structure is not.
#   :prepared_solve    repeats with the cross-solve structure cache left WARM, so
#                      the solve reuses the frozen symbolic structure exactly as a
#                      prepared session would. The cache hit/miss and
#                      symbolic-analysis deltas are recorded as the evidence that
#                      the phase was what it claims to be.
#
# A true **prepared-update replay** (new objective/RHS into an existing prepared
# object) needs the S04/S07 prepared-update API, which does not exist at this
# baseline. That replay is reported as `prepared_update_replay_status = "not_run"`
# with its reason. Per ADR-003 §3 an unreachable phase is `not_run` with a
# reason, not estimated and not zero.
#
# ## One arithmetic and one environment per process
#
# Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width and the
# BFLA/MPFR specialisations are compiled in the SAME process, so each arithmetic
# arm runs in its own process, and `-t1` is enforced (`scripts/provider_smoke.sh`
# documents the same rule for its `all` target). A figure measured under
# `$REBUILD_ENV` is NOT comparable to one measured in the default project: the
# environment is part of the measurement identity and is recorded in every
# payload, together with the executed route and provider of every single run, so
# a configuration difference can never be read as a speed-up.
#
# ## A case budget is allowed; deleting a case is not
#
# `--case-budget-seconds=N` bounds the wall time one case may consume (measured:
# an unbounded BigFloat-256 arm was still running after 23 min, allocation/GC
# bound). A case that exceeds the budget is NOT deleted and NOT quietly given
# fewer repeats: the samples taken are kept, unreached phases are `not_run` with
# `runs = 0`, and `cases[].budget` states the limit and the measured elapsed
# time. The artifact is streamed after every case (`partial = true`), so an
# interrupted arm keeps the evidence it already paid for.
#
#   julia --startup-file=no --project=. -t1 benchmark/rebuild/measure.jl \
#       [--arithmetic=float64|multifloat_x2|bigfloat_256] [--repeats=N] \
#       [--case-budget-seconds=N] [--cases=id1,id2] [--out=PATH] [--env-label=LABEL]
#
# Exit codes: 0 measured (case failures included — they are data),
#             2 harness/usage/infrastructure error,
#             3 the requested arithmetic arm is not runnable in this environment,
#             4 SDPX_MEASURE_REQUIRE_IDLE=1 and the host is not idle.

using SDPX
using Dates
using TOML
using SHA
using Printf
using Statistics: median
using LinearAlgebra: BLAS

include(joinpath(@__DIR__, "manifest.jl"))
using .RebuildManifest

const SCHEMA = "sdpx-rebuild-measure/2"

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
        julia_command=join(Base.julia_cmd().exec, " "),
        process_started_with_threads=Threads.nthreads(),
    )
end

"""
    source_fingerprint() -> String

SHA-256 over `(relative path, file SHA-256)` for every source file of the
package under test (`src/`, `ext/`, `Project.toml`), sorted by path.

`git rev-parse HEAD` is not sufficient for measurement integrity in this packet:
several tasks commit to this tree concurrently, and the working tree also carries
uncommitted work. A HEAD-only record cannot tell whether the code that produced
two numbers in the same arm was the same code. This can: it is computed at the
start and at the end of every arm, and a change is flagged.
"""
function source_fingerprint()
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    files = String[]
    for sub in ("src", "ext")
        root = joinpath(repo, sub)
        isdir(root) || continue
        for (directory, _, names) in walkdir(root)
            for name in names
                endswith(name, ".jl") || continue
                push!(files, relpath(joinpath(directory, name), repo))
            end
        end
    end
    push!(files, "Project.toml")
    sort!(files)
    buffer = IOBuffer()
    for relative in files
        path = joinpath(repo, relative)
        isfile(path) || continue
        print(buffer, relative, '\0', bytes2hex(SHA.sha256(read(path))), '\n')
    end
    return bytes2hex(SHA.sha256(take!(buffer)))
end

"""
    contention_facts() -> NamedTuple

Who else is on the machine. A performance number taken while other Julia
processes are running is not comparable to one taken on an idle host, so the
fact is recorded with every case rather than assumed. It is an OBSERVATION of
this host at this moment (a `pgrep` count and the platform load average), not a
measurement of another process's effect on this one.

`SDPX_MEASURE_REQUIRE_IDLE=1` turns the observation into a refusal: the harness
then exits 4 instead of producing numbers it cannot defend.
"""
function contention_facts()
    others = nothing
    try
        listing = read(`pgrep -f julia`, String)
        others = max(0, count(!isempty, split(listing, '\n')) - 1) # minus self
    catch
        others = nothing
    end
    load = try
        Sys.loadavg()
    catch
        nothing
    end
    return (
        other_julia_processes=others,
        loadavg_1m=load === nothing ? nothing : Float64(load[1]),
        loadavg_5m=load === nothing ? nothing : Float64(load[2]),
        loadavg_15m=load === nothing ? nothing : Float64(load[3]),
        cpu_threads=Sys.CPU_THREADS,
    )
end

"""The contention rule, stated so a reader can disagree with it explicitly."""
const CONTENTION_RULE = "contended = (other_julia_processes >= 1) || " *
    "(loadavg_1m > 0.75 * cpu_threads); an observation of this host, not a " *
    "measurement of another process's effect"

function _contended(facts)
    others = facts.other_julia_processes
    load = facts.loadavg_1m
    return (others !== nothing && others >= 1) ||
           (load !== nothing && load > 0.75 * facts.cpu_threads)
end

"""`git rev-parse HEAD` of the tree being measured, read at measurement time."""
function _head_sha()
    try
        return strip(read(`git -C $(normpath(joinpath(@__DIR__, "..", ".."))) rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

function _worktree_dirty()
    try
        return !isempty(strip(read(
            `git -C $(normpath(joinpath(@__DIR__, "..", ".."))) status --porcelain`, String)))
    catch
        return nothing
    end
end

# ---------------------------------------------------------------------------
# One timed solve
# ---------------------------------------------------------------------------

"""
    run_case(case, settings, T, arithmetic) -> (row, unavailable)

One timed solve. Allocation is the whole public call — result construction,
certificate assembly and recovery included — because that is what a caller pays.
It is NOT an inner-loop allocation figure and must not be reported as one.

Every fact needed to keep the phase honest is captured around the call:
structure-cache and symbolic-analysis deltas (was the structure really rebuilt?),
RSS peak, and — for BigFloat — the cell-identity set of the published values.
"""
function run_case(case::RebuildCase, settings, ::Type{T}, arithmetic::Symbol) where {T}
    stats_before = SDPX.structure_cache_stats()
    analyses_before = SDPX.symbolic_analysis_count()
    rss_before = _rss_bytes()
    started = time_ns()
    local result
    try
        allocated = @allocated begin
            result = SDPX.optimize!(case.build(T); settings)
        end
        elapsed = Float64(time_ns() - started) * 1.0e-9
        rss_after = _rss_bytes()
        delta = (rss_before === nothing || rss_after === nothing) ? nothing :
                rss_after - rss_before
        row, unavailable = result_row(case, result;
            arithmetic=arithmetic, seconds=elapsed,
            allocated_bytes=allocated, rss_delta_bytes=delta)
        stats_after = SDPX.structure_cache_stats()
        cache = (
            hits_delta=stats_after.hits - stats_before.hits,
            misses_delta=stats_after.misses - stats_before.misses,
            symbolic_analyses_delta=SDPX.symbolic_analysis_count() - analyses_before,
            entries_after=stats_after.entries,
        )
        return row, unavailable, cache, cell_identity_snapshot(result)
    catch exception
        elapsed = Float64(time_ns() - started) * 1.0e-9
        # A construction/solve failure is DATA. It is recorded and kept.
        row = failure_row(case, exception; arithmetic=arithmetic, phase=:solve,
            seconds=elapsed, precision_bits=nothing)
        stats_after = SDPX.structure_cache_stats()
        cache = (
            hits_delta=stats_after.hits - stats_before.hits,
            misses_delta=stats_after.misses - stats_before.misses,
            symbolic_analyses_delta=SDPX.symbolic_analysis_count() - analyses_before,
            entries_after=stats_after.entries,
        )
        return row, String[], cache, nothing
    end
end

# ---------------------------------------------------------------------------
# One case, three phases
# ---------------------------------------------------------------------------

"""
    measure_case(case, T, arithmetic; repeats, budget_seconds) -> NamedTuple

Run one case through the three phases and aggregate.

Aggregation discipline (ADR-003 §7): timing medians are taken over SUCCESSFUL
runs only, the success/failure counts are reported beside them, and the raw rows
are kept in full, so a fast failure can never masquerade as a speed win.

`budget_seconds > 0` bounds the *measured wall time this case may consume*. A case
that exceeds it is NOT deleted and NOT silently given fewer repeats: the samples
already taken are kept, every phase that could not be reached is reported
`not_run` with `runs = 0`, and `budget.reason` states the measured elapsed time
and the limit. That distinction matters — `not_run` with a measured cost and a
reason is a counted result, whereas a dropped case is a missing one.
"""
function measure_case(case::RebuildCase, ::Type{T}, arithmetic::Symbol;
    repeats::Int=3, budget_seconds::Float64=0.0,
) where {T}
    rows = NamedTuple[]
    unavailable = String[]
    phase_notes = Dict{String,Any}()
    spent = Ref(0.0)
    stopped_phases = String[]
    over_budget() = budget_seconds > 0.0 && spent[] >= budget_seconds

    function collect_phase!(phase::Symbol, clears_cache::Bool)
        phase_rows = NamedTuple[]
        cache_facts = NamedTuple[]
        seconds = Float64[]
        reused = Int[]
        previous_ids = nothing
        for repeat in 1:(phase === :first_compile ? 1 : repeats)
            if over_budget()
                push!(stopped_phases, String(phase))
                break
            end
            clears_cache && SDPX.clear_structure_cache!()
            row, row_unavailable, cache, ids = run_case(
                case, case_settings(case, T), T, arithmetic)
            append!(unavailable, row_unavailable)
            push!(phase_rows, row)
            push!(cache_facts, cache)
            push!(seconds, row.seconds)
            spent[] += row.seconds
            if ids !== nothing && previous_ids !== nothing
                push!(reused, length(intersect(previous_ids, ids)))
            elseif ids !== nothing
                push!(reused, -1) # first BigFloat sample: no predecessor
            end
            previous_ids = ids
        end
        append!(rows, phase_rows)
        success(row) = !row.threw && row.status == "optimal" &&
                       row.certificate_valid === true
        solved = filter(success, phase_rows)
        aggregate(field) = isempty(solved) ? nothing :
            median(getproperty(row, field) for row in solved)
        pick(field) = isempty(solved) ? nothing :
            [getproperty(row, field) for row in solved]
        finite_min(values) = (values === nothing || isempty(filter(!isnothing, values))) ?
            nothing : minimum(filter(!isnothing, values))
        finite_max(values) = (values === nothing || isempty(filter(!isnothing, values))) ?
            nothing : maximum(filter(!isnothing, values))
        entry = (
            runs=length(phase_rows),
            solved=length(solved),
            failed=length(phase_rows) - length(solved),
            seconds=aggregate(:seconds),
            seconds_min=finite_min(pick(:seconds)),
            seconds_max=finite_max(pick(:seconds)),
            allocated_bytes=aggregate(:allocated_bytes),
            rss_delta_bytes=aggregate(:rss_delta_bytes),
            iterations=aggregate(:iterations),
            factorizations=aggregate(:factorizations),
            workspace_bytes=aggregate(:workspace_bytes),
            # `init=0`: a phase stopped by the per-case budget before its first
            # sample contributes nothing, and summing an empty collection throws.
            structure_cache_hits_delta=sum((cache.hits_delta for cache in cache_facts); init=0),
            structure_cache_misses_delta=sum((cache.misses_delta for cache in cache_facts); init=0),
            symbolic_analyses_delta=sum((cache.symbolic_analyses_delta for cache in cache_facts); init=0),
            statuses=join(sort(unique(String(row.status) for row in phase_rows)), ","),
            cells_reused_from_previous_sample=isempty(reused) ? nothing : reused,
            rows=phase_rows,
        )
        phase_notes[String(phase)] = (
            clears_structure_cache=clears_cache,
            solves=length(phase_rows),
            note=phase === :first_compile ?
                "first solve of this case in this process; LLVM compilation included" :
                phase === :warm_fresh_setup ?
                "compiled code warm, symbolic structure rebuilt" :
                "compiled code warm, cross-solve structure cache reused",
        )
        entry
    end

    first_compile = collect_phase!(:first_compile, true)
    warm_fresh_setup = collect_phase!(:warm_fresh_setup, true)
    prepared_solve = collect_phase!(:prepared_solve, false)

    observed = try
        observed_shape_trace(case, T)
    catch exception
        Dict("observed_trace_status" => "not_run",
             "observed_trace_reason" => sprint(showerror, exception))
    end
    mismatches = try
        declared_vs_observed(case, observed isa NamedTuple ? observed : nothing)
    catch exception
        ["declared_vs_observed: not_run — $(sprint(showerror, exception))"]
    end
    fingerprint = try
        inputs_fingerprint(case, T, arithmetic_arm(arithmetic).precision_bits;
            tolerances=(primal=1e-8, dual=1e-8, gap=1e-8),
            limits=(iterations=400, time=180.0, threads=1))
    catch exception
        "not_run: $(sprint(showerror, exception))"
    end

    phase_entries = (("first_compile", first_compile),
                     ("warm_fresh_setup", warm_fresh_setup),
                     ("prepared_solve", prepared_solve))
    with_samples = [name for (name, entry) in phase_entries if entry.runs > 0]
    without_samples = [name for (name, entry) in phase_entries if entry.runs == 0]
    budget = (
        limit_seconds=budget_seconds > 0.0 ? budget_seconds : nothing,
        status=isempty(stopped_phases) ? "within_budget" :
               (isempty(with_samples) ? "exceeded_before_any_phase" : "exceeded"),
        measured_seconds=spent[],
        budget_tripped_in=stopped_phases,
        phases_without_samples=without_samples,
        phases_with_samples=with_samples,
        reason=isempty(stopped_phases) ?
            (budget_seconds > 0.0 ? "completed within the per-case budget" :
             "no per-case budget was set (unbounded)") :
            "per-case measurement budget of $(budget_seconds) s exceeded after " *
            "$(round(spent[]; digits=3)) s of measured wall clock; the case is KEPT in the " *
            "manifest and every sample already taken is kept. Phases without samples were " *
            "not run, and are recorded as not_run rather than estimated.",
    )
    return (
        id=case.id,
        family=case.family,
        arithmetic=String(arithmetic),
        budget=budget,
        shape=(
            declared=shape_trace(case),
            observed=observed,
            declared_vs_observed=mismatches,
        ),
        inputs_fingerprint=fingerprint,
        objective_oracle=(
            status=case.known_objective === nothing ? "no_independent_oracle" : "declared",
            value=case.known_objective === nothing ? "not_run" : case.known_objective,
        ),
        phases=(
            first_compile=first_compile,
            warm_fresh_setup=warm_fresh_setup,
            prepared_solve=prepared_solve,
        ),
        phase_notes=phase_notes,
        bigfloat_allocation=bigfloat_allocation_block(rows, T),
        unavailable=sort(unique(unavailable)),
        all_rows=rows,
    )
end

# ---------------------------------------------------------------------------
# BigFloat allocation disaggregation (ADR-003 §7 / packet §5.6)
# ---------------------------------------------------------------------------

"""
    bigfloat_allocation_block(rows, T) -> NamedTuple

The four axes ADR-003 §7 requires to be distinguished, each with its own
provenance, and never one reported as another:

1. `julia_heap_bytes`  — GC-heap bytes allocated by the whole public call
                         (`@allocated`); includes every non-BigFloat allocation
                         too, so it is an upper bound on BigFloat object churn.
2. `cell_identity`     — `objectid` counts of the published BigFloat values:
                         distinct cells, aliased pairs, and how many cells a
                         repeat shares with its predecessor. `@allocated == 0`
                         alone would not establish that MPFR did not allocate;
                         this axis is about object identity, not bytes.
3. `native_allocator`  — **not measured**, with the reason. Julia exposes no
                         in-process counter for MPFR/GMP `malloc` traffic here;
                         `@allocated` observes the GC heap only. Reporting RSS as
                         this number would be exactly the defect ADR-003 §7 names.
4. `rss`               — peak-RSS counter before/after (`Sys.maxrss()`). It is a
                         PEAK counter: a zero delta means "no new peak", not
                         "no memory touched". Recorded as its own axis.
"""
function bigfloat_allocation_block(rows, ::Type{T}) where {T}
    solved = filter(row -> !row.threw && row.status == "optimal" &&
                          row.certificate_valid === true, rows)
    if T !== BigFloat
        return (
            status="not_applicable",
            reason="arithmetic is $(T): no BigFloat cells and no MPFR allocations exist",
            julia_heap_bytes=nothing,
            cell_identity=nothing,
            native_allocator_bytes=nothing,
            native_allocator_status="not_applicable",
            rss=nothing,
        )
    end
    median_of(field) = isempty(solved) ? nothing :
        median(getproperty(row, field) for row in solved)
    elements = [row.bigfloat_value_cells for row in solved if row.bigfloat_value_cells !== nothing]
    distinct = [row.bigfloat_distinct_cells for row in solved if row.bigfloat_distinct_cells !== nothing]
    return (
        status=isempty(solved) ? "not_run" : "measured",
        reason=isempty(solved) ?
            "no successful run to disaggregate; failures are kept in the rows" : "none",
        julia_heap_bytes=median_of(:allocated_bytes),
        julia_heap_note="GC-heap bytes for the whole public call; includes non-BigFloat allocations",
        cell_identity=(
            elements=isempty(elements) ? nothing : Int(median(elements)),
            distinct_cells=isempty(distinct) ? nothing : Int(median(distinct)),
            aliased_pairs=(isempty(elements) || isempty(distinct)) ? nothing :
                Int(median(elements) - median(distinct)),
            note="objectid counts over published primal+dual values",
        ),
        native_allocator_bytes=nothing,
        native_allocator_status="not_run",
        native_allocator_reason="no in-process counter for MPFR/GMP native malloc in this Julia; @allocated observes the GC heap only, so RSS or heap bytes must not be reported as this axis",
        rss=(
            peak_delta_bytes=median_of(:rss_delta_bytes),
            note="peak-RSS counter delta; zero means no NEW peak, not zero memory use",
        ),
        resident_workspace_bytes=median_of(:workspace_bytes),
    )
end

# ---------------------------------------------------------------------------
# Arm and payload
# ---------------------------------------------------------------------------

"""Can this arithmetic arm run in this process? Returns `(ok, facts)`."""
function arm_readiness(arm, T)
    loaded = Dict{String,Bool}()
    reasons = String[]
    if T === nothing
        push!(reasons, "arithmetic types not resolvable: every package in " *
            "$(arm.provider_packages) must be loadable")
    end
    for name in arm.provider_packages
        loaded_module = load_provider!(name)
        loaded[name] = loaded_module !== nothing
        loaded_module === nothing && push!(reasons,
            "$name is not resolvable in this environment")
    end
    extension = arm.provider === :mfla ? :SDPXMultiFloatLinearAlgebraExt :
                arm.provider === :bfla ? :SDPXBigFloatLinearAlgebraExt : nothing
    extension_active = extension === nothing ? true :
        Base.get_extension(SDPX, extension) !== nothing
    if !extension_active
        push!(reasons, "SDPX extension $extension is not active: the provider " *
            "package is present but SDPX did not load its adapter")
    end
    ambient = T === BigFloat ? Base.precision(BigFloat) : nothing
    return isempty(reasons), (
        loaded=loaded,
        extension=extension === nothing ? "not_applicable" : String(extension),
        extension_active=extension_active,
        ambient_bigfloat_precision=ambient,
        reasons=reasons,
    )
end

function _parse_args(args)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || continue
        pair = split(arg[3:end], "="; limit=2)
        length(pair) == 2 && (options[pair[1]] = pair[2])
    end
    return options
end

"""
    prepare_stage(options) -> NamedTuple

Stage 1. Resolves the arithmetic arm and **loads its provider packages**.
This has to happen before the measurement runs, and outside the world age the
measurement will use: `MultiFloat{Float64,2}(::Float64)` does not exist until the
provider is loaded, so calling it from code entered before the load fails with a
world-age error. `Base.invokelatest` at both stage boundaries (see the bottom of
this file) is what makes a dynamically resolved arithmetic type usable.
"""
function prepare_stage(options)
    arm = arithmetic_arm(Symbol(get(options, "arithmetic", "float64")))
    T = resolve_arithmetic(arm.id)
    if T === BigFloat
        setprecision(BigFloat, arm.precision_bits)
    end
    ready, arm_facts = if T === nothing
        (false, (loaded=Dict{String,Bool}(), extension="unknown",
                 extension_active=false, ambient_bigfloat_precision=nothing,
                 reasons=["arithmetic types not resolvable"]))
    else
        arm_readiness(arm, T)
    end
    return (arm=arm, T=T, ready=ready, facts=arm_facts)
end

"""    run_stage(options, stage) -> exit code

Stage 2. Everything that touches the resolved arithmetic type.
"""
function run_stage(options, stage)
    repeats = something(tryparse(Int, get(options, "repeats", "3")), 3)
    repeats >= 1 || (repeats = 1)
    budget_seconds = something(tryparse(Float64, get(options, "case-budget-seconds", "0")), 0.0)
    budget_seconds > 0.0 || (budget_seconds = 0.0)
    arithmetic = stage.arm.id
    out = get(options, "out", joinpath(@__DIR__, "measure_result.toml"))
    if haskey(options, "env-label")
        ENV["SDPX_MEASURE_ENV_LABEL"] = options["env-label"]
    end
    arm = stage.arm
    T = stage.T
    ready = stage.ready
    arm_facts = stage.facts

    # The two-process rule: one arithmetic per process, single-threaded.
    if Threads.nthreads() != 1
        println(stderr, "measure.jl: this harness must run with -t1 (got " *
            "Threads.nthreads()=$(Threads.nthreads())). Julia 1.12 can exhaust " *
            "its inference compiler when provider specialisations compile " *
            "together, and the thread payload must be unambiguous.")
        return 2
    end

    environment = environment_facts()
    source_fingerprint_at_start = source_fingerprint()
    contention_at_start = contention_facts()
    if get(ENV, "SDPX_MEASURE_REQUIRE_IDLE", get(options, "require-idle", "0")) == "1" &&
       _contended(contention_at_start)
        println(stderr, "measure.jl: refusing to measure: the host is not idle " *
            "(other_julia_processes=$(contention_at_start.other_julia_processes), " *
            "loadavg_1m=$(contention_at_start.loadavg_1m)). " *
            "Unset SDPX_MEASURE_REQUIRE_IDLE to measure anyway and record the contention.")
        return 4
    end

    header = (
        schema=SCHEMA,
        generated=string(now()),
        sdpx_head=_head_sha(),
        worktree_dirty=_worktree_dirty(),
        cases_fingerprint=cases_fingerprint(rebuild_cases()),
        environment=environment,
        source_fingerprint_at_start=source_fingerprint_at_start,
        contention_rule=CONTENTION_RULE,
        contention_at_start=contention_at_start,
        arithmetic=(
            id=String(arm.id),
            julia_type=T === nothing ? "not_run" : string(T),
            precision_bits=arm.precision_bits,
            provider=String(arm.provider),
            provider_packages=arm.provider_packages,
            rounding_mode=arm.rounding_mode,
            note=arm.note,
            arm_status=ready ? "runnable" : "not_run",
            arm_status_reason=isempty(arm_facts.reasons) ? "none" :
                join(arm_facts.reasons, "; "),
            loaded=arm_facts.loaded,
            extension=arm_facts.extension,
            extension_active=arm_facts.extension_active,
            ambient_bigfloat_precision=arm_facts.ambient_bigfloat_precision,
        ),
        threads=_thread_facts(1),
        tolerances=(primal=1e-8, dual=1e-8, gap=1e-8),
        limits=(iterations=400, time=180.0, threads=1),
        repeats=repeats,
        case_budget_seconds=budget_seconds > 0.0 ? budget_seconds : nothing,
        phases=("first_compile", "warm_fresh_setup", "prepared_solve"),
        phase_semantics=(
            first_compile="first solve of the case in this process, structure cache cleared; compilation included",
            warm_fresh_setup="repeats with clear_structure_cache!() before each solve: compiled code warm, symbolic structure rebuilt",
            prepared_solve="repeats with the cross-solve structure cache left warm: frozen symbolic structure reused",
        ),
        phase_evidence=(
            discriminator="structure_cache_hits_delta / structure_cache_misses_delta per phase",
            caveat="the cross-solve structure cache is the ONLY difference between " *
                "warm_fresh_setup and prepared_solve. It does not eliminate the " *
                "provider symbolic analysis: symbolic_analyses_delta counts REAL " *
                "CHOLMOD/QDLDL factor-construction analyses and is 1 per solve in " *
                "BOTH phases on the bordered route. On these shapes the timing " *
                "difference between the two phases is within run-to-run noise and " *
                "no conclusion may be drawn from it.",
        ),
        prepared_update_replay_status="not_run",
        prepared_update_replay_reason="a prepared-update replay (new objective/RHS into an existing prepared object) needs " *
            "the S04/S07 prepared-update API, which does not exist at this baseline",
        allocation_schema=(
            julia_heap="GC-heap bytes of the whole public call (@allocated)",
            cell_identity="objectid sets over published BigFloat values",
            native_allocator="not measured; no in-process MPFR/GMP malloc counter exists here",
            rss="Sys.maxrss() peak counter delta; zero means no new peak",
        ),
    )

    if !ready
        payload = merge(header, (
            cases=NamedTuple[],
            warm_totals=(total=0, solved=0, failed=0, threw=0, without_certificate=0),
        ))
        @printf("Q01 measure — arm %s NOT RUNNABLE in this environment\n", arm.id)
        for reason in arm_facts.reasons
            @printf("  reason: %s\n", reason)
        end
        @printf("  environment=%s project=%s\n", environment.label,
            something(environment.project_path, "unknown"))
        open(out, "w") do io
            TOML.print(io, _tomlify(payload); sorted=true)
        end
        @printf("  wrote %s\n", out)
        return 3
    end

    cases = rebuild_cases()
    selected = get(options, "cases", "")
    if !isempty(selected)
        wanted = Set(Symbol(strip(id)) for id in split(selected, ","))
        cases = filter(case -> case.id in wanted, cases)
    end

    @printf("Q01 measure — SDPX %s (dirty=%s)\n", first(header.sdpx_head, 12),
        header.worktree_dirty)
    @printf("  environment=%s  project=%s\n", environment.label,
        something(environment.project_path, "unknown"))
    @printf("  arithmetic=%s (%s, %d bits, provider=%s)\n", arm.id,
        header.arithmetic.julia_type, arm.precision_bits, arm.provider)
    @printf("  threads: requested=%d julia=%d blas=%d cpu=%d\n",
        header.threads.requested_threads, header.threads.julia_threads,
        header.threads.blas_threads, header.threads.cpu_threads)
    @printf("  cases_fingerprint=%d  repeats=%d\n", header.cases_fingerprint, repeats)
    @printf("  source_fingerprint=%s\n", first(header.source_fingerprint_at_start, 16))
    @printf("  contention: other_julia=%s loadavg_1m=%s cpu_threads=%d -> contended=%s\n",
        something(header.contention_at_start.other_julia_processes, "not_run"),
        something(header.contention_at_start.loadavg_1m, "not_run"),
        header.contention_at_start.cpu_threads,
        _contended(header.contention_at_start))
    @printf("  %-18s %-22s %-22s %-22s %s\n", "case",
        "first_compile", "warm_fresh_setup", "prepared_solve", "cache h/m/sym")
    cell(phase) = @sprintf("%s (%d/%d)", _fmt(phase.seconds), phase.solved, phase.runs)
    results = NamedTuple[]
    for case in cases
        contention_before = contention_facts()
        @printf("  case %s: measuring\n", case.id)
        flush(stdout)
        entry = merge(measure_case(case, T, arithmetic; repeats=repeats,
                budget_seconds=budget_seconds),
            (contention=(
                before_case=contention_before,
                after_case=contention_facts(),
                contended_before_case=_contended(contention_before),
                rule=CONTENTION_RULE,
            ),))
        push!(results, entry)
        # Stream a partial artifact after every case: a bounded or interrupted arm
        # must leave the evidence it already paid for, not lose it with the
        # process. `partial` says whether more cases were still to come.
        partial_payload = merge(header, (
            partial=true,
            cases_completed=length(results),
            cases_planned=length(cases),
            cases=results,
        ))
        open(out, "w") do io
            TOML.print(io, _tomlify(partial_payload); sorted=true)
        end
        fresh = entry.phases.warm_fresh_setup
        prepared = entry.phases.prepared_solve
        @printf("  %-18s %-22s %-22s %-22s %d/%d/%d\n", entry.id,
            cell(entry.phases.first_compile), cell(fresh), cell(prepared),
            prepared.structure_cache_hits_delta,
            prepared.structure_cache_misses_delta,
            prepared.symbolic_analyses_delta)
    end
    all_rows = vcat([collect(entry.all_rows) for entry in results]...)
    summary = summarize(all_rows)
    # Attribution guard: sibling packet tasks commit to the provider checkouts
    # DURING a run. A revision read once at the start is not evidence for the
    # whole run, so both ends are recorded and a change is flagged.
    environment_end = environment_facts()
    source_fingerprint_at_end = source_fingerprint()
    provider_revisions_changed = [(p["name"], p["revision"], q["revision"])
        for (p, q) in zip(header.environment.providers, environment_end.providers)
        if p["revision"] != q["revision"]]
    payload = merge(header, (
        partial=false,
        cases_completed=length(results),
        cases_planned=length(cases),
        contention_at_end=contention_facts(),
        source_fingerprint_at_end=source_fingerprint_at_end,
        source_changed_during_run=source_fingerprint_at_end !=
            header.source_fingerprint_at_start,
        sdpx_head_at_end=_head_sha(),
        sdpx_head_changed_during_run=_head_sha() != header.sdpx_head,
        provider_revisions_at_end=[(name=p["name"], revision=p["revision"],
            worktree_dirty=p["worktree_dirty"]) for p in environment_end.providers],
        provider_revisions_changed_during_run=provider_revisions_changed,
        cases=results,
        warm_totals=(
            total=summary.total, solved=summary.solved, failed=summary.failed,
            threw=summary.threw, without_certificate=summary.without_certificate,
        ),
    ))
    @printf("  rows: total=%d solved=%d failed=%d threw=%d without_certificate=%d\n",
        summary.total, summary.solved, summary.failed, summary.threw,
        summary.without_certificate)
    for provider in header.environment.providers
        provider["resolved"] || continue
        @printf("  provider %-26s version=%s revision=%s dirty=%s\n",
            provider["name"], something(provider["version"], "not_run"),
            something(provider["revision"], "not_run"),
            something(provider["worktree_dirty"], "not_run"))
    end
    isempty(provider_revisions_changed) || @printf(
        "  WARNING provider revision changed during the run: %s\n",
        join(["$(n): $(a) -> $(b)" for (n, a, b) in provider_revisions_changed], "; "))
    payload.sdpx_head_changed_during_run && @printf(
        "  WARNING SDPX HEAD changed during the run: %s -> %s\n",
        first(payload.sdpx_head, 12), first(payload.sdpx_head_at_end, 12))
    payload.source_changed_during_run && @printf(
        "  WARNING package source changed during the run: %s -> %s\n",
        first(payload.source_fingerprint_at_start, 16),
        first(payload.source_fingerprint_at_end, 16))
    open(out, "w") do io
        TOML.print(io, _tomlify(payload); sorted=true)
    end
    @printf("  wrote %s\n", out)
    return 0
end

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

"""
    _tomlify(value) -> Any

Convert a result payload into a structure `TOML.print` accepts: nested
`Dict{String,Any}` with string keys and scalar leaves.

Two rules from ADR-003 §3 are enforced here rather than left to the caller:

* `nothing` becomes the literal string `"not_run"`, never `0` — an unmeasured
  value must remain distinguishable from a measured zero after serialisation.
  Every nullable field that matters is accompanied by an explicit status/reason
  field, so `"not_run"` never has to carry the explanation by itself.
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

if abspath(PROGRAM_FILE) == @__FILE__
    options = _parse_args(ARGS)
    # Two-stage entry, both stages through `invokelatest`: stage 1 loads the
    # provider packages, stage 2 runs in the world that includes their methods.
    stage = Base.invokelatest(prepare_stage, options)
    exit(Base.invokelatest(run_stage, options, stage))
end
