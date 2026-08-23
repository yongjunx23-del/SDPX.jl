#!/usr/bin/env julia
# One-shot driver for the benchmark-driven optimization loop on a development
# machine. See benchmark/WORKFLOW.md for the narrative.
#
#   julia benchmark/optimize.jl baseline [--base=HEAD^] [--suites=ladder] [--samples=3]
#   julia benchmark/optimize.jl candidate [--suites=ladder] [--samples=3]
#   julia benchmark/optimize.jl compare   [--suites=ladder]
#   julia benchmark/optimize.jl record    [--suites=ladder]
#   julia benchmark/optimize.jl loop      [--base=HEAD^] [--suites=ladder] [--samples=3]
#
# `baseline` checks <base> out into work/optimize/baseline-tree, copies the
# current Manifest.toml (pinning identical dependency versions), instantiates
# against the shared depot, and runs the suites there. Everything else runs on
# the current tree. Compare uses allow_dirty=true: this is the local
# diagnostics path, and any claim built on it must be re-validated by the
# fresh-process protocol before it is treated as evidence.

module OptimizeDriver

using Printf

const JULIA = joinpath(Sys.BINDIR, Base.julia_exename())
const REGISTRY_INCLUDE = joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl")

mutable struct Options
    mode::String
    base::String
    suites::Vector{String}
    samples::Int
    dir::String
end

function parse_args(args)
    opts = Options("loop", "HEAD^", ["ladder"], 3,
                   joinpath("work", "optimize"))
    isempty(args) && (opts.mode = "loop"; return _finalize(opts))
    opts.mode = args[1]
    opts.mode in ("baseline", "candidate", "compare", "record", "loop") ||
        throw(ArgumentError("unknown mode $(opts.mode); expected baseline|candidate|compare|record|loop"))
    for argument in args[2:end]
        startswith(argument, "--") || throw(ArgumentError("unexpected positional $argument"))
        key, value = split(argument[3:end], "="; limit=2)
        if key == "base"
            opts.base = value
        elseif key == "suites"
            opts.suites = String.(split(value, ","; keepempty=false))
            isempty(opts.suites) && throw(ArgumentError("--suites is empty"))
        elseif key == "samples"
            opts.samples = parse(Int, value)
            opts.samples == 1 || opts.samples >= 3 ||
                throw(ArgumentError("samples must be 1 or >= 3"))
        elseif key == "dir"
            opts.dir = value
        else
            throw(ArgumentError("unknown option --$key"))
        end
    end
    return _finalize(opts)
end

function _finalize(opts)
    mkpath(opts.dir)
    return opts
end

_git(arguments...) = readchomp(Cmd(["git", "-C", pwd(), arguments...]))

function _run(cmd::Cmd; cwd=pwd())
    println("+ ", join(string.(cmd.exec), " "), "  (cwd=$cwd)")
    flush(stdout)
    return run(setenv(cmd, dir=cwd))
end

_julia(args::Cmd; cwd=pwd()) = _run(`$JULIA --startup-file=no $args`; cwd)

"Check out `base` into a detached worktree with the current Manifest pinned."
function ensure_baseline_tree(opts)
    base_sha = _git("rev-parse", opts.base)
    tree = abspath(joinpath(opts.dir, "baseline-tree"))
    marker = joinpath(tree, ".optimize-base")
    if isfile(marker) && strip(read(marker, String)) == base_sha
        println("baseline tree up to date at ", base_sha[1:12])
        return tree
    end
    if ispath(tree)
        _run(`git worktree remove --force $tree`)
    end
    _run(`git worktree add --detach $tree $base_sha`)
    manifest = joinpath(pwd(), "Manifest.toml")
    isfile(manifest) && cp(manifest, joinpath(tree, "Manifest.toml"); force=true)
    _julia(`--project=$tree -e 'using Pkg; Pkg.instantiate()'`; cwd=tree)
    open(marker, "w") do io
        println(io, base_sha)
    end
    return tree
end

function run_suites(opts, project::AbstractString, cwd::AbstractString, tag::String)
    outputs = Pair{String,String}[]
    for suite in opts.suites
        output = abspath(joinpath(opts.dir, "$tag-$suite.toml"))
        _julia(`--project=$project benchmark/runner.jl $(Symbol(suite)) --samples=$(opts.samples) --output=$output`;
               cwd)
        push!(outputs, suite => output)
    end
    return outputs
end

"""Print one summary line per row plus the worst-moving phase, sorted by drift."""
function summarize_comparisons(comparisons)
    entries = Tuple{String,String,Float64,String}[]
    for (suite, rows) in comparisons
        for row in rows
            total = row.total_seconds_ratio
            total === missing && continue
            worst_phase, worst_ratio = "", total
            for field in propertynames(row)
                name = String(field)
                endswith(name, "_seconds_ratio") || continue
                value = getproperty(row, field)
                value === missing && continue
                if abs(log(value)) > abs(log(worst_ratio))
                    worst_phase, worst_ratio = name, value
                end
            end
            label = row.problem_id  # already carries its own prefix
            push!(entries, (label, string(row.candidate_status),
                            total, worst_phase === "" ? "-" :
                            "$worst_phase=$(round(worst_ratio; digits=3))"))
        end
    end
    sort!(entries; by=e -> -abs(log(e[3])))
    println()
    println("== drift summary (sorted by |log ratio|) ==")
    for (label, status, total, phase) in entries
        @printf("%-42s %-10s total=%6.3f  worst=%s\n", label, status, total, phase)
    end
    regressions = count(e -> e[3] > 1.05, entries)
    improvements = count(e -> e[3] < 0.95, entries)
    semantic = count(e -> e[2] != "Optimal", entries)
    println("rows=", length(entries), " regressions(>1.05)=", regressions,
            " improvements(<0.95)=", improvements, " non-optimal=", semantic)
    println(regressions == 0 && semantic == 0 ?
            "verdict: no regression detected -- human decision as planned" :
            "verdict: inspect the drifted phases above before deciding")
    return entries
end

function run_compare(opts, baseline_outputs, candidate_outputs)
    isdefined(@__MODULE__, :SDPXBenchmarkRegistry) ||
        include(REGISTRY_INCLUDE)
    candidate_by_suite = Dict(candidate_outputs)
    comparisons = Pair{String,Vector{Any}}[]
    for (suite, base_file) in baseline_outputs
        cand_file = get(candidate_by_suite, suite, nothing)
        cand_file === nothing && continue
        output = abspath(joinpath(opts.dir, "compare-$suite.tsv"))
        # compare_result_files is defined by the runtime include above, so the
        # call must pass through invokelatest (same pattern as runner_impl).
        rows = Base.invokelatest(
            SDPXBenchmarkRegistry.compare_result_files,
            base_file, cand_file; output=output, allow_dirty=true,
        )
        println("compared ", length(rows), " rows for $suite -> ", output)
        push!(comparisons, suite => rows)
    end
    isempty(comparisons) && error("nothing compared; run baseline and candidate first")
    return summarize_comparisons(comparisons)
end

function run_record(opts, candidate_outputs)
    for (suite, file) in candidate_outputs
        _julia(`benchmark/history_log.jl record $file --suite=$suite`)
    end
    return nothing
end

function main(args=ARGS)
    opts = parse_args(args)
    candidate_outputs = Pair{String,String}[]
    baseline_outputs = Pair{String,String}[]
    if opts.mode in ("baseline", "loop")
        tree = ensure_baseline_tree(opts)
        baseline_outputs = run_suites(opts, tree, tree, "baseline")
    end
    if opts.mode in ("candidate", "loop")
        candidate_outputs = run_suites(opts, pwd(), pwd(), "candidate")
    end
    if opts.mode == "compare" || opts.mode == "loop"
        isempty(baseline_outputs) &&
            (baseline_outputs = [(s, abspath(joinpath(opts.dir, "baseline-$s.toml")))
                                 for s in opts.suites])
        isempty(candidate_outputs) &&
            (candidate_outputs = [(s, abspath(joinpath(opts.dir, "candidate-$s.toml")))
                                  for s in opts.suites])
        run_compare(opts, baseline_outputs, candidate_outputs)
    end
    if opts.mode == "record" || opts.mode == "loop"
        isempty(candidate_outputs) &&
            (candidate_outputs = [(s, abspath(joinpath(opts.dir, "candidate-$s.toml")))
                                  for s in opts.suites])
        run_record(opts, candidate_outputs)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

end # module
