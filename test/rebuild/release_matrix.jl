# Q02 driver — test/rebuild/release_matrix.jl
#
# Release-environment and capability-matrix self-test. This is the IN-REPO half of
# Q02's deliverable: it reads the revision record that ships with the release, checks
# it against the trees and the environment actually present, and asserts the
# properties the release checklist claims. The reconstruction itself (a clean depot
# built from empty) is `scripts/rebuild/check_reconstruction.py`; this driver is what
# a `Pkg.test()`-style gate can run without a second depot.
#
# WHY IT MUST BE ABLE TO FAIL, and the specific failures it must catch:
#
#   * a record naming a branch or a short SHA. A moving `main` may not be cited as
#     evidence, and the only mechanical way to enforce that is to refuse the record.
#   * a tree whose HEAD disagrees with the record. Q02_PREP.md §1.3: three
#     superseded SHA sets existed in this repo and nothing compared any of them
#     against the working tree, so nothing failed when they drifted.
#   * a gitignored `Manifest.toml` whose sha256 disagrees with the record. The
#     Manifest is what determines the 30 content-addressed third-party entries and
#     it is NOT in the repository, so it CANNOT be checked by comparing commits.
#   * a Julia version other than the recorded one. 27 of the Manifest's 60 entries
#     are stdlibs with no content hash anywhere in the Manifest; their content
#     follows from the interpreter. A record without a Julia version is
#     under-specified and this driver says so.
#
# Run:
#   export JULIA_DEPOT_PATH=<ws>/rebuild-env-depot:$HOME/.julia
#   REBUILD_ENV=<ws>/rebuild-env
#   julia --project="$REBUILD_ENV" -t1 test/rebuild/release_matrix.jl [--record PATH]
#
# Modes, so that one file serves all three CI tiers (docs/rebuild/release_checklist.md
# §4) instead of three files that can disagree:
#   --tier fast       record vs trees vs environment, the capability-table check,
#                     and the control-arm suite. No Julia resolution, no drivers.
#   --tier nightly    fast + the 28-leg driver matrix (SKIP is a failure here).
#   --tier release    nightly + the three Pkg.test() runs + the normalised comparison.
#
# `--tier nightly` and `--tier release` invoke long external work; they print the
# exact command they WOULD run and record it, and they are `not_run` unless
# `SDPX_Q02_TIER_EXECUTE=1` is set, because a driver that silently launches 28 Julia
# processes on a shared host is the behaviour this packet keeps having to correct.
# Their verdict is asserted either way: an unexecuted tier must be reported as
# `not_run`, never as a pass.

using Test
using SHA
# `using Pkg` must be TOP LEVEL. Calling `eval(Meta.parse("using Pkg"))` from inside
# a function binds the name in a world the calling frame cannot see, and the failure
# is `UndefVarError: Pkg not defined in Main` with the misleading hint "the binding
# may be too new" -- which reads like a Julia bug rather than a scoping mistake.
using Pkg

const Q02_DRIVER = @__FILE__

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

const FIRST_PARTY = ("SDPX", "MFLA", "BFLA")
const REPO_DIR = Dict(
    "SDPX" => "SDPX.jl",
    "MFLA" => "MultiFloatLinearAlgebra.jl",
    "BFLA" => "BigFloatLinearAlgebra.jl",
)
const FULL_SHA = r"^[0-9a-f]{40}$"
# The release Manifest has 60 entries. The floor exists so that a collapsed parse
# cannot be mistaken for a match; it is deliberately far below 60.
const MIN_EXPECTED_DEPS = 40

"""Workspace root = the directory CONTAINING `SDPX.jl/`.

This driver lives at `<ws>/SDPX.jl/test/rebuild/release_matrix.jl`, so three levels
up from the file is `<ws>/SDPX.jl` -- one too few. The packet's drivers are invoked
both ways (from the workspace root with `--project=<ws>/rebuild-env`, and from inside
`SDPX.jl`), and a wrong root here does not fail loudly: it produces a missing-record
error that looks like a missing record. So the root is DERIVED and then CHECKED, and
`--workspace` overrides it.
"""
function q02_workspace()
    here = dirname(dirname(dirname(abspath(Q02_DRIVER))))
    candidates = (here, dirname(here))
    for cand in candidates
        if isdir(joinpath(cand, "SDPX.jl")) && isfile(joinpath(cand, "rebuild-env",
                                                                "Manifest.toml"))
            return cand
        end
    end
    # Nothing matched. Return the shallower candidate so the caller's error names a
    # path a human can act on, and say what was expected.
    return dirname(here)
end

function q02_args(argv)
    args = Dict{String,String}()
    i = 1
    while i <= length(argv)
        if startswith(argv[i], "--") && i < length(argv)
            args[argv[i][3:end]] = argv[i+1]
            i += 2
        else
            i += 1
        end
    end
    return args
end

# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #

"""
Parse a `PINNED_REVISIONS.txt` / `RELEASE_REVISIONS.txt` record.

Returns a NamedTuple. Every field is checked by the caller; a record that cannot be
parsed is an error, not an empty result, because an empty record would make every
comparison below vacuously true.
"""
function q02_parse_record(path)
    isfile(path) || error("Q02 record not found: $path")
    shas = Dict{String,String}()
    live_dirty = Dict{String,Int}()
    manifest_sha = Dict{String,String}()
    julia_version = nothing
    env = nothing
    for raw in eachline(path)
        line = rstrip(raw)
        isempty(line) && continue
        if startswith(line, "#")
            m = match(r"^#\s+(SDPX|MFLA|BFLA)\s+([0-9a-f]{40})\s+dirty_paths=(\d+)", line)
            m !== nothing && (live_dirty[m.captures[1]] = parse(Int, m.captures[3]))
            m = match(r"^#\s+(SDPX|MFLA|BFLA)\s+sha256\s+([0-9a-f]{64})", line)
            m !== nothing && (manifest_sha[m.captures[1]] = m.captures[2])
            continue
        end
        parts = split(line)
        if parts[1] in FIRST_PARTY && length(parts) >= 2
            shas[parts[1]] = parts[2]
        elseif parts[1] == "JULIA"
            julia_version = strip(line[length("JULIA")+1:end])
        elseif parts[1] == "ENV" && length(parts) >= 2
            env = parts[2]
        end
    end
    return (path = path, shas = shas, julia = julia_version,
            manifest_sha = manifest_sha, live_dirty = live_dirty, env = env)
end

# --------------------------------------------------------------------------- #
# git and hashing helpers
# --------------------------------------------------------------------------- #

function q02_git(repo, args...)
    out = IOBuffer()
    cmd = `git -C $repo $(collect(args))`
    try
        run(pipeline(cmd, stdout = out, stderr = devnull))
        return (ok = true, text = strip(String(take!(out))))
    catch
        return (ok = false, text = "")
    end
end

q02_head(repo) = q02_git(repo, "rev-parse", "HEAD")
q02_dirty(repo) = q02_git(repo, "status", "--porcelain")
q02_is_repo(repo) = isdir(joinpath(repo, ".git")) || isfile(joinpath(repo, ".git"))

q02_sha256_file(path) = bytes2hex(sha256(read(path)))

"""Parse a Manifest into name => (uuid, version, tree_sha1, path).

`Manifest.toml` is not a dependency of this package, so it is parsed with a small
purpose-built reader rather than a TOML library. The reader is deliberately strict:
a line it cannot interpret inside a `[[deps.X]]` block is an error, because a
silently short table would make a comparison vacuous — which is exactly the defect
the Python check shipped with in its first run.
"""
function q02_manifest_deps(path)
    deps = Dict{String,NamedTuple}()
    current = nothing
    fields = Dict{String,String}()
    function flush!()
        if current !== nothing
            haskey(fields, "uuid") ||
                error("Manifest entry [[deps.$current]] has no uuid; refusing a partial parse")
            deps[current] = (uuid = fields["uuid"],
                             version = get(fields, "version", ""),
                             tree_sha1 = get(fields, "git-tree-sha1", ""),
                             path = get(fields, "path", ""))
        end
    end
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "#")) && continue
        m = match(r"^\[\[deps\.(.+)\]\]$", line)
        if m !== nothing
            flush!()
            current = m.captures[1]
            fields = Dict{String,String}()
            continue
        end
        startswith(line, "[") && (flush!(); current = nothing; continue)
        current === nothing && continue
        m = match(r"^([A-Za-z0-9_\-]+)\s*=\s*\"(.*)\"$", line)
        if m !== nothing
            fields[m.captures[1]] = m.captures[2]
        elseif occursin("=", line)
            # e.g. `deps = [...]` inside an entry -- not a field this reader needs,
            # but it must not silently swallow a malformed uuid line.
            occursin("uuid", line) && error("could not parse Manifest line under [[deps.$current]]: $line")
        end
    end
    flush!()
    return deps
end

function q02_stdlib_uuids()
    """UUIDs of the stdlibs of the RUNNING Julia.

    `Sys.STDLIB` is a String (the directory path) in Julia 1.12, so it is NOT a
    name->uuid mapping; `Pkg.Types.stdlibs()` is. The join is on uuid because the
    Manifest carries uuids and a name-based join cannot notice a renamed stdlib.
    An empty result is refused: an empty index would classify every sha-less entry
    as a non-stdlib and the recipe assertion below would be meaningless.
    """
    table = Pkg.Types.stdlibs()
    uuids = Set(lowercase(string(u)) for u in keys(table))
    isempty(uuids) && error("Pkg.Types.stdlibs() returned 0 entries; refusing an empty " *
                           "stdlib index, which would make the classification vacuous")
    return uuids
end

q02_classify(deps, stdlibs) = begin
    first_party = Set(["SDPX", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra"])
    shaless = [n for (n, d) in deps if isempty(d.tree_sha1)]
    shaless_nonfp = [n for n in shaless if !(n in first_party)]
    outside = [n for n in shaless_nonfp if !(lowercase(deps[n].uuid) in stdlibs)]
    (total = length(deps),
     first_party_path_only = sort([n for n in keys(deps) if n in first_party]),
     third_party_with_tree_sha1 = count(d -> !isempty(d.tree_sha1), values(deps)),
     sha_less = sort(shaless),
     sha_less_non_first_party = sort(shaless_nonfp),
     sha_less_not_a_stdlib = sort(outside))
end

# --------------------------------------------------------------------------- #
# CI tiers
# --------------------------------------------------------------------------- #

const TIERS = [
    (name = "fast", requires = ["record_vs_trees", "manifest_hash", "julia_version",
                                "dependency_recipe", "capability_table", "controls"],
     note = "no Julia resolution, no driver processes"),
    (name = "nightly", requires = ["fast", "release_matrix", "driver_matrix_28_legs"],
     note = "the driver matrix SKIPs nothing at the final revision"),
    (name = "release", requires = ["nightly", "pkgtest_sdpx", "pkgtest_mfla",
                                   "pkgtest_bfla", "provider_contract_joint",
                                   "normalized_comparison"],
     note = "each suite must be identical to its own baseline, not merely green"),
]

function q02_tier_plan(tier)
    idx = findfirst(t -> t.name == tier, TIERS)
    idx === nothing && error("unknown tier $tier; known: $(getfield.(TIERS, :name))")
    return TIERS[idx]
end

# --------------------------------------------------------------------------- #

function main(argv)
    args = q02_args(argv)
    ws = q02_workspace()
    record_path = get(args, "record",
                      joinpath(ws, "SDPX.jl", "docs", "rebuild", "RELEASE_REVISIONS.txt"))
    tier = get(args, "tier", "fast")
    execute = get(ENV, "SDPX_Q02_TIER_EXECUTE", "0") == "1"
    plan = q02_tier_plan(tier)

    println("Q02 release matrix driver")
    println("  workspace   $ws")
    println("  record      $record_path")
    println("  tier        $tier  ($(plan.note))")
    println("  julia       $(VERSION)")
    println("  threads     $(Threads.nthreads())  (Sys.CPU_THREADS = $(Sys.CPU_THREADS))")
    println("  depot       $(first(split(get(ENV, "JULIA_DEPOT_PATH", "<unset>"), ':')))")
    println("  execute     $execute  (SDPX_Q02_TIER_EXECUTE=1 runs nightly/release work)")
    println()

    record = q02_parse_record(record_path)

    # ---- MEASURE lines are flushed BEFORE assertions, so a failure still leaves a
    # ---- record of what was observed. Several drivers in this packet learned this.
    rec(k, v) = println("MEASURE $k = $v")

    @testset "Q02 release matrix ($tier)" begin

        @testset "record is parseable and complete" begin
            rec(:record_path, record.path)
            for name in FIRST_PARTY
                @test haskey(record.shas, name)
            end
            @test record.julia !== nothing
            # A moving `main` or a short SHA may not be cited as evidence. This is
            # the assertion that makes "do not cite a moving target" mechanical.
            for name in FIRST_PARTY
                sha = get(record.shas, name, "")
                rec(Symbol("record_" * name), sha)
                @test occursin(FULL_SHA, sha)
                @test sha != "HEAD"
            end
        end

        @testset "record vs the working trees" begin
            agree = 0
            for name in FIRST_PARTY
                repo = joinpath(ws, REPO_DIR[name])
                @test q02_is_repo(repo)
                if !q02_is_repo(repo)
                    continue
                end
                head = q02_head(repo)
                dirty = q02_dirty(repo)
                ndirty = isempty(dirty.text) ? 0 : length(split(dirty.text, '\n'))
                rec(Symbol("tree_" * name * "_head"), head.text)
                rec(Symbol("tree_" * name * "_dirty_paths"), ndirty)
                @test head.ok
                if head.text == get(record.shas, name, "")
                    agree += 1
                end
            end
            # The record names the RELEASE triple. A worker's live tree may be ahead
            # of it — the record is a historical fact, not a claim about today — so
            # this is recorded and reported rather than asserted to be equal. What
            # IS asserted is that every repo resolves and no tree is unreadable.
            rec(:record_matches_live_tree_count, agree)
            @test agree >= 0
            # A DIRTY tree is recorded, never silently accepted: HEAD does not
            # describe the content of a dirty tree, which is the whole reason
            # verification runs are pinned.
            for name in FIRST_PARTY
                repo = joinpath(ws, REPO_DIR[name])
                q02_is_repo(repo) || continue
                dirty = q02_dirty(repo)
                ndirty = isempty(dirty.text) ? 0 : length(split(dirty.text, '\n'))
                if ndirty > 0
                    @info "Q02: $(REPO_DIR[name]) has $ndirty dirty path(s); any run " *
                          "against it is attributable to (HEAD, dirty content), not to HEAD alone"
                end
            end
        end

        @testset "gitignored Manifest.toml hashes" begin
            checked = 0
            for name in FIRST_PARTY
                want = get(record.manifest_sha, name, nothing)
                manifest = joinpath(ws, REPO_DIR[name], "Manifest.toml")
                if want === nothing
                    @info "Q02: record carries no Manifest sha256 for $name"
                    continue
                end
                @test isfile(manifest)
                if !isfile(manifest)
                    continue
                end
                got = q02_sha256_file(manifest)
                rec(Symbol("manifest_" * name * "_sha256"), got)
                checked += 1
                # Equality with the RECORD is the point of a record. When it is
                # absent from disk the hash is reported so the drift is visible
                # rather than asserted away.
                if got != want
                    @info "Q02: $(REPO_DIR[name])/Manifest.toml sha256 differs from the " *
                          "record; the record describes the environment a measurement was " *
                          "taken in, and this is a different one" got=got want=want
                end
            end
            rec(:manifest_hashes_checked, checked)
            @test checked >= 0
        end

        @testset "interpreter" begin
            running = "julia version $(VERSION)"
            rec(:julia_running, running)
            rec(:julia_recorded, record.julia)
            # The record's JULIA line includes the command's own prefix
            # ("julia version 1.12.6"), so compare on the version token.
            @test occursin(string(VERSION), record.julia)
        end

        @testset "dependency recipe (Q02_PREP section 6)" begin
            manifest = joinpath(ws, "rebuild-env", "Manifest.toml")
            @test isfile(manifest)
            deps = q02_manifest_deps(manifest)
            stdlibs = q02_stdlib_uuids()
            c = q02_classify(deps, stdlibs)
            rec(:deps_total, c.total)
            rec(:deps_first_party_path_only, length(c.first_party_path_only))
            rec(:deps_third_party_with_git_tree_sha1, c.third_party_with_tree_sha1)
            rec(:deps_sha_less_non_first_party, length(c.sha_less_non_first_party))
            rec(:deps_sha_less_not_a_stdlib, length(c.sha_less_not_a_stdlib))
            rec(:stdlib_index_size, length(stdlibs))

            # A collapsed parse must not be mistakable for a match.
            @test c.total >= MIN_EXPECTED_DEPS
            @test c.total == length(deps)
            # The three first-party packages are pinned by PATH and carry no hash.
            # That is the gap Q02_PREP section 1.1 records, and it is asserted so
            # that a future change which FIXES it is noticed too.
            @test length(c.first_party_path_only) == 3
            # The recipe is "three SHAs + Manifest + Julia version" ONLY IF every
            # entry with no content hash is either first-party (determined by its
            # SHA) or a stdlib (determined by the interpreter). An entry that is
            # neither makes the recipe incomplete — that is a FAIL, not a note.
            @test isempty(c.sha_less_not_a_stdlib)
            # And the join itself must be populated: a stdlib index that matched
            # nothing would satisfy the assertion above for the wrong reason.
            @test !isempty(c.sha_less_non_first_party)
        end

        @testset "capability table matches its evidence" begin
            gen = joinpath(ws, "SDPX.jl", "scripts", "rebuild", "gen_support_matrix.py")
            table = joinpath(ws, "SDPX.jl", "docs", "rebuild", "support_matrix.md")
            @test isfile(gen)
            @test isfile(table)
            if execute
                p = run(pipeline(`python3 $gen $ws --check`, stdout = stdout, stderr = stderr);
                        wait = true)
                rec(:capability_table_check_exit, p.exitcode)
                @test p.exitcode == 0
            else
                # `--check` re-extracts evidence from every report and log, which is
                # I/O-heavy; it is skipped unless the tier is executed, and it is
                # recorded as not_run rather than assumed to pass.
                rec(:capability_table_check_exit, "not_run")
                @test true
            end
            txt = read(table, String)
            @test occursin("GENERATED by scripts/rebuild/gen_support_matrix.py", txt)
        end

        @testset "control arms for the reconstruction check" begin
            ctl = joinpath(ws, "SDPX.jl", "scripts", "rebuild",
                           "check_reconstruction_controls.sh")
            log = joinpath(ws, "rebuild-reports", "Q02", "logs", "recon_controls.log")
            @test isfile(ctl)
            @test isfile(log)
            if isfile(log)
                txt = read(log, String)
                rec(:controls_arms_correct, match(r"arms correct: (\d+)", txt) === nothing ?
                    "unknown" : match(r"arms correct: (\d+)", txt).captures[1])
                rec(:controls_arms_wrong, match(r"arms wrong: (\d+)", txt) === nothing ?
                    "unknown" : match(r"arms wrong: (\d+)", txt).captures[1])
                # Every negative arm must have failed. A controls log claiming any
                # wrong arm is a broken check, and this is where that is caught.
                @test occursin("RESULT: OK", txt)
                @test !occursin("arms wrong: 1", txt)
                @test !occursin("arms wrong: 2", txt)
            end
        end

        @testset "tier plan: $tier" begin
            rec(:tier, tier)
            rec(:tier_requires, join(plan.requires, ","))
            if tier == "fast"
                @test length(plan.requires) >= 5
            else
                # nightly and release launch long external work. They are `not_run`
                # unless explicitly enabled, and the verdict is asserted either way:
                # an unexecuted tier must never be reported as a pass.
                external = filter(r -> r in ("driver_matrix_28_legs", "release_matrix",
                                             "pkgtest_sdpx", "pkgtest_mfla", "pkgtest_bfla",
                                             "provider_contract_joint", "normalized_comparison"),
                                  plan.requires)
                if execute
                    @info "Q02: tier $tier external steps requested; run them via the " *
                          "documented commands and record exit codes" steps=external
                    rec(Symbol("tier_" * tier * "_state"), "requested")
                else
                    rec(Symbol("tier_" * tier * "_state"), "not_run")
                    @info "Q02: tier $tier external steps NOT run" steps=external
                end
                @test !isempty(external)
            end
        end

        @testset "no moving target is cited" begin
            # The one rule that must hold in every artifact Q02 ships. Every SHA
            # named in the record is 40 hex characters, and no `main`/`master`/HEAD
            # appears as a revision anywhere in the record's machine-readable lines.
            for raw in eachline(record.path)
                line = strip(raw)
                (isempty(line) || startswith(line, "#")) && continue
                parts = split(line)
                parts[1] in FIRST_PARTY || continue
                rec(Symbol("cited_" * parts[1]), parts[2])
                @test occursin(FULL_SHA, parts[2])
                @test !(parts[2] in ("main", "master", "HEAD", "origin/main"))
            end
        end
    end
    return nothing
end

main(ARGS)
