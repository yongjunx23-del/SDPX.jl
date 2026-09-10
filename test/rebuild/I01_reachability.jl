# I01 acceptance 3 — reachability of the new modules, measured.
#
#   julia --project="$REBUILD_ENV" SDPX.jl/test/rebuild/I01_reachability.jl
#
# WHY A SCRIPT AND NOT A PARAGRAPH. Twice in this rebuild a reachability claim
# was asserted from reading rather than running and both times it was wrong:
# "the seven unlisted files are not loaded" (they are, transitively) and "the
# `git cat-file` failure proves the blob never existed" (it did not prove that).
# So this file measures the two things a claim needs:
#
#   LOAD reachability   — is the name defined in the loaded module at all?
#   CALL reachability   — is there a call site in source that SDPX actually loads?
#
# and it reports the second as a *call-site list*, not a boolean, because a
# boolean invites the reader to assume the call is on the production path.
#
# What this script deliberately does NOT claim: that a call site executes. A call
# site inside a branch that no default configuration reaches is still not
# production reachability. Only the driver matrix and the inherited suite speak
# to execution; this script bounds the question to "could it be reached without
# changing an include".

# @__DIR__ is SDPX.jl/test/rebuild.
const SDPX_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SRC = joinpath(SDPX_ROOT, "src")

using SDPX

# The 14 files whose `include` I01 section A added, in the order added.
const SECTION_A = [
    "core/compiled_problem.jl",
    "core/transforms.jl",
    "kkt/operator.jl",
    "kkt/session.jl",
    "kkt/strategy.jl",
    "kkt/refinement_policy.jl",
    "la/protocol.jl",
    "la/admission.jl",
    "la/factor_lease.jl",
    "solver/loop.jl",
    "certification/original.jl",
    "planning/costs.jl",
    "planning/resources.jl",
    "planning/setup.jl",
]

# The files section A did NOT list. Named here so the record shows they were
# checked rather than overlooked.
const NOT_IN_SECTION_A = [
    "certification/status.jl",
    "certification/direction.jl",
    "solver/iterate.jl",
    "solver/session.jl",
    "solver/residuals.jl",
    "solver/globalization.jl",
    "solver/recovery.jl",
]

# Files a `include` pulls in but section A's list does not name directly. Every
# path here was found by grepping for `include` targets, not assumed.
const TRANSITIVE_ONLY = Dict(
    "certification/status.jl" => "certification/original.jl:1408 (inside `module SDPXCertification`)",
    "certification/direction.jl" => "certification/original.jl:1409 (inside `module SDPXCertification`)",
    "solver/iterate.jl" => "solver/loop.jl:31",
    "solver/session.jl" => "solver/loop.jl:32",
    "solver/residuals.jl" => "solver/loop.jl:33",
    "solver/globalization.jl" => "solver/loop.jl:34",
    "solver/recovery.jl" => "solver/loop.jl:35",
)

const DEF_RE = r"^\s*(?:@inline\s+|@noinline\s+)?(?:function\s+([A-Za-z_][A-Za-z0-9_!]*)|(?:mutable\s+)?struct\s+([A-Za-z_][A-Za-z0-9_!]*)|abstract\s+type\s+([A-Za-z_][A-Za-z0-9_!]*)|const\s+([A-Za-z_][A-Za-z0-9_!]*)|@enum\s+([A-Za-z_][A-Za-z0-9_!]*))"m

"Names a file defines near the top of its nesting. Comments are stripped first,
so a name that only appears in prose is not mistaken for a definition."
function defined_names(path::AbstractString)
    text = read(path, String)
    text = replace(text, r"(?m)^\s*#.*$" => "")
    text = replace(text, r"(?s)#=.*?=#" => "")
    names = Set{String}()
    for m in eachmatch(DEF_RE, text)
        for g in m.captures
            g === nothing || push!(names, String(g))
        end
    end
    return sort!(collect(names))
end

"Every `.jl` under `src/` that `src/SDPX.jl` reaches, by following `include`
targets from the entry point. Files that exist but are never included are
excluded — that is the whole point of the function."
function loaded_source_files()
    loaded = Set{String}()
    queue = [joinpath(SRC, "SDPX.jl")]
    while !isempty(queue)
        file = pop!(queue)
        file = normpath(file)
        (isfile(file) && !(file in loaded)) || continue
        push!(loaded, file)
        text = read(file, String)
        for m in eachmatch(r"include\(\s*(?:joinpath\(@__DIR__,\s*)?\"([^\"]+)\"", text)
            push!(queue, normpath(joinpath(dirname(file), m.captures[1])))
        end
    end
    return sort!(collect(loaded))
end

"Call sites for `name` in the loaded sources, excluding `exclude`. A call site is
`name(` with no preceding `.` (which would be a different module's binding) and no
preceding identifier character."
function call_sites(name::AbstractString, loaded, exclude)
    pattern = Regex("(?<![A-Za-z0-9_!.])" * escape_string(name) * "\\s*\\(")
    sites = String[]
    for file in loaded
        file in exclude && continue
        for (i, line) in enumerate(eachline(file))
            startswith(strip(line), '#') && continue
            occursin(pattern, line) || continue
            push!(sites, string(relpath(file, SDPX_ROOT), ":", i))
        end
    end
    return sites
end

"""
Every module reachable from `root` by submodule nesting, keyed by qualified name.

WHY THIS REPLACED A FILE-BASED SCAN. The first version of this script flagged a
"duplicate definition" whenever two loaded files both contained a definition of
the same name, and it produced 109 hits. Every one of them was a false positive:
`src/` opens 17 submodules, so `LorentzCone` in `cone_algebra.jl` lives in
`ConeAlgebra` while `LorentzCone` in `modeling/domains.jl` lives in `SDPX`;
`refactor_numeric!` is one generic function in `SDPX` with methods contributed by
two files; and `record!` in `chordal.jl` and `planning/setup.jl` are both local
closures inside functions. Text cannot see any of that. The loaded module tree
can.
"""
function module_inventory(root::Module)
    mods = Dict{String, Module}()
    seen = Set{Module}()
    stack = Module[root]
    while !isempty(stack)
        m = pop!(stack)
        m in seen && continue
        push!(seen, m)
        for n in names(m; all = true)
            startswith(String(n), "#") && continue
            isdefined(m, n) || continue
            v = getfield(m, n)
            if v isa Module && parentmodule(v) === m
                mods[join(fullname(v), ".")] = v
                push!(stack, v)
            end
        end
    end
    return mods
end

"Modules in which `name` resolves to a binding, as qualified names."
function defining_modules(name::String, mods::Dict{String, Module})
    sym = Symbol(name)
    return sort!([q for (q, m) in mods if isdefined(m, sym)])
end

function main()
    loaded = loaded_source_files()
    loaded_rel = [relpath(f, SDPX_ROOT) for f in loaded]
    println("== load reachability ==")
    println("files under src/: ", count(f -> endswith(f, ".jl"),
        [joinpath(r, x) for (r, _, xs) in walkdir(SRC) for x in xs]))
    println("files reached from src/SDPX.jl by include: ", length(loaded))
    println()

    println("== the 14 section-A entry points ==")
    println(rpad("file", 32), rpad("defs", 6), rpad("in SDPX", 9), rpad("in SDPXCertification", 22), "calls from other loaded files")
    for rel in SECTION_A
        path = joinpath(SRC, rel)
        names = defined_names(path)
        in_main = count(n -> isdefined(SDPX, Symbol(n)), names)
        in_cert = count(n -> isdefined(SDPX.SDPXCertification, Symbol(n)), names)
        sites = String[]
        for n in names
            append!(sites, call_sites(n, loaded, [path]))
        end
        unique!(sites)
        shown = isempty(sites) ? "(none)" : join(sites[1:min(end, 3)], ", ") *
            (length(sites) > 3 ? " (+$(length(sites) - 3) more)" : "")
        println(rpad(rel, 32), rpad(length(names), 6), rpad(in_main, 9), rpad(in_cert, 22), shown)
    end
    println()

    println("== files NOT listed by section A: are they loaded anyway? ==")
    for rel in NOT_IN_SECTION_A
        path = joinpath(SRC, rel)
        names = defined_names(path)
        in_main = [n for n in names if isdefined(SDPX, Symbol(n))]
        in_cert = [n for n in names if isdefined(SDPX.SDPXCertification, Symbol(n))]
        how = get(TRANSITIVE_ONLY, rel, "NOT FOUND")
        verdict = !isempty(in_main) ? "LOADED into SDPX" :
                  !isempty(in_cert) ? "LOADED into SDPX.SDPXCertification" :
                  "not loaded"
        println(rpad(rel, 30), rpad(string(length(names)) * " defs", 10),
                rpad(verdict, 38), "via ", how)
    end
    println()

    mods = module_inventory(SDPX)
    println("== module inventory ==")
    println("modules reachable from SDPX by nesting: ", length(mods))
    println()

    println("== can any section-A name be reached from two modules at once? ==")
    # The I01 hazard is not "two files mention the name"; it is "one name with two
    # independent homes", which is what makes an old definition silently win.
    #
    # `Base`/`Core` are excluded: `DEF_RE` matches the `Base` in a `Base.foo`
    # method definition line, so they were reported as defined in all 17 modules.
    # They are the standard library, not an I01 binding.
    module_noise = ("Base", "Core", "Main")
    ambiguous = Tuple{String, Vector{String}}[]
    for rel in SECTION_A
        for n in defined_names(joinpath(SRC, rel))
            n in module_noise && continue
            homes = defining_modules(n, mods)
            length(homes) > 1 && push!(ambiguous, (n, homes))
        end
    end
    unique!(ambiguous)
    if isempty(ambiguous)
        println("no section-A name resolves in more than one module")
    else
        for (n, homes) in sort(ambiguous)
            println("MULTI-HOME ", n, " -> ", join(homes, ", "))
        end
    end
    println()

    println("== the bindings section A introduced, as actually resolved ==")
    println(rpad("name", 30), rpad("kind", 10), rpad("methods", 9), rpad("home", 22), "also in submodule")
    reported = 0
    for rel in SECTION_A
        for n in defined_names(joinpath(SRC, rel))
            isdefined(SDPX, Symbol(n)) || continue
            v = getfield(SDPX, Symbol(n))
            kind = v isa Type ? "Type" : v isa Function ? "Function" :
                   v isa Module ? "Module" : string(typeof(v))
            nm = v isa Function ? string(length(methods(v))) : "-"
            others = setdiff(defining_modules(n, mods), ["SDPX"])
            # Only the interesting rows: Types (the shadowing class), and any name
            # that also lives in a submodule.
            (kind == "Type" || !isempty(others)) || continue
            reported += 1
            println(rpad(n, 30), rpad(kind, 10), rpad(nm, 9), rpad("SDPX", 22),
                    isempty(others) ? "-" : join(others, ", "))
        end
    end
    println("(types and multi-home names shown: ", reported, ")")
    println()

    println("== method-overwrite check ==")
    # A same-signature redefinition would not show up above at all; it needs a
    # runtime observer. `Pkg.test()` loads with `--warn-overwrite=yes`, which
    # prints "WARNING: Method definition ... overwritten". Counted separately, so
    # the record carries the number rather than the reassurance.
    println("This script cannot observe method overwrite. The evidence is the")
    println("`--warn-overwrite=yes` run of the inherited suite:")
    println("  SDPX.jl/rebuild-reports/I01_prework/SDPX_pkgtest_*.log")
    println()
    println("loaded_file_count = ", length(loaded))
    println("modules = ", length(mods))
    println("multi_home_names = ", length(ambiguous))
    return nothing
end

main()
