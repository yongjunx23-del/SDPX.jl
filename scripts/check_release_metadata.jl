using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT = TOML.parsefile(joinpath(ROOT, "Project.toml"))
const VERSION = string(PROJECT["version"])

function require_compat(path::String)
    project = TOML.parsefile(path)
    compat = get(project, "compat", Dict{String,Any}())
    haskey(compat, "julia") ||
        error("$(relpath(path, ROOT)) is missing [compat].julia")
    for dependency in keys(get(project, "deps", Dict{String,Any}()))
        haskey(compat, dependency) || error(
            "$(relpath(path, ROOT)) is missing a compat bound for $dependency",
        )
    end
end

citation = read(joinpath(ROOT, "CITATION.cff"), String)
citation_match = match(r"(?m)^version:\s*([^\s]+)\s*$", citation)
citation_match === nothing &&
    error("CITATION.cff has no top-level version")
citation_match.captures[1] == VERSION || error(
    "CITATION.cff version $(citation_match.captures[1]) " *
    "does not match Project.toml version $VERSION",
)

for relative in (
    "examples/Project.toml",
    "benchmark/benchenv/Project.toml",
    "bin/Project.toml",
    "docs/Project.toml",
)
    require_compat(joinpath(ROOT, relative))
end

println("Release metadata is consistent for SDPX v$VERSION.")
