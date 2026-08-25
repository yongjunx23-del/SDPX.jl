module PhysicsBenchmarkHarness

using SDPX
using Dates
using LinearAlgebra
using SHA
using TOML

export PhysicsBenchmarkReference, PhysicsBenchmarkSpec, PhysicsBenchmarkEntry
export PhysicsBenchmarkCatalog, catalog_entries, catalog_spec
export build_problem, validate_result, load_catalog
export RESULT_COLUMNS, RESULT_SCHEMA_VERSION, write_results
export run_suite, compare_result_files, main

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, ".."))

const MULTIFLOAT_TYPES = let types = Dict{Symbol,DataType}()
    try
        @eval import MultiFloats
        types[:float64x2] = MultiFloats.Float64x2
        types[:float64x3] = MultiFloats.Float64x3
        types[:float64x4] = MultiFloats.Float64x4
    catch
        empty!(types)
    end
    types
end

Base.@kwdef struct PhysicsBenchmarkReference
    status::Symbol = :optimal
    objective::Any = nothing
    absolute_tolerance::Float64 = 1.0e-8
    relative_tolerance::Float64 = 1.0e-8
    note::String = ""
end

Base.@kwdef struct PhysicsBenchmarkSpec
    id::String
    name::String
    family::Symbol
    problem_type::Symbol
    source::Symbol = :physics
    purpose::Symbol = :regression
    seed::Union{Nothing,Int} = nothing
    parameters::NamedTuple = (;)
    tags::Tuple{Vararg{Symbol}} = ()
    reference::PhysicsBenchmarkReference = PhysicsBenchmarkReference()
    fingerprint::String
end

struct PhysicsBenchmarkEntry
    problem_id::String
    arithmetic::Symbol
    provider::Symbol
end

"""A benchmark catalog whose build and validation decisions are injected.

`build(spec, T)` must return a named tuple with at least `problem`, `expected`,
and `kind`. `validate(spec, built, result, metrics)` returns `nothing`, `true`,
or an empty collection on success; `false`, a string, or a collection of
strings reports catalog-specific semantic failures.
"""
struct PhysicsBenchmarkCatalog
    name::Symbol
    version::String
    specs::Dict{String,PhysicsBenchmarkSpec}
    suites::Dict{Symbol,Vector{PhysicsBenchmarkEntry}}
    build::Function
    validate::Function
end

function PhysicsBenchmarkCatalog(
    name::Symbol,
    version::AbstractString,
    specs::AbstractVector{PhysicsBenchmarkSpec},
    suites::AbstractDict,
    build::Function;
    validate::Function=(spec, built, result, metrics) -> nothing,
)
    table = Dict{String,PhysicsBenchmarkSpec}()
    for spec in specs
        isempty(spec.id) && throw(ArgumentError("benchmark id must be non-empty"))
        isempty(spec.fingerprint) && throw(ArgumentError(
            "benchmark $(spec.id) must declare a deterministic fingerprint",
        ))
        haskey(table, spec.id) && throw(ArgumentError(
            "duplicate benchmark id $(repr(spec.id))",
        ))
        table[spec.id] = spec
    end
    normalized = Dict{Symbol,Vector{PhysicsBenchmarkEntry}}()
    for (suite, raw_entries) in suites
        entries = PhysicsBenchmarkEntry[raw_entries...]
        isempty(entries) && throw(ArgumentError("suite $suite is empty"))
        for entry in entries
            haskey(table, entry.problem_id) || throw(ArgumentError(
                "suite $suite refers to unknown benchmark $(repr(entry.problem_id))",
            ))
        end
        normalized[Symbol(suite)] = entries
    end
    isempty(normalized) && throw(ArgumentError("catalog must contain a suite"))
    return PhysicsBenchmarkCatalog(
        name, String(version), table, normalized, build, validate,
    )
end

function catalog_entries(catalog::PhysicsBenchmarkCatalog, suite::Symbol)
    entries = get(catalog.suites, suite, nothing)
    entries === nothing && throw(ArgumentError(
        "unknown suite $suite; choices=$(join(sort!(string.(collect(keys(catalog.suites)))), ", "))",
    ))
    return copy(entries)
end

function catalog_spec(catalog::PhysicsBenchmarkCatalog, id::AbstractString)
    return get(catalog.specs, String(id)) do
        throw(KeyError("unknown benchmark id $(repr(id))"))
    end
end

build_problem(catalog::PhysicsBenchmarkCatalog, spec::PhysicsBenchmarkSpec,
              ::Type{T}) where {T} = catalog.build(spec, T)

function validate_result(catalog::PhysicsBenchmarkCatalog, spec, built, result, metrics)
    verdict = catalog.validate(spec, built, result, metrics)
    (verdict === nothing || verdict === true) && return String[]
    verdict === false && return ["catalog_validation"]
    verdict isa AbstractString && return [String(verdict)]
    verdict isa Symbol && return [string(verdict)]
    verdict isa Tuple || verdict isa AbstractVector || throw(ArgumentError(
        "catalog validator must return nothing, Bool, String, Symbol, Tuple, or Vector",
    ))
    return string.(collect(verdict))
end

_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

"""Load an injected catalog file defining `physics_benchmark_catalog()`."""
function load_catalog(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || throw(ArgumentError("catalog file does not exist: $absolute"))
    host = Module(gensym(:InjectedPhysicsBenchmarkCatalog))
    Core.eval(host, :(import Main.PhysicsBenchmarkHarness))
    Base.include(host, absolute)
    isdefined(host, :physics_benchmark_catalog) || throw(ArgumentError(
        "catalog file must define physics_benchmark_catalog()",
    ))
    catalog = Core.eval(
        host, :(Base.invokelatest(physics_benchmark_catalog)),
    )
    catalog isa PhysicsBenchmarkCatalog || throw(ArgumentError(
        "physics_benchmark_catalog() returned $(typeof(catalog)), expected PhysicsBenchmarkCatalog",
    ))
    return catalog
end

include("result_schema.jl")
include("runner_impl.jl")
include("compare_impl.jl")

end # module
