module PhysicsBenchmarkHarness

using SDPX
using Dates
using LinearAlgebra
using SHA
using Statistics
using TOML

export PhysicsBenchmarkReference, PhysicsBenchmarkSpec, PhysicsBenchmarkEntry
export PhysicsBenchmarkCatalog, catalog_entries, catalog_spec
export build_problem, validate_result, load_catalog
export RESULT_COLUMNS, RESULT_SCHEMA_VERSION, write_results
export run_suite, compare_result_files, main

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, "..", ".."))

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
struct PhysicsBenchmarkCatalog{B,V}
    name::Symbol
    version::String
    specs::Dict{String,PhysicsBenchmarkSpec}
    suites::Dict{Symbol,Vector{PhysicsBenchmarkEntry}}
    build::B
    validate::V
end

function PhysicsBenchmarkCatalog(
    name::Symbol,
    version::AbstractString,
    specs::AbstractVector{PhysicsBenchmarkSpec},
    suites::AbstractDict,
    build;
    validate=(spec, built, result, metrics) -> nothing,
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
              ::Type{T}) where {T} = Base.invokelatest(catalog.build, spec, T)

function validate_result(catalog::PhysicsBenchmarkCatalog, spec, built, result, metrics)
    # Catalogs are loaded into a fresh anonymous module at runtime.  Julia
    # 1.12's strict world-age rules therefore require the same dynamic-call
    # boundary here that the runner already uses for injected builders.
    verdict = Base.invokelatest(catalog.validate, spec, built, result, metrics)
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

# `load_catalog` records the injected catalog's exact source closure.  A
# directory-wide tree hash is deliberately not used: an unrelated file added
# beside a catalog must not change a benchmark identity, while a transitive
# include (including one with a non-`.jl` suffix) must change it.
mutable struct _CatalogIncludeContext
    host::Module
    root::String
    files::Vector{String}
    stack::Vector{String}
end

const _CATALOG_FILE_CONTEXT = IdDict{Any,_CatalogIncludeContext}()

function _path_under_root(path::AbstractString, root::AbstractString)
    relative = relpath(path, root)
    return relative == "." ||
           (relative != ".." && !startswith(relative, "../") &&
            !startswith(relative, "..\\"))
end

function _catalog_argument_error(exception)
    exception isa ArgumentError && return exception
    exception isa LoadError || return nothing
    return _catalog_argument_error(exception.error)
end

function _tracked_include(context::_CatalogIncludeContext, raw_path)
    raw_path isa AbstractString || throw(ArgumentError(
        "catalog include path must be a string, got $(typeof(raw_path))",
    ))
    base = isempty(context.stack) ? context.root : dirname(last(context.stack))
    candidate = isabspath(raw_path) ? String(raw_path) : joinpath(base, raw_path)
    isfile(candidate) || throw(ArgumentError(
        "catalog include does not name a file: $(abspath(candidate))",
    ))
    absolute = realpath(candidate)
    _path_under_root(absolute, context.root) || throw(ArgumentError(
        "catalog include escapes canonical root $(context.root): $absolute",
    ))
    absolute in context.files || push!(context.files, absolute)
    push!(context.stack, absolute)
    try
        # `mapexpr` rewrites both `include(...)` and explicit
        # `Base.include(...)` calls in the included source before evaluation.
        return Base.include(
            expression -> _rewrite_catalog_includes(
                expression, context,
            ),
            context.host,
            absolute,
        )
    catch exception
        argument_error = _catalog_argument_error(exception)
        argument_error === nothing || throw(argument_error)
        rethrow()
    finally
        pop!(context.stack)
    end
end

function _include_callee(callee)
    callee === :include && return true
    callee isa Expr && callee.head === :. || return false
    text = string(callee)
    return text == "Base.include" || text == "Core.include"
end

function _rewrite_catalog_includes(expression, context::_CatalogIncludeContext)
    expression isa Expr || return expression
    if expression.head === :call && !isempty(expression.args) &&
       _include_callee(expression.args[1])
        isempty(expression.args) && return expression
        path = _rewrite_catalog_includes(last(expression.args), context)
        return Expr(:call, :_sdpx_tracked_include, path)
    end
    return Expr(
        expression.head,
        (_rewrite_catalog_includes(argument, context)
         for argument in expression.args)...,
    )
end

function _catalog_manifest_sha256(context::_CatalogIncludeContext)
    paths = sort!(unique!(copy(context.files)))
    payload = IOBuffer()
    for source in paths
        current = realpath(source)
        _path_under_root(current, context.root) || throw(ArgumentError(
            "catalog source escaped canonical root $(context.root): $current",
        ))
        isfile(current) || throw(ArgumentError(
            "catalog source disappeared after load: $current",
        ))
        write(payload, replace(relpath(current, context.root), '\\' => '/'))
        write(payload, UInt8(0), read(current), UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

"""Load an injected catalog file defining `physics_benchmark_catalog()`."""
function load_catalog(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || throw(ArgumentError("catalog file does not exist: $absolute"))
    canonical = realpath(absolute)
    root = realpath(dirname(canonical))
    host = Module(gensym(:InjectedPhysicsBenchmarkCatalog))
    Core.eval(host, :(import Main.PhysicsBenchmarkHarness))
    context = _CatalogIncludeContext(host, root, String[], String[])
    Core.eval(host, :(
        _sdpx_tracked_include(path) =
            PhysicsBenchmarkHarness._tracked_include(
                __catalog_include_context__, path,
            )
    ))
    # Bind the context through a private host method.  A function body may be
    # world-age delayed, whereas a module constant is available to all mapped
    # expressions as soon as the first include is evaluated.
    Core.eval(host, :(
        const __catalog_include_context__ = $context
    ))
    _tracked_include(context, canonical)
    isdefined(host, :physics_benchmark_catalog) || throw(ArgumentError(
        "catalog file must define physics_benchmark_catalog()",
    ))
    catalog = Core.eval(
        host, :(Base.invokelatest(physics_benchmark_catalog)),
    )
    catalog isa PhysicsBenchmarkCatalog || throw(ArgumentError(
        "physics_benchmark_catalog() returned $(typeof(catalog)), expected PhysicsBenchmarkCatalog",
    ))
    _CATALOG_FILE_CONTEXT[catalog] = context
    return catalog
end

include("result_schema.jl")
include("runner_impl.jl")
include("compare_impl.jl")

end # module
