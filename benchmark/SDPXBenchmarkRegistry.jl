module SDPXBenchmarkRegistry

using SDPX
using LinearAlgebra
using Random
using SHA
using SparseArrays
using TOML
using Dates

# Load only the arithmetic type package while this benchmark module is being
# initialized. Keeping the optional provider packages unloaded preserves
# environment-independent `:auto` provider selection.
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

export BenchmarkSpec, ExternalSource, BenchmarkReference, SuiteEntry
export benchmark_registry, benchmark_spec, suite_names, suite_entries
export build_problem, prepare_external!, external_cache_status
export run_suite, compare_result_files, write_results, main

const ROOT = @__DIR__
const REPOSITORY = normpath(joinpath(ROOT, ".."))

struct ExternalSource
    project::String
    authoritative_url::String
    filename::String
    format::Symbol
    sha256::Union{Nothing,String}
    license_note::String
end

struct BenchmarkReference
    status::Symbol
    objective::Any
    absolute_tolerance::Float64
    relative_tolerance::Float64
    note::String
end

struct BenchmarkSpec
    id::String
    name::String
    family::Symbol
    problem_type::Symbol
    source::Symbol
    tiers::Tuple{Vararg{Symbol}}
    tags::Tuple{Vararg{Symbol}}
    purpose::Symbol
    seed::Union{Nothing,Int}
    loader::Symbol
    parameters::NamedTuple
    reference::BenchmarkReference
    size::NamedTuple
    external::Union{Nothing,ExternalSource}
end

struct SuiteEntry
    problem_id::String
    arithmetic::Symbol
    provider::Symbol
end

include("generators/problems.jl")
include("registry/public.jl")
include("registry/synthetic.jl")
include("registry/heavy.jl")

const REGISTRY = let entries = vcat(
        PUBLIC_SPECS,
        SYNTHETIC_SPECS,
        HEAVY_SPECS,
    )
    table = Dict{String,BenchmarkSpec}()
    for spec in entries
        haskey(table, spec.id) && error("duplicate benchmark id $(spec.id)")
        isempty(string(spec.purpose)) && error("benchmark $(spec.id) has no purpose")
        table[spec.id] = spec
    end
    table
end

include("suites/micro.jl")
include("suites/representative.jl")
include("suites/local_full.jl")
include("suites/heavy.jl")

const SUITES = Dict{Symbol,Vector{SuiteEntry}}(
    :micro => MICRO_SUITE,
    :representative => REPRESENTATIVE_SUITE,
    :local_full => LOCAL_FULL_SUITE,
    :heavy => HEAVY_SUITE,
)

include("cache.jl")
include("runner_impl.jl")
include("compare_impl.jl")

benchmark_registry() = sort!(collect(values(REGISTRY)); by=spec -> spec.id)
benchmark_spec(id::AbstractString) = get(REGISTRY, String(id)) do
    throw(KeyError("unknown benchmark id $(repr(id))"))
end
suite_names() = (:micro, :representative, :local_full, :heavy)

function suite_entries(name::Symbol)
    haskey(SUITES, name) || throw(ArgumentError(
        "unknown suite $name; choices=$(join(suite_names(), ", "))",
    ))
    return copy(SUITES[name])
end

function build_problem(spec::BenchmarkSpec, ::Type{T}) where {T}
    spec.source === :synthetic || throw(ArgumentError(
        "$(spec.id) is external metadata; prepare its cache and add a supported loader before execution",
    ))
    return build_generated_problem(spec.loader, T; spec.parameters...)
end

end # module
