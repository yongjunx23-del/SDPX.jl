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
export benchmark_registry, benchmark_spec, suite_names, campaign_names, suite_entries
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
include("generators/pathological.jl")
include("loaders/csdr_fixed_trace.jl")
include("loaders/netlib_mps.jl")
include("loaders/sdpa_sparse.jl")
include("loaders/cbf.jl")
include("loaders/external.jl")
include("registry/public.jl")
include("registry/synthetic.jl")
include("registry/pathological.jl")
include("registry/heavy.jl")
include("registry/full_unitarity_eft.jl")

const REGISTRY = let entries = vcat(
        PUBLIC_SPECS,
        SYNTHETIC_SPECS,
        PATHOLOGICAL_SPECS,
        HEAVY_SPECS,
        FULL_UNITARITY_EFT_SPECS,
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
include("suites/large.jl")
include("suites/heavy.jl")
include("suites/core_matrix.jl")

const SUITES = Dict{Symbol,Vector{SuiteEntry}}(
    :micro => MICRO_SUITE,
    :representative => REPRESENTATIVE_SUITE,
    :local_full => LOCAL_FULL_SUITE,
    :large => LARGE_SUITE,
    :heavy => HEAVY_SUITE,
    :core_matrix => CORE_MATRIX_SUITE,
)

include("cache.jl")
include("runner_impl.jl")
include("compare_impl.jl")

benchmark_registry() = sort!(collect(values(REGISTRY)); by=spec -> spec.id)
benchmark_spec(id::AbstractString) = get(REGISTRY, String(id)) do
    throw(KeyError("unknown benchmark id $(repr(id))"))
end
suite_names() = (:micro, :representative, :local_full, :large, :heavy)
campaign_names() = (:core_matrix,)

function suite_entries(name::Symbol)
    haskey(SUITES, name) || throw(ArgumentError(
        "unknown suite $name; choices=$(join((suite_names()..., campaign_names()...), ", "))",
    ))
    return copy(SUITES[name])
end

function build_problem(
    spec::BenchmarkSpec,
    ::Type{T};
    cache_dir=DEFAULT_CACHE,
) where {T}
    if spec.source === :synthetic
        startswith(string(spec.loader), "pathological_") &&
            return build_pathological_problem(spec.loader, T; spec.parameters...)
        return build_generated_problem(spec.loader, T; spec.parameters...)
    end
    status = external_cache_status(spec; cache_dir)
    status.loadable || throw(ArgumentError(
        "$(spec.id) is not executable: $(status.reason)",
    ))
    return build_external_problem(spec, T, status.path, status.checksum)
end

end # module
