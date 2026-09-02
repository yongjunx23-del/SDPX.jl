module V2TargetBridge

using TOML
import SDPX

# This bridge intentionally does not merge the V2 branch into the optimizer
# branch. It consumes an explicitly pinned V2 checkout as an input artifact.
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const V2_ROOT = normpath(get(ENV, "SDPX_V2_ROOT", ""))

function _require_v2_root()
    isempty(V2_ROOT) && throw(ArgumentError("SDPX_V2_ROOT is required"))
    isdir(V2_ROOT) || throw(ArgumentError("SDPX_V2_ROOT is not a directory: $V2_ROOT"))
    git(args...) = readchomp(Cmd(vcat(["git", "-C", V2_ROOT], String[string(arg) for arg in args])))
    source = git("rev-parse", "HEAD")
    occursin(r"^[0-9a-f]{40}$", source) ||
        throw(ArgumentError("V2 source must be an exact checked-out SHA"))
    tree = git("rev-parse", "HEAD^{tree}")
    occursin(r"^[0-9a-f]{40}$", tree) ||
        throw(ArgumentError("V2 source tree must be an exact SHA"))
    return (commit=source, tree=tree)
end

function _load_v2()
    identity = _require_v2_root()
    v2file = joinpath(V2_ROOT, "benchmark", "general", "v2", "GeneralBenchmarkV2.jl")
    adapterfile = joinpath(V2_ROOT, "benchmark", "optimization", "v2_schema9_adapter.jl")
    isfile(v2file) && isfile(adapterfile) ||
        throw(ArgumentError("pinned V2 checkout lacks schema-v9 adapter files"))
    isdefined(Main, :GeneralBenchmarkV2) || Base.include(Main, v2file)
    isdefined(Main, :V2Schema9Adapter) || Base.include(Main, adapterfile)
    return identity, Main.GeneralBenchmarkV2, Main.V2Schema9Adapter
end

"""Select the reviewed Float64 execution declaration from the pinned V2 module.

The declaration is intentionally selected rather than reconstructed here: this
keeps the bridge coupled to the V2 review matrix (including its provider label,
bit width, and decimal tolerances) and prevents drift between the two
validators.
"""
function _reviewed_float64_precision(v2)
    specs = Base.invokelatest(getfield(v2, :reviewed_precision_specs))
    matches = filter(spec -> getfield(spec, :name) === :Float64, specs)
    length(matches) == 1 || throw(ArgumentError(
        "pinned V2 review matrix must contain exactly one Float64 declaration"))
    return only(matches)
end

"""Build one real V2 target through the pinned V2 adapter.

The adapter performs one excluded warmup and exactly three rebuilt measured
samples, including independent V2 status/certificate/oracle gates. This
bridge then applies the dependent optimizer's own `validate_profile_row` live
validator; no receipt field is inferred or filled by this module.
"""
function profile_first_target(; case_id::Symbol=:v2_lp_box_small)
    identity, v2, adapter = _load_v2()
    isdefined(Main, :ProfileCatalog) ||
        Base.include(Main, joinpath(ROOT, "benchmark", "optimization", "profile_catalog.jl"))
    P = Main.ProfileCatalog
    catalog = Base.invokelatest(getfield(v2, :lp_tranche_catalog))
    instance = only(filter(x -> x.id === case_id, catalog.instances))
    precision = _reviewed_float64_precision(v2)
    row = Base.invokelatest(getfield(adapter, :profile_v2_target), catalog, instance, precision;
        warmup=true, samples=3)
    Base.invokelatest(getfield(P, :validate_profile_row), row; live=true) ||
        throw(ArgumentError("dependent optimizer rejected V2 row as live evidence"))
    return (row=row, v2_commit=identity.commit, v2_tree=identity.tree)
end

"""Emit the V2 row as schema-v9 TSV/TOML and revalidate the projected row."""
function emit_first_target(path::AbstractString; case_id::Symbol=:v2_lp_box_small)
    result = profile_first_target(; case_id)
    paths = Base.invokelatest(getfield(Main.V2Schema9Adapter, :write_schema9), path, [result.row])
    isfile(paths.tsv) && isfile(paths.toml) ||
        throw(ArgumentError("V2 adapter did not emit both schema-v9 files"))
    document = TOML.parsefile(paths.toml)
    document["schema_version"] == 9 ||
        throw(ArgumentError("V2 adapter emitted non-schema-v9 document"))
    rows = get(document, "result", Any[])
    length(rows) == 1 || throw(ArgumentError("V2 adapter emitted unexpected row count"))
    return merge(result, (paths=paths, schema9=document))
end

"""Return an honest readiness receipt; this does not mutate GitHub variables."""
function readiness_receipt(result)
    row = result.row
    return Dict{String,Any}(
        "local_target_ready" => true,
        "target_case_key" => row.case_key,
        "target_schema_version" => 9,
        "target_live_validator" => true,
        "v2_source_commit" => result.v2_commit,
        "v2_source_tree" => result.v2_tree,
        "warmup_excluded" => row.warmup_count,
        "sample_count" => length(row.sample_seconds),
        "all_certificates_valid" => all(row.sample_certificate_valid),
        "all_semantic_pass" => all(row.sample_semantic_pass),
        "iteration_deterministic" => length(unique(row.sample_iterations)) == 1,
        "objective_deterministic" => length(unique(row.sample_objective)) == 1,
        "repository_variable" => "SDPX_ENABLE_DEPENDENT_OPTIMIZATION",
        "repository_variable_state" => "disabled_not_mutated_locally",
        "remaining_open" => [
            "catalog workflow publication on main",
            "fresh-process samples (current adapter is same_process_three_sample)",
            "full catalog breadth and external holdout completion",
            "Stage-B performance and memory gates",
        ],
    )
end

export profile_first_target, emit_first_target, readiness_receipt
end
