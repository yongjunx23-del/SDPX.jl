#!/usr/bin/env julia
# Stage-B fresh-process schema-v9 profile runner.
#
# Parent mode launches one excluded warmup child and three measured child
# processes. Child mode rebuilds and solves exactly one V2 target, then writes
# one atomic TOML receipt. No Serialization or solver/tolerance changes are
# involved. Include this file in tests without invoking main().
module V2FreshProcessProfile

using SHA
using TOML
using Dates
using Statistics
using LinearAlgebra
using SparseArrays
import SDPX

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SCRIPT = abspath(@__FILE__)

if !isdefined(Main, :GeneralBenchmarkV2)
    Base.include(Main, joinpath(ROOT, "benchmark", "general", "v2", "GeneralBenchmarkV2.jl"))
end
if !isdefined(Main, :ProfileCatalog)
    Base.include(Main, joinpath(ROOT, "benchmark", "optimization", "profile_catalog.jl"))
end
if !isdefined(Main, :V2Schema9Adapter)
    Base.include(Main, joinpath(ROOT, "benchmark", "optimization", "v2_schema9_adapter.jl"))
end
const V2 = Main.GeneralBenchmarkV2
const P = Main.ProfileCatalog
const A = Main.V2Schema9Adapter

const PROTOCOL_VERSION = 1
const FRESH_EXECUTION_MODE = "fresh_process_three_sample"

_sha_bytes(bytes) = bytes2hex(SHA.sha256(bytes))
_sha(value) = _sha_bytes(codeunits(string(value)))
_file_sha(path) = isfile(path) ? _sha_bytes(read(path)) : "missing"
function _directory_sha(root)
    io = IOBuffer()
    for (directory, _, names) in walkdir(root), name in sort(names)
        path = joinpath(directory, name)
        isfile(path) || continue
        write(io, relpath(path, root)); write(io, UInt8(0)); write(io, read(path))
    end
    _sha_bytes(take!(io))
end
_git(args...) = readchomp(Cmd(vcat(["git", "-C", ROOT], String[string(arg) for arg in args])))
_source_clean() = isempty(_git("status", "--porcelain"))
function _require_clean_source(stage)
    _source_clean() || throw(ArgumentError("source worktree became dirty at $stage"))
    true
end

function _arg(name::String, default=nothing)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function _canonical_destination(path::AbstractString)
    absolute = abspath(path)
    ispath(absolute) && throw(ArgumentError("refusing to overwrite existing artifact: $absolute"))
    parent = dirname(absolute)
    suffix = String[]
    while !ispath(parent)
        pushfirst!(suffix, basename(parent))
        next = dirname(parent)
        next == parent && throw(ArgumentError("no existing parent for artifact path: $absolute"))
        parent = next
    end
    resolved = realpath(parent)
    for component in suffix
        resolved = joinpath(resolved, component)
    end
    candidate = normpath(joinpath(resolved, basename(absolute)))
    root = realpath(ROOT)
    relative = relpath(candidate, root)
    inside = relative == "." || (!isabspath(relative) && relative != ".." &&
        !startswith(relative, ".." * Base.Filesystem.path_separator))
    inside && throw(ArgumentError("generated artifact resolves inside source worktree: $candidate"))
    candidate
end

function _sync_file(io)
    flush(io)
    result = Sys.iswindows() ? ccall(:_commit, Cint, (Cint,), Base.fd(io)) :
        ccall(:fsync, Cint, (Cint,), Base.fd(io))
    result == 0 || throw(SystemError("fsync artifact", Libc.errno()))
end

function _sync_directory(path)
    Sys.iswindows() && return
    fd = ccall(:open, Cint, (Cstring, Cint), path, 0)
    fd >= 0 || throw(SystemError("open artifact directory", Libc.errno()))
    try
        ccall(:fsync, Cint, (Cint,), fd) == 0 ||
            throw(SystemError("fsync artifact directory", Libc.errno()))
    finally
        ccall(:close, Cint, (Cint,), fd)
    end
end

function _atomic_bytes(path::AbstractString, bytes::Vector{UInt8})
    canonical = _canonical_destination(path)
    mkpath(dirname(canonical))
    # Re-resolve after mkdir so a symlink swap cannot redirect publication.
    canonical == _canonical_destination(canonical) || throw(ArgumentError(
        "artifact canonical path changed during preflight"))
    temporary = string(canonical, ".tmp.", getpid(), ".", rand(UInt))
    open(temporary, "x") do io
        write(io, bytes)
        _sync_file(io)
    end
    try
        mv(temporary, canonical; force=false)
        _sync_directory(dirname(canonical))
    catch
        rm(temporary; force=true)
        rethrow()
    end
    canonical
end

function _atomic_toml(path::AbstractString, value::Dict{String,Any})
    io = IOBuffer()
    TOML.print(io, value; sorted=true)
    _atomic_bytes(path, take!(io))
end

function _environment(precision)
    source_project = joinpath(ROOT, "Project.toml")
    source_manifest = joinpath(ROOT, "Manifest.toml")
    active_project = Base.active_project()
    active_project === nothing && throw(ArgumentError("fresh runner requires an active project"))
    active_manifest = joinpath(dirname(active_project), "Manifest.toml")
    isfile(active_project) || throw(ArgumentError("active Project.toml is missing"))
    isfile(active_manifest) || throw(ArgumentError("active Manifest.toml is missing"))
    return Dict{String,Any}(
        # Canonical runtime environment identity: the temp/developed project
        # and manifest used by every child. Source-project hashes are retained
        # separately; a missing repository Manifest is not misrepresented.
        "project_sha256" => _file_sha(active_project),
        "manifest_sha256" => _file_sha(active_manifest),
        "source_project_sha256" => _file_sha(source_project),
        "source_manifest_sha256" => _file_sha(source_manifest),
        "julia" => string(VERSION),
        "os" => string(Sys.KERNEL),
        "cpu" => string(Sys.CPU_NAME),
        "machine" => string(Sys.MACHINE),
        "julia_threads" => Threads.nthreads(),
        "gc_threads" => (isdefined(Threads, :ngcthreads) ? Threads.ngcthreads() :
            parse(Int, get(ENV, "JULIA_NUM_GC_THREADS", "1"))),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "omp_threads" => get(ENV, "OMP_NUM_THREADS", "unset"),
        "arithmetic" => string(precision.arithmetic),
        "precision_bits" => precision.bits,
        "provider" => string(precision.provider),
    )
end

function _route_dict(route)
    Dict(string(name) => string(getproperty(route, name)) for name in propertynames(route))
end

function _precision()
    specs = V2.reviewed_precision_specs()
    matches = filter(spec -> spec.name === :Float64, specs)
    length(matches) == 1 || throw(ArgumentError("reviewed Float64 declaration is not unique"))
    spec = only(matches)
    spec.arithmetic isa Type{<:AbstractFloat} || throw(ArgumentError(
        "fresh Stage-B runner requires a concrete Float64 execution declaration"))
    V2._validate_precision_spec(spec)
    spec
end

function _catalog_instance(case_id::Symbol)
    catalog = V2.lp_tranche_catalog()
    matches = filter(instance -> instance.id === case_id, catalog.instances)
    length(matches) == 1 || throw(ArgumentError("unknown or non-unique V2 case $case_id"))
    instance = only(matches)
    instance.split === :train || throw(ArgumentError("fresh profile requires a train instance"))
    get(instance.provenance, :solve_eligible, false) === true ||
        throw(ArgumentError("fresh profile requires a solve-eligible instance"))
    instance.reference.status === :optimal ||
        throw(ArgumentError("fresh profile requires an optimal reference"))
    catalog, instance
end

function _child_receipt(case_id::Symbol)
    _require_clean_source("child_start")
    precision = _precision()
    catalog, instance = _catalog_instance(case_id)
    blas_threads = parse(Int, get(ENV, "SDPX_BLAS_THREADS", "1"))
    SDPX.set_blas_threads!(blas_threads)
    outputs = SDPX.Outputs(:all, :all, :all; diagnostics=:full,
        certificate=:summary, objectives=true, history=false, trace=false)
    started = time_ns()
    result = V2.run_instance(catalog, instance, precision; outputs)
    total_seconds = (time_ns() - started) * 1e-9
    env = _environment(precision)
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    catalog_fp = V2.catalog_fingerprint(catalog)
    route = _route_dict(result.route_receipt)
    reference = instance.reference.objective_interval
    reference === nothing && throw(ArgumentError("optimal instance lacks independent objective interval"))
    lower, upper = reference
    solver_version = string(Base.pkgversion(SDPX))
    provider_version = string("CHOLMOD_BUILD_VERSION=", SparseArrays.CHOLMOD.BUILD_VERSION)
    provider_fp = _sha((precision.provider, precision.name, precision.bits,
                        string(precision.arithmetic), provider_version))
    environment_fp = _sha(env)
    driver_sha = _file_sha(SCRIPT)
    solver_source_sha = _directory_sha(joinpath(ROOT, "src"))
    harness_sha = _file_sha(joinpath(ROOT, "benchmark", "optimization", "profile_catalog.jl"))
    schema_sha = _file_sha(joinpath(ROOT, "benchmark", "bootstrap", "result_schema.jl"))
    contract_fp = _sha((instance.id, result.input_fingerprint, instance.reference,
                        instance.provenance.transform))
    campaign_id = _sha((commit, tree, catalog_fp, instance.id, precision.name,
                        FRESH_EXECUTION_MODE))
    validation = result.validation
    semantic = validation.reference && isempty(validation.failures)
    _require_clean_source("child_after_solve")
    receipt = Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "pid" => getpid(),
        "source_commit" => commit,
        "source_dirty" => false,
        "tree_fingerprint" => tree,
        "catalog" => String(catalog.name),
        "family" => String(instance.family),
        "instance" => String(instance.id),
        "case_key" => join((catalog.name, instance.family, instance.tier.name,
                              instance.id, precision.name), "|"),
        "input_fingerprint" => result.input_fingerprint,
        "execution_fingerprint" => result.execution_fingerprint,
        "catalog_artifact_sha256" => catalog_fp,
        "project_sha256" => env["project_sha256"],
        "manifest_sha256" => env["manifest_sha256"],
        "source_project_sha256" => env["source_project_sha256"],
        "source_manifest_sha256" => env["source_manifest_sha256"],
        "environment_fingerprint" => environment_fp,
        "provider_fingerprint" => provider_fp,
        "provider" => string(precision.provider),
        "provider_version" => provider_version,
        "solver_version" => solver_version,
        "precision_name" => string(precision.name),
        "precision_bits" => precision.bits,
        "solver_tolerance" => precision.solver_tolerance,
        "certificate_limit" => precision.certificate_limit,
        "environment" => env,
        "benchmark_driver_sha256" => driver_sha,
        "solver_source_sha256" => solver_source_sha,
        "harness_source_sha256" => harness_sha,
        "schema_source_sha256" => schema_sha,
        "contract_fingerprint" => contract_fp,
        "campaign_id" => campaign_id,
        "shard_id" => "local",
        "shard_index" => 1,
        "shard_count" => 1,
        "external_checksum" => result.input_fingerprint,
        "status" => string(result.status),
        "certificate_valid" => result.certificate_valid,
        "validation_certificate" => validation.certificate,
        "validation_reference" => validation.reference,
        "semantic_pass" => semantic,
        "validation_failures" => string.(validation.failures),
        "certificate_kind" => "optimal",
        "certificate_failures" => string.(validation.failures),
        "objective" => result.objective,
        "dual_objective" => result.dual_objective,
        "primal_residual" => result.primal_residual,
        "dual_residual" => result.dual_residual,
        "relative_gap" => result.relative_gap,
        "certificate_metrics" => Dict(string(name) => value
            for (name, value) in pairs(result.certificate_metrics)),
        "objective_interval" => Dict("lower" => lower, "upper" => upper),
        "reference_objective" => string((parse(BigFloat, lower) + parse(BigFloat, upper)) / BigFloat(2)),
        "iterations" => result.iterations,
        "total_seconds" => total_seconds,
        "route_receipt" => route,
        "maxrss_bytes" => Int(Sys.maxrss()),
        "execution_mode" => "single_child_process",
    )
    for (key, value) in (("setup_seconds", result.setup_seconds),
                         ("core_seconds", result.core_seconds),
                         ("recovery_seconds", result.recovery_seconds),
                         ("allocated_bytes", result.allocated_bytes))
        value === nothing || (receipt[key] = value)
    end
    receipt
end

function child_main()
    output = _arg("output")
    case_text = _arg("case-id", "v2_lp_box_small")
    output === nothing && throw(ArgumentError("--output=PATH is required in child mode"))
    case_id = Symbol(case_text)
    receipt = _child_receipt(case_id)
    _atomic_toml(output, receipt)
    println("CHILD_RECEIPT_WRITTEN path=", output, " pid=", receipt["pid"])
end

function _required_identity(receipt)
    keys = ("protocol_version", "source_commit", "tree_fingerprint", "catalog",
            "family", "instance", "case_key", "input_fingerprint",
            "execution_fingerprint", "catalog_artifact_sha256", "project_sha256",
            "manifest_sha256", "benchmark_driver_sha256", "solver_source_sha256",
            "harness_source_sha256", "schema_source_sha256", "contract_fingerprint",
            "campaign_id", "shard_id", "shard_index", "shard_count",
            "environment", "environment_fingerprint", "provider_fingerprint",
            "provider", "provider_version", "precision_name", "precision_bits",
            "solver_tolerance", "certificate_limit", "route_receipt")
    all(haskey(receipt, key) for key in keys) || return false
    receipt["protocol_version"] == PROTOCOL_VERSION || return false
    occursin(r"^[0-9a-f]{40}$", receipt["source_commit"]) || return false
    occursin(r"^[0-9a-f]{40}$", receipt["tree_fingerprint"]) || return false
    for key in ("input_fingerprint", "execution_fingerprint",
                "catalog_artifact_sha256", "project_sha256", "manifest_sha256",
                "benchmark_driver_sha256", "solver_source_sha256",
                "harness_source_sha256", "schema_source_sha256",
                "contract_fingerprint", "environment_fingerprint", "provider_fingerprint")
        occursin(r"^[0-9a-f]{64}$", receipt[key]) || return false
    end
    receipt["precision_bits"] isa Integer && receipt["precision_bits"] > 0 || return false
    receipt["shard_index"] isa Integer && receipt["shard_count"] isa Integer &&
        1 <= receipt["shard_index"] <= receipt["shard_count"] || return false
    receipt["environment"] isa AbstractDict || return false
    receipt["route_receipt"] isa AbstractDict || return false
    true
end

function _child_hash(path)
    _sha_bytes(read(path))
end

function _read_child(path)
    isfile(path) || throw(ArgumentError("child artifact missing: $path"))
    bytes = read(path)
    digest = _sha_bytes(bytes)
    receipt = try
        TOML.parse(String(bytes))
    catch error
        throw(ArgumentError("child artifact TOML parse failed for $path: $(sprint(showerror, error))"))
    end
    _required_identity(receipt) || throw(ArgumentError(
        "child artifact failed identity/protocol validation: $path"))
    (receipt=receipt, sha256=digest, path=String(path))
end

function _launch_child(path, case_id)
    active = Base.active_project()
    active === nothing && throw(ArgumentError("parent must run under an active project"))
    julia = Base.julia_cmd()
    cmd = `$julia --startup-file=no --gcthreads=1 --project=$active $SCRIPT --child --case-id=$(String(case_id)) --output=$path`
    threads = get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads()))
    blas_threads = get(ENV, "SDPX_BLAS_THREADS", "1")
    run(setenv(cmd,
        "JULIA_NUM_THREADS" => threads,
        "JULIA_NUM_GC_THREADS" => "1",
        "SDPX_BLAS_THREADS" => blas_threads,
        "OPENBLAS_NUM_THREADS" => blas_threads,
        "OMP_NUM_THREADS" => get(ENV, "OMP_NUM_THREADS", "1")))
    _require_clean_source("parent_after_child")
end

function _median_or_nothing(values)
    all(x -> x !== nothing, values) || return nothing
    vals = Float64[x for x in values]
    median(vals)
end

function aggregate_child_receipts(warmup, measured, catalog, instance, precision;
                                  child_paths=String[], child_hashes=String[],
                                  warmup_path::String="", warmup_hash::String="")
    length(measured) == 3 || throw(ArgumentError("exactly three measured child receipts are required"))
    all(_required_identity, [warmup; measured]) || throw(ArgumentError("invalid child protocol receipt"))
    pids = [Int(r["pid"]) for r in [warmup; measured]]
    length(unique(pids)) == 4 || throw(ArgumentError("warmup and measured child PIDs must be distinct"))
    length(child_paths) == 3 && length(child_hashes) == 3 || throw(ArgumentError(
        "three durable measured child artifacts and hashes are required"))
    all(isfile, child_paths) || throw(ArgumentError("measured child artifact path is not durable"))
    all(hash -> occursin(r"^[0-9a-f]{64}$", hash), child_hashes) || throw(ArgumentError(
        "invalid measured child artifact hash"))
    all(_child_hash(path) == hash for (path, hash) in zip(child_paths, child_hashes)) ||
        throw(ArgumentError("measured child artifact hash mismatch"))
    isfile(warmup_path) || throw(ArgumentError("warmup child artifact path is not durable"))
    occursin(r"^[0-9a-f]{64}$", warmup_hash) || throw(ArgumentError(
        "invalid warmup child artifact hash"))
    _child_hash(warmup_path) == warmup_hash || throw(ArgumentError(
        "warmup child artifact hash mismatch"))
    identities = ("protocol_version", "source_commit", "tree_fingerprint", "catalog",
        "family", "instance", "case_key", "input_fingerprint", "execution_fingerprint",
        "catalog_artifact_sha256", "project_sha256", "manifest_sha256",
        "benchmark_driver_sha256", "solver_source_sha256", "harness_source_sha256",
        "schema_source_sha256", "contract_fingerprint", "campaign_id", "shard_id",
        "shard_index", "shard_count", "environment", "environment_fingerprint",
        "provider_fingerprint", "provider", "provider_version", "precision_name", "precision_bits",
        "solver_tolerance", "certificate_limit", "route_receipt")
    for key in identities
        first_value = measured[1][key]
        all(r -> r[key] == first_value, measured) ||
            throw(ArgumentError("measured child identity differs for $key"))
        warmup[key] == first_value || throw(ArgumentError("warmup identity differs for $key"))
    end
    warmup["status"] == "optimal" && warmup["certificate_valid"] === true &&
        warmup["validation_certificate"] === true && warmup["validation_reference"] === true &&
        warmup["semantic_pass"] === true && isempty(warmup["validation_failures"]) ||
        throw(ArgumentError("excluded warmup child failed status/certificate/reference/semantic gates"))
    all(r -> r["status"] == "optimal" && r["certificate_valid"] === true &&
            r["validation_certificate"] === true && r["validation_reference"] === true &&
            r["semantic_pass"] === true && isempty(r["validation_failures"]), measured) ||
        throw(ArgumentError("every measured child must pass status/certificate/reference/semantic gates"))
    objectives = [String(r["objective"]) for r in measured]
    iterations = [Int(r["iterations"]) for r in measured]
    objectives == fill(objectives[1], 3) || throw(ArgumentError("measured objective strings are nondeterministic"))
    iterations == fill(iterations[1], 3) || throw(ArgumentError("measured iterations are nondeterministic"))
    String(warmup["objective"]) == objectives[1] || throw(ArgumentError(
        "warmup objective differs from measured children"))
    Int(warmup["iterations"]) == iterations[1] || throw(ArgumentError(
        "warmup iterations differ from measured children"))
    interval = measured[1]["objective_interval"]
    lower, upper = String(interval["lower"]), String(interval["upper"])
    expected_interval = instance.reference.objective_interval
    expected_interval === nothing && throw(ArgumentError("instance lacks independent reference interval"))
    (lower, upper) == (String(expected_interval[1]), String(expected_interval[2])) ||
        throw(ArgumentError("child reference interval differs from immutable catalog reference"))
    public_half_width = abs(parse(BigFloat, upper) - parse(BigFloat, lower)) / BigFloat(2)
    allowance = max(public_half_width, parse(BigFloat, precision.certificate_limit))
    center = (parse(BigFloat, lower) + parse(BigFloat, upper)) / BigFloat(2)
    all(r -> begin
        value = parse(BigFloat, String(r["objective"]))
        center - allowance <= value <= center + allowance
    end, measured) || throw(ArgumentError("measured objectives leave the independent reference interval/precision allowance"))
    totals = Float64[r["total_seconds"] for r in measured]
    all(x -> isfinite(x) && x > 0, totals) || throw(ArgumentError("measured total timings are invalid"))
    get_optional(r, key) = haskey(r, key) ? Float64(r[key]) : nothing
    setups = [get_optional(r, "setup_seconds") for r in measured]
    cores = [get_optional(r, "core_seconds") for r in measured]
    recoveries = [get_optional(r, "recovery_seconds") for r in measured]
    allocs = [haskey(r, "allocated_bytes") ? Int(r["allocated_bytes"]) : nothing for r in measured]
    all(x -> x === nothing || (isfinite(x) && x >= 0), [setups; cores; recoveries]) ||
        throw(ArgumentError("optional phase timings are invalid"))
    all(x -> x !== nothing && x >= 0, allocs) ||
        throw(ArgumentError("measured allocation timing is unavailable; refusing to encode it as zero"))
    rss = [Int(r["maxrss_bytes"]) for r in measured]
    all(>(0), rss) || throw(ArgumentError("measured peak RSS must be positive"))
    first_result = measured[1]
    route = first_result["route_receipt"]
    row_receipt = Dict{String,Any}(first_result)
    # Schema-v9 execution_mode is the semantic operation; process isolation
    # is a separate receipt fact.
    row_receipt["execution_mode"] = "profile"
    row_receipt["process_isolation"] = FRESH_EXECUTION_MODE
    row_receipt["fresh_process"] = true
    row_receipt["warmup_excluded"] = 1
    row_receipt["sample_count"] = 3
    row_receipt["sample_pids"] = Int[r["pid"] for r in measured]
    row_receipt["warmup_pid"] = Int(warmup["pid"])
    row_receipt["sample_total_seconds"] = totals
    row_receipt["sample_setup_seconds"] = Any[has_key_or_missing(r, "setup_seconds") for r in measured]
    row_receipt["sample_core_seconds"] = Any[has_key_or_missing(r, "core_seconds") for r in measured]
    row_receipt["sample_recovery_seconds"] = Any[has_key_or_missing(r, "recovery_seconds") for r in measured]
    row_receipt["sample_allocated_bytes"] = Any[has_key_or_missing(r, "allocated_bytes") for r in measured]
    row_receipt["sample_peak_rss_bytes"] = rss
    row_receipt["process_peak_rss_bytes"] = maximum(rss)
    row_receipt["child_artifact_sha256"] = child_hashes
    row_receipt["child_artifact_paths"] = child_paths
    row_receipt["warmup_artifact_sha256"] = warmup_hash
    row_receipt["warmup_artifact_path"] = warmup_path
    row_receipt["phase_accounting_complete"] = false
    row_receipt["production_invariants_valid"] = "not_declared_by_api"
    row_receipt["full_numerical_gate_valid"] = true
    row_receipt["catalog_validation_pass"] = true
    row_receipt["reference_objective"] = first_result["reference_objective"]
    row_receipt["actual_objective"] = objectives
    row_receipt["catalog_run_id"] = _sha((first_result["source_commit"],
        first_result["tree_fingerprint"], first_result["catalog_artifact_sha256"],
        first_result["case_key"], FRESH_EXECUTION_MODE))
    environment = first_result["environment"]
    for key in ("cpu", "julia_threads", "blas_threads", "omp_threads", "gc_threads")
        row_receipt[key] = environment[key]
    end
    row_receipt["resolved_tolerances"] = Dict("primal"=>precision.solver_tolerance,
        "dual"=>precision.solver_tolerance, "gap"=>precision.solver_tolerance)
    row_receipt["trajectory_semantics"] = "not_applicable"
    row_receipt["trajectory_reason"] = "V2 target has no published per-iterate trajectory"
    row_receipt["certificate_kind"] = "optimal"
    row_receipt["certificate_failures"] = String[]
    row_receipt["sample_certificate_metrics"] = [r["certificate_metrics"] for r in measured]
    row_receipt["sample_primal_residual"] = [r["primal_residual"] for r in measured]
    row_receipt["sample_dual_residual"] = [r["dual_residual"] for r in measured]
    row_receipt["sample_relative_gap"] = [r["relative_gap"] for r in measured]
    row_receipt["objective_error"] = [string(abs(parse(BigFloat, x) - parse(BigFloat, first_result["reference_objective"]))) for x in objectives]
    for key in ("requested_route", "planned_route", "executed_route",
                "requested_formulation", "planned_formulation", "executed_formulation",
                "requested_backend", "planned_backend", "executed_backend",
                "requested_provider", "planned_provider", "executed_provider",
                "requested_kernel", "planned_kernel", "executed_kernel", "reuse")
        row_receipt[key] = route_value(route, key)
    end
    reference_objective = Float64(parse(BigFloat, first_result["reference_objective"]))
    sample_allocs = Union{Nothing,Int}[x for x in allocs]
    transform_fingerprint = _sha(instance.provenance.transform)
    row_receipt["transform_fingerprint"] = transform_fingerprint
    P.ProfileRow(case_key=String(first_result["case_key"]), catalog=String(catalog.name),
        id=String(instance.id), family=String(instance.family), tier=String(instance.tier.name),
        arithmetic=String(precision.name), solve_eligible=true, build_only=false,
        source=instance.source, status="optimal", certificate_valid=true, semantic_pass=true,
        objective=parse(Float64, objectives[1]), iterations=iterations[1], sample_seconds=totals,
        sample_core_seconds=Union{Nothing,Float64}[x for x in cores],
        setup_seconds=_median_or_nothing(setups), allocation_bytes=Int[x for x in sample_allocs],
        sample_iterations=iterations, peak_rss_bytes=maximum(rss), reference_status="optimal",
        reference_objective=reference_objective,
        objective_tolerance=parse(Float64, precision.certificate_limit),
        transform_exactness=String(instance.provenance.transform.exactness),
        transform_fingerprint=transform_fingerprint, requested_route=route_value(route,"requested_route"),
        planned_route=route_value(route,"planned_route"), executed_route=route_value(route,"executed_route"),
        input_fingerprint=String(first_result["input_fingerprint"]), source_commit=String(first_result["source_commit"]),
        tree_fingerprint=String(first_result["tree_fingerprint"]), catalog_fingerprint=String(first_result["catalog_artifact_sha256"]),
        environment_fingerprint=String(first_result["environment_fingerprint"]), provider_fingerprint=String(first_result["provider_fingerprint"]),
        trajectory_reason="V2 target has no published per-iterate trajectory", warmup_count=1,
        sample_status=[String(r["status"]) for r in measured],
        sample_certificate_valid=Bool[r["certificate_valid"] for r in measured],
        sample_semantic_pass=Bool[r["semantic_pass"] for r in measured], sample_objective=Float64[parse(Float64, x) for x in objectives],
        sample_trajectory_sha=String[], reference_lower=parse(Float64, lower), reference_upper=parse(Float64, upper),
        resolved_tolerances=string(Dict("primal"=>precision.solver_tolerance,"dual"=>precision.solver_tolerance,"gap"=>precision.solver_tolerance)),
        receipt=row_receipt)
end

# TOML has no null literal. Missing optional values are represented by a
# deliberately explicit string in the parent receipt and remain `nothing` in
# ProfileRow/schema-v9 aggregation; they are never replaced by zero.
has_key_or_missing(r, key) = haskey(r, key) ? r[key] : "not_declared_by_api"
route_value(route, key) = get(route, key, "not_declared_by_api")

function _validate_schema_semantics(path)
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        Base.include(Main, joinpath(ROOT, "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl"))
    end
    harness = Main.PhysicsBenchmarkHarness
    rows = Base.invokelatest(getfield(harness, :compare_result_files), path, path)
    length(rows) == 1 || throw(ArgumentError("schema semantic validator returned unexpected row count"))
    getproperty(only(rows), :comparison_valid) === true || throw(ArgumentError(
        "schema-v9 row failed repository semantic comparison validation"))
    true
end

function parent_main()
    _require_clean_source("parent_start")
    case_id = Symbol(_arg("case-id", "v2_lp_box_small"))
    requested_prefix = _arg("prefix", joinpath(tempdir(), "SDPX_STAGE_B_FRESH_PROFILE"))
    receipt_path = _canonical_destination(string(requested_prefix, ".receipt.toml"))
    prefix = chop(receipt_path; tail=length(".receipt.toml"))
    final_tsv = _canonical_destination(string(prefix, ".tsv"))
    final_toml = _canonical_destination(string(prefix, ".toml"))
    catalog, instance = _catalog_instance(case_id)
    precision = _precision()
    work = string(prefix, ".children")
    ispath(work) && throw(ArgumentError("refusing to overwrite existing child artifact directory: $work"))
    mkpath(work)
    warmup_path = joinpath(work, "warmup.toml")
    measured_paths = [joinpath(work, "sample_$(i).toml") for i in 1:3]
    _launch_child(warmup_path, case_id)
    warmup_snapshot = _read_child(warmup_path)
    for path in measured_paths
        _launch_child(path, case_id)
    end
    measured_snapshots = [_read_child(path) for path in measured_paths]
    # Bind aggregation to the exact bytes parsed above; fail if any artifact
    # changes before publication.
    _child_hash(warmup_path) == warmup_snapshot.sha256 || throw(ArgumentError(
        "warmup artifact changed after parse"))
    all(_child_hash(snapshot.path) == snapshot.sha256 for snapshot in measured_snapshots) ||
        throw(ArgumentError("measured child artifact changed after parse"))
    measured = [snapshot.receipt for snapshot in measured_snapshots]
    hashes = [snapshot.sha256 for snapshot in measured_snapshots]
    row = aggregate_child_receipts(warmup_snapshot.receipt, measured,
        catalog, instance, precision; child_paths=measured_paths,
        child_hashes=hashes, warmup_path=warmup_path,
        warmup_hash=warmup_snapshot.sha256)
    Base.invokelatest(getfield(P, :validate_profile_row), row; live=true) ||
        throw(ArgumentError("fresh aggregate failed the live ProfileRow validator"))
    schema_stage = mktempdir()
    staged_schema = A.write_schema9(joinpath(schema_stage, "profile"), [row])
    schema_document = TOML.parsefile(staged_schema.toml)
    schema_document["schema_version"] == 9 || throw(ArgumentError(
        "fresh aggregate emitted a non-schema-v9 document"))
    length(get(schema_document, "result", Any[])) == 1 || throw(ArgumentError(
        "fresh aggregate emitted an unexpected schema row count"))
    _validate_schema_semantics(staged_schema.toml)
    _require_clean_source("parent_before_publication")
    published_receipt = _atomic_toml(receipt_path, row.receipt)
    published_tsv = _atomic_bytes(final_tsv, read(staged_schema.tsv))
    published_toml = _atomic_bytes(final_toml, read(staged_schema.toml))
    rm(schema_stage; recursive=true, force=true)
    _require_clean_source("parent_after_publication")
    println("STAGE_B_FRESH_RECEIPT receipt=$(published_receipt) tsv=$(published_tsv) toml=$(published_toml)")
    println("STAGE_B_FRESH_SUMMARY pid=", row.receipt["sample_pids"],
        " iterations=", row.sample_iterations, " objectives=", row.sample_objective,
        " peak_rss=", row.peak_rss_bytes)
    println("STAGE_B_FRESH_IDENTITY source_commit=$(row.source_commit) tree=$(row.tree_fingerprint) child_hashes=$(hashes)")
    true
end

function main()
    "--child" in ARGS ? child_main() : parent_main()
end

end # module V2FreshProcessProfile

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    V2FreshProcessProfile.main()
end
