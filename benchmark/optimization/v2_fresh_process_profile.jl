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
_git(args...) = readchomp(Cmd(vcat(["git", "-C", ROOT], String[string(arg) for arg in args])))

function _arg(name::String, default=nothing)
    prefix = "--" * name * "="
    for arg in ARGS
        startswith(arg, prefix) && return arg[length(prefix)+1:end]
    end
    return default
end

function _atomic_toml(path::AbstractString, value::Dict{String,Any})
    mkpath(dirname(path))
    temporary = string(path, ".tmp.", getpid(), ".", rand(UInt))
    open(temporary, "w") do io
        TOML.print(io, value; sorted=true)
    end
    mv(temporary, path; force=true)
    path
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
    isempty(_git("status", "--porcelain")) || throw(ArgumentError(
        "fresh-process source worktree must be clean"))
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
    provider_version = string(Base.pkgversion(SDPX))
    provider_fp = _sha((precision.provider, precision.name, precision.bits,
                        string(precision.arithmetic), provider_version))
    environment_fp = _sha(env)
    validation = result.validation
    semantic = validation.reference && isempty(validation.failures)
    receipt = Dict{String,Any}(
        "protocol_version" => PROTOCOL_VERSION,
        "pid" => getpid(),
        "source_commit" => commit,
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
        "precision_name" => string(precision.name),
        "precision_bits" => precision.bits,
        "solver_tolerance" => precision.solver_tolerance,
        "certificate_limit" => precision.certificate_limit,
        "environment" => env,
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
            "manifest_sha256", "environment", "environment_fingerprint",
            "provider_fingerprint", "provider", "provider_version",
            "precision_name", "precision_bits",
            "solver_tolerance", "certificate_limit", "route_receipt")
    all(haskey(receipt, key) for key in keys) || return false
    receipt["protocol_version"] == PROTOCOL_VERSION || return false
    occursin(r"^[0-9a-f]{40}$", receipt["source_commit"]) || return false
    occursin(r"^[0-9a-f]{40}$", receipt["tree_fingerprint"]) || return false
    for key in ("input_fingerprint", "execution_fingerprint",
                "catalog_artifact_sha256", "project_sha256", "manifest_sha256",
                "environment_fingerprint", "provider_fingerprint")
        occursin(r"^[0-9a-f]{64}$", receipt[key]) || return false
    end
    receipt["precision_bits"] isa Integer && receipt["precision_bits"] > 0 || return false
    receipt["environment"] isa AbstractDict || return false
    receipt["route_receipt"] isa AbstractDict || return false
    true
end

function _child_hash(path)
    _sha_bytes(read(path))
end

function _read_child(path)
    isfile(path) || throw(ArgumentError("child artifact missing: $path"))
    receipt = TOML.parsefile(path)
    _required_identity(receipt) || throw(ArgumentError("child artifact failed identity/protocol validation: $path"))
    receipt
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
        "environment", "environment_fingerprint", "provider_fingerprint", "provider",
        "provider_version", "precision_name", "precision_bits",
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
    row_receipt["execution_mode"] = FRESH_EXECUTION_MODE
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
    row_receipt["certificate_kind"] = "optimal"
    row_receipt["certificate_failures"] = String[]
    row_receipt["sample_certificate_metrics"] = [r["certificate_metrics"] for r in measured]
    row_receipt["sample_primal_residual"] = [r["primal_residual"] for r in measured]
    row_receipt["sample_dual_residual"] = [r["dual_residual"] for r in measured]
    row_receipt["sample_relative_gap"] = [r["relative_gap"] for r in measured]
    row_receipt["objective_error"] = [string(abs(parse(BigFloat, x) - parse(BigFloat, first_result["reference_objective"]))) for x in objectives]
    row_receipt["requested_route"] = route_value(route, "requested_route")
    row_receipt["planned_route"] = route_value(route, "planned_route")
    row_receipt["executed_route"] = route_value(route, "executed_route")
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

function parent_main()
    isempty(_git("status", "--porcelain")) || throw(ArgumentError(
        "fresh-process parent requires a clean source worktree"))
    case_id = Symbol(_arg("case-id", "v2_lp_box_small"))
    prefix = abspath(_arg("prefix", joinpath(tempdir(), "SDPX_STAGE_B_FRESH_PROFILE")))
    startswith(prefix, abspath(ROOT) * Base.Filesystem.path_separator) && throw(ArgumentError(
        "generated Stage-B artifacts must be outside the source worktree"))
    catalog, instance = _catalog_instance(case_id)
    precision = _precision()
    work = string(prefix, ".children")
    ispath(work) && throw(ArgumentError("refusing to overwrite existing child artifact directory: $work"))
    mkpath(work)
    warmup_path = joinpath(work, "warmup.toml")
    measured_paths = [joinpath(work, "sample_$(i).toml") for i in 1:3]
    _launch_child(warmup_path, case_id)
    warmup = _read_child(warmup_path)
    for path in measured_paths
        _launch_child(path, case_id)
    end
    measured = [_read_child(path) for path in measured_paths]
    hashes = [_child_hash(path) for path in measured_paths]
    warmup_hash = _child_hash(warmup_path)
    row = aggregate_child_receipts(warmup, measured, catalog, instance, precision;
        child_paths=measured_paths, child_hashes=hashes,
        warmup_path=warmup_path, warmup_hash=warmup_hash)
    Base.invokelatest(getfield(P, :validate_profile_row), row; live=true) ||
        throw(ArgumentError("fresh aggregate failed the live ProfileRow validator"))
    receipt_path = string(prefix, ".receipt.toml")
    schema_prefix = string(prefix, ".schema9")
    _atomic_toml(receipt_path, row.receipt)
    schema_paths = A.write_schema9(schema_prefix, [row])
    schema_document = TOML.parsefile(schema_paths.toml)
    schema_document["schema_version"] == 9 || throw(ArgumentError(
        "fresh aggregate emitted a non-schema-v9 document"))
    length(get(schema_document, "result", Any[])) == 1 || throw(ArgumentError(
        "fresh aggregate emitted an unexpected schema row count"))
    println("STAGE_B_FRESH_RECEIPT receipt=$(receipt_path) tsv=$(schema_paths.tsv) toml=$(schema_paths.toml)")
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
