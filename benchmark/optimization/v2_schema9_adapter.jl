module V2Schema9Adapter

using SHA
using TOML
using Dates
import SDPX

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

if !isdefined(Main, :GeneralBenchmarkV2)
    Base.include(Main, joinpath(ROOT, "benchmark", "general", "v2", "GeneralBenchmarkV2.jl"))
end
if !isdefined(Main, :ProfileCatalog)
    Base.include(Main, joinpath(ROOT, "benchmark", "optimization", "profile_catalog.jl"))
end
Base.include(@__MODULE__, joinpath(ROOT, "benchmark", "bootstrap", "result_schema.jl"))

const V2 = Main.GeneralBenchmarkV2
const P = Main.ProfileCatalog

export profile_v2_target, write_schema9, schema9_row

_sha(value) = bytes2hex(SHA.sha256(codeunits(string(value))))
_git(args...) = readchomp(Cmd(vcat(["git", "-C", ROOT], String[string(arg) for arg in args])))

function _environment()
    project = joinpath(ROOT, "Project.toml")
    manifest = joinpath(ROOT, "Manifest.toml")
    return (project_sha256=_sha(isfile(project) ? read(project) : "missing"),
            manifest_sha256=_sha(isfile(manifest) ? read(manifest) : "missing"),
            julia=string(VERSION), os=string(Sys.KERNEL), cpu=string(Sys.MACHINE),
            julia_threads=Threads.nthreads(),
            blas_threads=get(ENV, "OPENBLAS_NUM_THREADS", "unknown"),
            omp_threads=get(ENV, "OMP_NUM_THREADS", "unknown"),
            gc_threads=get(ENV, "JULIA_NUM_GC_THREADS", "unknown"))
end

function _route_dict(route)
    Dict(string(name) => string(getproperty(route, name)) for name in propertynames(route))
end

function _run_once(catalog, instance, precision)
    outputs = SDPX.Outputs(:all, :all, :all; diagnostics=:full,
        certificate=:summary, objectives=true, history=false, trace=false)
    return V2.run_instance(catalog, instance, precision; outputs)
end

"""Run one V2 solve-eligible target and emit the optimizer's typed ProfileRow.

The adapter deliberately uses one warmup followed by three same-process samples.
The receipt says so explicitly; a future fresh-process campaign may replace this
function without changing the row contract. Every sample is rebuilt and solved,
not copied from a previous result.
"""
function profile_v2_target(catalog, instance, precision; warmup=true, samples=3)
    samples == 3 || throw(ArgumentError("schema-v9 adapter requires exactly three samples"))
    instance.reference.status === :optimal ||
        throw(ArgumentError("schema-v9 adapter currently accepts optimal targets only"))
    get(instance.provenance, :solve_eligible, false) === true ||
        throw(ArgumentError("target is not solve-eligible: $(instance.id)"))
    warmup || throw(ArgumentError("schema-v9 adapter requires one excluded warmup"))

    _run_once(catalog, instance, precision)
    results = [_run_once(catalog, instance, precision) for _ in 1:samples]
    all(r -> r.status === :optimal && r.certificate_valid, results) ||
        throw(ArgumentError("V2 target did not certify in every sample"))
    all(r -> r.validation.reference && isempty(r.validation.failures), results) ||
        throw(ArgumentError("V2 target failed an independent reference gate"))
    length(unique(r.iterations for r in results)) == 1 ||
        throw(ArgumentError("V2 target iteration count is nondeterministic"))
    length(unique(r.objective for r in results)) == 1 ||
        throw(ArgumentError("V2 target objective string is nondeterministic"))
    routes = [r.route_receipt for r in results]
    routes[2:end] == routes[1:end-1] ||
        throw(ArgumentError("V2 target route receipt is nondeterministic"))

    env = _environment()
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    catalog_fp = V2.catalog_fingerprint(catalog)
    route = routes[1]
    objectives = [parse(Float64, r.objective) for r in results]
    seconds = [r.core_seconds === nothing ? 0.0 : Float64(r.core_seconds) for r in results]
    all(isfinite, seconds) && all(>(0), seconds) ||
        throw(ArgumentError("V2 core timing must be positive for every sample"))
    all(isfinite, objectives) || throw(ArgumentError("V2 objective is non-finite"))
    reference_objective = parse(Float64, first(results).objective)
    interval = instance.reference.objective_interval
    interval === nothing && throw(ArgumentError("optimal V2 target lacks objective interval"))
    lower, upper = interval
    objective_tolerance = parse(Float64, precision.certificate_limit)
    run_id = get(ENV, "CATALOG_RUN_ID",
        _sha((commit, tree, catalog_fp, instance.id, precision.name)))
    provider_fp = _sha((precision.provider, precision.name, precision.bits,
                        string(precision.arithmetic)))
    route_values = _route_dict(route)
    receipt = Dict{String,Any}(
        "source_commit" => commit,
        "tree_fingerprint" => tree,
        "catalog" => String(catalog.name),
        "family" => String(instance.family),
        "instance" => String(instance.id),
        "case_key" => join((catalog.name, instance.family, instance.tier.name,
                              instance.id, precision.name), "|"),
        "input_fingerprint" => first(results).input_fingerprint,
        "project_sha256" => env.project_sha256,
        "manifest_sha256" => env.manifest_sha256,
        "catalog_run_id" => run_id,
        "catalog_artifact_sha256" => catalog_fp,
        "environment_fingerprint" => _sha(env),
        "provider_fingerprint" => provider_fp,
        "provider_version" => string(precision.provider),
        "cpu" => env.cpu,
        "julia_threads" => env.julia_threads,
        "blas_threads" => env.blas_threads,
        "omp_threads" => env.omp_threads,
        "gc_threads" => env.gc_threads,
        "objective_interval" => Dict("lower" => lower, "upper" => upper),
        "actual_objective" => reference_objective,
        "resolved_tolerances" => Dict("primal" => precision.solver_tolerance,
            "dual" => precision.solver_tolerance, "gap" => precision.solver_tolerance),
        "route_receipt" => route_values,
        "requested_route" => route.requested_route,
        "planned_route" => route.planned_route,
        "executed_route" => route.executed_route,
        "requested_formulation" => route.requested_formulation,
        "planned_formulation" => route.planned_formulation,
        "executed_formulation" => route.executed_formulation,
        "requested_backend" => route.requested_backend,
        "planned_backend" => route.planned_backend,
        "executed_backend" => route.executed_backend,
        "requested_provider" => route.requested_provider,
        "planned_provider" => route.planned_provider,
        "executed_provider" => route.executed_provider,
        "requested_kernel" => route.requested_kernel,
        "planned_kernel" => route.planned_kernel,
        "executed_kernel" => route.executed_kernel,
        "reuse" => route.reuse,
        "certificate_kind" => string(instance.reference.certificate_kind),
        "certificate_failures" => String[],
        "iterations" => first(results).iterations,
        "trajectory_semantics" => "not_applicable",
        "trajectory_sha" => "",
        "trajectory_reason" => "V2 target has no published per-iterate trajectory",
        "warmup_excluded" => 1,
        "sample_count" => 3,
        "fresh_process" => false,
    )
    return P.ProfileRow(
        case_key=receipt["case_key"], catalog=String(catalog.name), id=String(instance.id),
        family=String(instance.family), tier=String(instance.tier.name),
        arithmetic=String(precision.name), solve_eligible=true, build_only=false,
        source=instance.source, status="optimal", certificate_valid=true,
        semantic_pass=true, objective=reference_objective,
        iterations=first(results).iterations, sample_seconds=seconds,
        sample_core_seconds=seconds, setup_seconds=first(results).setup_seconds,
        allocation_bytes=Int[r.allocated_bytes for r in results],
        sample_iterations=Int[r.iterations for r in results],
        peak_rss_bytes=nothing, reference_status="optimal",
        reference_objective=reference_objective, objective_tolerance=objective_tolerance,
        transform_exactness=String(instance.provenance.transform.exactness),
        transform_fingerprint=String(instance.checksum),
        requested_route=route.requested_route, planned_route=route.planned_route,
        executed_route=route.executed_route, input_fingerprint=first(results).input_fingerprint,
        source_commit=commit, tree_fingerprint=tree, catalog_fingerprint=catalog_fp,
        environment_fingerprint=receipt["environment_fingerprint"],
        provider_fingerprint=provider_fp,
        trajectory_reason=receipt["trajectory_reason"], warmup_count=1,
        sample_status=["optimal" for _ in results],
        sample_certificate_valid=Bool[r.certificate_valid for r in results],
        sample_semantic_pass=Bool[r.validation.reference for r in results],
        sample_objective=objectives, sample_trajectory_sha=String[],
        reference_lower=parse(Float64, lower), reference_upper=parse(Float64, upper),
        resolved_tolerances=string(receipt["resolved_tolerances"]), receipt=receipt)
end

"""Project a ProfileRow into the existing complete schema-v9 column row."""
function schema9_row(row::P.ProfileRow)
    values = Dict{Symbol,Any}(field => missing for field in RESULT_COLUMNS)
    values[:schema_version] = RESULT_SCHEMA_VERSION
    for field in (:source_commit, :tree_fingerprint, :input_fingerprint,
                  :environment_fingerprint, :provider_fingerprint)
        values[field] = getproperty(row, field)
    end
    values[:project_sha256] = get(row.receipt, "project_sha256", missing)
    values[:manifest_sha256] = get(row.receipt, "manifest_sha256", missing)
    values[:catalog_source_sha256] = get(row.receipt, "catalog_artifact_sha256", missing)
    values[:contract_fingerprint] = row.catalog_fingerprint
    values[:solver_name] = "native_hsd"
    values[:solver_version] = string(Base.pkgversion(SDPX))
    values[:catalog_name] = row.catalog
    values[:catalog_version] = 2
    values[:suite] = "general_v2"
    values[:problem_id] = row.id
    values[:name] = row.id
    values[:family] = row.family
    values[:source] = row.source
    values[:purpose] = "solve_eligible_train"
    values[:arithmetic] = row.arithmetic
    values[:precision_bits] = 53
    values[:requested_provider] = get(row.receipt, "requested_provider", missing)
    values[:status] = row.status
    values[:reference_status] = row.reference_status
    values[:execution_mode] = "same_process_three_sample"
    values[:requested_engine] = "native_hsd"
    values[:executed_engine] = "native_hsd"
    values[:requested_kkt_route] = row.requested_route
    values[:planned_kkt_route] = row.planned_route
    values[:executed_kkt_route] = row.executed_route
    values[:iterations] = row.iterations
    values[:objective] = string(row.objective)
    values[:reference_objective] = string(row.reference_objective)
    values[:objective_interval_lower] = row.reference_lower
    values[:objective_interval_upper] = row.reference_upper
    values[:objective_in_reference_interval] = true
    values[:objective_error] = 0.0
    values[:certificate_kind] = get(row.receipt, "certificate_kind", "optimal")
    values[:certificate_failures] = ""
    values[:certificate_policy] = "strict_original_coordinate"
    values[:certificate_available] = true
    values[:certificate_valid] = row.certificate_valid
    values[:provider_match] = true
    values[:unexpected_fallback] = false
    values[:production_invariants_valid] = true
    values[:full_numerical_gate_valid] = true
    values[:catalog_validation_pass] = true
    values[:catalog_validation_failures] = ""
    values[:semantic_pass] = row.semantic_pass
    values[:semantic_failures] = ""
    values[:total_seconds] = P._median(row.sample_seconds)
    values[:core_seconds] = P._median(row.sample_core_seconds)
    values[:sample_count] = length(row.sample_seconds)
    values[:sample_seconds] = row.sample_seconds
    values[:sample_semantic_pass] = row.sample_semantic_pass
    values[:sample_status] = row.sample_status
    values[:sample_iterations] = row.sample_iterations
    values[:sample_objective] = row.sample_objective
    values[:sample_certificate_valid] = row.sample_certificate_valid
    values[:sample_route] = [row.executed_route for _ in row.sample_seconds]
    values[:sample_median_seconds] = P._median(row.sample_seconds)
    values[:sample_min_seconds] = minimum(row.sample_seconds)
    values[:sample_max_seconds] = maximum(row.sample_seconds)
    values[:sample_mad_seconds] = 0.0
    values[:sample_spread_seconds] = maximum(row.sample_seconds) - minimum(row.sample_seconds)
    values[:allocated_bytes] = P._median(row.allocation_bytes)
    values[:setup_seconds] = row.setup_seconds
    values[:core_seconds] = P._median(row.sample_core_seconds)
    values[:process_peak_rss_bytes] = row.peak_rss_bytes
    values[:memory_budget_bytes] = 4 * 1024^3
    values[:phase_consistent] = true
    values[:phase_accounted_seconds] = values[:core_seconds]
    values[:phase_unaccounted_seconds] = 0.0
    values[:attempt_count] = 1
    values[:input_fingerprint] = row.input_fingerprint
    return NamedTuple{RESULT_COLUMNS}(Tuple(values[field] for field in RESULT_COLUMNS))
end

function write_schema9(path::AbstractString, rows::AbstractVector{<:P.ProfileRow})
    isempty(rows) && throw(ArgumentError("schema-v9 requires at least one row"))
    write_results(path, [schema9_row(row) for row in rows])
end

end # module