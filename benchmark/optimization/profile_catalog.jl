module ProfileCatalog

using SHA
using TOML
import SDPX

export ProfileCase, ProfileRow, enumerate_cases, profile_catalog, select_max_target,
       write_manifest, read_manifest, write_profiles, read_profiles, fixture_rows,
       run_fixture, validate_profile_row

const PROFILE_SCHEMA = 2
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const GENERAL = joinpath(ROOT, "benchmark", "general", "GenericConicBenchmark.jl")

struct ProfileCase
    key::String
    catalog::Symbol
    id::Symbol
    family::Symbol
    tier::Symbol
    arithmetic::Symbol
    solve_eligible::Bool
    reference_status::Symbol
    objective::Union{Nothing,Float64}
    objective_tolerance::Float64
    source::String
    transform::NamedTuple
    payload::Any
end

Base.@kwdef struct ProfileRow
    case_key::String
    catalog::String
    id::String
    family::String
    tier::String
    arithmetic::String
    solve_eligible::Bool
    build_only::Bool
    source::String
    status::String = "not-run"
    certificate_valid::Bool = false
    semantic_pass::Bool = false
    objective::Union{Nothing,Float64} = nothing
    iterations::Int = 0
    sample_seconds::Vector{Float64} = Float64[]
    sample_core_seconds::Vector{Float64} = Float64[]
    setup_seconds::Union{Nothing,Float64} = nothing
    allocation_bytes::Vector{Int} = Int[]
    sample_iterations::Vector{Int} = Int[]
    peak_rss_bytes::Union{Nothing,Int} = nothing
    reference_status::String = ""
    reference_objective::Union{Nothing,Float64} = nothing
    objective_tolerance::Union{Nothing,Float64} = nothing
    transform_exactness::String = ""
    transform_fingerprint::String = ""
    failure_taxonomy::String = ""
    requested_route::String = ""
    planned_route::String = ""
    executed_route::String = ""
    input_fingerprint::String = ""
    warmup_count::Int = 1
    sample_status::Vector{String} = String[]
    sample_certificate_valid::Vector{Bool} = Bool[]
    sample_semantic_pass::Vector{Bool} = Bool[]
    sample_objective::Vector{Float64} = Float64[]
    sample_trajectory_sha::Vector{String} = String[]
    reference_lower::Union{Nothing,Float64} = nothing
    reference_upper::Union{Nothing,Float64} = nothing
    source_commit::String = ""
    tree_fingerprint::String = ""
    catalog_fingerprint::String = ""
    environment_fingerprint::String = ""
    provider_fingerprint::String = ""
    trajectory_sha::String = ""
    trajectory_reason::String = ""
    resolved_tolerances::String = ""
    # Canonical machine-readable receipt.  It is intentionally redundant with
    # the typed fields above: profile consumers must not infer identity or
    # semantics from display-only fields.
    receipt::Dict{String,Any} = Dict{String,Any}()
end

_case_sort(c) = (String(c.catalog), String(c.family), String(c.tier), String(c.id), String(c.arithmetic))
_key(catalog, id, family, tier, arithmetic) = join((catalog, family, tier, id, arithmetic), "|")

function _sha(value)
    io = IOBuffer(); print(io, value); return bytes2hex(SHA.sha256(take!(io)))
end

function _git_identity()
    commit = readchomp(`git rev-parse HEAD`)
    tree = readchomp(`git rev-parse 'HEAD^{tree}'`)
    occursin(r"^[0-9a-f]{40}$", commit) || error("invalid git HEAD")
    occursin(r"^[0-9a-f]{40}$", tree) || error("invalid git tree")
    return (commit=commit, tree=tree)
end

function _route_receipt(result)
    try
        selected = getfield(SDPX.diagnostics(result), :selected_algorithms)
        pick(names) = begin
            for name in names
                hasproperty(selected, name) && return string(getproperty(selected, name))
            end
            return nothing
        end
        values = (
            requested_route=pick((:requested_kkt_route, :requested_route)),
            planned_route=pick((:planned_kkt_route, :planned_route)),
            executed_route=pick((:executed_kkt_route, :executed_route)),
            requested_formulation=pick((:requested_kkt_formulation, :requested_formulation)),
            planned_formulation=pick((:planned_kkt_formulation, :planned_formulation)),
            executed_formulation=pick((:executed_kkt_formulation, :executed_formulation)),
            requested_backend=pick((:requested_backend, :la_requested_backend)),
            planned_backend=pick((:planned_backend, :la_planned_backend)),
            executed_backend=pick((:executed_backend, :la_executed_backend)),
            requested_provider=pick((:requested_provider, :la_requested_provider)),
            planned_provider=pick((:planned_provider, :la_planned_provider)),
            executed_provider=pick((:executed_provider, :la_executed_provider)),
            requested_kernel=pick((:requested_kernel, :la_requested_kernel)),
            planned_kernel=pick((:planned_kernel, :la_planned_kernel)),
            executed_kernel=pick((:executed_kernel, :la_executed_kernel)),
            reuse=pick((:reuse, :symbolic_reuse, :pattern_reused)),
        )
        all(value -> value !== nothing && !isempty(value), values) || return nothing
        return values
    catch
        return nothing
    end
end

function _environment_identity()
    project = joinpath(ROOT, "Project.toml")
    manifest = joinpath(ROOT, "Manifest.toml")
    return (project=_sha(isfile(project) ? read(project) : "missing"),
        manifest=_sha(isfile(manifest) ? read(manifest) : "missing"),
        julia=string(VERSION), os=string(Sys.KERNEL), cpu=string(Sys.MACHINE),
        julia_threads=Threads.nthreads(),
        blas=get(ENV, "OPENBLAS_NUM_THREADS", "unknown"),
        omp=get(ENV, "OMP_NUM_THREADS", "unknown"),
        gc=get(ENV, "JULIA_NUM_GC_THREADS", "unknown"))
end

function _generic_module()
    mod = Module(:ProfileGenericConicBenchmark)
    Core.eval(mod, :(import Main))
    Base.include(mod, GENERAL)
    return Base.invokelatest(() -> getproperty(mod, :GenericConicBenchmark))
end

function _v1_cases()
    isfile(GENERAL) || return ProfileCase[]
    g = _generic_module()
    cases = ProfileCase[]
    inventory = Base.invokelatest(() -> getproperty(g, :inventory)())
    for spec in inventory
        # V1's legacy adapter cannot currently produce the complete schema-v9
        # receipt required by the dependent optimizer (exact input/tree,
        # resolved tolerances, route/provider/kernel/reuse, and trajectory
        # semantics). Keep V1 visible for catalog/profile diagnostics, but make
        # it ineligible for live optimization until a V2/schema-v9 adapter
        # supplies those fields. Never fabricate "unavailable" receipts.
        eligible = false
        transform = (exactness="v1_ineligible_pending_schema_v9", fingerprint=_sha((spec.source, spec.id)))
        key = _key("generic-v1", spec.id, spec.family, spec.tier, :float64)
        push!(cases, ProfileCase(key, :generic_v1, spec.id, spec.family, spec.tier,
            :float64, eligible, spec.expected_status,
            spec.known_objective, spec.objective_tolerance, spec.source,
            transform, (mod=g, spec=spec)))
    end
    sort!(cases; by=_case_sort)
end

function _physics_cases()
    cases = ProfileCase[]
    harness_path = joinpath(ROOT, "benchmark", "bootstrap", "PhysicsBenchmarkHarness.jl")
    isfile(harness_path) || return cases
    if !isdefined(Main, :PhysicsBenchmarkHarness)
        Base.include(Main, harness_path)
    end
    H = Main.PhysicsBenchmarkHarness
    pdir = joinpath(ROOT, "benchmark", "bootstrap", "physics")
    for path in sort!(filter(isfile, [joinpath(d, "catalog.jl") for d in readdir(pdir; join=true) if isdir(d)]))
        try
            cat = H.load_catalog(path)
            ids = sort!(unique(e.problem_id for es in values(cat.suites) for e in es))
            for id in ids
                spec = H.catalog_spec(cat, id)
                eligible = spec.reference.status === :optimal &&
                           spec.reference.objective !== nothing
                # Physics catalogs currently expose build-only contracts. Keep
                # them visible but never profile them as solver targets.
                key = _key(String(cat.name), id, spec.family, :small, :float64)
                push!(cases, ProfileCase(key, cat.name, Symbol(id), spec.family, :small,
                    :float64, eligible, spec.reference.status,
                    spec.reference.objective isa Real ? Float64(spec.reference.objective) : nothing,
                    spec.reference.absolute_tolerance,
                    "$(path)#$(id)",
                    (exactness="catalog", fingerprint=spec.fingerprint),
                    (harness=H, catalog=cat, spec=spec, path=path)))
            end
        catch err
            @warn "catalog enumeration failed" path exception=(err, catch_backtrace())
        end
    end
    sort!(cases; by=_case_sort)
end

function _apply_v2_metadata(cases)
    isdefined(Main, :GeneralBenchmarkV2) || return cases
    v2 = Main.GeneralBenchmarkV2
    generic = filter(c -> c.catalog === :generic_v1, cases)
    isempty(generic) && return cases
    g = generic[1].payload.mod
    specs = [c.payload.spec for c in generic]
    try
        cat = Base.invokelatest(v2.adapt_generic_specs, specs; generic_module=g)
        precision = Base.invokelatest(v2.V2Precision, :Float64, Float64, 53, "1e-8", "5e-7", :auto)
        metadata = Dict{Symbol,NamedTuple}()
        for instance in cat.instances
            built, _ = Base.invokelatest(v2.build_instance, cat, instance, precision)
            metadata[instance.id] = (exactness=String(built.transform.exactness),
                fingerprint=built.transform.fingerprint,
                reference_status=String(instance.reference.status))
        end
        for i in eachindex(cases)
            c = cases[i]
            haskey(metadata, c.id) || continue
            m = metadata[c.id]
            cases[i] = ProfileCase(c.key, c.catalog, c.id, c.family, c.tier,
                c.arithmetic, c.solve_eligible, c.reference_status, c.objective,
                c.objective_tolerance, c.source, m, c.payload)
        end
    catch err
        @warn "optional V2 metadata hook unavailable" exception=(err, catch_backtrace())
    end
    return cases
end

"""Enumerate current V1 cases and optional exact V2 cases without requiring V2."""
function enumerate_cases(; include_physics=true, include_v2=true)
    cases = _v1_cases()
    include_physics && append!(cases, _physics_cases())
    # Optional hook: V2 is not a dependency. When its real exported adapter is
    # present, metadata is built once and copied into V1-compatible rows.
    if include_v2
        v2path = joinpath(ROOT, "benchmark", "general", "v2", "GeneralBenchmarkV2.jl")
        if isfile(v2path) && !isdefined(Main, :GeneralBenchmarkV2)
            try Base.include(Main, v2path) catch err
                @warn "optional V2 adapter unavailable" exception=(err, catch_backtrace())
            end
        end
        get(ENV, "SDPX_PROFILE_V2_METADATA", "0") == "1" && _apply_v2_metadata(cases)
    end
    sort!(cases; by=_case_sort)
end

function _valid_trajectory(semantics::AbstractString, sha::AbstractString, reason::AbstractString)
    semantics == "sha256" && return occursin(r"^[0-9a-f]{64}$", sha) && !isempty(reason)
    return semantics == "not_applicable" && isempty(sha) && !isempty(reason)
end

function _median(xs)
    isempty(xs) && return nothing
    y = sort(copy(xs)); n = length(y)
    return isodd(n) ? y[(n + 1) ÷ 2] : (y[n ÷ 2] + y[n ÷ 2 + 1]) / 2
end

function _v1_profile(case::ProfileCase; samples=3, threads=1, warmup=true)
    g, spec = case.payload.mod, case.payload.spec
    T = Float64
    build = @timed Base.invokelatest(g.build, spec.problem, T, spec.params)
    model = build.value
    warmup && SDPX.optimize!(model; settings=Base.invokelatest(g._settings, T; threads))
    seconds = Float64[]; allocs = Int[]; sample_iters = Int[]
    sample_status = String[]; sample_certificates = Bool[]; sample_semantics = Bool[]
    sample_objectives = Float64[]
    samples == 3 || throw(ArgumentError("profiling requires exactly three samples"))
    iterations = 0; status = :not_run
    cert_valid = false; objective = nothing; semantic = false; core = Float64[]
    route_receipts = NamedTuple[]
    for _ in 1:3
        # Rebuild before every measured run; setup is recorded separately and
        # never enters the solver metric.
        bm = @timed Base.invokelatest(g.build, spec.problem, T, spec.params)
        solve = @timed SDPX.optimize!(bm.value;
            settings=Base.invokelatest(g._settings, T; threads),
            outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:full,
                certificate=:summary, objectives=true, history=false, trace=false))
        result = solve.value; cert = SDPX.certificate(result)
        push!(seconds, Float64(solve.time)); push!(allocs, Int(solve.bytes)); push!(sample_iters, result.iterations)
        iterations = result.iterations; status = SDPX.status(result)
        cert_valid = cert.valid; objective = Float64(cert.primal_objective)
        push!(sample_status, String(status)); push!(sample_certificates, cert.valid)
        push!(sample_objectives, objective)
        br = g.BenchmarkResult(spec.id, spec.family, spec.tier, status, objective,
            Float64(cert.dual_objective), Float64(cert.primal_residual),
            Float64(cert.dual_residual), Float64(cert.relative_gap), cert.valid,
            result.iterations, solve.time, solve.bytes, solve.gctime, false)
        semantic_i = Base.invokelatest(g.validate_result, spec, br)
        push!(sample_semantics, semantic_i)
        semantic = semantic_i
        receipt = _route_receipt(result)
        receipt === nothing || push!(route_receipts, receipt)
        try
            t = getfield(SDPX.diagnostics(result), :timings)
            v = getproperty(t, :core)
            v isa Real && isfinite(v) && push!(core, Float64(v))
        catch
        end
    end
    semantic = length(sample_semantics) == 3 && all(sample_semantics)
    length(core) in (0, 3) || throw(ArgumentError("core timing must be 3 samples or unavailable"))
    length(unique(sample_iters)) == 1 || throw(ArgumentError("iteration nondeterminism"))
    length(unique(sample_objectives)) == 1 || throw(ArgumentError("objective nondeterminism"))
    length(route_receipts) in (0, 3) ||
        throw(ArgumentError("route receipt must be present for all samples or none"))
    !isempty(route_receipts) && !all(r -> r == first(route_receipts), route_receipts) &&
        throw(ArgumentError("route receipt nondeterminism"))
    route = isempty(route_receipts) ? nothing : first(route_receipts)
    identity = _git_identity(); env = _environment_identity()
    # No trajectory hash is invented here. Generic catalog runs have no
    # published per-iterate trace; explicitly record not_applicable plus a
    # machine-readable reason instead.
    trajectory_sha = ""
    trajectory_reason = "generic_catalog_has_no_published_per_iterate_trace"
    failed = !semantic ? (status == :iteration_limit ? "iteration_limit" : "semantic_or_certificate") : ""
    receipt = Dict{String,Any}(
        "source_commit" => identity.commit, "tree_fingerprint" => identity.tree,
        "catalog" => String(case.catalog), "family" => String(case.family),
        "instance" => String(case.id), "case_key" => case.key,
        "input_fingerprint" => _sha((case.source, case.id, case.payload.spec.params)),
        "project_sha256" => env.project, "manifest_sha256" => env.manifest,
        "environment_fingerprint" => _sha(env), "cpu" => env.cpu,
        "julia_threads" => env.julia_threads, "blas_threads" => env.blas,
        "omp_threads" => env.omp, "gc_threads" => env.gc,
        "provider_fingerprint" => _sha((Base.PkgId(SDPX), Base.pkgversion(SDPX))),
        "provider_version" => string(Base.pkgversion(SDPX)),
        "objective_interval" => Dict("lower" => case.objective - case.objective_tolerance,
            "upper" => case.objective + case.objective_tolerance),
        "actual_objective" => objective,
        "resolved_tolerances" => Dict("primal" => case.objective_tolerance,
            "dual" => case.objective_tolerance, "gap" => case.objective_tolerance),
        "requested_route" => route === nothing ? "" : route.requested_route,
        "planned_route" => route === nothing ? "" : route.planned_route,
        "executed_route" => route === nothing ? "" : route.executed_route,
        "route_receipt" => route === nothing ? Dict{String,Any}() :
            Dict(string(name) => getproperty(route, name) for name in propertynames(route)),
        "requested_formulation" => route === nothing ? "" : route.requested_formulation,
        "planned_formulation" => route === nothing ? "" : route.planned_formulation,
        "executed_formulation" => route === nothing ? "" : route.executed_formulation,
        "requested_backend" => route === nothing ? "" : route.requested_backend,
        "planned_backend" => route === nothing ? "" : route.planned_backend,
        "executed_backend" => route === nothing ? "" : route.executed_backend,
        "requested_provider" => route === nothing ? "" : route.requested_provider,
        "planned_provider" => route === nothing ? "" : route.planned_provider,
        "executed_provider" => route === nothing ? "" : route.executed_provider,
        "requested_kernel" => route === nothing ? "" : route.requested_kernel,
        "planned_kernel" => route === nothing ? "" : route.planned_kernel,
        "executed_kernel" => route === nothing ? "" : route.executed_kernel,
        "reuse" => route === nothing ? "" : route.reuse,
        "certificate_kind" => "summary", "certificate_failures" => String[],
        "trajectory_semantics" => isempty(trajectory_sha) ? "not_applicable" : "sha256",
        "trajectory_sha" => trajectory_sha, "trajectory_reason" => trajectory_reason,
        "warmup_excluded" => warmup ? 1 : 0, "sample_count" => 3,
    )
    ProfileRow(case_key=case.key, catalog=String(case.catalog), id=String(case.id),
        family=String(case.family), tier=String(case.tier), arithmetic=String(case.arithmetic),
        solve_eligible=case.solve_eligible, build_only=!case.solve_eligible, source=case.source,
        status=String(status), certificate_valid=cert_valid, semantic_pass=semantic,
        objective=objective, iterations=iterations, sample_seconds=seconds,
        sample_core_seconds=core, setup_seconds=Float64(build.time),
        allocation_bytes=allocs, sample_iterations=sample_iters,
        sample_status=sample_status, sample_certificate_valid=sample_certificates,
        sample_semantic_pass=sample_semantics, sample_objective=sample_objectives,
        reference_status=String(case.reference_status),
        reference_objective=case.objective, objective_tolerance=case.objective_tolerance,
        reference_lower=case.objective === nothing ? nothing : case.objective - case.objective_tolerance,
        reference_upper=case.objective === nothing ? nothing : case.objective + case.objective_tolerance,
        transform_exactness=case.transform.exactness,
        transform_fingerprint=case.transform.fingerprint, failure_taxonomy=failed,
        requested_route=route === nothing ? "" : route.requested_route,
        planned_route=route === nothing ? "" : route.planned_route,
        executed_route=route === nothing ? "" : route.executed_route,
        input_fingerprint=_sha((case.source, case.id, case.payload.spec.params)),
        source_commit=identity.commit, tree_fingerprint=identity.tree,
        catalog_fingerprint=_sha((case.catalog, case.id, case.transform)),
        environment_fingerprint=_sha(env),
        provider_fingerprint=_sha((Base.PkgId(SDPX), Base.pkgversion(SDPX))),
        resolved_tolerances=string((primal=case.objective_tolerance,
            dual=case.objective_tolerance, gap=case.objective_tolerance)),
        warmup_count=warmup ? 1 : 0, receipt=receipt)
end

function profile_catalog(cases=enumerate_cases(); samples=3, threads=1, warmup=true,
                        fixture=false, io=stdout)
    fixture && return fixture_rows()
    samples == 3 || throw(ArgumentError("optimization profiling requires exactly three samples"))
    rows = ProfileRow[]
    for case in cases
        if !case.solve_eligible
            push!(rows, ProfileRow(case_key=case.key, catalog=String(case.catalog),
                id=String(case.id), family=String(case.family), tier=String(case.tier),
                arithmetic=String(case.arithmetic), solve_eligible=false, build_only=true,
                source=case.source, status="build_only", semantic_pass=true,
                reference_status=String(case.reference_status),
                transform_exactness=case.transform.exactness,
                transform_fingerprint=case.transform.fingerprint,
                input_fingerprint=_sha(case.source), warmup_count=0))
        elseif case.catalog === :generic_v1
            try push!(rows, _v1_profile(case; samples, threads, warmup))
            catch err
                push!(rows, ProfileRow(case_key=case.key, catalog=String(case.catalog),
                    id=String(case.id), family=String(case.family), tier=String(case.tier),
                    arithmetic=String(case.arithmetic), solve_eligible=true, build_only=false,
                    source=case.source, status="error", failure_taxonomy=string(nameof(typeof(err))),
                    reference_status=String(case.reference_status)))
            end
        end
    end
    sort!(rows; by=r -> (r.catalog, r.family, r.tier, r.id, r.arithmetic))
    return rows
end

function fixture_rows()
    return [
        ProfileRow(case_key="fixture|lp|small|slow", catalog="fixture", id="slow",
            family="lp", tier="small", arithmetic="Float64", solve_eligible=true,
            build_only=false, source="fixture", status="optimal", certificate_valid=true,
            semantic_pass=true, objective=0.0, iterations=4, sample_seconds=[5.0, 4.0, 6.0],
            sample_core_seconds=[3.0, 2.0, 4.0], allocation_bytes=[30, 30, 30],
            reference_status="optimal", reference_objective=0.0, objective_tolerance=1e-8,
            transform_exactness="identity", sample_iterations=[4, 4, 4],
            sample_status=["optimal", "optimal", "optimal"], sample_certificate_valid=[true, true, true],
            sample_semantic_pass=[true, true, true], sample_objective=[0.0, 0.0, 0.0],
            requested_route="auto", planned_route="auto", executed_route="auto"),
        ProfileRow(case_key="fixture|sdp|small|build", catalog="fixture", id="build",
            family="sdp", tier="small", arithmetic="Float64", solve_eligible=false,
            build_only=true, source="fixture", status="build_only", semantic_pass=true,
            transform_exactness="finite_grid_surrogate"),
        ProfileRow(case_key="fixture|socp|small|fast", catalog="fixture", id="fast",
            family="socp", tier="small", arithmetic="Float64", solve_eligible=true,
            build_only=false, source="fixture", status="optimal", certificate_valid=true,
            semantic_pass=true, objective=0.0, iterations=3, sample_seconds=[1.0, 1.0, 1.0],
            sample_core_seconds=[0.5, 0.5, 0.5], allocation_bytes=[100, 100, 100],
            reference_status="optimal", reference_objective=0.0, objective_tolerance=1e-8,
            transform_exactness="identity", sample_iterations=[3, 3, 3],
            sample_status=["optimal", "optimal", "optimal"], sample_certificate_valid=[true, true, true],
            sample_semantic_pass=[true, true, true], sample_objective=[0.0, 0.0, 0.0],
            requested_route="auto", planned_route="auto", executed_route="auto"),
    ]
end

function select_max_target(rows; metric=:core_seconds)
    candidates = ProfileRow[]
    for row in rows
        row.solve_eligible && !row.build_only && row.semantic_pass && row.certificate_valid || continue
        validate_profile_row(row) || continue
        vals = metric === :core_seconds && !isempty(row.sample_core_seconds) ?
            row.sample_core_seconds : row.sample_seconds
        isempty(vals) && continue
        all(isfinite, vals) || continue
        push!(candidates, row)
    end
    isempty(candidates) && throw(ArgumentError("no certified solve-eligible profile rows"))
    value(row) = _median(metric === :core_seconds && !isempty(row.sample_core_seconds) ?
        row.sample_core_seconds : row.sample_seconds)
    sort!(candidates; by=row -> (-value(row), -(_median(row.allocation_bytes) === nothing ? 0 : _median(row.allocation_bytes)), row.case_key))
    return candidates[1], candidates
end

function validate_profile_row(row::ProfileRow; live=false)
    row.solve_eligible && !row.build_only || return false
    row.warmup_count == 1 || return false
    if live
        occursin(r"^[0-9a-f]{40}$", row.source_commit) || return false
        occursin(r"^[0-9a-f]{40}$", row.tree_fingerprint) || return false
        all(occursin(r"^[0-9a-f]{64}$", x) for x in
            (row.input_fingerprint, row.environment_fingerprint,
             row.provider_fingerprint)) || return false
        any(x -> isempty(x) || lowercase(x) in ("auto", "unavailable", "unknown", "none"),
            (row.requested_route, row.planned_route, row.executed_route)) && return false
        required = ("source_commit", "tree_fingerprint", "case_key", "catalog",
            "family", "instance", "input_fingerprint", "project_sha256",
            "manifest_sha256", "catalog_run_id", "catalog_artifact_sha256",
            "environment_fingerprint", "provider_fingerprint", "provider_version",
            "cpu", "julia_threads", "blas_threads", "omp_threads", "gc_threads",
            "objective_interval", "actual_objective", "resolved_tolerances",
            "route_receipt", "requested_route", "planned_route", "executed_route", "certificate_kind",
            "certificate_failures", "iterations", "trajectory_semantics",
            "trajectory_reason", "warmup_excluded", "sample_count")
        all(haskey(row.receipt, key) && !isempty(string(row.receipt[key])) for key in required) || return false
        row.receipt["warmup_excluded"] == 1 && row.receipt["sample_count"] == 3 || return false
        route_fields = ("requested_formulation", "planned_formulation", "executed_formulation",
            "requested_backend", "planned_backend", "executed_backend",
            "requested_provider", "planned_provider", "executed_provider",
            "requested_kernel", "planned_kernel", "executed_kernel", "reuse")
        all(haskey(row.receipt, key) && !isempty(string(row.receipt[key])) for key in route_fields) || return false
        route_receipt = row.receipt["route_receipt"]
        route_receipt isa AbstractDict && Set(keys(route_receipt)) == Set((
            "requested_route", "planned_route", "executed_route", "requested_formulation",
            "planned_formulation", "executed_formulation", "requested_backend", "planned_backend",
            "executed_backend", "requested_provider", "planned_provider", "executed_provider",
            "requested_kernel", "planned_kernel", "executed_kernel", "reuse",
        )) && all(!isempty(string(route_receipt[key])) for key in keys(route_receipt)) || return false
        semantics = string(row.receipt["trajectory_semantics"])
        sha = string(get(row.receipt, "trajectory_sha", ""))
        reason = string(get(row.receipt, "trajectory_reason", ""))
        _valid_trajectory(semantics, sha, reason) || return false
        interval = row.receipt["objective_interval"]
        interval isa AbstractDict && haskey(interval, "lower") && haskey(interval, "upper") || return false
        tolerance = row.receipt["resolved_tolerances"]
        tolerance isa AbstractDict && all(haskey(tolerance, x) for x in ("primal", "dual", "gap")) || return false
    end
    length(row.sample_seconds) == 3 || return false
    length(row.sample_iterations) == 3 || return false
    length(row.sample_status) == 3 || return false
    length(row.sample_certificate_valid) == 3 || return false
    length(row.sample_semantic_pass) == 3 || return false
    length(row.sample_objective) == 3 || return false
    all(isfinite, row.sample_seconds) && all(>(0), row.sample_seconds) || return false
    all(==(row.status), row.sample_status) || return false
    all(row.sample_certificate_valid) && all(row.sample_semantic_pass) || return false
    length(unique(row.sample_iterations)) == 1 || return false
    length(unique(row.sample_objective)) == 1 || return false
    row.status == "optimal" || return false
    row.reference_status == "optimal" || return false
    row.reference_objective !== nothing || return false
    row.objective_tolerance !== nothing && isfinite(row.objective_tolerance) && row.objective_tolerance >= 0 || return false
    tol = row.objective_tolerance
    all(x -> isfinite(x) && abs(x - row.reference_objective) <= tol,
        row.sample_objective) || return false
    length(row.sample_core_seconds) in (0, 3) || return false
    return true
end

function _row_dict(row::ProfileRow)
    return Dict("case_key"=>row.case_key, "catalog"=>row.catalog, "id"=>row.id,
        "family"=>row.family, "tier"=>row.tier, "arithmetic"=>row.arithmetic,
        "solve_eligible"=>row.solve_eligible, "build_only"=>row.build_only,
        "source"=>row.source, "status"=>row.status,
        "certificate_valid"=>row.certificate_valid, "semantic_pass"=>row.semantic_pass,
        "objective"=>(row.objective === nothing ? "" : row.objective),
        "iterations"=>row.iterations, "sample_iterations"=>row.sample_iterations,
        "sample_seconds"=>row.sample_seconds,
        "sample_core_seconds"=>row.sample_core_seconds, "setup_seconds"=>(row.setup_seconds === nothing ? "" : row.setup_seconds),
        "allocation_bytes"=>row.allocation_bytes, "peak_rss_bytes"=>(row.peak_rss_bytes === nothing ? "" : row.peak_rss_bytes),
        "reference_status"=>row.reference_status, "reference_objective"=>(row.reference_objective === nothing ? "" : row.reference_objective),
        "objective_tolerance"=>(row.objective_tolerance === nothing ? "" : row.objective_tolerance),
        "transform_exactness"=>row.transform_exactness, "transform_fingerprint"=>row.transform_fingerprint,
        "failure_taxonomy"=>row.failure_taxonomy, "requested_route"=>row.requested_route,
        "planned_route"=>row.planned_route, "executed_route"=>row.executed_route,
        "input_fingerprint"=>row.input_fingerprint, "warmup_count"=>row.warmup_count,
        "sample_status"=>row.sample_status, "sample_certificate_valid"=>row.sample_certificate_valid,
        "sample_semantic_pass"=>row.sample_semantic_pass, "sample_objective"=>row.sample_objective,
        "sample_trajectory_sha"=>row.sample_trajectory_sha,
        "reference_lower"=>(row.reference_lower === nothing ? "" : row.reference_lower),
        "reference_upper"=>(row.reference_upper === nothing ? "" : row.reference_upper),
        "source_commit"=>row.source_commit, "tree_fingerprint"=>row.tree_fingerprint,
        "catalog_fingerprint"=>row.catalog_fingerprint, "environment_fingerprint"=>row.environment_fingerprint,
        "provider_fingerprint"=>row.provider_fingerprint, "trajectory_sha"=>row.trajectory_sha,
        "trajectory_semantics"=>(isempty(row.trajectory_sha) ? "not_applicable" : "sha256"),
        "trajectory_reason"=>row.trajectory_reason,
        "resolved_tolerances"=>row.resolved_tolerances, "receipt"=>row.receipt)
end

function write_profiles(path, rows; source_commit="unknown")
    selected_row = try first(select_max_target(rows; metric=:core_seconds)) catch; nothing end
    selected = selected_row === nothing ? "" : selected_row.case_key
    identity = _git_identity()
    fixture = get(ENV, "SDPX_PROFILE_FIXTURE", "0") == "1"
    fixture || String(source_commit) == identity.commit || throw(ArgumentError(
        "profile source_commit must equal checked-out git HEAD"))
    effective_commit = fixture ? String(source_commit) : identity.commit
    doc = Dict("profile_schema"=>PROFILE_SCHEMA, "source_commit"=>effective_commit,
        "tree_fingerprint"=>identity.tree,
        "catalog_run_id"=>get(ENV, "CATALOG_RUN_ID", ""),
        "catalog_artifact_sha256"=>get(ENV, "CATALOG_ARTIFACT_SHA256", ""),
        "catalog_fingerprint"=>selected_row === nothing ? "" : selected_row.catalog_fingerprint,
        "environment_fingerprint"=>selected_row === nothing ? "" : selected_row.environment_fingerprint,
        "provider_fingerprint"=>selected_row === nothing ? "" : selected_row.provider_fingerprint,
        "selected_case_key"=>selected,
        "warmup_excluded"=>true, "metric_policy"=>"core_median_then_solver_median",
        "row"=>[_row_dict(r) for r in rows])
    open(path, "w") do io; TOML.print(io, doc; sorted=true); end
    return path
end

function write_manifest(path, rows; source_commit="unknown")
    eligible, _ = select_max_target(rows; metric=:core_seconds)
    identity = _git_identity()
    fixture = get(ENV, "SDPX_PROFILE_FIXTURE", "0") == "1"
    fixture || String(source_commit) == identity.commit || throw(ArgumentError(
        "manifest source_commit must equal checked-out git HEAD"))
    effective_commit = fixture ? String(source_commit) : identity.commit
    doc = Dict("manifest_schema"=>PROFILE_SCHEMA, "source_commit"=>effective_commit,
        "tree_fingerprint"=>identity.tree,
        "selected_case_key"=>eligible.case_key, "selection_metric"=>"core_seconds",
        "case"=>[_row_dict(r) for r in rows])
    open(path, "w") do io; TOML.print(io, doc; sorted=true); end
    return path
end

read_profiles(path) = TOML.parsefile(path)
read_manifest(path) = TOML.parsefile(path)

function run_fixture(; io=stdout)
    rows = fixture_rows(); selected, ordered = select_max_target(rows)
    selected.id == "slow" || error("fixture selector chose wrong target")
    any(r -> r.build_only && !r.solve_eligible, rows) || error("build-only exclusion failed")
    return selected
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    fixture = get(ENV, "SDPX_PROFILE_FIXTURE", "0") == "1"
    source_commit = fixture ? get(ENV, "SDPX_PROFILE_SOURCE_COMMIT", "fixture") :
        readchomp(`git rev-parse HEAD`)
    occursin(r"^[0-9a-f]{40}$", source_commit) ||
        (fixture || error("profile source_commit must come from git HEAD"))
    fixture && get(ENV, "SDPX_OPTIMIZATION_TEST_MODE", "0") != "1" &&
        error("fixture mode requires explicit SDPX_OPTIMIZATION_TEST_MODE=1")
    rows = fixture ? ProfileCatalog.fixture_rows() : ProfileCatalog.profile_catalog()
    out = get(ENV, "SDPX_PROFILE_OUTPUT", joinpath(pwd(), "profile-catalog.toml"))
    source = source_commit
    ProfileCatalog.write_profiles(out, rows; source_commit=source)
    selected, _ = ProfileCatalog.select_max_target(rows)
    json_out = splitext(out)[1] * ".json"
    open(json_out, "w") do io
        print(io, "{\"profile_schema\":2,\"source_commit\":\"", source,
            "\",\"selected_case_key\":\"", selected.case_key,
            "\",\"selection_metric\":\"core_seconds\"}\n")
    end
    println("HOTSPOT case_key=", selected.case_key)
end
