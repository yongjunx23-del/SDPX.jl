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
    resolved_tolerances::String = ""
end

_case_sort(c) = (String(c.catalog), String(c.family), String(c.tier), String(c.id), String(c.arithmetic))
_key(catalog, id, family, tier, arithmetic) = join((catalog, family, tier, id, arithmetic), "|")

function _sha(value)
    io = IOBuffer(); print(io, value); return bytes2hex(SHA.sha256(take!(io)))
end

function _generic_module()
    mod = Module(:ProfileGenericConicBenchmark)
    Core.eval(mod, :(import Main))
    Base.include(mod, GENERAL)
    return getproperty(mod, :GenericConicBenchmark)
end

function _v1_cases()
    isfile(GENERAL) || return ProfileCase[]
    g = _generic_module()
    cases = ProfileCase[]
    for spec in Base.invokelatest(g.inventory)
        eligible = spec.expected_status === :optimal &&
                   spec.known_objective !== nothing &&
                   spec.known_objective isa Real
        transform = (exactness="identity", fingerprint=_sha((spec.source, spec.id)))
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
    iterations = 0; status = :not_run
    cert_valid = false; objective = nothing; semantic = false; core = Float64[]
    for _ in 1:samples
        # Rebuild before every measured run; setup is recorded separately and
        # never enters the solver metric.
        bm = @timed Base.invokelatest(g.build, spec.problem, T, spec.params)
        solve = @timed SDPX.optimize!(bm.value; settings=Base.invokelatest(g._settings, T; threads))
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
        semantic = Base.invokelatest(g.validate_result, spec, br)
        push!(sample_semantics, semantic)
        try
            t = getfield(SDPX.diagnostics(result), :timings)
            v = getproperty(t, :core)
            v isa Real && push!(core, Float64(v))
        catch
        end
    end
    failed = !semantic ? (status == :iteration_limit ? "iteration_limit" : "semantic_or_certificate") : ""
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
        requested_route="auto", planned_route="auto", executed_route="auto",
        input_fingerprint=_sha(case.source), source_commit=get(ENV, "GITHUB_SHA", "local"),
        catalog_fingerprint=_sha(case.catalog), environment_fingerprint=_sha((VERSION, Sys.MACHINE)),
        provider_fingerprint=_sha((SDPX,)), resolved_tolerances=string(case.objective_tolerance),
        warmup_count=warmup ? 1 : 0)
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

function validate_profile_row(row::ProfileRow)
    row.solve_eligible && !row.build_only || return false
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
    if row.reference_objective !== nothing
        tol = something(row.objective_tolerance, Inf)
        all(x -> isfinite(x) && abs(x - row.reference_objective) <= tol,
            row.sample_objective) || return false
    end
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
        "resolved_tolerances"=>row.resolved_tolerances)
end

function write_profiles(path, rows; source_commit="unknown")
    selected = try first(select_max_target(rows; metric=:core_seconds)).case_key catch; "" end
    doc = Dict("profile_schema"=>PROFILE_SCHEMA, "source_commit"=>String(source_commit),
        "selected_case_key"=>selected,
        "warmup_excluded"=>true, "metric_policy"=>"core_median_then_solver_median",
        "row"=>[_row_dict(r) for r in rows])
    open(path, "w") do io; TOML.print(io, doc; sorted=true); end
    return path
end

function write_manifest(path, rows; source_commit="unknown")
    eligible, _ = select_max_target(rows; metric=:core_seconds)
    doc = Dict("manifest_schema"=>PROFILE_SCHEMA, "source_commit"=>String(source_commit),
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
    source_commit = get(ENV, "SDPX_PROFILE_SOURCE_COMMIT", get(ENV, "GITHUB_SHA", ""))
    isempty(source_commit) && (source_commit = try readchomp(`git rev-parse HEAD`) catch; "local" end)
    fixture = get(ENV, "SDPX_PROFILE_FIXTURE", "0") == "1"
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
