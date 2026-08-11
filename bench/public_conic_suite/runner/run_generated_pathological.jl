#!/usr/bin/env julia

"""
P0 generated-pathological cluster runner for the public conic suite.

The runner builds every case from `generators/SDPXPathologicalBenchmarks.jl`
and `runner/campaign.jl`; it performs no downloads, no Pkg operations, and
refuses to compute unless `JULIA_PKG_OFFLINE=true` is set.

Resource contract (enforced by `resource_gate`):
- regular: PBS ppn=5, Julia `-t 4`, solver threads 4, BLAS threads 1,
  Float64 and Float64x4;
- bigfloat: PBS ppn=1, Julia `-t 1`, solver threads 1, BLAS threads 1,
  BigFloat (default 256 bits).

Each row records the full audit trail defined in `result_schema.jl`.  A
`SUCCESS` marker is written only when every requested repetition passes the
identity, resource, status, objective, and certificate gates; otherwise a
`FAILED` marker is written and the process exits non-zero.
"""

using Dates
using LinearAlgebra
import MathOptInterface as MOI
using Printf
using SHA

const RUNNER_ROOT = normpath(@__DIR__)
const SUITE_ROOT = normpath(joinpath(RUNNER_ROOT, ".."))
include(joinpath(RUNNER_ROOT, "result_schema.jl"))
include(joinpath(RUNNER_ROOT, "campaign.jl"))
include(joinpath(SUITE_ROOT, "generators", "SDPXPathologicalBenchmarks.jl"))
using .GeneratedPathologicalResultSchema
using .GeneratedPathologicalCampaign
using .SDPXPathologicalBenchmarks

using JuMP
using SDPX
try
    @eval using MultiFloats
    @eval using MultiFloats: Float64x4
catch err
    error(
        "Float64x4 campaign requires MultiFloats in the Julia environment " *
        "(scripts/setup_benchmark_env.jl installs it): " *
        sprint(showerror, err),
    )
end

const CERTIFICATE_UNAVAILABLE = (
    available=false,
    valid=false,
    kind=:unknown,
    failures=(:certificate_unavailable,),
    primal_residual=NaN,
    dual_residual=NaN,
    primal_affine_residual=NaN,
    dual_affine_residual=NaN,
    primal_cone_violation=NaN,
    dual_cone_violation=NaN,
    primal_residual_scaled=NaN,
    dual_residual_scaled=NaN,
    equality_backward_error=NaN,
    dual_backward_error=NaN,
    gap_relative=NaN,
    complementarity_relative=NaN,
)

struct RunContext
    source::String
    output::String
    class::String
    expected_resources::NamedTuple
    spec::String
    bits::Int
    tol::Float64
    time_limit::Float64
    max_iterations::Int
    solver_threads::Int
    git_sha::String
    archive_sha::String
    source_sha256::String
    environment_sha256::String
    environment_project_path::String
    environment_project_sha256::String
    environment_manifest_sha256::String
    candidate_pathof::String
    candidate_pathof_sha256::String
    candidate_pathof_match::Bool
    jump_pathof::String
    multifloots_pathof::String
    julia_version::String
    os::String
    cpu_model::String
    blas_vendor::String
    pbs_ppn::String
    pbs_job_id::String
    hostname::String
    run_token::String
end

function parse_cli(args)
    values = Dict{String,String}()
    for argument in args
        startswith(argument, "--") || continue
        body = argument[3:end]
        parts = split(body, '='; limit=2)
        name = parts[1]
        value = length(parts) == 2 ? parts[2] : "true"
        values["--" * name] = value
    end
    return values
end

function cli(cfg, name, default)
    return get(cfg, name, default)
end

function parse_int(value, name)
    parsed = tryparse(Int, value)
    parsed === nothing && error("$name must be an integer, got: $value")
    return parsed
end

function parse_float(value, name)
    parsed = tryparse(Float64, value)
    parsed === nothing && error("$name must be a number, got: $value")
    return parsed
end

function parse_arithmetic_spec(spec)
    specs = filter(x -> !isempty(x), split(spec, ','))
    isempty(specs) && error("--arithmetic must not be empty")
    for item in specs
        supported = item == "float64" || item == "float64x4" ||
                    (startswith(item, "bigfloat") &&
                     length(item) > 8 &&
                     all(isdigit, item[9:end]))
        supported || error(
            "unsupported arithmetic '$item'; use float64, float64x4, or bigfloat<bits>",
        )
    end
    return specs
end

function _try_string(f)
    try
        return string(f())
    catch
        return ""
    end
end

function _capture(command::Cmd)
    try
        return strip(read(command, String))
    catch
        return ""
    end
end

function _num(value)
    value === nothing && return NaN
    value === missing && return NaN
    try
        return Float64(value)
    catch
        return NaN
    end
end

function _get_field(value, key, default)
    value === nothing && return default
    hasproperty(value, key) || return default
    field = getproperty(value, key)
    return field === nothing ? default : field
end

function _cpu_model()
    try
        name = getfield(Base.Sys, :CPU_NAME)
        isempty(name) || return name
    catch
    end
    try
        for line in eachline("/proc/cpuinfo")
            occursin(r"^model name\s*:", line) &&
                return strip(split(line, ':', limit=2)[2])
        end
    catch
    end
    return "unknown"
end

function _blas_vendor()
    try
        return string(BLAS.vendor())
    catch
        return "unknown"
    end
end

function _path_under(root, path)
    isempty(path) && return false
    r = normpath(root)
    p = normpath(path)
    p == r && return true
    return startswith(p, r * "/")
end

function _file_sha256(path)
    try
        isfile(path) || return ""
        return bytes2hex(sha256(read(path)))
    catch
        return ""
    end
end

function _source_subset_hash(root)
    files = String[]
    for rel in (
        "src",
        joinpath("bench", "public_conic_suite"),
        "Project.toml",
        "Manifest.toml",
    )
        path = joinpath(root, rel)
        if isfile(path)
            push!(files, path)
        elseif isdir(path)
            for (dirpath, dirnames, filenames) in
                walkdir(path; follow_symlinks=false)
                filter!(dirnames) do name
                    name != ".git" && name != "results" &&
                        name != "external" && name != ".julia"
                end
                for name in filenames
                    push!(files, joinpath(dirpath, name))
                end
            end
        end
    end
    sort!(files; by=path -> relpath(path, root))
    io = IOBuffer()
    for path in files
        write(io, relpath(path, root))
        write(io, UInt8(0))
        write(io, read(path))
        write(io, UInt8(0))
    end
    return bytes2hex(sha256(take!(io)))
end

function _environment_snapshot(
    spec,
    bits,
    tol,
    expected,
    julia_version,
    os,
    cpu_model,
    blas_vendor,
)
    project_path = _try_string(() -> Base.active_project())
    manifest_path = isempty(project_path) ? "" :
                    joinpath(dirname(project_path), "Manifest.toml")
    project_sha = _file_sha256(project_path)
    manifest_sha = _file_sha256(manifest_path)
    facts = join((
        "julia=$julia_version",
        "os=$os",
        "cpu=$cpu_model",
        "blas=$blas_vendor",
        "active_project=$project_path",
        "project_sha256=$project_sha",
        "manifest_sha256=$manifest_sha",
        "julia_threads=$(Threads.nthreads())",
        "blas_threads=$(BLAS.get_num_threads())",
        "solver_threads=$(expected.solver_threads)",
        "ppn=$(expected.ppn)",
        "arithmetic=$spec",
        "precision_bits=$bits",
        "tolerance=$tol",
    ), "\n")
    return (
        hash=bytes2hex(sha256(facts)),
        project_path=project_path,
        project_sha256=project_sha,
        manifest_sha256=manifest_sha,
    )
end

function _arithmetic_bits(spec)
    spec == "float64" && return 53
    spec == "float64x4" && return try Int(precision(Float64x4)) catch; 209 end
    startswith(spec, "bigfloat") && return parse(Int, spec[9:end])
    error("unsupported arithmetic: $spec")
end

function _resolve_arithmetic(spec)
    spec == "float64" && return (T=Float64, bits=53)
    spec == "float64x4" && return (T=Float64x4, bits=_arithmetic_bits(spec))
    if startswith(spec, "bigfloat")
        bits = parse(Int, spec[9:end])
        bits >= 128 || error("BigFloat precision must be at least 128 bits")
        return (T=BigFloat, bits=bits)
    end
    error("unsupported arithmetic: $spec")
end

function _default_tolerance(spec)
    spec == "float64" && return 1.0e-8
    spec == "float64x4" && return 1.0e-24
    startswith(spec, "bigfloat") && return 1.0e-32
    error("unsupported arithmetic: $spec")
end

function _resource_expectations(class)
    class == "regular" && return RESOURCE_MATRIX.regular
    class == "bigfloat" && return RESOURCE_MATRIX.bigfloat
    error("--resource-class must be 'regular' or 'bigfloat', got: $class")
end

function _expected_resources(spec, class_expectations)
    startswith(spec, "bigfloat") &&
        return (ppn=1, julia_threads=1, solver_threads=1, blas_threads=1)
    return class_expectations
end

function _solver_from_model(model)
    candidates = Any[]
    try
        push!(candidates, unsafe_backend(model))
    catch
    end
    try
        push!(candidates, JuMP.backend(model))
    catch
    end
    for candidate in candidates
        candidate === nothing && continue
        candidate isa SDPX.Optimizer && return candidate
        if hasproperty(candidate, :optimizer)
            inner = getproperty(candidate, :optimizer)
            inner isa SDPX.Optimizer && return inner
        end
    end
    return nothing
end

function _objective_sense(model, solver)
    if solver !== nothing && hasproperty(solver, :sense)
        sense = getproperty(solver, :sense)
        sense == MOI.MAX_SENSE && return "max"
        sense == MOI.MIN_SENSE && return "min"
    end
    text = lowercase(_try_string(() -> MOI.get(model, MOI.ObjectiveSense())))
    occursin("max", text) && return "max"
    occursin("min", text) && return "min"
    return ""
end

function _int_value(value)
    try
        return Int(value)
    catch
        return 0
    end
end

function _typed_value(value, ::Type{T}) where {T}
    value === nothing && return missing
    value === missing && return missing
    value isa Real && value != value && return missing
    try
        return convert(T, value)
    catch
        return missing
    end
end

function _max_typed(values...)
    finite_values = [value for value in values
                     if !(value isa Missing) &&
                        !(value isa Real && value != value)]
    isempty(finite_values) && return missing
    return maximum(finite_values)
end

function _certificate_margin(::Type{T}, details) where {T}
    isempty(details) && return missing
    margin = missing
    for detail in details
        detail isa NamedTuple || return missing
        ok = _get_field(detail, :ok, false)
        shift_resolved = _get_field(detail, :shift_resolved, false)
        # An unresolved failing block has no certified PSD lower bound.
        if !shift_resolved && !ok
            return missing
        end
        required_shift = _typed_value(
            _get_field(detail, :required_shift, NaN),
            T,
        )
        scale = _typed_value(_get_field(detail, :scale, 1), T)
        required_shift isa Real || return missing
        scale isa Real || return missing
        block_margin = -required_shift / scale
        margin = margin isa Missing ? block_margin : min(margin, block_margin)
    end
    return margin
end

"""
Normalized cone margin lower bounds derived from SDPX's target-arithmetic PSD
certificate (`-required_shift / scale` per block), never from a generic LAPACK
eigmin that BigFloat may not support.
"""
function _certificate_cone_margins(::Type{T}, certificate) where {T}
    primal_details = _get_field(
        _get_field(certificate, :primal_psd, (details=NamedTuple[],)),
        :details,
        NamedTuple[],
    )
    dual_details = _get_field(
        _get_field(certificate, :dual_psd, (details=NamedTuple[],)),
        :details,
        NamedTuple[],
    )
    return (
        primal=_certificate_margin(T, primal_details),
        dual=_certificate_margin(T, dual_details),
    )
end

function _formulation_labels(algorithm, lp_formulation)
    algo = string(algorithm)
    lp = string(lp_formulation)
    planned = if algo == "lp_primal_dual"
        "lp_native"
    elseif algo == "socp_fixed_trace_q3"
        "socp_native_fixed_trace"
    elseif algo == "socp_psd2"
        "socp_psd_arrow_reference"
    elseif algo == "socp_psd_lift"
        "socp_general_frontend_psd_lift"
    elseif algo == "sdp_primal_dual"
        "sdp_native"
    else
        algo
    end
    executed = planned
    if algo == "lp_primal_dual" &&
       lp != "not_applicable" && !isempty(lp)
        executed = lp
    end
    return (planned=planned, executed=executed, lp_formulation=lp)
end

function _candidate_metadata(source::String)
    # The cluster git does not support `git -C`; run from the source directory.
    commit = _capture(cd(`git rev-parse HEAD 2>/dev/null`, source))
    commit_file = "git:HEAD"
    if isempty(commit)
        commit_file = joinpath(dirname(source), "metadata", "source_commit.txt")
        if isfile(commit_file)
            commit = strip(read(commit_file, String))
        else
            commit_file = ""
        end
    end
    if isempty(commit)
        for file in (
            joinpath(source, "candidate_metadata.toml"),
            joinpath(source, "bench", "public_conic_suite", "candidate_metadata.toml"),
        )
            isfile(file) || continue
            for line in eachline(file)
                match = match(r"^\s*(?:commit|commit_sha|sdpx_git_sha)\s*=\s*[\"']?([0-9a-fA-F]{7,})", line)
                if match !== nothing
                    commit = match.captures[1]
                    commit_file = file
                    break
                end
            end
            isempty(commit) || break
        end
    end
    commit = strip(commit)
    archive_sha = ""
    for file in (
        joinpath(dirname(source), "metadata", "archive_sha.txt"),
        joinpath(dirname(source), "metadata", "archive_sha256.txt"),
        joinpath(dirname(source), "metadata", "archive_sha"),
    )
        isfile(file) || continue
        archive_sha = strip(read(file, String))
        isempty(archive_sha) || break
    end
    return (commit=commit, archive_sha=archive_sha, commit_file=commit_file)
end

function _certificate(ctx, problem, result, options)
    try
        return SDPX.result_certificate(problem, result, options)
    catch err
        return merge(
            CERTIFICATE_UNAVAILABLE,
            (
                valid=false,
                kind=:certificate_error,
                failures=(:certificate_exception,),
                exception=sprint(showerror, err),
            ),
        )
    end
end

function run_case(ctx::RunContext, T, case_cfg, case_index, repetition,
                  compile_seconds)
    expected = ctx.expected_resources
    expected_status = case_cfg.expected_status
    expected_norm = expected_normalized_status(expected_status)
    input_sha = campaign_input_hash(case_cfg.case, case_cfg.kwargs)
    kwargs = merge(
        case_cfg.kwargs,
        (
            tol=ctx.tol,
            max_iter=ctx.max_iterations,
            threads=ctx.solver_threads,
        ),
    )

    model = nothing
    meta = nothing
    setup_seconds = 0.0
    exception = ""
    try
        started = time()
        model, meta = build_case(case_cfg.case, T; kwargs...)
        setup_seconds = time() - started
    catch err
        exception = sprint(showerror, err)
    end

    solver = nothing
    result = nothing
    problem = nothing
    options = nothing
    total_seconds = 0.0
    raw_moi = ""
    raw_primal = ""
    raw_dual = ""
    raw_message = ""
    if model !== nothing
        solver = _solver_from_model(model)
        if solver !== nothing
            options = solver.options
            names = fieldnames(typeof(options))
            base_fields = NamedTuple{names}(
                Tuple(getproperty(options, name) for name in names),
            )
            overrides = (
                timing=true,
                diagnostics=true,
                max_time=ctx.time_limit,
            )
            merged = merge(base_fields, overrides)
            solver.options = SDPX.SolverOptions{T}(; merged...)
            options = solver.options
        end
        GC.gc()
        try
            started = time()
            optimize!(model)
            total_seconds = time() - started
            solver === nothing && (solver = _solver_from_model(model))
            result = solver === nothing ? nothing : solver.result
            raw_moi = _try_string(() -> termination_status(model))
            raw_primal = _try_string(() -> primal_status(model))
            raw_dual = _try_string(() -> dual_status(model))
            raw_message = _try_string(() -> raw_status(model))
            if solver !== nothing
                isempty(raw_message) &&
                    (raw_message = _try_string(() -> solver.result.message))
                problem = solver.problem
                options = solver.options
            end
        catch err
            exception = isempty(exception) ? sprint(showerror, err) : exception
        end
    end

    certificate = CERTIFICATE_UNAVAILABLE
    if result !== nothing && problem !== nothing && options !== nothing
        certificate = _certificate(ctx, problem, result, options)
    end

    raw_status = result === nothing ? "" : _try_string(() -> result.status)
    objective_sense = model === nothing ? "" : _objective_sense(model, solver)
    objective_primal = result === nothing ?
                       missing : natural_objective(result.pObj, objective_sense)
    objective_dual = result === nothing ?
                     missing : natural_objective(result.dObj, objective_sense)
    relative_gap = result === nothing ? missing : result.gap_rel
    expected_objective = _get_field(meta, :expected_objective, nothing)
    has_expected = expected_objective !== nothing &&
                   expected_status === :optimal
    objective_error = has_expected ?
                      abs(objective_primal - expected_objective) : missing
    objective_relative_error = has_expected ?
                               objective_error /
                               max(one(T), abs(expected_objective)) : missing

    margins = _get_field(certificate, :available, false) ?
              _certificate_cone_margins(T, certificate) :
              (primal=missing, dual=missing)
    cert_valid = _get_field(certificate, :valid, false)
    cert_kind = _get_field(certificate, :kind, :unknown)
    cert_type = _get_field(certificate, :available, false) ?
                string(cert_kind) : "not_available"
    cert_residual = _max_typed(
        _typed_value(
            _get_field(certificate, :primal_residual_scaled, NaN),
            T,
        ),
        _typed_value(
            _get_field(certificate, :dual_residual_scaled, NaN),
            T,
        ),
        _typed_value(
            _get_field(certificate, :gap_relative, NaN),
            T,
        ),
    )
    cert_failures = join(
        string.(_get_field(certificate, :failures, Symbol[])),
        ";",
    )

    timings = result === nothing ? nothing : result.timings
    solve_seconds = _num(_get_field(timings, :total, NaN))
    if isnan(solve_seconds) && solver !== nothing
        solve_seconds = _num(solver.solve_time)
    end
    presolve_seconds = NaN
    memory = nothing
    selected = nothing
    d_timings = nothing
    if result !== nothing && result.diagnostics !== nothing
        memory = result.diagnostics.memory
        selected = result.diagnostics.selected_algorithms
        d_timings = result.diagnostics.timings
        presolve_seconds = _num(_get_field(d_timings, :presolve, NaN))
    end
    factorization_seconds = _num(_get_field(timings, :kkt_factorization, NaN))
    schur_assembly_seconds = _num(_get_field(timings, :schur_assembly, NaN))
    refinement_seconds = NaN
    refinement_steps = result === nothing ? 0 :
                       _num(_get_field(
                           result.termination,
                           :total_refinement_steps,
                           0,
                       ))
    iterations = result === nothing ? 0 : result.iterations
    restarts = result === nothing ? 0 : result.restarts
    regularizations = result === nothing ? 0 : result.regularizations
    workspace_bytes = result === nothing ? 0 :
                      _int_value(_get_field(memory, :workspace_bytes, 0))
    process_peak_rss_bytes = result === nothing ? 0 :
                             _int_value(_get_field(
                                 memory,
                                 :process_peak_rss_bytes,
                                 0,
                             ))

    planned_backend = _get_field(selected, :planned_backend, :not_available)
    executed_backend = _get_field(selected, :executed_backend, :not_executed)
    lp_formulation = _get_field(selected, :lp_formulation, :not_applicable)
    kkt_backend = _get_field(selected, :kkt, :not_available)
    gram_kernel = _get_field(selected, :gram, :not_available)
    equality_method = _get_field(selected, :equality, :not_available)
    solver_algorithm = _get_field(selected, :solver, :not_available)
    backend_resolution = _get_field(selected, :backend_resolution, :planned)
    fallback_reason = _get_field(selected, :fallback_reason, :none)
    fallback_text = string(fallback_reason)
    fallback = fallback_text != "none" && !isempty(fallback_text)
    plan_algorithm = result !== nothing &&
                     result.diagnostics !== nothing ?
                     result.diagnostics.plan.algorithm : solver_algorithm
    planned_labels = _formulation_labels(plan_algorithm, lp_formulation)
    executed_labels = _formulation_labels(solver_algorithm, lp_formulation)
    planned_formulation = planned_labels.planned
    executed_formulation = executed_labels.executed
    lp_formulation_text = executed_labels.lp_formulation
    planned_backend_text = string(planned_backend)
    executed_backend_text = string(executed_backend)
    resolution_text = string(backend_resolution)
    route_authorized_resolution = resolution_text in (
        "post_presolve",
        "analytic_equality_only",
        "resolved_no_iteration",
        "not_resolved",
    )
    legacy_route = planned_backend_text == "not_available" &&
                   executed_backend_text == "not_executed"
    if legacy_route
        # Archive baselines predate the planned/executed backend fields.  Derive
        # route provenance from the actual legacy diagnostics instead of
        # weakening the strict current-candidate route gate.
        legacy_actual = (solver_algorithm, kkt_backend, gram_kernel)
        legacy_actual_ok = all(legacy_actual) do field
            text = string(field)
            !isempty(text) &&
            text != "not_available" &&
            text != "not_executed" &&
            text != "none"
        end
        legacy_fallback_ok = !fallback && fallback_text == "none"
        gate_route = legacy_actual_ok && legacy_fallback_ok
    elseif planned_backend_text == executed_backend_text
        gate_route = true
    elseif planned_backend_text == "lp_deferred" &&
           route_authorized_resolution
        no_iter = resolution_text in ("resolved_no_iteration", "not_resolved")
        gate_route = no_iter ?
                     executed_backend_text == "not_executed" :
                     executed_backend_text != "not_executed" &&
                     !isempty(executed_backend_text)
    else
        gate_route = fallback &&
                     fallback_text != "none" &&
                     !isempty(fallback_text)
    end

    ppn_value = tryparse(Int, ctx.pbs_ppn)
    ppn_ok = ppn_value !== nothing && ppn_value == expected.ppn
    gate_resource = Threads.nthreads() == expected.julia_threads &&
                    BLAS.get_num_threads() == expected.blas_threads &&
                    ctx.solver_threads == expected.solver_threads && ppn_ok
    gate_identity = ctx.candidate_pathof_match &&
                    !isempty(ctx.git_sha) &&
                    !isempty(ctx.archive_sha) &&
                    !isempty(ctx.source_sha256) &&
                    !isempty(input_sha) &&
                    !isempty(ctx.environment_sha256) &&
                    !isempty(ctx.environment_project_path) &&
                    !isempty(ctx.environment_project_sha256) &&
                    !isempty(ctx.environment_manifest_sha256) &&
                    !isempty(ctx.candidate_pathof_sha256)
    normalized_status = normalize_status(
        raw_status;
        certificate_valid=cert_valid,
        certificate_kind=string(cert_kind),
        raw_moi_status=raw_moi,
        allow_unresolved=expected_status === :weakly_infeasible,
    )
    weak_unresolved =
        expected_status === :weakly_infeasible &&
        normalized_status == "unresolved"
    weak_infeasible_ok =
        expected_status === :weakly_infeasible &&
        normalized_status == "certified_infeasible"
    gate_status = weak_unresolved || weak_infeasible_ok ||
                  normalized_status == expected_norm

    tol_T = convert(T, ctx.tol)
    objective_tol = 10 * tol_T
    expected_abs = has_expected ? abs(expected_objective) : one(T)
    gate_objective = !has_expected ||
                     (
                        objective_error isa Real &&
                        objective_relative_error isa Real &&
                        isfinite(objective_error) &&
                        isfinite(objective_relative_error) &&
                        objective_error <=
                            objective_tol * max(one(T), expected_abs) &&
                        objective_relative_error <= objective_tol
                     )
    gap_ok = relative_gap isa Missing ?
             false :
             isfinite(relative_gap) && relative_gap <= tol_T
    if weak_unresolved
        gate_certificate = true
    elseif expected_norm == "certified_infeasible" ||
           weak_infeasible_ok
        gate_certificate = cert_valid &&
                           string(cert_kind) in (
                               "primal_infeasibility",
                               "structural_infeasibility",
                               "auxiliary_dual_infeasibility",
                           )
    else
        gate_certificate = cert_valid &&
                           cert_residual isa Real &&
                           isfinite(cert_residual) &&
                           cert_residual <= tol_T && gap_ok
    end

    gate_failures = String[]
    gate_identity || push!(gate_failures, "identity")
    gate_resource || push!(gate_failures, "resource")
    gate_route || push!(gate_failures, "route")
    gate_status || push!(gate_failures, "status")
    gate_objective || push!(gate_failures, "objective")
    gate_certificate || push!(gate_failures, "certificate")
    isempty(exception) || push!(gate_failures, "exception")
    gate_pass = isempty(gate_failures)

    values = Dict{Symbol,Any}(
        :run_id => "$(ctx.run_token)-$(ctx.spec)-$(case_cfg.case)-r$(repetition)",
        :suite => "generated_pathological",
        :tier => "p0",
        :family => string(case_cfg.family),
        :problem => string(case_cfg.case),
        :case_index => case_index,
        :repetition => repetition,
        :severity => case_cfg.severity,
        :severity_rank => case_index,
        :arithmetic => ctx.spec,
        :precision_bits => ctx.bits,
        :julia_threads => Threads.nthreads(),
        :blas_threads => BLAS.get_num_threads(),
        :blas_vendor => ctx.blas_vendor,
        :solver_threads => ctx.solver_threads,
        :pbs_ppn => ctx.pbs_ppn,
        :pbs_job_id => ctx.pbs_job_id,
        :hostname => ctx.hostname,
        :resource_class => ctx.class,
        :expected_julia_threads => expected.julia_threads,
        :expected_solver_threads => expected.solver_threads,
        :expected_blas_threads => expected.blas_threads,
        :expected_ppn => expected.ppn,
        :resource_gate => gate_resource,
        :source_path => ctx.source,
        :source_sha256 => ctx.source_sha256,
        :archive_sha => ctx.archive_sha,
        :input_sha256 => input_sha,
        :environment_sha256 => ctx.environment_sha256,
        :environment_project_path => ctx.environment_project_path,
        :environment_project_sha256 => ctx.environment_project_sha256,
        :environment_manifest_sha256 => ctx.environment_manifest_sha256,
        :candidate_pathof => ctx.candidate_pathof,
        :candidate_pathof_sha256 => ctx.candidate_pathof_sha256,
        :candidate_pathof_match => ctx.candidate_pathof_match,
        :jump_pathof => ctx.jump_pathof,
        :multifloots_pathof => ctx.multifloots_pathof,
        :sdpx_git_sha => ctx.git_sha,
        :julia_version => ctx.julia_version,
        :os => ctx.os,
        :cpu_model => ctx.cpu_model,
        :max_iterations => ctx.max_iterations,
        :time_limit_seconds => ctx.time_limit,
        :tolerance_primal => ctx.tol,
        :tolerance_dual => ctx.tol,
        :tolerance_gap => ctx.tol,
        :planned_backend => string(planned_backend),
        :executed_backend => string(executed_backend),
        :planned_formulation => planned_formulation,
        :executed_formulation => executed_formulation,
        :lp_formulation => lp_formulation_text,
        :kkt_backend => string(kkt_backend),
        :gram_kernel => string(gram_kernel),
        :equality_method => string(equality_method),
        :solver_algorithm => string(solver_algorithm),
        :backend_resolution => string(backend_resolution),
        :fallback => fallback,
        :fallback_reason => fallback_text,
        :raw_status => raw_status,
        :raw_moi_status => raw_moi,
        :raw_primal_status => raw_primal,
        :raw_dual_status => raw_dual,
        :raw_message => raw_message,
        :normalized_status => normalized_status,
        :expected_status => string(expected_status),
        :expected_normalized_status => expected_norm,
        :objective_primal => objective_primal,
        :objective_dual => objective_dual,
        :objective_expected => has_expected ? expected_objective : missing,
        :objective_error => objective_error,
        :objective_relative_error => objective_relative_error,
        :relative_gap => relative_gap,
        :primal_residual_original =>
            _typed_value(_get_field(certificate, :primal_residual, NaN), T),
        :dual_residual_original =>
            _typed_value(_get_field(certificate, :dual_residual, NaN), T),
        :primal_affine_residual_original =>
            _typed_value(_get_field(certificate, :primal_affine_residual, NaN), T),
        :dual_affine_residual_original =>
            _typed_value(_get_field(certificate, :dual_affine_residual, NaN), T),
        :primal_cone_violation_original =>
            _typed_value(_get_field(certificate, :primal_cone_violation, NaN), T),
        :dual_cone_violation_original =>
            _typed_value(_get_field(certificate, :dual_cone_violation, NaN), T),
        :primal_residual_scaled_original =>
            _typed_value(_get_field(certificate, :primal_residual_scaled, NaN), T),
        :dual_residual_scaled_original =>
            _typed_value(_get_field(certificate, :dual_residual_scaled, NaN), T),
        :equality_backward_error_original =>
            _typed_value(_get_field(certificate, :equality_backward_error, NaN), T),
        :dual_backward_error_original =>
            _typed_value(_get_field(certificate, :dual_backward_error, NaN), T),
        :complementarity_relative =>
            _typed_value(_get_field(certificate, :complementarity_relative, NaN), T),
        :cone_margin_primal => margins.primal,
        :cone_margin_dual => margins.dual,
        :certificate_type => cert_type,
        :certificate_valid => cert_valid,
        :certificate_residual => cert_residual,
        :certificate_failures => cert_failures,
        :certificate_validation_precision_bits => ctx.bits,
        :setup_seconds => setup_seconds,
        :solve_seconds => solve_seconds,
        :total_seconds => total_seconds,
        :compile_seconds => compile_seconds,
        :presolve_seconds => presolve_seconds,
        :factorization_seconds => factorization_seconds,
        :schur_assembly_seconds => schur_assembly_seconds,
        :refinement_seconds => refinement_seconds,
        :iterations => iterations,
        :restarts => restarts,
        :regularizations => regularizations,
        :refinement_steps => refinement_steps,
        :workspace_bytes => workspace_bytes,
        :process_peak_rss_bytes => process_peak_rss_bytes,
        :exception => exception,
        :gate_identity => gate_identity,
        :gate_resource => gate_resource,
        :gate_route => gate_route,
        :gate_status => gate_status,
        :gate_objective => gate_objective,
        :gate_certificate => gate_certificate,
        :gate_failures => join(gate_failures, ";"),
        :gate_pass => gate_pass,
        :timestamp_utc => string(Dates.now(Dates.UTC)),
    )
    names = Tuple(RESULT_COLUMNS)
    return NamedTuple{names}(Tuple(
        get(values, column, nothing) for column in RESULT_COLUMNS
    ))
end

function run_arithmetic(ctx::RunContext, T, case_filter, warmup, repetitions)
    compile_seconds = 0.0
    campaign_rows = campaign_rows_for(ctx.bits)
    if warmup
        if !isempty(campaign_rows)
            first_cfg = first(campaign_rows)
            try
                started = time()
                run_case(ctx, T, first_cfg, 0, 0, 0.0)
                compile_seconds = time() - started
            catch err
                @warn "warmup failed; timed rows will carry compile cost" exception=(
                    err,
                    catch_backtrace(),
                )
            end
        end
    end
    rows = NamedTuple[]
    for (case_index, case_cfg) in enumerate(campaign_rows)
        isempty(case_filter) ||
            (string(case_cfg.case) in case_filter) ||
            continue
        for repetition in 1:repetitions
            push!(rows, run_case(
                ctx,
                T,
                case_cfg,
                case_index,
                repetition,
                compile_seconds,
            ))
        end
    end
    return rows
end

function main(args)
    cfg = parse_cli(args)
    offline = lowercase(get(ENV, "JULIA_PKG_OFFLINE", ""))
    offline == "true" ||
        error("compute must run offline: set JULIA_PKG_OFFLINE=true")

    source = cli(cfg, "--source", "")
    isempty(source) && error("--source=<candidate checkout> is required")
    source = normpath(abspath(source))
    isfile(joinpath(source, "src", "SDPX.jl")) ||
        error("candidate source is missing src/SDPX.jl: $source")

    output = cli(cfg, "--output", "")
    isempty(output) && error("--output=<result directory> is required")
    output = abspath(output)
    mkpath(output)

    class = cli(cfg, "--resource-class", "regular")
    expectations = _resource_expectations(class)
    specs = parse_arithmetic_spec(
        cli(cfg, "--arithmetic", expectations.default_arithmetic),
    )
    repetitions = parse_int(cli(cfg, "--repetitions", "1"), "--repetitions")
    repetitions >= 1 || error("--repetitions must be at least 1")
    time_limit = parse_float(cli(cfg, "--time-limit", "900"), "--time-limit")
    max_iterations = parse_int(
        cli(cfg, "--max-iterations", "300"),
        "--max-iterations",
    )
    warmup = lowercase(cli(cfg, "--warmup", "true")) == "true"
    tol_override = cli(cfg, "--tol", "")
    case_filter = filter(x -> !isempty(x), split(cli(cfg, "--case-filter", ""), ','))

    candidate = _candidate_metadata(source)
    git_sha = candidate.commit
    archive_sha = candidate.archive_sha
    isempty(git_sha) &&
        error("candidate commit not found; add metadata/source_commit.txt or use a git checkout: $source")
    source_sha256 = _source_subset_hash(source)
    pbs_ppn = get(ENV, "PBS_NP", get(ENV, "SDPX_PPN", "unknown"))
    pbs_job_id = get(ENV, "PBS_JOBID", "unknown")
    hostname = _try_string(() -> gethostname())
    julia_version = string(VERSION)
    os = string(Sys.KERNEL)
    cpu_model = _cpu_model()
    blas_vendor = _blas_vendor()
    candidate_pathof = pathof(SDPX)
    candidate_pathof_sha256 = _file_sha256(candidate_pathof)
    candidate_pathof_match = _path_under(source, candidate_pathof)
    jump_pathof = _try_string(() -> pathof(JuMP))
    multifloots_pathof = _try_string(() -> pathof(MultiFloats))
    run_token = Dates.format(now(UTC), "yyyymmddTHHMMSSZ")

    all_rows = NamedTuple[]
    environment_hashes = Dict{String,String}()
    for spec in specs
        T, bits = _resolve_arithmetic(spec)
        tol = isempty(tol_override) ?
              _default_tolerance(spec) :
              parse_float(tol_override, "--tol")
        expected = _expected_resources(spec, expectations)
        environment = _environment_snapshot(
            spec,
            bits,
            tol,
            expected,
            julia_version,
            os,
            cpu_model,
            blas_vendor,
        )
        environment_sha = environment.hash
        environment_hashes[spec] = environment_sha
        ctx = RunContext(
            source,
            output,
            class,
            expected,
            spec,
            bits,
            tol,
            time_limit,
            max_iterations,
            expected.solver_threads,
            git_sha,
            archive_sha,
            source_sha256,
            environment_sha,
            environment.project_path,
            environment.project_sha256,
            environment.manifest_sha256,
            candidate_pathof,
            candidate_pathof_sha256,
            candidate_pathof_match,
            jump_pathof,
            multifloots_pathof,
            julia_version,
            os,
            cpu_model,
            blas_vendor,
            pbs_ppn,
            pbs_job_id,
            hostname,
            run_token,
        )
        rows = if T === BigFloat
            setprecision(BigFloat, bits) do
                run_arithmetic(ctx, T, case_filter, warmup, repetitions)
            end
        else
            run_arithmetic(ctx, T, case_filter, warmup, repetitions)
        end
        append!(all_rows, rows)
    end

    for row in all_rows
        shape = check_columns(row)
        shape.ok || error(
            "row shape mismatch: missing=$(shape.missing) extra=$(shape.extra)",
        )
    end
    failures = [row for row in all_rows if !getproperty(row, :gate_pass)]
    gate_pass = !isempty(all_rows) && isempty(failures)

    results_csv = joinpath(output, "results.csv")
    failures_csv = joinpath(output, "failures.csv")
    manifest_path = joinpath(output, "benchmark_manifest.toml")
    report_path = joinpath(output, "report.md")
    success_path = joinpath(output, "SUCCESS")
    failure_marker_path = joinpath(output, "FAILED")

    write_csv(results_csv, all_rows)
    write_csv(failures_csv, failures)
    combined_environment_sha256 = bytes2hex(
        sha256(join(sort!(collect(values(environment_hashes))), "\n")),
    )
    manifest = Dict{String,Any}(
        "campaign_version" => CAMPAIGN_VERSION,
        "campaign_rows_defined" => length(CAMPAIGN),
        "campaign_rows_executed" => length(all_rows),
        "campaign_failures" => length(failures),
        "resource_class" => class,
        "arithmetic_specs" => specs,
        "repetitions" => repetitions,
        "source_path" => source,
        "source_sha256" => source_sha256,
        "sdpx_git_sha" => git_sha,
        "candidate_pathof" => candidate_pathof,
        "candidate_pathof_sha256" => candidate_pathof_sha256,
        "candidate_pathof_match" => candidate_pathof_match,
        "jump_pathof" => jump_pathof,
        "multifloots_pathof" => multifloots_pathof,
        "julia_version" => julia_version,
        "os" => os,
        "cpu_model" => cpu_model,
        "blas_vendor" => blas_vendor,
        "pbs_ppn" => pbs_ppn,
        "pbs_job_id" => pbs_job_id,
        "hostname" => hostname,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "time_limit_seconds" => time_limit,
        "max_iterations" => max_iterations,
        "environment_sha256" => combined_environment_sha256,
        "environment_sha256_by_arithmetic" => join(
            ("$key=$value" for (key, value) in sort!(collect(environment_hashes))),
            ";",
        ),
    )
    write_manifest(manifest_path, manifest)
    write_markdown_report(
        report_path;
        manifest=manifest,
        rows=all_rows,
        failures=failures,
        gate_pass=gate_pass,
        success_path=success_path,
    )

    if gate_pass
        write_success_marker(success_path; manifest=manifest, row_count=length(all_rows))
        ispath(failure_marker_path) && rm(failure_marker_path; force=true)
    else
        write_failure_marker(
            failure_marker_path;
            reason="$(length(failures)) of $(length(all_rows)) rows failed gates",
        )
        ispath(success_path) && rm(success_path; force=true)
    end

    @printf(
        "rows=%d failures=%d gate=%s results=%s\n",
        length(all_rows),
        length(failures),
        gate_pass ? "PASS" : "FAIL",
        results_csv,
    )
    for row in failures
        @printf(
            "FAIL %s family=%s severity=%s gates=%s\n",
            getproperty(row, :run_id),
            getproperty(row, :family),
            getproperty(row, :severity),
            getproperty(row, :gate_failures),
        )
    end
    exit(gate_pass ? 0 : 1)
end

main(ARGS)
