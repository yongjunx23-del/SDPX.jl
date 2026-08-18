const RESULT_COLUMNS = (
    :schema_version, :source_commit, :source_dirty, :julia_version, :os,
    :cpu_name, :hostname, :pbs_job_id, :julia_threads, :blas_threads,
    :project_sha256, :manifest_sha256, :benchmark_driver_sha256,
    :mfla_commit, :bfla_commit,
    :suite, :problem_id, :name, :family,
    :problem_type, :conic_formulation, :source, :purpose, :seed, :arithmetic,
    :precision_bits, :requested_provider, :status, :reference_status,
    :reference_absolute_tolerance, :reference_relative_tolerance,
    :skip_reason, :termination_reason,
    :variables, :equalities, :blocks, :block_sizes, :planned_formulation,
    :executed_formulation, :planned_backend, :executed_backend,
    :requested_scaling, :planned_scaling, :executed_scaling,
    :planned_provider, :executed_provider, :executed_specialization,
    :psd_lift_used, :fallback_reason,
    :la_fallback_reason, :iterations, :objective, :reference_objective,
    :physical_objective, :objective_interval_lower, :objective_interval_upper,
    :objective_in_reference_interval, :benchmark_scale,
    :input_generation_precision_bits, :original_equalities, :source_parameters,
    :objective_error, :primal_residual, :dual_residual, :relative_gap,
    :objective_relative_error, :b_correct, :analytic_family, :analytic_kind,
    :analytic_direction, :analytic_equivalence_group,
    :analytic_monotonic_group, :analytic_bound_group, :analytic_reference,
    :analytic_absolute_tolerance, :analytic_relative_tolerance,
    :complementarity, :relative_complementarity,
    :primal_cone_violation, :dual_cone_violation,
    :certificate_policy, :certificate_available, :certificate_valid,
    :certificate_kind, :certificate_failures, :certificate_provenance,
    :validation_precision_bits, :certificate_error,
    :provider_match, :unexpected_fallback,
    :production_invariants_valid, :full_numerical_gate_valid,
    :classification, :semantic_pass, :eligible_for_performance,
    :semantic_failures, :equivalence_gate_valid,
    :monotonicity_gate_valid, :bound_pair_gate_valid, :group_failures,
    :solve_settings, :nonzeros, :coefficient_dynamic_range,
    :setup_seconds, :equilibration_seconds, :certification_seconds,
    :independent_validation_seconds, :end_to_end_seconds,
    :factorization_count, :refinement_count,
    :total_seconds, :seconds_per_iteration,
    :allocated_bytes, :gc_seconds,
    :workspace_bytes, :process_peak_rss_bytes, :memory_budget_bytes,
    :sample_count, :sample_seconds, :sample_semantic_pass,
    :sample_status, :sample_iterations, :sample_objective,
    :sample_certificate_valid, :sample_route,
    :sample_semantic_parity, :sample_parity_failures,
    :sample_median_seconds, :sample_min_seconds, :sample_max_seconds,
    :sample_mad_seconds, :sample_spread_seconds,
    :assembly_seconds, :factor_seconds, :solve_seconds,
    :refinement_seconds, :local_metric_seconds, :local_factor_seconds,
    :panel_transform_seconds, :equality_gram_seconds, :equality_factor_seconds,
    :predictor_rhs_seconds, :corrector_rhs_seconds, :block_residual_seconds,
    :block_recovery_seconds, :local_metric_preparations,
    :equality_gram_assemblies, :equality_factorizations, :rhs_solves,
    :cone_composition, :input_fingerprint, :external_checksum,
)

const RESULT_SCHEMA_VERSION = 4

_cell(value) = value === missing || value === nothing ? "" :
               replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _source_commit()
    return try
        readchomp(`git -C $REPOSITORY rev-parse HEAD`)
    catch
        "unknown"
    end
end

function _source_dirty()
    return try
        !isempty(readchomp(`git -C $REPOSITORY status --porcelain`))
    catch
        true
    end
end

_cpu_name() = try
    Sys.CPU_NAME
catch
    "unknown"
end

_blas_threads() = try
    LinearAlgebra.BLAS.get_num_threads()
catch
    missing
end

_hostname() = try
    gethostname()
catch
    "unknown"
end

function _sha256_if_file(path)
    path === nothing && return missing
    isfile(path) || return missing
    return _sha256_file(path)
end

function _manifest_sha256()
    project = Base.active_project()
    project === nothing && return missing
    return _sha256_if_file(joinpath(dirname(project), "Manifest.toml"))
end

function _package_commit(package::AbstractString)
    source = try
        Base.find_package(package)
    catch
        nothing
    end
    source === nothing && return :not_loaded
    root = normpath(joinpath(dirname(source), ".."))
    return try
        readchomp(`git -C $root rev-parse HEAD`)
    catch
        :not_a_git_checkout
    end
end

_built_value(built, name::Symbol, default) =
    hasproperty(built, name) ? getproperty(built, name) : default

function _solve_settings(built)
    return _built_value(built, :solve_settings, (;))
end

function _arithmetic_label(::Type{T}) where {T}
    T === Float64 && return :float64
    T === BigFloat && return Symbol("bigfloat", precision(BigFloat))
    if SDPX.is_multifloat_arithmetic(T)
        parameters = Base.unwrap_unionall(T).parameters
        return Symbol("float64x", parameters[2])
    end
    return Symbol(lowercase(string(nameof(T))))
end

_precision_bits(::Type{BigFloat}) = precision(BigFloat)
_precision_bits(::Type{T}) where {T} = SDPX.sig_bits(T)

function _load_multifloat_type(arithmetic::Symbol)
    return get(MULTIFLOAT_TYPES, arithmetic, nothing)
end

function _arithmetic_type(arithmetic::Symbol)
    arithmetic === :float64 && return Float64
    startswith(string(arithmetic), "bigfloat") && return BigFloat
    T = _load_multifloat_type(arithmetic)
    T === nothing && throw(ArgumentError(
        "arithmetic $arithmetic is unavailable; install/load MultiFloats for fixed multiword arithmetic",
    ))
    return T
end

function _load_requested_provider(provider::Symbol)
    if provider === :bfla
        @eval import BigFloatLinearAlgebra
    elseif provider === :multifloat
        @eval import MultiFloatLinearAlgebra
    end
    return nothing
end

function _options(provider::Symbol, ::Type{T}, built=nothing; verbose=false) where {T}
    settings = built === nothing ? (;) : _solve_settings(built)
    default_tolerance = T === Float64 ? T(1.0e-8) : T(1.0e-20)
    tolerance = hasproperty(settings, :tolerance) ?
                parse(T, string(settings.tolerance)) : default_tolerance
    maximum_iterations = get(settings, :maximum_iterations, 200)
    max_time = get(settings, :max_time, Inf)
    return SDPX.SolveOptions(
        duality_gap_threshold=tolerance,
        primal_error_threshold=tolerance,
        dual_error_threshold=tolerance,
        maximum_iterations=maximum_iterations,
        max_runtime=max_time,
        threads=Threads.nthreads(),
        verbosity=0,
        presolve=get(settings, :presolve, :auto),
        scaling=get(settings, :scaling, :auto),
        algorithm=get(settings, :algorithm, :auto),
        sparse=get(settings, :sparse, :auto),
        formulation=get(settings, :formulation, :auto),
        equality_solver=get(settings, :equality_solver, :auto),
        linear_algebra_backend=provider,
        working_precision_policy=get(settings, :working_precision_policy, :auto),
        diagnostics=true,
        timing=true,
        certification=true,
    )
end

function _solve_built(built, ::Type{T}, provider; verbose=false) where {T}
    options = _options(provider, T, built; verbose=verbose)
    if built.kind === :socp
        settings = _solve_settings(built)
        specialization = get(settings, :specialization, :auto)
        if specialization !== :auto
            tolerance = parse(T, string(settings.tolerance))
            return SDPX.solve_socp(
                built.problem;
                specialization,
                tolerance,
                maximum_iterations=get(settings, :maximum_iterations, 200),
                max_time=get(settings, :max_time, Inf),
                verbosity=0,
                timing=true,
                diagnostics=true,
                certification=true,
                linear_algebra_backend=provider,
                threads=Threads.nthreads(),
            )
        end
        return SDPX.solve_socp(built.problem, options)
    end
    return SDPX.solve(built.problem, options)
end

function _safe_certificate(problem, result, ::Type{T}, built=nothing) where {T}
    settings = built === nothing ? (;) : _solve_settings(built)
    tolerance = hasproperty(settings, :tolerance) ?
                parse(T, string(settings.tolerance)) :
                (T === Float64 ? T(1e-8) : T(1e-20))
    try
        return (value=SDPX.result_certificate(
            problem,
            result,
            SDPX.SolverOptions{T}(
                ϵ_gap=tolerance,
                ϵ_primal=tolerance,
                ϵ_dual=tolerance,
            ),
        ), error=missing)
    catch exception
        return (value=nothing, error=sprint(showerror, exception))
    end
end

function _problem_facts(problem)
    if problem isa SDPX.SDPProblem
        return (
            variables=problem.dims.m,
            equalities=problem.dims.n,
            blocks=problem.dims.L,
            block_sizes=Tuple(problem.dims.k),
        )
    elseif problem isa SDPX.ConicProblem
        return (
            variables=length(problem.c),
            equalities=size(problem.Aeq, 1),
            blocks=length(problem.cones),
            block_sizes=Tuple(size(cone.A, 1) for cone in problem.cones),
        )
    end
    throw(ArgumentError(
        "benchmark facts require SDPProblem or ConicProblem, got $(typeof(problem))",
    ))
end

function _problem_fingerprint(spec, built, arithmetic)
    facts = _problem_facts(built.problem)
    builder_paths = if spec.external === nothing
        paths = [joinpath(ROOT, "generators", "problems.jl")]
        spec.loader === :analytic && push!(
            paths, joinpath(ROOT, "analytic", "AnalyticFamilies.jl"),
        )
        paths
    else
        [joinpath(ROOT, "loaders", "csdr_fixed_trace.jl")]
    end
    builder_digest = bytes2hex(SHA.sha256(vcat(
        (read(path) for path in builder_paths)...,
    )))
    payload = join((
        spec.id,
        string(arithmetic),
        string(spec.seed),
        repr(spec.parameters),
        repr(facts),
        string(_built_value(built, :external_checksum, "")),
        builder_digest,
    ), "|")
    return bytes2hex(SHA.sha256(payload))
end

function _problem_nonzeros(problem)
    if problem isa SDPX.ConicProblem
        total = Base.count(!iszero, problem.c) + Base.count(!iszero, problem.beq)
        total += Base.count(!iszero, problem.Aeq)
        for cone in problem.cones
            total += Base.count(!iszero, cone.A) + Base.count(!iszero, cone.b)
        end
        return total
    elseif problem isa SDPX.SDPProblem
        total = Base.count(!iszero, problem.c) + Base.count(!iszero, problem.b)
        total += Base.count(!iszero, problem.B)
        total += sum(Base.count(!iszero, block) for block in problem.C)
        if problem.cons isa SDPX.DenseCons
            total += sum(Base.count(!iszero, block) for block in problem.cons.Av)
        elseif problem.cons isa SDPX.SparseCons
            total += sum(length(block.val) for block in problem.cons.coo)
        end
        return total
    end
    return missing
end

function _cone_composition(problem)
    if problem isa SDPX.ConicProblem
        dimensions = sort!(collect(size(cone.A, 1) for cone in problem.cones))
        counts = Dict{Int,Int}()
        for dimension in dimensions
            counts[dimension] = get(counts, dimension, 0) + 1
        end
        return join(("lorentz$(dimension)x$(counts[dimension])"
                     for dimension in sort!(collect(keys(counts)))), ",")
    elseif problem isa SDPX.SDPProblem
        counts = Dict{Int,Int}()
        for dimension in problem.dims.k
            counts[dimension] = get(counts, dimension, 0) + 1
        end
        return join(("psd$(dimension)x$(counts[dimension])"
                     for dimension in sort!(collect(keys(counts)))), ",")
    end
    return missing
end

function _coefficient_dynamic_range(problem)
    minimum_value = nothing
    maximum_value = nothing
    function scan(values)
        for value in values
            magnitude = abs(value)
            iszero(magnitude) && continue
            minimum_value = minimum_value === nothing ? magnitude :
                            min(minimum_value, magnitude)
            maximum_value = maximum_value === nothing ? magnitude :
                            max(maximum_value, magnitude)
        end
    end
    if problem isa SDPX.ConicProblem
        scan(problem.c)
        scan(problem.Aeq)
        scan(problem.beq)
        for cone in problem.cones
            scan(cone.A)
            scan(cone.b)
        end
    elseif problem isa SDPX.SDPProblem
        scan(problem.c)
        scan(problem.b)
        scan(problem.B)
        foreach(scan, problem.C)
        if problem.cons isa SDPX.DenseCons
            foreach(scan, problem.cons.Av)
        elseif problem.cons isa SDPX.SparseCons
            foreach(block -> scan(block.val), problem.cons.coo)
        end
    else
        return missing
    end
    minimum_value === nothing && return zero(eltype(problem))
    return maximum_value / minimum_value
end

_analytic_contract(built) = _built_value(built, :analytic_contract, nothing)

_trace_value(value) = SDPX.isavailable(value) ? value : missing
_trace_field(record, name::Symbol) = hasproperty(record, name) ?
    _trace_value(getproperty(record, name)) : missing
_sym(value) = value isa Symbol ? value : missing

_normalized_status(status) = Symbol(lowercase(string(status)))

function _classification(status, semantic_pass::Bool)
    semantic_pass && return :PASS
    normalized = _normalized_status(status)
    normalized in (
        :optimal,
        :primalinfeasible,
        :dualinfeasible,
        :infeasiblecert,
        :error,
    ) && return :FAIL
    return :UNRESOLVED
end

function _unexpected_fallback(spec, fallback_reason, la_fallback_reason)
    planned_reasons = (
        :none,
        :not_executed,
        :route_not_migrated,
        :requested_legacy,
        :compatibility,
        :unsupported_arithmetic,
    )
    fallback_reason === missing && return true
    la_fallback_reason === missing && return true
    fallback_reason in planned_reasons || return true
    la_fallback_reason in planned_reasons && return false
    :expected_la_factor_failure in spec.tags &&
        return la_fallback_reason !== :la_equality_factor_failed
    if :rank_deficient in spec.tags || :rank_ladder in spec.tags
        return !(la_fallback_reason in (:la_equality_factor_failed, :la_factor_failed))
    end
    return true
end

function _semantic_failures(
    spec,
    status,
    objective,
    expected,
    analytic_contract,
    primal_residual,
    dual_residual,
    relative_gap,
    certificate,
    provider_match,
    unexpected_fallback,
)
    failures = String[]
    _normalized_status(status) === _normalized_status(spec.reference.status) ||
        push!(failures, "status")
    actual_optimal = _normalized_status(status) === :optimal
    if actual_optimal
        if analytic_contract !== nothing
            reference = get(analytic_contract, :reference, nothing)
            direction = get(analytic_contract, :direction, :exact)
            if reference !== nothing
                absolute = get(analytic_contract, :absolute_tolerance, 0.0)
                relative = get(analytic_contract, :relative_tolerance, 0.0)
                allowed = absolute + relative * abs(reference)
                error = abs(objective - reference)
                objective_ok = if direction === :lower
                    objective <= reference + allowed
                elseif direction === :upper
                    objective >= reference - allowed
                else
                    isfinite(error) && error <= allowed
                end
                objective_ok || push!(failures, direction === :exact ?
                    "objective" : "objective_bound")
            end
        elseif expected !== nothing
            error = abs(objective - expected)
            allowed = spec.reference.absolute_tolerance +
                      spec.reference.relative_tolerance * max(one(error), abs(expected))
            (!isfinite(error) || error > allowed) && push!(failures, "objective")
        end
    end
    residual_tolerance = analytic_contract === nothing ?
        max(spec.reference.absolute_tolerance,
            spec.reference.relative_tolerance) :
        get(analytic_contract, :residual_tolerance,
            spec.reference.relative_tolerance)
    residual_limit = 10 * residual_tolerance
    if actual_optimal
        for (name, value) in (
            ("primal_residual", primal_residual),
            ("dual_residual", dual_residual),
            ("relative_gap", relative_gap),
        )
            if value === missing
                push!(failures, name * "_missing")
            elseif !isfinite(value) || abs(value) > residual_limit
                push!(failures, name)
            end
        end
    end
    if actual_optimal && spec.family in (:lp, :socp, :sdp)
        (certificate === nothing || !certificate.valid) &&
            push!(failures, "certificate")
    end
    provider_match || push!(failures, "provider_mismatch")
    unexpected_fallback && push!(failures, "unexpected_fallback")
    return failures
end

function _explicit_provider_matches(requested, executed, ::Type{T}) where {T}
    requested === :auto && return true
    executed === missing && return false
    expected = requested === :multifloat ? :multifloat_linear_algebra :
               requested === :bfla ? :bigfloat_linear_algebra :
               requested === :legacy ? :sdpx_legacy_la :
               requested === :standard ?
                   (T <: Union{Float32,Float64} ? :blas_lapack :
                    :generic_linear_algebra) : requested
    return executed === expected
end

function _formulation(trace, result, planned::Bool)
    diagnostics = result isa SDPX.ConicResult ? result.diagnostics : result.diagnostics
    diagnostics === nothing && return missing
    selected = diagnostics.selected_algorithms
    if planned
        return get(selected, :planned_kkt_formulation, missing)
    end
    return get(selected, :executed_kkt_formulation,
               get(selected, :planned_kkt_formulation, missing))
end

function _specialization(result)
    diagnostics = result.diagnostics
    diagnostics === nothing && return missing
    selected = diagnostics.selected_algorithms
    return get(selected, :soc_specialization, missing)
end

function _physical_objective(built, result, default)
    objective_map = _built_value(built, :physical_objective, nothing)
    objective_map === nothing && return default
    return objective_map(result.x)
end

function _reference_interval(built, ::Type{T}) where {T}
    interval = _built_value(built, :objective_interval, nothing)
    interval === nothing && return nothing
    return (
        lower=parse(T, string(interval.lower)),
        upper=parse(T, string(interval.upper)),
    )
end

function _result_row(
    spec, suite, arithmetic, provider, built, result, elapsed;
    allocated_bytes=missing,
    gc_seconds=missing,
)
    trace = SDPX.performance_trace(result)
    T = eltype(built.problem)
    certificate_measurement = @timed _safe_certificate(
        built.problem, result, T, built,
    )
    certificate_record = certificate_measurement.value
    certificate = certificate_record.value
    certificate_error = certificate_record.error
    native_objective = trace.final.primal_objective
    objective = _physical_objective(built, result, native_objective)
    interval = _reference_interval(built, T)
    interval_pass = interval === nothing ? missing :
                    interval.lower <= objective <= interval.upper
    analytic = _analytic_contract(built)
    analytic_reference = analytic === nothing ? nothing :
                         get(analytic, :reference, nothing)
    expected = analytic_reference !== nothing ?
               (get(analytic, :kind, :exact) === :exact ? analytic_reference : nothing) :
               (built.expected === nothing ? spec.reference.objective : built.expected)
    objective_error_reference = analytic_reference !== nothing ?
                                analytic_reference : expected
    objective_error = objective_error_reference === nothing ? missing :
                      abs(objective - objective_error_reference)
    objective_error_scale = if objective_error_reference === nothing
        nothing
    elseif iszero(objective_error_reference)
        eps(one(objective_error_reference))
    else
        max(abs(objective_error_reference), eps(abs(objective_error_reference)))
    end
    objective_relative_error = objective_error === missing ? missing :
        objective_error / objective_error_scale
    b_correct = if objective_error === missing
        missing
    elseif iszero(objective_error)
        Inf
    else
        Float64(-log2(objective_relative_error))
    end
    primal_residual = trace.final.primal_residual
    dual_residual = trace.final.dual_residual
    relative_gap = trace.final.relative_gap
    fallback_reason = _trace_value(trace.setup.fallback_reason)
    la_fallback_reason = _trace_value(trace.setup.executed_la_fallback_reason)
    executed_provider = _trace_value(trace.setup.executed_la_provider)
    provider_match = _explicit_provider_matches(
        provider, executed_provider, eltype(built.problem),
    )
    unexpected_fallback = _unexpected_fallback(
        spec, fallback_reason, la_fallback_reason,
    )
    failures = _semantic_failures(
        spec,
        trace.final.status,
        objective,
        expected,
        _analytic_contract(built),
        primal_residual,
        dual_residual,
        relative_gap,
        certificate,
        provider_match,
        unexpected_fallback,
    )
    specialization = _specialization(result)
    required_specialization = _built_value(
        built, :required_specialization, nothing,
    )
    required_specialization !== nothing &&
        specialization !== required_specialization &&
        push!(failures, "specialization")
    # NativeSOCDiagnostics is the positive evidence that a ConicResult came
    # from direct Lorentz execution. Test/reference PSD helpers deliberately
    # retain SDP diagnostics, so the benchmark gate remains able to detect a
    # reference lift without restoring the removed compatibility payload.
    psd_lift_used =
        result isa SDPX.ConicResult &&
        result.diagnostics !== nothing &&
        !(result.diagnostics isa SDPX.NativeSOCDiagnostics)
    _built_value(built, :forbid_psd_lift, false) && psd_lift_used &&
        push!(failures, "psd_lift")
    interval_pass === false && push!(failures, "objective_interval")
    maximum_relative_gap = _built_value(
        built, :maximum_relative_gap, nothing,
    )
    if maximum_relative_gap !== nothing
        limit = parse(T, string(maximum_relative_gap))
        (relative_gap === missing || !isfinite(relative_gap) ||
         abs(relative_gap) > limit) && push!(failures, "relative_gap_strict")
    end
    production_invariants =
        (required_specialization === nothing ||
         specialization === required_specialization) &&
        provider_match && !psd_lift_used && !unexpected_fallback
    if _built_value(built, :require_no_fallback, false)
        exact_no_fallback = fallback_reason === :none && la_fallback_reason === :none
        exact_no_fallback || push!(failures, "fallback")
        production_invariants &= exact_no_fallback
    end
    facts = _problem_facts(built.problem)
    certificate_failures = certificate === nothing ? missing :
        get(certificate, :failures, missing)
    certificate_kind = certificate === nothing ? missing :
        get(certificate, :kind, missing)
    certificate_provenance = certificate === nothing ? nothing :
        get(certificate, :provenance, nothing)
    validation_precision_bits = certificate_provenance === nothing ? missing :
        get(certificate_provenance, :validation_precision_bits,
            get(certificate_provenance, :precision_bits, missing))
    semantic_pass = isempty(failures)
    classification = _classification(trace.final.status, semantic_pass)
    solve_seconds = begin
        predictor = _trace_value(trace.iteration.predictor_solve_seconds)
        corrector = _trace_value(trace.iteration.corrector_solve_seconds)
        predictor === missing || corrector === missing ? missing : predictor + corrector
    end
    return (
        schema_version=RESULT_SCHEMA_VERSION,
        source_commit=_source_commit(),
        source_dirty=_source_dirty(),
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        cpu_name=_cpu_name(),
        hostname=_hostname(),
        pbs_job_id=get(ENV, "PBS_JOBID", missing),
        julia_threads=Threads.nthreads(),
        blas_threads=_blas_threads(),
        project_sha256=_sha256_if_file(Base.active_project()),
        manifest_sha256=_manifest_sha256(),
        benchmark_driver_sha256=_sha256_file(joinpath(ROOT, "runner_impl.jl")),
        mfla_commit=_package_commit("MultiFloatLinearAlgebra"),
        bfla_commit=_package_commit("BigFloatLinearAlgebra"),
        suite=suite,
        problem_id=spec.id,
        name=spec.name,
        family=spec.family,
        problem_type=spec.problem_type,
        conic_formulation=built.kind === :socp ? :native_lorentz :
                          spec.family === :lp ? :lp_native : :sdp_native,
        source=spec.source,
        purpose=spec.purpose,
        seed=spec.seed,
        arithmetic=arithmetic,
        precision_bits=_precision_bits(eltype(built.problem)),
        requested_provider=provider,
        status=trace.final.status,
        reference_status=spec.reference.status,
        reference_absolute_tolerance=spec.reference.absolute_tolerance,
        reference_relative_tolerance=spec.reference.relative_tolerance,
        skip_reason=missing,
        termination_reason=_trace_value(trace.final.termination_reason),
        variables=facts.variables,
        equalities=facts.equalities,
        blocks=facts.blocks,
        block_sizes=facts.block_sizes,
        planned_formulation=_formulation(trace, result, true),
        executed_formulation=_formulation(trace, result, false),
        planned_backend=_trace_value(trace.setup.planned_backend),
        executed_backend=_trace_value(trace.setup.executed_backend),
        requested_scaling=get(_solve_settings(built), :scaling, :auto),
        planned_scaling=_trace_value(trace.setup.planned_scaling),
        executed_scaling=_trace_value(trace.setup.executed_scaling),
        planned_provider=_trace_value(trace.setup.planned_la_provider),
        executed_provider=executed_provider,
        executed_specialization=specialization,
        psd_lift_used,
        fallback_reason=fallback_reason,
        la_fallback_reason=la_fallback_reason,
        iterations=trace.counters.iterations,
        objective=string(objective),
        reference_objective=expected === nothing ? missing : string(expected),
        physical_objective=
            _built_value(built, :physical_objective, nothing) === nothing ?
            missing : string(objective),
        objective_interval_lower=interval === nothing ? missing : string(interval.lower),
        objective_interval_upper=interval === nothing ? missing : string(interval.upper),
        objective_in_reference_interval=interval_pass,
        benchmark_scale=_built_value(built, :benchmark_scale, missing),
        input_generation_precision_bits=_built_value(
            built, :input_generation_precision_bits, missing,
        ),
        original_equalities=_built_value(built, :original_equalities, missing),
        source_parameters=begin
            parameters = _built_value(built, :source_parameters, nothing)
            parameters === nothing ? missing : repr(parameters)
        end,
        objective_error=objective_error === missing ? missing : string(objective_error),
        primal_residual=string(primal_residual),
        dual_residual=string(dual_residual),
        relative_gap=string(relative_gap),
        objective_relative_error=objective_relative_error === missing ? missing :
                                 string(objective_relative_error),
        b_correct=b_correct,
        analytic_family=analytic === nothing ? missing : get(analytic, :family, missing),
        analytic_kind=analytic === nothing ? missing : get(analytic, :kind, missing),
        analytic_direction=analytic === nothing ? missing : get(analytic, :direction, missing),
        analytic_equivalence_group=analytic === nothing ? missing :
                                   get(analytic, :equivalence_group, missing),
        analytic_monotonic_group=analytic === nothing ? missing :
                                 get(analytic, :monotonic_group, missing),
        analytic_bound_group=analytic === nothing ? missing :
                             get(analytic, :bound_group, missing),
        analytic_reference=analytic_reference === nothing ? missing :
                           string(analytic_reference),
        analytic_absolute_tolerance=analytic === nothing ? missing :
            string(get(analytic, :absolute_tolerance, missing)),
        analytic_relative_tolerance=analytic === nothing ? missing :
            string(get(analytic, :relative_tolerance, missing)),
        complementarity=certificate === nothing ? missing :
                        get(certificate, :complementarity, missing),
        relative_complementarity=certificate === nothing ? missing :
                                 get(certificate, :complementarity_relative, missing),
        primal_cone_violation=certificate === nothing ? missing :
                              get(certificate, :primal_cone_violation, missing),
        dual_cone_violation=certificate === nothing ? missing :
                            get(certificate, :dual_cone_violation, missing),
        certificate_policy=:original_coordinate_required,
        certificate_available=certificate !== nothing,
        certificate_valid=certificate === nothing ? missing : certificate.valid,
        certificate_kind=certificate_kind,
        certificate_failures=certificate_failures,
        certificate_provenance=certificate_provenance === nothing ? missing :
                               repr(certificate_provenance),
        validation_precision_bits=validation_precision_bits,
        certificate_error=certificate_error,
        provider_match=provider_match,
        unexpected_fallback=unexpected_fallback,
        production_invariants_valid=production_invariants,
        full_numerical_gate_valid=isempty(failures),
        classification=classification,
        semantic_pass=semantic_pass,
        eligible_for_performance=classification === :PASS,
        semantic_failures=join(failures, ","),
        equivalence_gate_valid=missing,
        monotonicity_gate_valid=missing,
        bound_pair_gate_valid=missing,
        group_failures="",
        solve_settings=repr(_solve_settings(built)),
        nonzeros=_problem_nonzeros(built.problem),
        coefficient_dynamic_range=string(
            _coefficient_dynamic_range(built.problem),
        ),
        setup_seconds=_trace_value(trace.setup.setup_seconds),
        equilibration_seconds=_trace_value(trace.setup.equilibration_seconds),
        certification_seconds=_trace_value(trace.final.certification_seconds),
        independent_validation_seconds=certificate_measurement.time,
        end_to_end_seconds=elapsed + certificate_measurement.time,
        factorization_count=_trace_field(trace.counters, :numeric_factorizations),
        refinement_count=_trace_field(trace.counters, :refinement_solves),
        total_seconds=elapsed,
        seconds_per_iteration=elapsed / max(trace.counters.iterations, 1),
        allocated_bytes,
        gc_seconds,
        workspace_bytes=_trace_value(trace.final.workspace_bytes),
        process_peak_rss_bytes=_trace_value(trace.final.process_peak_rss_bytes),
        memory_budget_bytes=_trace_value(trace.final.memory_budget_bytes),
        sample_count=1,
        sample_seconds=missing,
        sample_semantic_pass=missing,
        sample_status=missing,
        sample_iterations=missing,
        sample_objective=missing,
        sample_certificate_valid=missing,
        sample_route=missing,
        sample_semantic_parity=missing,
        sample_parity_failures=missing,
        sample_median_seconds=missing,
        sample_min_seconds=missing,
        sample_max_seconds=missing,
        sample_mad_seconds=missing,
        sample_spread_seconds=missing,
        assembly_seconds=_trace_value(trace.iteration.schur_assembly_seconds),
        factor_seconds=_trace_value(trace.iteration.kkt_factorization_seconds),
        solve_seconds=solve_seconds,
        refinement_seconds=_trace_value(trace.iteration.refinement_seconds),
        local_metric_seconds=_trace_value(trace.iteration.fixed_local_metric_seconds),
        local_factor_seconds=_trace_value(trace.iteration.fixed_local_factor_seconds),
        panel_transform_seconds=
            _trace_value(trace.iteration.equality_panel_transform_seconds),
        equality_gram_seconds=
            _trace_value(trace.iteration.equality_gram_syrk_seconds),
        equality_factor_seconds=
            _trace_value(trace.iteration.equality_factor_seconds),
        predictor_rhs_seconds=_trace_value(trace.iteration.predictor_rhs_seconds),
        corrector_rhs_seconds=_trace_value(trace.iteration.corrector_rhs_seconds),
        block_residual_seconds=
            _trace_value(trace.iteration.fixed_block_residual_seconds),
        block_recovery_seconds=
            _trace_value(trace.iteration.fixed_block_recovery_seconds),
        local_metric_preparations=
            _trace_field(trace.counters, :local_metric_preparations),
        equality_gram_assemblies=
            _trace_field(trace.counters, :equality_gram_assemblies),
        equality_factorizations=
            _trace_field(trace.counters, :equality_factorizations),
        rhs_solves=_trace_field(trace.counters, :rhs_solves),
        cone_composition=_cone_composition(built.problem),
        input_fingerprint=_problem_fingerprint(spec, built, arithmetic),
        external_checksum=_built_value(built, :external_checksum, missing),
    )
end

function _json_escape(value)
    escaped = replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
    return replace(escaped, "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
end

function _sample_json_list(values)
    isempty(values) && return missing
    payload = join((
        value isa AbstractString ? "\"" * _json_escape(value) * "\"" :
        value === true ? "true" :
        value === false ? "false" :
        value === nothing || value === missing ? "null" :
        repr(value)
        for value in values
    ), ",")
    return "[" * payload * "]"
end

function _median(values)
    ordered = sort!(collect(Float64, values))
    isempty(ordered) && return NaN
    midpoint = (length(ordered) + 1) ÷ 2
    return isodd(length(ordered)) ? ordered[midpoint] :
           (ordered[midpoint] + ordered[midpoint + 1]) / 2
end

function _sampling_summary(sample_seconds)
    seconds = [Float64(value) for value in sample_seconds]
    ordered = sort(seconds)
    center = _median(ordered)
    deviations = sort(abs.(ordered .- center))
    return (
        median_seconds=center,
        min_seconds=first(ordered),
        max_seconds=last(ordered),
        mad_seconds=_median(deviations),
        spread_seconds=last(ordered) - first(ordered),
    )
end

function _sample_route_key(row)
    fields = (
        :conic_formulation, :planned_formulation, :executed_formulation,
        :planned_backend, :executed_backend, :planned_provider,
        :executed_provider, :executed_specialization, :psd_lift_used,
        :fallback_reason, :la_fallback_reason,
    )
    return join((_cell(getproperty(row, field)) for field in fields), "|")
end

function _reference_tolerance(row)
    analytic_absolute = getproperty(row, :analytic_absolute_tolerance)
    analytic_relative = getproperty(row, :analytic_relative_tolerance)
    analytic = analytic_absolute !== missing && analytic_relative !== missing
    absolute = analytic ? analytic_absolute :
               getproperty(row, :reference_absolute_tolerance)
    relative = analytic ? analytic_relative :
               getproperty(row, :reference_relative_tolerance)
    if analytic
        try
            parse(BigFloat, string(absolute)) >= 0 &&
                parse(BigFloat, string(relative)) >= 0 || return nothing, nothing, false
            return string(absolute), string(relative), false
        catch
            return nothing, nothing, false
        end
    end
    if absolute isa Real && isfinite(absolute) && absolute >= 0 &&
       relative isa Real && isfinite(relative) && relative >= 0
        return string(absolute), string(relative), true
    end
    return nothing, nothing, true
end

function _parity_precision_bits(values...)
    lengths = Int[]
    for value in values
        value === nothing && continue
        push!(lengths, length(string(value)))
    end
    return max(256, ceil(Int, 4 * maximum(lengths; init=0)))
end

function _objective_parity(rows)
    # Semantic agreement, not bitwise string equality: exact equality is the
    # fast path, otherwise compare through BigFloat against the benchmark
    # row's reference absolute/relative tolerances. Missing/unusable
    # tolerances fall back to exact equality; unparseable or non-finite
    # objective values fail closed. Precision is derived from the decimal
    # string lengths (>=256 bits) and scoped locally, so long BigFloat
    # objectives are never collapsed at the ambient working precision.
    try
        objective_strings = [
            string(getproperty(row, :objective)) for row in rows
        ]
        absolute, relative, unit_scale = _reference_tolerance(first(rows))
        bits = _parity_precision_bits(
            objective_strings..., absolute, relative,
        )
        return setprecision(BigFloat, bits) do
            numeric = [
                parse(BigFloat, objective)
                for objective in objective_strings
            ]
            all(isfinite, numeric) || return (
                ok=false, message="objective_nonfinite",
            )
            first_value = first(numeric)
            all(value -> value == first_value, numeric) && return (
                ok=true, message="",
            )
            if absolute === nothing || relative === nothing
                return (
                    ok=false,
                    message="objective_differ_no_reference_tolerance",
                )
            end
            absolute_tolerance = parse(BigFloat, absolute)
            relative_tolerance = parse(BigFloat, relative)
            if all(numeric) do value
                error = abs(value - first_value)
                scale = unit_scale ? max(one(BigFloat), abs(first_value)) :
                        abs(first_value)
                allowed = absolute_tolerance + relative_tolerance * scale
                error <= allowed
            end
                return (ok=true, message="")
            end
            return (ok=false, message="objective")
        end
    catch exception
        return (
            ok=false,
            message="objective_unparseable",
        )
    end
end

function _sample_parity(rows)
    failures = String[]
    statuses = [_cell(getproperty(row, :status)) for row in rows]
    iterations = [getproperty(row, :iterations) for row in rows]
    objectives = [_cell(getproperty(row, :objective)) for row in rows]
    certificates = [_cell(getproperty(row, :certificate_valid)) for row in rows]
    routes = [_sample_route_key(row) for row in rows]
    length(unique(statuses)) == 1 || push!(failures, "status")
    length(unique(iterations)) == 1 || push!(failures, "iterations")
    objective_parity = _objective_parity(rows)
    objective_parity.ok || push!(failures, objective_parity.message)
    length(unique(certificates)) == 1 || push!(failures, "certificate")
    length(unique(routes)) == 1 || push!(failures, "route")
    all(row -> row.semantic_pass, rows) || push!(failures, "semantic_pass")
    return (
        parity=isempty(failures),
        failures=join(failures, ","),
        statuses=statuses,
        iterations=iterations,
        objectives=objectives,
        certificates=certificates,
        routes=routes,
    )
end

function _sampling_row(rows, sample_seconds; sample_count=length(rows))
    summary = _sampling_summary(sample_seconds)
    selected = findmin(abs.(sample_seconds .- summary.median_seconds))[2]
    parity = _sample_parity(rows)
    base = rows[selected]
    values = Dict{Symbol,Any}(
        :sample_count => sample_count,
        :sample_seconds => _sample_json_list(sample_seconds),
        :sample_semantic_pass =>
            _sample_json_list([row.semantic_pass for row in rows]),
        :sample_status => _sample_json_list(parity.statuses),
        :sample_iterations => _sample_json_list(parity.iterations),
        :sample_objective => _sample_json_list(parity.objectives),
        :sample_certificate_valid =>
            _sample_json_list(parity.certificates),
        :sample_route => _sample_json_list(parity.routes),
        :sample_semantic_parity => parity.parity,
        :sample_parity_failures => parity.failures,
        :sample_median_seconds => summary.median_seconds,
        :sample_min_seconds => summary.min_seconds,
        :sample_max_seconds => summary.max_seconds,
        :sample_mad_seconds => summary.mad_seconds,
        :sample_spread_seconds => summary.spread_seconds,
    )
    return NamedTuple{RESULT_COLUMNS}(
        Tuple(
            haskey(values, field) ? values[field] : getproperty(base, field)
            for field in RESULT_COLUMNS
        ),
    )
end

function _skip_row(spec, suite, arithmetic, provider, reason, checksum=missing)
    values = Dict{Symbol,Any}(field => missing for field in RESULT_COLUMNS)
    analytic_parameters = spec.loader === :analytic ? spec.parameters : (;)
    family_parameters = get(analytic_parameters, :analytic_parameters, (;))
    merge!(values, Dict(
        :schema_version => RESULT_SCHEMA_VERSION,
        :source_commit => _source_commit(),
        :source_dirty => _source_dirty(),
        :julia_version => string(VERSION),
        :os => string(Sys.KERNEL),
        :cpu_name => _cpu_name(),
        :hostname => _hostname(),
        :pbs_job_id => get(ENV, "PBS_JOBID", missing),
        :julia_threads => Threads.nthreads(),
        :blas_threads => _blas_threads(),
        :project_sha256 => _sha256_if_file(Base.active_project()),
        :manifest_sha256 => _manifest_sha256(),
        :benchmark_driver_sha256 => _sha256_file(joinpath(ROOT, "runner_impl.jl")),
        :mfla_commit => _package_commit("MultiFloatLinearAlgebra"),
        :bfla_commit => _package_commit("BigFloatLinearAlgebra"),
        :suite => suite,
        :problem_id => spec.id,
        :name => spec.name,
        :family => spec.family,
        :problem_type => spec.problem_type,
        :conic_formulation => missing,
        :source => spec.source,
        :purpose => spec.purpose,
        :seed => spec.seed,
        :arithmetic => arithmetic,
        :requested_provider => provider,
        :status => :skipped,
        :reference_status => spec.reference.status,
        :reference_absolute_tolerance => spec.reference.absolute_tolerance,
        :reference_relative_tolerance => spec.reference.relative_tolerance,
        :analytic_family => get(analytic_parameters, :analytic_family, missing),
        :analytic_direction => get(family_parameters, :bound, :exact),
        :analytic_equivalence_group =>
            get(analytic_parameters, :equivalence_group, missing),
        :analytic_monotonic_group =>
            get(analytic_parameters, :monotonic_group, missing),
        :analytic_bound_group => get(analytic_parameters, :bound_group, missing),
        :certificate_policy => :original_coordinate_required,
        :classification => :UNRESOLVED,
        :eligible_for_performance => false,
        :group_failures => "",
        :solve_settings => repr(get(analytic_parameters, :solve_settings, (;))),
        :skip_reason => reason,
        :external_checksum => checksum === missing && spec.external !== nothing ?
                              something(spec.external.sha256, missing) : checksum,
        :input_fingerprint => spec.external === nothing ? missing :
                              bytes2hex(SHA.sha256(join((
                                  spec.id,
                                  spec.external.authoritative_url,
                                  string(spec.external.sha256),
                                  string(spec.external.format),
                              ), "|"))),
    ))
    return NamedTuple{RESULT_COLUMNS}(Tuple(values[field] for field in RESULT_COLUMNS))
end

function _error_row(spec, suite, arithmetic, provider, exception)
    skipped = _skip_row(
        spec, suite, arithmetic, provider,
        "execution_error: " * sprint(showerror, exception),
    )
    values = Dict{Symbol,Any}(
        field => getproperty(skipped, field) for field in RESULT_COLUMNS
    )
    values[:status] = :error
    values[:classification] = :FAIL
    values[:semantic_pass] = false
    values[:eligible_for_performance] = false
    values[:semantic_failures] = "execution_error"
    values[:sample_count] = 1
    return NamedTuple{RESULT_COLUMNS}(
        Tuple(values[field] for field in RESULT_COLUMNS),
    )
end

function _replace_result_row(row, updates::Pair...)
    replacement = Dict{Symbol,Any}(updates)
    values = Any[
        haskey(replacement, field) ? replacement[field] :
        getproperty(row, field)
        for field in RESULT_COLUMNS
    ]
    return NamedTuple{RESULT_COLUMNS}(Tuple(values))
end

function _append_failure(existing, failure)
    text = existing === missing || existing === nothing ? "" : string(existing)
    isempty(text) && return string(failure)
    failure in split(text, ',') && return text
    return text * "," * string(failure)
end

_classification_rank(value) = value === :FAIL ? 3 :
                              value === :UNRESOLVED ? 2 :
                              value === :PASS ? 1 : 0

function _worse_classification(left, right)
    return _classification_rank(left) >= _classification_rank(right) ? left : right
end

function _analytic_groups(rows, field)
    groups = Dict{Tuple{String,String,String},Vector{Int}}()
    for (index, row) in pairs(rows)
        value = getproperty(row, field)
        (value === missing || value === nothing) && continue
        row.status === :skipped && continue
        isempty(string(value)) && continue
        key = (
            string(value),
            string(getproperty(row, :arithmetic)),
            string(getproperty(row, :requested_provider)),
        )
        push!(get!(groups, key, Int[]), index)
    end
    return groups
end

function _analytic_group_tolerance(group_rows, values)
    absolute = BigFloat(0)
    relative = BigFloat(0)
    for row in group_rows
        absolute = max(
            absolute,
            parse(BigFloat, string(row.analytic_absolute_tolerance)),
        )
        relative = max(
            relative,
            parse(BigFloat, string(row.analytic_relative_tolerance)),
        )
    end
    scale = maximum(abs, values; init=BigFloat(0))
    return absolute + relative * scale
end

function _group_precision(group_rows)
    recorded = [row.precision_bits for row in group_rows
                if row.precision_bits isa Integer]
    strings = String[]
    for row in group_rows
        for field in (:objective, :analytic_absolute_tolerance,
                      :analytic_relative_tolerance)
            value = getproperty(row, field)
            value === missing || push!(strings, string(value))
        end
    end
    return max(maximum(recorded; init=128) * 2,
               _parity_precision_bits(strings...))
end

function _mark_group!(rows, indices, field, valid, failure, severity)
    for index in indices
        row = rows[index]
        if valid
            rows[index] = _replace_result_row(row, field => true)
            continue
        end
        classification = _worse_classification(row.classification, severity)
        rows[index] = _replace_result_row(
            row,
            field => false,
            :classification => classification,
            :semantic_pass => false,
            :eligible_for_performance => false,
            :semantic_failures => _append_failure(row.semantic_failures, failure),
            :group_failures => _append_failure(row.group_failures, failure),
        )
    end
    return rows
end

function _apply_equivalence_gates!(rows)
    for indices in values(_analytic_groups(rows, :analytic_equivalence_group))
        length(indices) >= 2 || continue
        group_rows = rows[indices]
        # Formulation invariance is an independent diagnostic: two certified
        # Optimal rows can agree with each other even when both miss the
        # analytic oracle. Do not conflate those two failure mechanisms.
        ready = all(
            row -> _normalized_status(row.status) === :optimal &&
                   row.certificate_valid === true,
            group_rows,
        )
        valid = false
        failure = "equivalence_unresolved"
        severity = any(row -> row.classification === :FAIL, group_rows) ?
                   :FAIL : :UNRESOLVED
        if ready
            bits = _group_precision(group_rows)
            valid = setprecision(BigFloat, bits) do
                objectives = [parse(BigFloat, string(row.objective)) for row in group_rows]
                all(isfinite, objectives) || return false
                maximum(objectives) - minimum(objectives) <=
                    _analytic_group_tolerance(group_rows, objectives)
            end
            failure = "equivalence_objective"
            severity = :FAIL
        end
        _mark_group!(
            rows, indices, :equivalence_gate_valid,
            valid, failure, severity,
        )
    end
    return rows
end

function _analytic_order(row)
    spec = benchmark_spec(row.problem_id)
    parameters = get(spec.parameters, :analytic_parameters, (;))
    return get(parameters, :order, nothing)
end

function _has_nonfinite_objective(rows)
    try
        objectives = [parse(BigFloat, string(row.objective)) for row in rows]
        return any(!isfinite, objectives)
    catch
        return true
    end
end

function _apply_monotonicity_gates!(rows)
    for indices in values(_analytic_groups(rows, :analytic_monotonic_group))
        length(indices) >= 2 || continue
        sort!(indices; by=index -> something(_analytic_order(rows[index]), typemax(Int)))
        group_rows = rows[indices]
        ready = all(row -> row.classification === :PASS, group_rows)
        valid = false
        failure = "monotonicity_unresolved"
        severity = any(row -> row.classification === :FAIL, group_rows) ?
                   :FAIL : :UNRESOLVED
        if _has_nonfinite_objective(group_rows)
            failure = "monotonicity"
            severity = :FAIL
            ready = false
        end
        if ready
            bits = _group_precision(group_rows)
            valid = setprecision(BigFloat, bits) do
                objectives = [parse(BigFloat, string(row.objective)) for row in group_rows]
                allowed = _analytic_group_tolerance(group_rows, objectives)
                direction = first(group_rows).analytic_direction
                if direction === :lower
                    all(objectives[index + 1] + allowed >= objectives[index]
                        for index in 1:(length(objectives) - 1))
                elseif direction === :upper
                    all(objectives[index + 1] <= objectives[index] + allowed
                        for index in 1:(length(objectives) - 1))
                else
                    false
                end
            end
            failure = "monotonicity"
            severity = :FAIL
        end
        _mark_group!(
            rows, indices, :monotonicity_gate_valid,
            valid, failure, severity,
        )
    end
    return rows
end

function _apply_bound_pair_gates!(rows)
    for indices in values(_analytic_groups(rows, :analytic_bound_group))
        length(indices) >= 2 || continue
        group_rows = rows[indices]
        directions = Dict(row.analytic_direction => row for row in group_rows)
        haskey(directions, :lower) && haskey(directions, :upper) || continue
        ready = all(row -> row.classification === :PASS, group_rows)
        valid = false
        failure = "bound_pair_unresolved"
        severity = any(row -> row.classification === :FAIL, group_rows) ?
                   :FAIL : :UNRESOLVED
        bound_rows = (directions[:lower], directions[:upper])
        if _has_nonfinite_objective(bound_rows)
            failure = "bound_pair"
            severity = :FAIL
            ready = false
        end
        if ready
            bits = _group_precision(group_rows)
            valid = setprecision(BigFloat, bits) do
                lower = parse(BigFloat, string(directions[:lower].objective))
                upper = parse(BigFloat, string(directions[:upper].objective))
                values = BigFloat[lower, upper]
                lower <= upper + _analytic_group_tolerance(group_rows, values)
            end
            failure = "bound_pair"
            severity = :FAIL
        end
        _mark_group!(
            rows, indices, :bound_pair_gate_valid,
            valid, failure, severity,
        )
    end
    return rows
end

function _apply_analytic_group_gates(rows)
    updated = NamedTuple[row for row in rows]
    _apply_equivalence_gates!(updated)
    _apply_monotonicity_gates!(updated)
    _apply_bound_pair_gates!(updated)
    return updated
end

function _write_tsv(path, rows)
    open(path, "w") do io
        println(io, join(string.(RESULT_COLUMNS), '\t'))
        for row in rows
            println(io, join((_cell(getproperty(row, field)) for field in RESULT_COLUMNS), '\t'))
        end
    end
end

function _toml_value(value)
    if value === missing || value === nothing
        return ""
    end
    value isa Symbol && return string(value)
    value isa Tuple && return join(string.(value), ",")
    value isa AbstractVector && return [
        item isa Symbol ? string(item) : item for item in value
    ]
    return value
end

function write_results(path::AbstractString, rows)
    root, extension = splitext(path)
    tsv = extension == ".tsv" ? path : root * ".tsv"
    toml = extension == ".toml" ? path : root * ".toml"
    mkpath(dirname(tsv))
    _write_tsv(tsv, rows)
    document = Dict(
        "schema_version" => RESULT_SCHEMA_VERSION,
        "generated_at" => string(Dates.now()),
        "result" => [Dict(string(field) => _toml_value(getproperty(row, field))
                         for field in RESULT_COLUMNS) for row in rows],
    )
    open(toml, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return (tsv=tsv, toml=toml)
end

function _write_failure_map(path::AbstractString, rows)
    failed = filter(row -> row.status != :skipped &&
                    (row.semantic_pass === false ||
                     row.sample_semantic_parity === false), rows)
    document = Dict(
        "schema_version" => RESULT_SCHEMA_VERSION,
        "generated_at" => string(Dates.now()),
        "failure" => [Dict(
            "problem_id" => row.problem_id,
            "status" => string(row.status),
            "classification" => _cell(row.classification),
            "semantic_failures" => string(row.semantic_failures),
            "group_failures" => _cell(row.group_failures),
            "analytic_family" => _cell(row.analytic_family),
            "analytic_direction" => _cell(row.analytic_direction),
            "objective" => _cell(row.objective),
            "analytic_reference" => _cell(row.analytic_reference),
        ) for row in failed],
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return path
end

function _selected_entries(suite, problem, arithmetic, provider)
    entries = suite_entries(suite)
    problem !== nothing && (entries = filter(e -> e.problem_id == problem, entries))
    arithmetic !== nothing && (entries = [SuiteEntry(e.problem_id, arithmetic,
                                                     e.provider) for e in entries])
    provider !== nothing && (entries = [SuiteEntry(e.problem_id, e.arithmetic,
                                                   provider) for e in entries])
    seen = Set{Tuple{String,Symbol,Symbol}}()
    entries = filter(entries) do entry
        key = (entry.problem_id, entry.arithmetic, entry.provider)
        key in seen && return false
        push!(seen, key)
        return true
    end
    isempty(entries) && throw(ArgumentError("selection contains no benchmark entries"))
    return entries
end

function run_suite(
    suite::Symbol;
    problem=nothing,
    arithmetic=nothing,
    provider=nothing,
    output=joinpath(ROOT, "out", string(suite), "rows.toml"),
    verbose=false,
    warmup=true,
    strict_semantics=true,
    samples=1,
    cache_dir=DEFAULT_CACHE,
    allow_large=false,
    allow_semantic_failures=false,
)
    suite === :heavy && throw(ArgumentError(
        "the heavy suite is register-only and cannot be executed",
    ))
    suite === :large && !(allow_large || haskey(ENV, "PBS_JOBID")) &&
        throw(ArgumentError(
            "the large suite is cluster-only; run it inside PBS or pass " *
            "allow_large=true for an explicitly authorized non-cluster diagnostic",
        ))
    suite === :analytic_stress && !allow_large &&
        throw(ArgumentError(
            "the analytic stress suite is gated; pass allow_large=true for an " *
            "explicitly authorized stress run",
        ))
    samples isa Integer || throw(ArgumentError(
        "samples must be an integer count of timed solves, got $samples",
    ))
    samples == 1 || samples >= 3 || throw(ArgumentError(
        "samples must be 1 (default single run) or >= 3 timed solves; " *
        "got $samples; a two-run observation cannot support a timing statistic",
    ))
    entries = _selected_entries(suite, problem, arithmetic, provider)
    rows = NamedTuple[]
    for entry in entries
        spec = benchmark_spec(entry.problem_id)
        cache = spec.external === nothing ? nothing :
                external_cache_status(spec; cache_dir=cache_dir)
        if cache !== nothing && !cache.loadable
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                cache.reason, something(cache.checksum, missing),
            ))
            verbose && println("skip ", spec.id, ": ", cache.reason)
            continue
        end
        T = try
            _arithmetic_type(entry.arithmetic)
        catch exception
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                :arithmetic_unavailable,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        try
            _load_requested_provider(entry.provider)
        catch exception
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider,
                :provider_unavailable,
            ))
            verbose && println("skip ", spec.id, ": ", exception)
            continue
        end
        if :capability_skip in spec.tags
            reason = Symbol(get(
                spec.parameters, :skip_reason,
                :backend_capability_unavailable,
            ))
            push!(rows, _skip_row(
                spec, suite, entry.arithmetic, entry.provider, reason,
            ))
            verbose && println("skip ", spec.id, ": ", reason)
            continue
        end
        run = function ()
            built = build_problem(spec, T; cache_dir)
            result = _solve_built(
                built, T, entry.provider; verbose=verbose,
            )
            return built, result
        end
        local rows_for_entry
        try
            if samples >= 3
                _measure_sample = function ()
                    # Canonical O0 boundary: problem build, JIT compilation,
                    # and the untimed warm-up run are excluded from every
                    # timed sample. A fresh deterministic build per sample
                    # keeps problem generation out of the measurement while
                    # preserving identical problem fingerprints.
                    built = Base.invokelatest(
                        build_problem, spec, T; cache_dir=cache_dir,
                    )
                    measurement = @timed Base.invokelatest(
                        _solve_built, built, T, entry.provider;
                        verbose=verbose,
                    )
                    result = measurement.value
                    row = _result_row(
                        spec, suite, entry.arithmetic, entry.provider,
                        built, result, measurement.time;
                        allocated_bytes=measurement.bytes,
                        gc_seconds=measurement.gctime,
                    )
                    return row, measurement.time
                end
                timed = Float64[]
                if T === BigFloat
                    bits = parse(Int, replace(string(entry.arithmetic), "bigfloat" => ""))
                    rows_for_entry = setprecision(BigFloat, bits) do
                        warmup && Base.invokelatest(run)
                        sample_rows = NamedTuple[]
                        for _ in 1:samples
                            row, elapsed = _measure_sample()
                            push!(sample_rows, row)
                            push!(timed, elapsed)
                        end
                        [_sampling_row(
                            sample_rows,
                            timed,
                            sample_count=samples,
                        )]
                    end
                else
                    warmup && Base.invokelatest(run)
                    sample_rows = NamedTuple[]
                    for _ in 1:samples
                        row, elapsed = _measure_sample()
                        push!(sample_rows, row)
                        push!(timed, elapsed)
                    end
                    rows_for_entry = [_sampling_row(
                        sample_rows,
                        timed,
                        sample_count=samples,
                    )]
                end
            elseif T === BigFloat
                bits = parse(Int, replace(string(entry.arithmetic), "bigfloat" => ""))
                rows_for_entry = [setprecision(BigFloat, bits) do
                    warmup && Base.invokelatest(run)
                    measurement = @timed Base.invokelatest(run)
                    built, result = measurement.value
                    _result_row(spec, suite, entry.arithmetic, entry.provider,
                                built, result, measurement.time;
                                allocated_bytes=measurement.bytes,
                                gc_seconds=measurement.gctime)
                end]
            else
                warmup && Base.invokelatest(run)
                measurement = @timed Base.invokelatest(run)
                built, result = measurement.value
                row = _result_row(
                    spec, suite, entry.arithmetic, entry.provider,
                    built, result, measurement.time;
                    allocated_bytes=measurement.bytes,
                    gc_seconds=measurement.gctime,
                )
                rows_for_entry = [row]
            end
        catch exception
            rows_for_entry = [_error_row(
                spec, suite, entry.arithmetic, entry.provider, exception,
            )]
        end
        append!(rows, rows_for_entry)
        if verbose
            row = first(rows_for_entry)
            if row.sample_count isa Integer && row.sample_count >= 3 &&
               row.sample_median_seconds isa Real
                elapsed_label = string(
                    "median=", round(row.sample_median_seconds; digits=3), "s ",
                    "spread=", round(row.sample_spread_seconds; digits=3), "s ",
                    "samples=", row.sample_count,
                )
            else
                elapsed_label = row.total_seconds isa Real ?
                    string(round(row.total_seconds; digits=3), "s") : "unavailable"
            end
            println(spec.id, " ", row.status, " ", elapsed_label)
        end
    end
    rows = _apply_analytic_group_gates(rows)
    paths = write_results(output, rows)
    failed = filter(
        row -> row.status != :skipped &&
               (row.semantic_pass === false ||
                row.sample_semantic_parity === false),
        rows,
    )
    failure_map = nothing
    if !isempty(failed) && allow_semantic_failures
        root, extension = splitext(output)
        failure_map = _write_failure_map(
            (extension == ".toml" || extension == ".tsv" ? root : output) *
            ".failure-map.toml", rows,
        )
    end
    if strict_semantics && !allow_semantic_failures && !isempty(failed)
        details = join((
            let failures = String[]
                isempty(row.semantic_failures) ||
                    push!(failures, row.semantic_failures)
                row.sample_semantic_parity === false &&
                    push!(failures, "sample_parity:$(row.sample_parity_failures)")
                join(failures, ",")
            end
            for row in failed
        ), "; ")
        throw(ErrorException(
            "semantic regression in $(length(failed)) benchmark result(s): $details; " *
            "results were written to $(paths.toml)",
        ))
    end
    return (rows=rows, paths=paths, failure_map=failure_map)
end

function _parse_symbol(value)
    value === nothing && return nothing
    return Symbol(lowercase(strip(value)))
end

function _parse_cli(args)
    suite = :micro
    problem = nothing
    arithmetic = nothing
    provider = nothing
    output = nothing
    cache_dir = DEFAULT_CACHE
    verbose = false
    prepare = false
    allow_large = false
    allow_semantic_failures = false
    samples = 1
    warmup = true
    positional = String[]
    for argument in args
        if startswith(argument, "--problem=")
            problem = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--arithmetic=")
            arithmetic = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--provider=")
            provider = _parse_symbol(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--output=")
            output = split(argument, "="; limit=2)[2]
        elseif startswith(argument, "--cache-dir=")
            cache_dir = abspath(split(argument, "="; limit=2)[2])
        elseif startswith(argument, "--samples=")
            samples = parse(Int, split(argument, "="; limit=2)[2])
        elseif argument == "--verbose"
            verbose = true
        elseif argument == "--prepare"
            prepare = true
        elseif argument == "--allow-large"
            allow_large = true
        elseif argument == "--allow-semantic-failures"
            allow_semantic_failures = true
        elseif argument == "--no-warmup"
            warmup = false
        elseif startswith(argument, "--")
            throw(ArgumentError("unknown option $argument"))
        else
            push!(positional, argument)
        end
    end
    !isempty(positional) && (suite = Symbol(lowercase(first(positional))))
    length(positional) > 1 && problem === nothing && (problem = positional[2])
    return (; suite, problem, arithmetic, provider, output, cache_dir, verbose,
            prepare, allow_large, allow_semantic_failures, samples, warmup)
end

function main(args=ARGS)
    options = _parse_cli(args)
    if options.prepare
        options.suite === :heavy && throw(ArgumentError(
            "the heavy suite is register-only; prepare selected non-heavy cases explicitly",
        ))
        ids = options.problem === nothing ?
              [entry.problem_id for entry in suite_entries(options.suite)
               if benchmark_spec(entry.problem_id).external !== nothing] :
              [options.problem]
        return prepare_external!(
            ids; cache_dir=options.cache_dir, verbose=options.verbose,
        )
    end
    output = something(
        options.output,
        joinpath(ROOT, "out", string(options.suite), "rows.toml"),
    )
    result = run_suite(
        options.suite;
        problem=options.problem,
        arithmetic=options.arithmetic,
        provider=options.provider,
        output=output,
        verbose=options.verbose,
        samples=options.samples,
        warmup=options.warmup,
        cache_dir=options.cache_dir,
        allow_large=options.allow_large,
        allow_semantic_failures=options.allow_semantic_failures,
    )
    solved = count(row -> !(row.status in (:skipped, :error)), result.rows)
    errors = count(row -> row.status === :error, result.rows)
    skipped = count(row -> row.status === :skipped, result.rows)
    println("suite=", options.suite, " solved=", solved,
            " skipped=", skipped, " errors=", errors,
            " output=", result.paths.toml)
    return result
end
