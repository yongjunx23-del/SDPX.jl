using LinearAlgebra
using MultiFloats: Float64x2, Float64x4
using MultiFloatLinearAlgebra
using SDPX
using Serialization
using SparseArrays
using TOML

const STUDY_ROOT = get(ENV, "CSDR_STUDY_ROOT", "")
const CSDR_SOURCE = isempty(STUDY_ROOT) ? "" : joinpath(
    STUDY_ROOT, "source", "src", "CSDRBootstrap.jl",
)

function _load_csdr_module()
    isfile(CSDR_SOURCE) || error(
        "CSDRBootstrap source not found at $CSDR_SOURCE; " *
        "set CSDR_STUDY_ROOT or pass a neutral reduced payload",
    )
    isdefined(Main, :CSDRBootstrap) || Base.include(Main, CSDR_SOURCE)
    return Main.CSDRBootstrap
end

function _native_fixed_trace_problem(payload, csdr)
    elimination = csdr._eliminate_low_energy_variables(payload)
    reduced = elimination.problem
    T = eltype(reduced.c)
    blocks = reduced.dims.L
    variables = reduced.dims.m
    variables == 2blocks || error(
        "fixed-trace conversion requires two local variables per block",
    )

    cones = Vector{SDPX.SOCConstraint{T}}(undef, blocks)
    @inbounds for block in 1:blocks
        r_index = 2block - 1
        q_index = 2block
        # [[q,r],[r,2-q]] >= 0  iff  (1,q-1,r) belongs to Q3.
        affine = sparse(
            [2, 3], [q_index, r_index], T[one(T), one(T)], 3, variables,
        )
        offset = T[one(T), -one(T), zero(T)]
        cones[block] = SDPX.SOCConstraint(affine, offset; T)
    end

    equality = sparse(transpose(reduced.B))
    problem = SDPX.second_order_program(
        reduced.c,
        cones;
        Aeq=equality,
        beq=reduced.b,
        T,
    )
    return problem, elimination
end

function _native_fixed_trace_problem(payload::NamedTuple, ::Nothing)
    payload.schema === :csdr_fixed_trace_reduced_v1 || error(
        "unsupported neutral CSDR payload schema",
    )
    T = eltype(payload.reduced_c)
    variables = length(payload.reduced_c)
    iseven(variables) || error("fixed-trace reduced variable count must be even")
    blocks = variables ÷ 2
    cones = Vector{SDPX.SOCConstraint{T}}(undef, blocks)
    @inbounds for block in 1:blocks
        r_index = 2block - 1
        q_index = 2block
        affine = sparse(
            [2, 3], [q_index, r_index], T[one(T), one(T)], 3, variables,
        )
        cones[block] = SDPX.SOCConstraint(
            affine, T[one(T), -one(T), zero(T)]; T,
        )
    end
    problem = SDPX.second_order_program(
        payload.reduced_c,
        cones;
        Aeq=sparse(transpose(payload.reduced_B)),
        beq=payload.reduced_b,
        T,
    )
    return problem, payload
end

function _convert_neutral_arithmetic(payload::NamedTuple, ::Type{T}) where {T}
    return merge(payload, (
        reduced_c=T.(payload.reduced_c),
        reduced_B=T.(payload.reduced_B),
        reduced_b=T.(payload.reduced_b),
        coefficient_constant=T.(payload.coefficient_constant),
        coefficient_from_spectrum=T.(payload.coefficient_from_spectrum),
    ))
end

function _physical_objective(payload, elimination, x)
    T = eltype(x)
    coefficients = elimination.coefficient_constant +
                   elimination.coefficient_from_spectrum * x
    full_x = vcat(coefficients, x)
    label_index = Dict(
        label => index for (index, label) in
        enumerate(payload.coefficient_labels)
    )
    value = zero(T)
    for (label, coefficient) in payload.config.objective
        beta = parse(T, coefficient)
        if haskey(label_index, label)
            value += beta * full_x[label_index[label]]
        else
            value += beta * parse(T, payload.config.fixed_coefficients[label])
        end
    end
    return value
end

function _physical_objective(payload::NamedTuple, ::NamedTuple, x)
    T = eltype(x)
    coefficients = payload.coefficient_constant +
                   payload.coefficient_from_spectrum * x
    full_x = vcat(coefficients, x)
    label_index = Dict(
        label => index for (index, label) in enumerate(payload.coefficient_labels)
    )
    value = zero(T)
    for (label, coefficient) in payload.objective
        beta = parse(T, coefficient)
        if haskey(label_index, label)
            value += beta * full_x[label_index[label]]
        else
            value += beta * parse(T, payload.fixed_coefficients[label])
        end
    end
    return value
end

function _warmup(::Type{T}, options) where {T}
    first = sparse([2, 3], [2, 1], T[one(T), one(T)], 3, 4)
    second = sparse([2, 3], [4, 3], T[one(T), one(T)], 3, 4)
    cones = [
        SDPX.SOCConstraint(first, T[one(T), -one(T), zero(T)]; T),
        SDPX.SOCConstraint(second, T[one(T), -one(T), zero(T)]; T),
    ]
    problem = SDPX.second_order_program(
        T[-one(T), zero(T), -one(T), zero(T)],
        cones;
        Aeq=T[one(T) zero(T) zero(T) zero(T);
              zero(T) zero(T) one(T) zero(T)],
        beq=T[T(1) / T(5), -T(1) / T(10)],
        T,
    )
    warm = SDPX.SolverOptions(
        T;
        tolerance=T(1e-8),
        maximum_iterations=20,
        verbosity=0,
        timing=false,
        diagnostics=false,
        linear_algebra_backend=options.linear_algebra_backend,
        threads=options.threads,
    )
    SDPX.solve_socp(
        problem;
        specialization=:fixed_trace,
        tolerance=T(1e-8),
        maximum_iterations=20,
        verbosity=0,
        timing=false,
        diagnostics=false,
        linear_algebra_backend=warm.linear_algebra_backend,
        threads=warm.threads,
    )
    return nothing
end

function _string_status(status)
    return string(status)
end

function _write_report(path, facts)
    open(path, "w") do io
        println(io, "# NativeSOC FixedTrace CSDR result")
        println(io)
        println(io, "```text")
        for key in (
            "status", "arithmetic", "physical_objective", "reference_objective",
            "objective_error", "objective_in_historical_interval",
            "certificate_valid", "relative_gap", "iterations",
            "seconds_per_iteration", "solve_allocations",
            "solve_allocated_bytes", "solve_gc_seconds", "solve_compile_seconds",
            "specialization", "equality_method", "la_provider", "scaling",
            "solve_seconds", "timing_total", "timing_schur_assembly",
            "timing_fixed_local_metric", "timing_fixed_local_factor",
            "timing_equality_panel_transform", "timing_equality_gram_syrk",
            "timing_equality_factor", "timing_predictor_rhs",
            "timing_corrector_rhs", "timing_fixed_block_residual",
            "timing_fixed_block_recovery", "timing_kkt_factorization",
            "timing_predictor", "timing_corrector",
            "local_metric_preparations", "equality_gram_assemblies",
            "equality_factorizations", "kkt_rhs_solves",
            "production_invariants_valid", "full_numerical_gate_valid",
            "workspace_bytes",
        )
            println(io, rpad(key, 32), facts[key])
        end
        println(io, "```")
        println(io)
        println(io, "The model is solved directly as 4,200 Lorentz Q3 blocks;")
        println(io, "no PSD matrices are created by the production solve.")
    end
end

function main(args=ARGS)
    study_root = STUDY_ROOT
    model_path = if isempty(args)
        isempty(study_root) && error(
            "set CSDR_STUDY_ROOT or pass a neutral reduced payload path",
        )
        joinpath(study_root, "results", "generate", "model.bin")
    else
        abspath(args[1])
    end
    output_dir = length(args) >= 2 ? abspath(args[2]) :
                 joinpath(
                     @__DIR__, "..", "..", "out",
                     "bootstrap_full_unitraity_eft",
                 )
    mkpath(output_dir)

    payload = open(deserialize, model_path)
    neutral = payload isa NamedTuple && hasproperty(payload, :schema) &&
              payload.schema === :csdr_fixed_trace_reduced_v1
    csdr = neutral ? nothing : _load_csdr_module()
    arithmetic = lowercase(get(ENV, "SDPX_BENCH_ARITHMETIC", "float64x4"))
    target_type = arithmetic == "float64x2" ? Float64x2 :
                  arithmetic == "float64x4" ? Float64x4 :
                  error("SDPX_BENCH_ARITHMETIC must be float64x2 or float64x4")
    neutral && (payload = _convert_neutral_arithmetic(payload, target_type))
    problem, elimination = neutral ?
        _native_fixed_trace_problem(payload, nothing) :
        _native_fixed_trace_problem(payload, csdr)
    T = eltype(problem.c)
    T === target_type || error("expected $target_type model, found $T")

    max_iterations = parse(Int, get(ENV, "SDPX_BENCH_MAX_ITERATIONS", "200"))
    max_time = parse(Float64, get(ENV, "SDPX_BENCH_MAX_TIME", "900"))
    require_optimal = lowercase(get(
        ENV, "SDPX_BENCH_REQUIRE_OPTIMAL", "true",
    )) in ("1", "true", "yes")
    options = SDPX.SolverOptions(
        T;
        tolerance=T(1e-12),
        maximum_iterations=max_iterations,
        max_time,
        verbosity=1,
        timing=true,
        diagnostics=true,
        linear_algebra_backend=:multifloat,
        threads=Threads.nthreads(),
    )
    plan = SDPX.plan_native_soc(problem, options; specialization=:fixed_trace)
    plan.cone.specialization === :fixed_trace_q3 || error(
        "NativeSOC did not select the fixed-trace Q3 specialization",
    )
    plan.la_config.selected === :multifloat || error(
        "MFLA provider was not selected",
    )

    _warmup(T, options)
    timed_solve = @timed SDPX.solve_socp(
        problem;
        specialization=:fixed_trace,
        tolerance=T(1e-12),
        maximum_iterations=max_iterations,
        max_time,
        verbosity=options.verbosity,
        timing=true,
        diagnostics=true,
        linear_algebra_backend=options.linear_algebra_backend,
        threads=options.threads,
    )
    result = timed_solve.value
    solve_seconds = timed_solve.time
    solve_allocations = timed_solve.gcstats.malloc +
                        timed_solve.gcstats.realloc +
                        timed_solve.gcstats.poolalloc +
                        timed_solve.gcstats.bigalloc
    certificate = SDPX.result_certificate(problem, result, options)
    physical = _physical_objective(payload, elimination, result.x)
    reference = parse(T, "30.4732058529286002611344264526887472")
    reference_upper = parse(T, "30.4732058529379858416561499463130361")
    timings = result.timings
    selected = result.diagnostics.selected_algorithms
    memory = result.diagnostics.memory
    termination = result.termination
    objective_in_interval = reference <= physical <= reference_upper
    production_invariants =
        plan.cone.specialization === :fixed_trace_q3 &&
        selected.la_executed_provider === :multifloat_linear_algebra &&
        result.lifted === nothing &&
        selected.fallback_reason === :none &&
        selected.la_fallback_reason === :none
    full_numerical_gate =
        result.status === SDPX.Optimal && certificate.valid &&
        objective_in_interval && abs(result.gap_rel) <= T(1e-12) &&
        production_invariants

    facts = Dict{String,Any}(
        "schema_version" => 1,
        "status" => _string_status(result.status),
        "arithmetic" => arithmetic,
        "physical_objective" => string(physical),
        "reference_objective" => string(reference),
        "objective_error" => string(abs(physical - reference)),
        "historical_interval_upper" => string(reference_upper),
        "objective_in_historical_interval" => objective_in_interval,
        "primal_objective" => string(result.pObj),
        "dual_objective" => string(result.dObj),
        "relative_gap" => string(result.gap_rel),
        "primal_residual" => string(result.p_res),
        "dual_residual" => string(result.d_res),
        "certificate_valid" => certificate.valid,
        "certificate_failures" => join(string.(certificate.failures), ","),
        "iterations" => result.iterations,
        "seconds_per_iteration" => solve_seconds / max(result.iterations, 1),
        "variables" => problem.variables,
        "soc_blocks" => length(problem.cones),
        "equality_count" => size(problem.Aeq, 1),
        "specialization" => string(plan.cone.specialization),
        "equality_method" => string(selected.equality),
        "la_provider" => string(selected.la_executed_provider),
        "scaling" => string(selected.scaling),
        "julia_threads" => Threads.nthreads(),
        "solve_seconds" => solve_seconds,
        "solve_allocations" => solve_allocations,
        "solve_allocated_bytes" => timed_solve.bytes,
        "solve_gc_seconds" => timed_solve.gctime,
        "solve_compile_seconds" => timed_solve.compile_time,
        "timing_total" => get(timings, :total, NaN),
        "timing_schur_assembly" => get(timings, :schur_assembly, NaN),
        "timing_fixed_local_metric" => get(timings, :fixed_local_metric, NaN),
        "timing_fixed_local_factor" => get(timings, :fixed_local_factor, NaN),
        "timing_equality_panel_transform" => get(
            timings, :equality_panel_transform, NaN,
        ),
        "timing_equality_gram_syrk" => get(timings, :equality_gram_syrk, NaN),
        "timing_equality_factor" => get(timings, :equality_factor, NaN),
        "timing_predictor_rhs" => get(timings, :predictor_rhs, NaN),
        "timing_corrector_rhs" => get(timings, :corrector_rhs, NaN),
        "timing_fixed_block_residual" => get(
            timings, :fixed_block_residual, NaN,
        ),
        "timing_fixed_block_recovery" => get(
            timings, :fixed_block_recovery, NaN,
        ),
        "timing_kkt_factorization" => get(timings, :kkt_factorization, NaN),
        "timing_predictor" => get(timings, :predictor, NaN),
        "timing_corrector" => get(timings, :corrector, NaN),
        "local_metric_preparations" => get(
            termination, :local_metric_preparations, 0,
        ),
        "equality_panel_transforms" => get(
            termination, :equality_panel_transforms, 0,
        ),
        "equality_gram_assemblies" => get(
            termination, :equality_gram_assemblies, 0,
        ),
        "equality_factorizations" => get(
            termination, :equality_factorizations, 0,
        ),
        "kkt_rhs_solves" => get(termination, :kkt_rhs_solves, 0),
        "production_invariants_valid" => production_invariants,
        "full_numerical_gate_valid" => full_numerical_gate,
        "require_optimal" => require_optimal,
        "workspace_bytes" => get(memory, :workspace_bytes, 0),
        "model_path" => model_path,
        "sdpx_commit" => get(ENV, "SDPX_BENCH_SOURCE", "working_tree"),
    )

    open(joinpath(output_dir, "result.toml"), "w") do io
        TOML.print(io, facts; sorted=true)
    end
    _write_report(joinpath(output_dir, "RESULTS.md"), facts)
    println("NativeSOC CSDR result written to $output_dir")
    production_invariants || error(
        "NativeSOC CSDR benchmark violated a production routing invariant",
    )
    require_optimal && !full_numerical_gate && error(
        "NativeSOC CSDR benchmark did not pass its full numerical gate",
    )
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
