#!/usr/bin/env julia

"""
    benchmark_float64.jl

Optional Float64 reference for the fixed-trace CSDR J40 payload.  The
reference is deliberately small in scope: it loads the same reduced payload
as `../benchmark.jl`, proves that every 2x2 block is eligible for the native
fixed-trace Q3 map, builds one direct MOI `SecondOrderCone(3)` model, and then
uses that model with Clarabel.jl and/or MosekTools.jl.

The source payload SHA-256 and the SHA-256 of the reduced serialized
`SDPProblem` are recorded in every report.  No fallback PSD model is built.
If the reduced model is not accepted by SDPX's exact Q3 eligibility predicate,
the run stops with the predicate's reason rather than benchmarking a model
whose relationship to the CSDR trace is uncertain.

Examples:

    julia --project=. docs/evidence/bench/soc_fixed_trace/reference/benchmark_float64.jl \
      --solver=clarabel --model=/data/J40.bin --release=/data/CSDR \
      --expected-hash=... --reps=1 --warmup=0 --output=/tmp/j40-clarabel.toml

Use `--preflight-only` to validate provenance, geometry, and Q3 conversion
without loading either optional solver package.
"""

using LinearAlgebra
using SHA
using Serialization
using SparseArrays
using TOML
using SDPX
import MathOptInterface as MOI

# Reuse the immutable-model loader and its low-energy-elimination path.  It is
# included in a private module so this optional script does not redefine the
# main fixed-trace benchmark's names in `Main`.
module FixedTraceSupport
const BENCHMARK_PATH = normpath(joinpath(@__DIR__, "..", "benchmark.jl"))
include(BENCHMARK_PATH)
end

const HELP = """
Optional Float64 direct-Q3 reference (J40 only)

Required:
  --model=PATH                 serialized CSDR payload
  --expected-hash=SHA256       SHA-256 of the exact source payload

Common:
  --release=PATH               pinned CSDR release for exact reduction
  --solver=clarabel|mosek|both external solver(s), default both
  --case=J40                   geometry gate (only J40 is accepted)
  --reps=N                     timed solves per solver (default 1)
  --warmup=N                   complete untimed solves (default 0)
  --tolerance=VALUE            Float64 certificate/solver tolerance (default 1e-8)
  --max-iterations=N           solver iteration cap (default 500)
  --time-limit-seconds=S       per-solver time limit (default 43200)
  --threads=N                  requested external solver threads (default 1)
  --output=PATH                TOML report (default: print only)
  --preflight-only             stop after exact load/conversion checks
  --quiet                      suppress solver output where supported
"""

function usage_error(message)
    error(message * "\n\n" * HELP)
end

function parse_bool(value::AbstractString)
    lowercase(value) in ("1", "true", "yes", "on") && return true
    lowercase(value) in ("0", "false", "no", "off") && return false
    usage_error("expected a boolean, got '$value'")
end

function parse_cli(args=ARGS)
    values = Dict{String,String}()
    positional = String[]
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            println(HELP)
            exit(0)
        elseif startswith(argument, "--") && occursin("=", argument)
            key, value = split(argument[3:end], "="; limit=2)
            values[key] = value
        elseif startswith(argument, "--")
            key = argument[3:end]
            if key in ("preflight-only", "quiet")
                values[key] = "true"
            else
                index += 1
                index <= length(args) || usage_error("missing value for --$key")
                values[key] = args[index]
            end
        else
            push!(positional, argument)
        end
        index += 1
    end

    model = get(values, "model", get(ENV, "CSDR_MODEL", ""))
    isempty(model) && !isempty(positional) && (model = first(positional))
    isempty(model) && usage_error("--model=PATH is required")
    model = abspath(model)
    isfile(model) || usage_error("serialized model does not exist: $model")

    expected_hash = lowercase(get(
        values,
        "expected-hash",
        get(ENV, "SDPX_EXPECTED_MODEL_SHA256",
            get(ENV, "SDPX_EXPECTED_HASH", "")),
    ))
    isempty(expected_hash) && usage_error(
        "--expected-hash=SHA256 is required; provenance must be explicit",
    )
    (length(expected_hash) == 64 && all(c -> c in '0':'9' || c in 'a':'f', expected_hash)) ||
        usage_error("expected-hash must be exactly 64 lowercase hexadecimal characters")

    case_name = uppercase(get(values, "case", get(ENV, "SDPX_CASE", "J40")))
    case_name == "J40" || usage_error(
        "this bounded reference only accepts --case=J40 (got $case_name)",
    )
    solver = lowercase(get(values, "solver", get(ENV, "SDPX_REFERENCE_SOLVER", "both")))
    solver in ("clarabel", "mosek", "both") || usage_error(
        "solver must be clarabel, mosek, or both",
    )

    release = get(values, "release", get(ENV, "CSDR_RELEASE", ""))
    release = isempty(release) ? "" : abspath(release)
    isempty(release) && usage_error(
        "--release=PATH is required to deserialize and reduce the pinned CSDR payload",
    )
    isdir(release) || usage_error("CSDR release directory does not exist: $release")

    reps = parse(Int, get(values, "reps", get(ENV, "SDPX_REPS", "1")))
    warmup = parse(Int, get(values, "warmup", get(ENV, "SDPX_WARMUP", "0")))
    threads = parse(Int, get(values, "threads", get(ENV, "SDPX_THREADS", "1")))
    tolerance_text = get(values, "tolerance", get(ENV, "SDPX_TOLERANCE", "1e-8"))
    tolerance = try
        parse(Float64, tolerance_text)
    catch
        usage_error("tolerance must be a Float64, got '$tolerance_text'")
    end
    max_iterations = parse(Int, get(values, "max-iterations", get(ENV, "SDPX_MAX_ITERATIONS", "500")))
    time_limit = parse(Float64, get(
        values,
        "time-limit-seconds",
        get(ENV, "SDPX_TIME_LIMIT_SECONDS", "43200"),
    ))
    output = get(values, "output", get(ENV, "SDPX_OUTPUT", ""))
    preflight_only = parse_bool(get(values, "preflight-only", "false"))
    quiet = parse_bool(get(values, "quiet", get(ENV, "SDPX_REFERENCE_QUIET", "true")))

    reps >= 1 || usage_error("reps must be positive")
    warmup >= 0 || usage_error("warmup must be nonnegative")
    threads >= 1 || usage_error("threads must be positive")
    isfinite(tolerance) && tolerance > 0 || usage_error("tolerance must be finite and positive")
    max_iterations >= 1 || usage_error("max-iterations must be positive")
    isfinite(time_limit) && time_limit > 0 || usage_error("time-limit-seconds must be finite and positive")

    return (; model, expected_hash, case_name, solver, release, reps, warmup,
        threads, tolerance, tolerance_text, max_iterations, time_limit, output,
        preflight_only, quiet)
end

function sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function source_tree_sha256(root::AbstractString)
    isempty(root) && return "none"
    isdir(root) || return "unavailable"
    files = String[]
    for relative_directory in (
        "src",
        "ext",
        joinpath("bench", "soc_fixed_trace"),
    )
        source = joinpath(root, relative_directory)
        isdir(source) || continue
        for (directory, subdirectories, names) in walkdir(source)
            filter!(name -> name != "results" && name != ".julia-depot", subdirectories)
            for name in names
                push!(files, joinpath(directory, name))
            end
        end
    end
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(root, name)
        isfile(path) && push!(files, path)
    end
    isempty(files) && return "unavailable"
    io = IOBuffer()
    for path in sort!(files)
        write(io, relpath(path, root))
        write(io, UInt8(0))
        write(io, read(path))
        write(io, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function serialized_sha256(value)
    return mktemp() do _, io
        serialize(io, value)
        flush(io)
        seekstart(io)
        bytes2hex(SHA.sha256(io))
    end
end

function load_reduced_model(config)
    # Serialization resolves the payload's concrete CSDRBootstrap types before
    # it returns an object. Load the pinned release first; inspecting the raw
    # payload and deciding to load afterward is already too late.
    if !isdefined(Main, :CSDRBootstrap)
        source = joinpath(config.release, "src", "CSDRBootstrap.jl")
        isfile(source) || error(
            "CSDR release has no src/CSDRBootstrap.jl: $source",
        )
        # The shared loader intentionally looks for the dynamically included
        # release in Main.  Include it there once so its exact reducer is used
        # even though this script keeps the loader itself in a private module.
        Base.include(Main, source)
    end
    # The shared loader deserializes once, prefers the release-owned exact
    # elimination when relation metadata is present, and otherwise accepts an
    # already-reduced SDPProblem. Geometry, source hash, fixed-trace identity,
    # and Q3 eligibility are checked immediately afterward.
    return FixedTraceSupport._load_payload(config.model, config.release)
end

function preflight_q3!(problem, config)
    expected_blocks = 4_200
    expected_variables = 8_400
    expected_equalities = 170
    geometry = FixedTraceSupport.geometry(problem)
    geometry.all_psd2 || error(
        "J40 preflight rejected non-PSD2 blocks: $(geometry.block_dimensions)",
    )
    geometry.blocks == expected_blocks || error(
        "J40 geometry mismatch: expected $expected_blocks blocks, got $(geometry.blocks)",
    )
    geometry.variables == expected_variables || error(
        "J40 geometry mismatch: expected $expected_variables variables, got $(geometry.variables)",
    )
    geometry.equalities == expected_equalities || error(
        "J40 geometry mismatch: expected $expected_equalities equalities, got $(geometry.equalities)",
    )
    analysis = SDPX.analyze_fixed_trace(problem)
    analysis.fixed_blocks == geometry.blocks || error(
        "fixed-trace preflight failed: $(analysis.fixed_blocks)/$(geometry.blocks) blocks have a constant trace",
    )
    analysis.soc_blocks == geometry.blocks || error(
        "fixed-trace preflight failed: $(analysis.soc_blocks)/$(geometry.blocks) blocks are Q3 candidates",
    )
    rejection = SDPX._fixed_trace_q3_rejection(problem)
    rejection === :eligible || error(
        "direct Q3 MOI conversion is blocked by SDPX's exact eligibility predicate: $rejection",
    )
    layout = SDPX._compile_fixed_trace_q3(problem)
    return geometry, analysis, layout
end

function check_q3_equivalence!(problem, layout)
    problem.cons isa SDPX.SparseCons{Float64} || error(
        "direct Q3 conversion requires SparseCons{Float64}; got $(typeof(problem.cons))",
    )
    cons = problem.cons::SDPX.SparseCons{Float64}
    max_error = 0.0
    scale = 1.0
    for block in 1:problem.dims.L
        constant = problem.C[block]
        predicted = (
            layout.head[block] + layout.offset_u[block],
            layout.offset_v[block],
            layout.head[block] - layout.offset_u[block],
        )
        actual = (-constant[1, 1], -constant[1, 2], -constant[2, 2])
        for index in 1:3
            scale = max(scale, abs(actual[index]), abs(predicted[index]))
            max_error = max(max_error, abs(predicted[index] - actual[index]))
        end
        active = cons.active[block]
        length(active) == 2 || error(
            "block $block has $(length(active)) active variables after Q3 eligibility",
        )
        for position in 1:2
            variable = layout.variables[position, block]
            variable == active[position] || error(
                "block $block active-variable ordering changed during Q3 compilation",
            )
            coefficient = cons.Asp[block][variable]
            expected = (
                layout.coefficient_u[position, block],
                layout.coefficient_v[position, block],
                -layout.coefficient_u[position, block],
            )
            actual_coefficient = (
                coefficient[1, 1], coefficient[1, 2], coefficient[2, 2],
            )
            for index in 1:3
                scale = max(scale, abs(actual_coefficient[index]), abs(expected[index]))
                max_error = max(max_error, abs(actual_coefficient[index] - expected[index]))
            end
        end
    end
    allowed = 128 * eps(Float64) * scale
    max_error <= allowed || error(
        "Q3 conversion identity check failed: max coefficient error $max_error > $allowed",
    )
    return (; max_absolute_error=max_error, allowed_error=allowed, scale)
end

function build_direct_q3_model(problem, layout; vector_equalities::Bool=true)
    model = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(model, problem.dims.m)
    objective_terms = MOI.ScalarAffineTerm{Float64}[]
    for variable in eachindex(problem.c)
        coefficient = Float64(problem.c[variable])
        iszero(coefficient) || push!(objective_terms, MOI.ScalarAffineTerm(coefficient, variables[variable]))
    end
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(objective_terms, 0.0),
    )

    soc_constraints = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.SecondOrderCone}[]
    for block in 1:problem.dims.L
        terms = MOI.VectorAffineTerm{Float64}[]
        for position in 1:2
            variable = variables[layout.variables[position, block]]
            coefficient_u = Float64(layout.coefficient_u[position, block])
            coefficient_v = Float64(layout.coefficient_v[position, block])
            iszero(coefficient_u) || push!(terms,
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(coefficient_u, variable)))
            iszero(coefficient_v) || push!(terms,
                MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(coefficient_v, variable)))
        end
        constants = Float64[
            layout.head[block],
            layout.offset_u[block],
            layout.offset_v[block],
        ]
        push!(soc_constraints, MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(terms, constants),
            MOI.SecondOrderCone(3),
        ))
    end

    equality_constraint = nothing
    equality_constraints = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}[]
    if problem.dims.n > 0 && vector_equalities
        terms = MOI.VectorAffineTerm{Float64}[]
        for equality in 1:problem.dims.n
            if problem.B isa SparseMatrixCSC
                for index in nzrange(problem.B, equality)
                    value = Float64(nonzeros(problem.B)[index])
                    iszero(value) || push!(terms, MOI.VectorAffineTerm(
                        equality,
                        MOI.ScalarAffineTerm(value, variables[rowvals(problem.B)[index]]),
                    ))
                end
            else
                for variable in axes(problem.B, 1)
                    value = Float64(problem.B[variable, equality])
                    iszero(value) || push!(terms, MOI.VectorAffineTerm(
                        equality,
                        MOI.ScalarAffineTerm(value, variables[variable]),
                    ))
                end
            end
        end
        equality_constraint = MOI.add_constraint(
            model,
            MOI.VectorAffineFunction(terms, -Float64.(problem.b)),
            MOI.Zeros(problem.dims.n),
        )
    elseif problem.dims.n > 0
        for equality in 1:problem.dims.n
            terms = MOI.ScalarAffineTerm{Float64}[]
            if problem.B isa SparseMatrixCSC
                for index in nzrange(problem.B, equality)
                    value = Float64(nonzeros(problem.B)[index])
                    iszero(value) || push!(terms, MOI.ScalarAffineTerm(
                        value,
                        variables[rowvals(problem.B)[index]],
                    ))
                end
            else
                for variable in axes(problem.B, 1)
                    value = Float64(problem.B[variable, equality])
                    iszero(value) || push!(terms, MOI.ScalarAffineTerm(
                        value,
                        variables[variable],
                    ))
                end
            end
            push!(equality_constraints, MOI.add_constraint(
                model,
                MOI.ScalarAffineFunction(terms, 0.0),
                MOI.EqualTo(Float64(problem.b[equality])),
            ))
        end
    end
    return (; model, variables, soc_constraints, equality_constraint, equality_constraints)
end

function _optional_module(name::Symbol)
    if name === :clarabel
        try
            return Base.require(Main, :Clarabel)
        catch exception
            error(
                "Clarabel.jl is unavailable; install Clarabel in the benchmark " *
                "environment or choose --solver=mosek. Original error: $(sprint(showerror, exception))",
            )
        end
    elseif name === :mosek
        try
            Base.require(Main, :Mosek)
            Base.require(Main, :MosekTools)
            return getfield(Main, :Mosek)
        catch exception
            error(
                "MosekTools.jl/Mosek.jl is unavailable; install both optional " *
                "packages (and a MOSEK license) or choose --solver=clarabel. " *
                "Original error: $(sprint(showerror, exception))",
            )
        end
    end
    error("unknown optional solver $name")
end

function module_provenance(module_object)
    source = try
        pathof(module_object)
    catch
        nothing
    end
    version = try
        Base.pkgversion(module_object)
    catch
        nothing
    end
    return Dict{String,Any}(
        "module" => string(nameof(module_object)),
        "version" => version === nothing ? "unknown" : string(version),
        "source_path" => source === nothing ? "unknown" : realpath(source),
    )
end

# Solver packages are intentionally optional and loaded only after the model
# preflight.  Julia 1.12 assigns methods added by a dynamic `import` to a
# newer world, so these tiny wrappers keep calls from a previously compiled
# harness function world-age safe.
moi_get(optimizer, attribute) = Base.invokelatest(MOI.get, optimizer, attribute)
moi_get(optimizer, attribute, index) = Base.invokelatest(MOI.get, optimizer, attribute, index)
moi_set(optimizer, attribute, value) = Base.invokelatest(MOI.set, optimizer, attribute, value)
moi_optimize!(optimizer) = Base.invokelatest(MOI.optimize!, optimizer)
moi_copy_to(optimizer, model) = Base.invokelatest(MOI.copy_to, optimizer, model)

function make_optimizer(name::Symbol, config)
    if name === :clarabel
        solver_module = _optional_module(:clarabel)
        tolerance = config.tolerance
        # Clarabel has no MOI NumberOfThreads attribute in current releases;
        # max_threads is the solver-native setting.
        optimizer_ctor = Base.invokelatest(getfield, solver_module, :Optimizer)
        optimizer = try
            Base.invokelatest(optimizer_ctor;
                verbose=!config.quiet,
                max_iter=config.max_iterations,
                time_limit=config.time_limit,
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
                tol_infeas_abs=tolerance,
                tol_infeas_rel=tolerance,
                tol_ktratio=tolerance,
                reduced_tol_gap_abs=tolerance,
                reduced_tol_gap_rel=tolerance,
                reduced_tol_feas=tolerance,
                reduced_tol_infeas_abs=tolerance,
                reduced_tol_infeas_rel=tolerance,
                reduced_tol_ktratio=tolerance,
                chordal_decomposition_enable=false,
                presolve_enable=false,
            )
        catch exception
            # Clarabel releases before the chordal-decomposition extension do
            # not accept that keyword.  Their SOC path is still equivalent;
            # only omit the unavailable optional setting.
            text = sprint(showerror, exception)
            occursin("chordal_decomposition_enable", text) || rethrow()
            Base.invokelatest(optimizer_ctor;
                verbose=!config.quiet,
                max_iter=config.max_iterations,
                time_limit=config.time_limit,
                tol_gap_abs=tolerance,
                tol_gap_rel=tolerance,
                tol_feas=tolerance,
                tol_infeas_abs=tolerance,
                tol_infeas_rel=tolerance,
                tol_ktratio=tolerance,
                reduced_tol_gap_abs=tolerance,
                reduced_tol_gap_rel=tolerance,
                reduced_tol_feas=tolerance,
                reduced_tol_infeas_abs=tolerance,
                reduced_tol_infeas_rel=tolerance,
                reduced_tol_ktratio=tolerance,
                presolve_enable=false,
            )
        end
        # `max_threads` is available in recent Clarabel builds.  Older
        # Clarabel.jl releases are single-threaded and simply do not expose
        # this MOI attribute; keep the bounded reference usable at one thread
        # while failing explicitly for an unsupported multi-thread request.
        try
            moi_set(optimizer, MOI.RawOptimizerAttribute("max_threads"), config.threads)
        catch exception
            config.threads == 1 || error(
                "this Clarabel.jl release has no max_threads setting; " *
                "rerun with --threads=1. Original error: $(sprint(showerror, exception))",
            )
        end
        return optimizer
    end
    solver_module = _optional_module(:mosek)
    optimizer_ctor = Base.invokelatest(getfield, solver_module, :Optimizer)
    optimizer = Base.invokelatest(optimizer_ctor)
    moi_set(optimizer, MOI.Silent(), config.quiet)
    moi_set(optimizer, MOI.TimeLimitSec(), config.time_limit)
    # MosekTools forwards the canonical MOSEK parameter names unchanged.
    for (parameter, value) in (
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS" => config.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS" => config.tolerance,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => config.tolerance,
        "MSK_IPAR_NUM_THREADS" => config.threads,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS" => config.max_iterations,
        "MSK_IPAR_PRESOLVE_USE" => 0,
    )
        moi_set(optimizer, MOI.RawOptimizerAttribute(parameter), value)
    end
    return optimizer
end

function q3_primal_to_psd(vector)
    length(vector) == 3 || error("external solver returned a non-Q3 primal vector")
    t, u, v = Float64.(vector)
    return [t + u v; v t - u]
end

function q3_dual_to_psd(vector)
    length(vector) == 3 || error("external solver returned a non-Q3 dual vector")
    z0, z1, z2 = Float64.(vector)
    # The factor 1/2 is required because the Q3 Euclidean dot product and the
    # PSD Frobenius product differ by a factor of two under this fixed-trace
    # isomorphism.  Omitting it makes the stationarity certificate fail.
    half = 0.5
    return [half * (z0 + z1) half * z2; half * z2 half * (z0 - z1)]
end

function external_result(problem, built, optimizer, index_map, config, status)
    status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error(
        "external solver terminated with $status; no certificate is claimed",
    )
    variables = built.variables
    x, X, Y, y = try
        primal = Float64[
            moi_get(optimizer, MOI.VariablePrimal(), index_map[variable])
            for variable in variables
        ]
        primal_blocks = Matrix{Float64}[
            q3_primal_to_psd(moi_get(
                optimizer,
                MOI.ConstraintPrimal(),
                index_map[index],
            )) for index in built.soc_constraints
        ]
        dual_blocks = Matrix{Float64}[
            q3_dual_to_psd(moi_get(
                optimizer,
                MOI.ConstraintDual(),
                index_map[index],
            )) for index in built.soc_constraints
        ]
        equality_dual = if built.equality_constraint !== nothing
            Float64.(moi_get(
                optimizer,
                MOI.ConstraintDual(),
                index_map[built.equality_constraint],
            ))
        else
            Float64[
                moi_get(optimizer, MOI.ConstraintDual(), index_map[index])
                for index in built.equality_constraints
            ]
        end
        primal, primal_blocks, dual_blocks, equality_dual
    catch exception
        error(
            "external $status solver did not expose a complete MOI primal/dual " *
            "iterate needed for SDPX.result_certificate: $(sprint(showerror, exception))",
        )
    end
    p_objective = dot(problem.c, x)
    d_objective = SDPX.dual_objective(problem, y, Y)
    result = SDPX.SDPResult{Float64}(
        SDPX.Optimal,
        "external $(status)",
        x,
        X,
        y,
        Y,
        p_objective,
        d_objective,
        abs(p_objective - d_objective) /
            max(1.0, (abs(p_objective) + abs(d_objective)) / 2),
        0.0,
        0.0,
        0,
        0,
        0,
        nothing,
        NamedTuple[],
        nothing,
        (reason=:external_reference, solver_status=string(status)),
    )
    options = SDPX.SolverOptions{Float64}(
        ϵ_gap=config.tolerance,
        ϵ_primal=config.tolerance,
        ϵ_dual=config.tolerance,
        iter_max=config.max_iterations,
        max_time=config.time_limit,
        verbosity=0,
    )
    certificate = SDPX.result_certificate(problem, result, options)
    certificate.valid || error(
        "external $(status) result failed SDPX.result_certificate: " *
        join(string.(certificate.failures), ", "),
    )
    return result, certificate
end

function run_solver(name::Symbol, problem, layout, config, shared_seconds)
    solver_started = time_ns()
    local built
    vector_equalities = name === :clarabel
    build_seconds = @elapsed built = build_direct_q3_model(
        problem,
        layout;
        vector_equalities,
    )
    warmup_seconds = 0.0
    for warmup in 1:config.warmup
        started = time_ns()
        optimizer = make_optimizer(name, config)
        index_map = moi_copy_to(optimizer, built.model)
        moi_optimize!(optimizer)
        warmup_seconds += (time_ns() - started) / 1.0e9
        external_result(
            problem,
            built,
            optimizer,
            index_map,
            config,
            moi_get(optimizer, MOI.TerminationStatus()),
        )
    end

    rows = Dict{String,Any}[]
    for repetition in 1:config.reps
        GC.gc()
        local optimizer
        local index_map
        setup_seconds = @elapsed begin
            optimizer = make_optimizer(name, config)
            index_map = moi_copy_to(optimizer, built.model)
        end
        solve_started = time_ns()
        moi_optimize!(optimizer)
        solver_seconds = (time_ns() - solve_started) / 1.0e9
        validation_started = time_ns()
        status = moi_get(optimizer, MOI.TerminationStatus())
        result, certificate = external_result(
            problem, built, optimizer, index_map, config, status,
        )
        validation_seconds = (time_ns() - validation_started) / 1.0e9
        end_to_end = shared_seconds + build_seconds + setup_seconds +
                     solver_seconds + validation_seconds
        push!(rows, Dict{String,Any}(
            "repetition" => repetition,
            "status" => string(status),
            "raw_status" => string(moi_get(optimizer, MOI.RawStatusString())),
            "iterations" => try Int(moi_get(optimizer, MOI.BarrierIterations())) catch; -1 end,
            "model_build_seconds" => build_seconds,
            "solver_setup_seconds" => setup_seconds,
            "warmup_seconds" => warmup_seconds,
            "solver_seconds" => solver_seconds,
            "validation_seconds" => validation_seconds,
            "end_to_end_seconds" => end_to_end,
            "objective_primal" => string(certificate.primal_objective),
            "objective_dual" => string(certificate.dual_objective),
            "relative_gap" => string(certificate.gap_relative),
            "primal_residual" => string(certificate.primal_residual),
            "dual_residual" => string(certificate.dual_residual),
            "equality_backward_error" => string(certificate.equality_backward_error),
            "primal_block_backward_error" => string(certificate.primal_block_backward_error),
            "dual_backward_error" => string(certificate.dual_backward_error),
            "certificate_valid" => certificate.valid,
            "minimum_primal_psd2_margin" => string(minimum_psd2_margin(result.X)),
            "minimum_dual_psd2_margin" => string(minimum_psd2_margin(result.Y)),
        ))
    end
    total_seconds = (time_ns() - solver_started) / 1.0e9 + shared_seconds
    return Dict{String,Any}(
        "solver" => string(name),
        "equality_encoding" => vector_equalities ? "vector_zeros" : "scalar_equal_to",
        "model_build_seconds" => build_seconds,
        "solver_setup_seconds" => isempty(rows) ? 0.0 :
                                  rows[1]["solver_setup_seconds"],
        "warmup_seconds" => warmup_seconds,
        "end_to_end_seconds" => total_seconds,
        "runs" => rows,
    )
end

function minimum_psd2_margin(blocks)
    isempty(blocks) && return NaN
    return minimum(blocks) do block
        a, b, c = block[1, 1], (block[1, 2] + block[2, 1]) / 2, block[2, 2]
        (a + c - sqrt((a - c)^2 + 4b^2)) / 2
    end
end

function _toml_value(value)
    value isa AbstractFloat && !isfinite(value) && return string(value)
    value isa Number || value isa Bool || value isa AbstractString || value === nothing || return string(value)
    return value
end

function write_report(path, report)
    isempty(path) && return
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        TOML.print(io, report; sorted=true)
    end
end

function print_summary(report)
    println("fixed-trace direct-Q3 Float64 reference")
    println("case=", report["case"], " model_sha256=", report["model_sha256"])
    println("reduced_model_sha256=", report["reduced_model_sha256"])
    geometry = report["geometry"]
    println(
        "geometry blocks=", geometry["blocks"],
        " variables=", geometry["variables"],
        " equalities=", geometry["equalities"],
    )
    for solver in report["solvers"]
        runs = solver["runs"]
        if isempty(runs)
            println("solver=", solver["solver"], " preflight-only")
            continue
        end
        row = first(runs)
        println(
            "solver=", solver["solver"], " status=", row["status"],
            " solver_seconds=", row["solver_seconds"],
            " certificate_valid=", row["certificate_valid"],
            " objective=", row["objective_primal"],
        )
    end
end

function main(args=ARGS)
    config = parse_cli(args)
    expected_source = get(ENV, "SDPX_SOURCE", "")
    if !isempty(expected_source)
        actual_source = realpath(pathof(SDPX))
        expected_root = realpath(expected_source)
        startswith(actual_source, joinpath(expected_root, "src")) || error(
            "loaded SDPX from $actual_source, expected candidate $expected_root",
        )
    end
    campaign_started = time_ns()
    hash_started = time_ns()
    model_hash = sha256_file(config.model)
    hash_seconds = (time_ns() - hash_started) / 1.0e9
    model_hash == config.expected_hash || error(
        "model SHA-256 mismatch: expected $(config.expected_hash), got $model_hash",
    )

    load_started = time_ns()
    reduced_raw = load_reduced_model(config)
    model_load_seconds = (time_ns() - load_started) / 1.0e9

    conversion_started = time_ns()
    problem = FixedTraceSupport._problem_as_type(reduced_raw, Float64, true)
    model_conversion_seconds = (time_ns() - conversion_started) / 1.0e9
    # Hash the compact, type-stable reduced problem rather than a legacy raw
    # SparseCons that may retain an m-entry reference vector per block.
    reduced_hash_started = time_ns()
    reduced_hash = serialized_sha256(problem)
    reduced_hash_seconds = (time_ns() - reduced_hash_started) / 1.0e9
    preflight_started = time_ns()
    geometry, analysis, layout = preflight_q3!(problem, config)
    conversion_check = check_q3_equivalence!(problem, layout)
    preflight_seconds = (time_ns() - preflight_started) / 1.0e9

    selected = config.solver == "both" ?
               (:clarabel, :mosek) : (Symbol(config.solver),)
    solver_provenance = Dict{String,Any}()
    if !config.preflight_only
        # Resolve every requested optional package before starting the first
        # solver. This prevents a partial campaign and records the exact
        # external implementation whose timing is reported.
        for solver in selected
            module_object = _optional_module(solver)
            if solver === :clarabel
                solver_provenance["clarabel"] = module_provenance(module_object)
            else
                solver_provenance["mosek"] = module_provenance(module_object)
                solver_provenance["mosek_tools"] = module_provenance(
                    getfield(Main, :MosekTools),
                )
            end
        end
    end

    active_project = Base.active_project()
    active_manifest = active_project === nothing ? "" :
                      joinpath(dirname(active_project), "Manifest.toml")

    shared_seconds = hash_seconds + model_load_seconds + reduced_hash_seconds +
                     model_conversion_seconds + preflight_seconds
    report = Dict{String,Any}(
        "case" => config.case_name,
        "arithmetic" => "Float64",
        "julia_version" => string(VERSION),
        "active_project" => active_project === nothing ? "none" : active_project,
        "active_manifest_sha256" => isfile(active_manifest) ?
                                    sha256_file(active_manifest) : "unavailable",
        "solver_provenance" => solver_provenance,
        "sdpx_source_path" => realpath(pathof(SDPX)),
        "sdpx_source_sha256" => source_tree_sha256(
            normpath(joinpath(dirname(pathof(SDPX)), "..")),
        ),
        "benchmark_driver_sha256" => sha256_file(@__FILE__),
        "shared_benchmark_driver_sha256" =>
            sha256_file(FixedTraceSupport.BENCHMARK_PATH),
        "model_path" => config.model,
        "model_sha256" => model_hash,
        "expected_model_sha256" => config.expected_hash,
        "reduced_model_sha256" => reduced_hash,
        "release_path" => isempty(config.release) ? "none" : config.release,
        "release_source_sha256" => source_tree_sha256(config.release),
        "tolerance" => config.tolerance_text,
        "max_iterations" => config.max_iterations,
        "time_limit_seconds" => config.time_limit,
        "threads_requested" => config.threads,
        "repetitions" => config.reps,
        "warmup" => config.warmup,
        "geometry" => Dict(
            "blocks" => geometry.blocks,
            "variables" => geometry.variables,
            "equalities" => geometry.equalities,
            "block_dimensions" => Int.(geometry.block_dimensions),
        ),
        "fixed_trace_blocks" => analysis.fixed_blocks,
        "fixed_trace_q3_blocks" => analysis.soc_blocks,
        "q3_eligibility" => "eligible",
        "q3_conversion_check" => Dict(
            "max_absolute_error" => conversion_check.max_absolute_error,
            "allowed_error" => conversion_check.allowed_error,
            "scale" => conversion_check.scale,
        ),
        "timing" => Dict(
            "model_hash_seconds" => hash_seconds,
            "model_load_seconds" => model_load_seconds,
            "reduced_model_hash_seconds" => reduced_hash_seconds,
            "model_conversion_seconds" => model_conversion_seconds,
            "preflight_seconds" => preflight_seconds,
            "shared_seconds" => shared_seconds,
        ),
        "solvers" => Dict{String,Any}[],
    )

    if !config.preflight_only
        for solver in selected
            push!(report["solvers"], run_solver(
                solver, problem, layout, config, shared_seconds,
            ))
        end
    end
    flat_runs = Dict{String,Any}[]
    for solver_record in report["solvers"]
        for row in solver_record["runs"]
            flattened = copy(row)
            flattened["solver"] = solver_record["solver"]
            push!(flat_runs, flattened)
        end
    end
    report["runs"] = flat_runs
    report["campaign_end_to_end_seconds"] = (time_ns() - campaign_started) / 1.0e9
    write_report(config.output, report)
    print_summary(report)
    isempty(config.output) || println("output=$(abspath(config.output))")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
