#!/usr/bin/env julia
#=====================================================================
    SDPX command-line bridge (schema v1)

        julia --project=bin bin/sdpx_solve.jl problem.json result.json

    A language-independent entry point: any environment that can write
    a JSON file and run a process can use the solver. The first
    consumer is the Mathematica package in `mathematica/SDPXLink.wl`.

    Design constraints, in order:

    1.  The solver core must not grow a JSON dependency. This script
        lives outside `src/` with its own environment (`bin/Project.toml`)
        and touches only the public API (`ingest`, `SolverOptions`,
        `solve!`, `solve_summary`).
    2.  Numbers cross the boundary as *strings* whenever the precision
        exceeds Float64, because JSON numbers are IEEE doubles in most
        parsers and silently round anything wider. Plain JSON numbers
        are also accepted on input for Float64 convenience.
    3.  A failure must still produce a structured result file. The
        caller is a foreign runtime that cannot parse a Julia stack
        trace; it can parse `{"success": false, "error": "..."}`.

    The schema is documented in `docs/bridge-schema.md`. This is a
    subprocess bridge by design — one process per solve, no state. The
    documented upgrade path (same document) is a persistent server or
    LibraryLink once the schema has proven itself; the schema is the
    part meant to outlive the transport.
=====================================================================#

module SDPXSolveCLI

using JSON
using SDPX

const SCHEMA_VERSION = 1

const PRECISIONS = Dict{String,DataType}()
const _PRECISIONS_READY = Ref(false)

function _ensure_precisions!()
    _PRECISIONS_READY[] && return PRECISIONS
    PRECISIONS["Float64"] = Float64
    # Extended types register lazily so the bridge works without the
    # extensions installed; requesting one without its package is a
    # structured error, not a MethodError. This helper is called from both
    # `__init__` and `solve_specification`, so include-based tests/embedders do
    # not depend on package-loader `__init__` semantics.
    try
        @eval using MultiFloats
        @eval PRECISIONS["Float64x2"] = MultiFloats.Float64x2
        @eval PRECISIONS["Float64x3"] = MultiFloats.Float64x3
        @eval PRECISIONS["Float64x4"] = MultiFloats.Float64x4
    catch exception
        exception isa InterruptException && rethrow()
    end
    PRECISIONS["BigFloat"] = BigFloat
    _PRECISIONS_READY[] = true
    return PRECISIONS
end

function __init__()
    _ensure_precisions!()
    return
end

"""Parse one schema number: a JSON number, or a string for precisions JSON
cannot carry. `1e-30` written as a JSON number has already been rounded to a
Float64 by the parser, which is exactly the silent loss the strings exist to
avoid — hence strings are accepted for every precision and required above
Float64 only by convention."""
_number(::Type{T}, value::Real) where {T} = T(value)
_number(::Type{T}, value::AbstractString) where {T} = parse(T, value)
_number(::Type{T}, value) where {T} =
    error("expected a number or numeric string, got $(typeof(value))")

"""Render one number for the result file. `string` on BigFloat and MultiFloat
values round-trips every bit; scientific text is what every consumer parses."""
_render(value) = string(value)

_json_safe(value::Symbol) = string(value)
_json_safe(value::AbstractFloat) = isfinite(value) ? value : string(value)
_json_safe(value) = value

function _coo_matrix(::Type{T}, entry, dimension::Int, label::String) where {T}
    matrix = zeros(T, dimension, dimension)
    rows = get(entry, "rows", Any[])
    cols = get(entry, "cols", Any[])
    values = get(entry, "values", Any[])
    length(rows) == length(cols) == length(values) ||
        error("$label: rows/cols/values lengths differ")
    for (r, c, v) in zip(rows, cols, values)
        (1 <= r <= dimension && 1 <= c <= dimension) ||
            error("$label: index ($r,$c) outside 1:$dimension")
        matrix[Int(r), Int(c)] = _number(T, v)
    end
    return matrix
end

function _build_problem(::Type{T}, spec) where {T}
    haskey(spec, "objective") || error("problem is missing \"objective\"")
    haskey(spec, "blocks") || error("problem is missing \"blocks\"")
    objective = T[_number(T, v) for v in spec["objective"]]
    m = length(objective)
    m > 0 || error("objective is empty")

    blocks = spec["blocks"]
    isempty(blocks) && error("\"blocks\" is empty")
    coefficients = Vector{Array{T,3}}(undef, length(blocks))
    constants = Vector{Matrix{T}}(undef, length(blocks))
    for (index, block) in enumerate(blocks)
        haskey(block, "dimension") || error("block $index is missing \"dimension\"")
        dimension = Int(block["dimension"])
        dimension > 0 || error("block $index: dimension must be positive")
        slab = zeros(T, m, dimension, dimension)
        for entry in get(block, "coefficients", Any[])
            haskey(entry, "variable") || error("block $index: coefficient entry missing \"variable\"")
            variable = Int(entry["variable"])
            (1 <= variable <= m) ||
                error("block $index: variable $variable outside 1:$m")
            slab[variable, :, :] =
                _coo_matrix(T, entry, dimension, "block $index, variable $variable")
        end
        coefficients[index] = slab
        constants[index] = haskey(block, "constant") ?
            _coo_matrix(T, block["constant"], dimension, "block $index constant") :
            zeros(T, dimension, dimension)
    end

    if haskey(spec, "equalities") && spec["equalities"] !== nothing
        eq = spec["equalities"]
        rhs = T[_number(T, v) for v in get(eq, "rhs", Any[])]
        n = length(rhs)
        B = zeros(T, m, n)
        for (r, c, v) in zip(get(eq, "rows", Any[]), get(eq, "cols", Any[]),
                             get(eq, "values", Any[]))
            (1 <= r <= m && 1 <= c <= n) ||
                error("equalities: index ($r,$c) outside ($m,$n)")
            B[Int(r), Int(c)] = _number(T, v)
        end
    else
        B = Matrix{T}(undef, m, 0)
        rhs = T[]
    end
    return objective, coefficients, constants, B, rhs
end

function _setting(settings, names...; default=:auto)
    for name in names
        haskey(settings, name) && return settings[name]
    end
    return default
end

function _solver_options(::Type{T}, settings) where {T}
    common = _setting(settings, "tolerance"; default=:auto)
    gap = _setting(
        settings,
        "dualityGapThreshold", "duality_gap_threshold", "gap_tolerance";
        default=common,
    )
    primal = _setting(
        settings,
        "primalErrorThreshold", "primal_error_threshold", "primal_tolerance";
        default=common,
    )
    dual = _setting(
        settings,
        "dualErrorThreshold", "dual_error_threshold", "dual_tolerance";
        default=common,
    )
    precision_request = _setting(settings, "precision_bits"; default=:auto)
    frontend = SDPX.SolveOptions(
        precision=precision_request,
        duality_gap_threshold=gap,
        primal_error_threshold=primal,
        dual_error_threshold=dual,
        maximum_iterations=_setting(
            settings, "maximumIterations", "maximum_iterations"; default=:auto,
        ),
        max_runtime=_setting(
            settings, "maxRuntime", "time_limit"; default=:auto,
        ),
        threads=_setting(settings, "threads"; default=:auto),
        verbosity=_setting(settings, "verbosity"; default=:auto),
        presolve=_setting(settings, "presolve"; default=:auto),
        scaling=_setting(settings, "scaling"; default=:auto),
        algorithm=_setting(settings, "algorithm"; default=:auto),
        sparse=_setting(settings, "sparse"; default=:auto),
        formulation=_setting(settings, "formulation"; default=:auto),
        equality_solver=_setting(
            settings, "equalitySolver", "equality_solver"; default=:auto,
        ),
        working_precision_policy=_setting(
            settings, "workingPrecisionPolicy", "working_precision_policy"; default=:auto,
        ),
        diagnostics=_setting(settings, "diagnostics"; default=:auto),
        timing=_setting(settings, "timing"; default=:auto),
        certification=_setting(settings, "certificate", "certification"; default=:auto),
    )
    return SDPX.Experimental.resolve_solve_options(T, frontend)
end

function _precision_name_and_bits(spec, settings)
    raw = get(spec, "precision", "auto")
    bits = _setting(settings, "precision_bits"; default=:auto)
    if raw isa Integer
        raw > 0 || error("precision bit count must be positive")
        return "BigFloat", Int(raw)
    end
    name = String(raw)
    lower = lowercase(strip(name))
    if lower == "auto"
        if bits isa Integer && bits > 53
            return "BigFloat", Int(bits)
        elseif bits isa AbstractString && lowercase(strip(bits)) != "auto"
            parsed = parse(Int, bits)
            parsed > 53 && return "BigFloat", parsed
        end
        return "Float64", 53
    end
    aliases = Dict(
        "float64" => "Float64",
        "float64x2" => "Float64x2",
        "float64x3" => "Float64x3",
        "float64x4" => "Float64x4",
        "bigfloat" => "BigFloat",
    )
    canonical = get(aliases, lower, name)
    resolved_bits = if canonical == "BigFloat"
        if bits === :auto || (bits isa AbstractString && lowercase(strip(bits)) == "auto")
            256
        else
            Int(bits isa Integer ? bits : parse(Int, bits))
        end
    else
        0
    end
    return canonical, resolved_bits
end

function _plan_response(result)
    diagnostics = result.diagnostics
    diagnostics === nothing && return nothing
    plan = diagnostics.plan
    classification = plan.classification
    return Dict{String,Any}(
        "cone" => string(classification.cone),
        "storage" => string(classification.storage),
        "arithmetic" => string(classification.arithmetic),
        "size" => string(classification.size),
        "variables" => classification.variables,
        "equalities" => classification.equalities,
        "cone_rows" => classification.cone_rows,
        "maximum_block_size" => classification.maximum_block_size,
        "algorithm" => string(plan.algorithm),
        "scaling" => string(plan.scaling),
        "kkt_backend" => string(plan.kkt_backend),
        "gram_kernel" => string(plan.gram_kernel),
        "schedule" => string(plan.schedule),
        "threads" => plan.threads,
        "parameter_profile" => string(plan.parameter_profile),
        "memory_budget_bytes" => plan.memory_budget_bytes,
    )
end

function solve_specification(spec)
    _ensure_precisions!()
    Int(get(spec, "sdpx_schema", 0)) == SCHEMA_VERSION ||
        error("unsupported or missing \"sdpx_schema\" (this bridge speaks version $SCHEMA_VERSION)")
    settings = get(spec, "settings", Dict{String,Any}())
    name, requested_bits = _precision_name_and_bits(spec, settings)
    haskey(PRECISIONS, name) ||
        error("unknown precision \"$name\"; available: $(sort!(collect(keys(PRECISIONS)))) " *
              "(Float64x2/Float64x3/Float64x4 need MultiFloats in the bridge environment)")
    T = PRECISIONS[name]
    if T === BigFloat
        settings["precision_bits"] = requested_bits
    end

    run_solve = function ()
        c, A, C, B, b = _build_problem(T, spec)
        problem = SDPX.ingest(c, A, C, B, b; verbosity=0)
        resolved = _solver_options(T, settings)
        result = SDPX.solve!(problem, resolved.core)
        return problem, resolved, result
    end
    # BigFloat data must be *parsed* at the working precision, not converted
    # to it afterwards; setprecision therefore wraps the whole build+solve.
    problem, resolved, result = T === BigFloat ?
        setprecision(run_solve, BigFloat, Int(get(settings, "precision_bits", 256))) :
        run_solve()
    options = resolved.core

    response = Dict{String,Any}(
        "sdpx_schema" => SCHEMA_VERSION,
        "success" => true,
        "error" => nothing,
        "precision" => name,
        "status" => string(result.status),
        "optimal" => result.status == SDPX.Optimal,
        "message" => result.message,
        "objective" => _render(result.pObj),
        "dual_objective" => _render(result.dObj),
        "relative_gap" => _render(result.gap_rel),
        "primal_residual" => _render(result.p_res),
        "dual_residual" => _render(result.d_res),
        "iterations" => result.iterations,
        "x" => [_render(v) for v in result.x],
        "y" => [_render(v) for v in result.y],
        "resolved_options" => Dict(string(k) => _json_safe(v) for (k, v) in pairs(resolved.summary)),
    )
    plan_response = _plan_response(result)
    plan_response === nothing || (response["plan"] = plan_response)
    if get(settings, "return_matrices", false) == true
        response["X"] = [[_render(v) for v in vec(block)] for block in result.X]
        response["Y"] = [[_render(v) for v in vec(block)] for block in result.Y]
        response["block_dimensions"] = [size(block, 1) for block in result.X]
    end
    if resolved.certification
        certificate = SDPX.result_certificate(problem, result, options)
        response["certificate"] = Dict{String,Any}(
            "valid" => certificate.valid,
            "gap" => _render(certificate.gap),
            "primal_residual" => _render(certificate.primal_residual),
            "dual_residual" => _render(certificate.dual_residual),
            "primal_psd" => certificate.primal_psd.ok,
            "dual_psd" => certificate.dual_psd.ok,
            "failures" => [string(f) for f in certificate.failures],
        )
    end
    return response
end

failure_response(message) = Dict{String,Any}(
    "sdpx_schema" => SCHEMA_VERSION,
    "success" => false,
    "status" => "Error",
    "error" => message,
)

function main(arguments)
    if length(arguments) != 2
        println(stderr, "usage: sdpx_solve.jl problem.json result.json")
        return 2
    end
    input_path, output_path = arguments
    response, code = try
        isfile(input_path) || error("input file not found: $input_path")
        spec = JSON.parsefile(input_path)
        solve_specification(spec), 0
    catch exception
        exception isa InterruptException && rethrow()
        # The consumer is a foreign runtime: it gets a structured error it
        # can parse, and the exit code says "look at the error field".
        failure_response(sprint(showerror, exception)), 1
    end
    open(output_path, "w") do io
        JSON.print(io, response)
    end
    return code
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(SDPXSolveCLI.main(ARGS))
end
