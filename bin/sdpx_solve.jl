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

function __init__()
    PRECISIONS["Float64"] = Float64
    # Extended types register lazily so the bridge works without the
    # extensions installed; requesting one without its package is a
    # structured error, not a MethodError.
    try
        @eval using MultiFloats
        @eval PRECISIONS["Float64x2"] = MultiFloats.Float64x2
        @eval PRECISIONS["Float64x4"] = MultiFloats.Float64x4
    catch exception
        exception isa InterruptException && rethrow()
    end
    PRECISIONS["BigFloat"] = BigFloat
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

function _solver_options(::Type{T}, settings) where {T}
    tolerance = _number(T, get(settings, "tolerance", "1e-8"))
    return SDPX.SolverOptions{T}(
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        iter_max=Int(get(settings, "maximum_iterations", 200)),
        max_time=Float64(get(settings, "time_limit", Inf)),
        threads=Int(get(settings, "threads", 1)),
        verbosity=Int(get(settings, "verbosity", 0)),
        precision_bits=Int(get(settings, "precision_bits", 256)),
    )
end

function solve_specification(spec)
    Int(get(spec, "sdpx_schema", 0)) == SCHEMA_VERSION ||
        error("unsupported or missing \"sdpx_schema\" (this bridge speaks version $SCHEMA_VERSION)")
    name = String(get(spec, "precision", "Float64"))
    haskey(PRECISIONS, name) ||
        error("unknown precision \"$name\"; available: $(sort!(collect(keys(PRECISIONS)))) " *
              "(Float64x2/Float64x4 need the MultiFloats package in the bridge environment)")
    T = PRECISIONS[name]
    settings = get(spec, "settings", Dict{String,Any}())

    run_solve = function ()
        c, A, C, B, b = _build_problem(T, spec)
        problem = SDPX.ingest(c, A, C, B, b; verbosity=0)
        options = _solver_options(T, settings)
        result = SDPX.solve!(problem, options)
        return problem, options, result
    end
    # BigFloat data must be *parsed* at the working precision, not converted
    # to it afterwards; setprecision therefore wraps the whole build+solve.
    problem, options, result = T === BigFloat ?
        setprecision(run_solve, BigFloat, Int(get(settings, "precision_bits", 256))) :
        run_solve()

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
    )
    if get(settings, "return_matrices", false) == true
        response["X"] = [[_render(v) for v in vec(block)] for block in result.X]
        response["Y"] = [[_render(v) for v in vec(block)] for block in result.Y]
        response["block_dimensions"] = [size(block, 1) for block in result.X]
    end
    if get(settings, "certificate", true) == true
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
