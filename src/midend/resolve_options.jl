"""
    ResolvedSolveOptions{T}

Result of lowering a user-facing [`SolveOptions`](@ref) to the concrete
configuration consumed by the current solver core.  `summary` contains only
stable, human-readable policy decisions and is suitable for diagnostics,
benchmark provenance and CLI output.
"""
struct ResolvedSolveOptions{T}
    core::SolverOptions{T}
    certification::Bool
    summary::NamedTuple
end

@inline _is_auto(value) = value === :auto ||
    (value isa AbstractString && lowercase(strip(value)) == "auto")

function _frontend_number(::Type{T}, value, label::AbstractString) where {T}
    _is_auto(value) && throw(ArgumentError("$label is still :auto"))
    if value isa AbstractString
        try
            return parse(T, value)
        catch exception
            exception isa InterruptException && rethrow()
            throw(ArgumentError("$label must be numeric, got $(repr(value))"))
        end
    elseif value isa Real
        return T(value)
    end
    throw(ArgumentError("$label must be numeric or :auto, got $(typeof(value))"))
end

function _frontend_bool(value, default::Bool, label::AbstractString)
    _is_auto(value) && return default
    value isa Bool && return value
    value isa Symbol && value in (:on, :true) && return true
    value isa Symbol && value in (:off, :false) && return false
    if value isa AbstractString
        normalized = lowercase(strip(value))
        normalized in ("on", "true", "yes", "1") && return true
        normalized in ("off", "false", "no", "0") && return false
    end
    throw(ArgumentError("$label must be :auto, on/off, or Bool"))
end

function _frontend_integer(value, default::Int, label::AbstractString; minimum::Int=0)
    result = if _is_auto(value)
        default
    elseif value isa Integer
        Int(value)
    elseif value isa AbstractString
        try
            parse(Int, value)
        catch exception
            exception isa InterruptException && rethrow()
            throw(ArgumentError("$label must be an integer or :auto"))
        end
    else
        throw(ArgumentError("$label must be an integer or :auto"))
    end
    result >= minimum || throw(ArgumentError("$label must be >= $minimum"))
    return result
end

function _frontend_float(value, default::Float64, label::AbstractString; minimum=-Inf)
    result = if _is_auto(value)
        default
    elseif value isa Real
        Float64(value)
    elseif value isa AbstractString
        try
            parse(Float64, value)
        catch exception
            exception isa InterruptException && rethrow()
            throw(ArgumentError("$label must be numeric or :auto"))
        end
    else
        throw(ArgumentError("$label must be numeric or :auto"))
    end
    result >= minimum || throw(ArgumentError("$label must be >= $minimum"))
    return result
end

function _frontend_policy(value, allowed::Tuple, label::AbstractString)
    candidate = if value isa AbstractString
        Symbol(lowercase(strip(value)))
    elseif value isa Symbol
        value
    elseif value isa Bool
        value
    else
        throw(ArgumentError("$label must be one of $(allowed)"))
    end
    candidate in allowed || throw(ArgumentError(
        "$label must be one of $(allowed), got $(repr(value))",
    ))
    return candidate
end

"""
    auto_tolerance(T, precision_bits=sig_bits(T))

Conservative frontend stopping target used only when the user leaves a
threshold at `:auto`.

The target uses roughly one third of the available decimal precision, with a
Float64-compatible floor of `1e-8`.  This deliberately leaves substantial
headroom for KKT conditioning, equilibration and cone algebra.  For example,
840-bit BigFloat resolves to about `1e-84`, close to the high-precision
thresholds commonly supplied explicitly in SDPB-style workflows.

This is a frontend policy, not a mathematical guarantee.  Final status remains
subject to SDPX's original-coordinate certificate.
"""
function auto_tolerance(::Type{T}, precision_bits::Integer=sig_bits(T)) where {T}
    bits = max(Int(precision_bits), 1)
    decimal_digits = max(1, floor(Int, bits * log10(2.0)))
    exponent = max(8, fld(decimal_digits, 3))
    exponent = min(exponent, max(1, decimal_digits - 6))
    text = "1e-$(exponent)"
    try
        return parse(T, text)
    catch exception
        exception isa InterruptException && rethrow()
        return T(10.0)^(-exponent)
    end
end

function _arithmetic_symbol(::Type{T}) where {T}
    T === Float64 && return :float64
    T === BigFloat && return :bigfloat
    name = lowercase(string(T))
    occursin("float64x2", name) && return :float64x2
    occursin("float64x3", name) && return :float64x3
    occursin("float64x4", name) && return :float64x4
    return Symbol(lowercase(replace(string(T), '.' => '_')))
end

function _resolve_precision_bits(::Type{T}, requested) where {T}
    native = T === BigFloat ? Base.precision(BigFloat) : sig_bits(T)
    _is_auto(requested) && return native
    if requested isa Integer ||
       (requested isa AbstractString && all(isdigit, strip(requested)))
        bits = requested isa Integer ? Int(requested) : parse(Int, strip(requested))
        bits > 0 || throw(ArgumentError("precision must be a positive bit count"))
        T === BigFloat || throw(ArgumentError(
            "an integer precision requests BigFloat input, but the problem is stored as $T; " *
            "re-ingest the model at BigFloat precision or use the CLI so parsing occurs at the requested precision",
        ))
        return bits
    end
    requested_symbol = requested isa Symbol ? requested : Symbol(lowercase(strip(String(requested))))
    requested_symbol === :auto && return native
    requested_symbol === _arithmetic_symbol(T) || throw(ArgumentError(
        "requested precision $(repr(requested)) is incompatible with stored arithmetic $T",
    ))
    return native
end

function _resolve_threshold(::Type{T}, value, automatic::T, label::AbstractString) where {T}
    threshold = _is_auto(value) ? automatic : _frontend_number(T, value, label)
    threshold > zero(T) || throw(ArgumentError("$label must be positive"))
    return threshold
end

"""
    resolve_solve_options(T, options=SolveOptions()) -> ResolvedSolveOptions{T}

Lower the small, all-auto user policy to the concrete expert options consumed
by the current backend.  This function is deterministic: the same arithmetic,
problem-independent frontend policy and process thread count resolve to the
same values.  Structural backend choices remain the responsibility of SDPX's
`ExecutionPlan` so there is a single authoritative planner.
"""
function resolve_solve_options(
    ::Type{T},
    options::SolveOptions=SolveOptions(),
) where {T}
    precision_bits = _resolve_precision_bits(T, options.precision)
    automatic = auto_tolerance(T, precision_bits)
    gap = _resolve_threshold(T, options.duality_gap_threshold, automatic,
                             "duality_gap_threshold")
    primal = _resolve_threshold(T, options.primal_error_threshold, automatic,
                                "primal_error_threshold")
    dual = _resolve_threshold(T, options.dual_error_threshold, automatic,
                              "dual_error_threshold")

    maximum_iterations = _frontend_integer(
        options.maximum_iterations, 200, "maximum_iterations"; minimum=1,
    )
    max_runtime = _frontend_float(
        options.max_runtime, Inf, "max_runtime"; minimum=0.0,
    )
    threads = _frontend_integer(
        options.threads, Base.Threads.nthreads(), "threads"; minimum=1,
    )
    verbosity = _frontend_integer(options.verbosity, 1, "verbosity"; minimum=0)

    presolve = _frontend_policy(options.presolve, (:auto, :on, :off, true, false), "presolve")
    presolve = presolve === :on ? true : presolve === :off ? false : presolve
    sparse = _frontend_policy(options.sparse, (:auto, :on, :off, true, false), "sparse")
    sparse = sparse === :on ? true : sparse === :off ? false : sparse
    scaling = _frontend_policy(options.scaling, (:auto, :none, :equilibrate), "scaling")
    algorithm = _frontend_policy(options.algorithm, (:auto, :lp, :socp, :sdp), "algorithm")
    formulation = _frontend_policy(options.formulation, (:auto, :primal, :dual), "formulation")
    chordal = _frontend_policy(options.chordal_decomposition, (:auto, :on, :off), "chordal_decomposition")
    equality_solver = _frontend_policy(
        options.equality_solver, (:auto, :normal_equations, :qr), "equality_solver",
    )
    linear_algebra_backend = _frontend_policy(
        options.linear_algebra_backend,
        (:auto, :standard, :multifloat, :legacy),
        "linear_algebra_backend",
    )
    working_precision_policy = _frontend_policy(
        options.working_precision_policy, (:auto, :fixed), "working_precision_policy",
    )
    diagnostics = _frontend_bool(options.diagnostics, true, "diagnostics")
    timing = _frontend_bool(options.timing, true, "timing")
    certification = _frontend_bool(options.certification, true, "certification")

    core = SolverOptions{T}(
        ϵ_gap=gap,
        ϵ_primal=primal,
        ϵ_dual=dual,
        iter_max=maximum_iterations,
        precision_bits=precision_bits,
        max_time=max_runtime,
        threads=threads,
        verbosity=verbosity,
        diagnostics=diagnostics,
        timing=timing,
        presolve=presolve,
        scaling=scaling,
        sparse=sparse,
        algorithm=algorithm,
        formulation=formulation,
        chordal_decomposition=chordal,
        equality_solver=equality_solver,
        linear_algebra_backend=linear_algebra_backend,
        working_precision_policy=working_precision_policy,
        parameter_policy=:auto,
        parameter_strategy=:adaptive,
        refine_policy=:auto,
        step_rule=:auto,
        certification=certification,
    )

    summary = (
        arithmetic=_arithmetic_symbol(T),
        precision_bits=precision_bits,
        duality_gap_threshold=string(gap),
        primal_error_threshold=string(primal),
        dual_error_threshold=string(dual),
        maximum_iterations=maximum_iterations,
        max_runtime=max_runtime,
        threads=threads,
        verbosity=verbosity,
        presolve=presolve,
        scaling=scaling,
        algorithm=algorithm,
        sparse=sparse,
        formulation=formulation,
        chordal_decomposition=chordal,
        equality_solver=equality_solver,
        linear_algebra_backend=linear_algebra_backend,
        working_precision_policy=working_precision_policy,
        diagnostics=diagnostics,
        timing=timing,
        certification=certification,
    )
    return ResolvedSolveOptions{T}(core, certification, summary)
end
