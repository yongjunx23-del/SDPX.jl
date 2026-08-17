using SDPX
using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats

"""Bound the normalized second moment of a quartic integral with an SDP.

For `exp(-x^2/2-g*x^4/4)`, integration by parts gives

    (2n+1) W_(2n) - W_(2n+2) - g W_(2n+4) = 0.

`w[n+1]` stores `W_(2n)`, `W_0=1`, and the parity-aware Hankel moment matrix
is constrained positive semidefinite. Minimizing and maximizing `W_2` gives
finite-order certified bounds. The script reconstructs the numerical Hankel
matrix after solving and reports its smallest eigenvalue.
"""

const _QUARTIC_REFERENCE_W2 = "0.467919916973665188637421298330615640"

function _precision_scope(f::Function, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        f()
    end
end

function _precision_scope(f::Function, ::Type{T}, bits) where {T<:AbstractFloat}
    return f()
end

function _decimal(::Type{BigFloat}, text::AbstractString)
    return BigFloat(text)
end

function _decimal(::Type{T}, text::AbstractString) where {T<:AbstractFloat}
    return T(parse(Float64, text))
end

function _arithmetic_type(name::AbstractString)
    key = lowercase(name)
    key == "f64" && return (Float64, nothing)
    key == "bf256" && return (BigFloat, 256)
    key == "bf512" && return (BigFloat, 512)
    key in ("f64x2", "f64x4") || error(
        "unknown arithmetic '$name'; choose f64, f64x2, f64x4, bf256, or bf512",
    )
    type_name = key == "f64x2" ? :Float64x2 : :Float64x4
    isdefined(MultiFloats, type_name) || error(
        "arithmetic $key is unavailable: MultiFloats does not provide $type_name",
    )
    return (getfield(MultiFloats, type_name), nothing)
end

function _quartic_model(
    ::Type{T},
    bits,
    g_text::AbstractString,
    order::Int,
    sense,
) where {T<:AbstractFloat}
    model = T === BigFloat ?
        SDPX.Model(T; precision_bits=bits, name="quartic_integral") :
        SDPX.Model(T; name="quartic_integral")
    g = _decimal(T, g_text)
    # w[1], ..., w[order+1] are W_0, ..., W_(2 order).
    w = SDPX.variable!(model, :w, order + 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :normalization, w[1] - one(T), SDPX.ZeroCone())
    for n in 0:(order - 2)
        recurrence = (2n + 1) * w[n + 1] - w[n + 2] - g * w[n + 3]
        SDPX.constraint!(model, Symbol("recurrence_", n), recurrence, SDPX.ZeroCone())
    end

    entry_type = typeof(w[1])
    hankel = Matrix{Union{T,entry_type}}(undef, order + 1, order + 1)
    for i in 0:order, j in 0:order
        hankel[i + 1, j + 1] = iseven(i + j) ? w[(i + j) ÷ 2 + 1] : zero(T)
    end
    SDPX.constraint!(model, :hankel_psd, hankel, SDPX.PSDCone())
    SDPX.objective!(model, sense, w[2])
    return model, w
end

function _quartic_settings(model, threads::Int, blas_threads::Int, max_iterations::Int)
    return SDPX.Settings(
        model;
        algorithm=:sdp,
        limits=SDPX.Limits(iterations=max_iterations, time=120.0, threads=threads),
        verbosity=0,
        timing=false,
        diagnostics=:summary,
        certification=true,
        blas_threads=blas_threads,
    )
end

function _quartic_outputs()
    return SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
end

function _quartic_hankel(moments::AbstractVector{T}, order::Int) where {T}
    matrix = Matrix{T}(undef, order + 1, order + 1)
    for i in 0:order, j in 0:order
        matrix[i + 1, j + 1] =
            iseven(i + j) ? moments[(i + j) ÷ 2 + 1] : zero(T)
    end
    return matrix
end

function _quartic_bound(
    ::Type{T},
    bits,
    g_text::AbstractString,
    order::Int,
    sense,
    threads::Int,
    blas_threads::Int,
    max_iterations::Int,
) where {T<:AbstractFloat}
    model, w = _quartic_model(T, bits, g_text, order, sense)
    result = SDPX.optimize!(
        model;
        settings=_quartic_settings(model, threads, blas_threads, max_iterations),
        outputs=_quartic_outputs(),
    )
    status = SDPX.status(result)
    status === :optimal || error("quartic bound did not reach optimal: status=$status")
    cert = SDPX.certificate(result)
    cert.valid || error(
        "quartic original-coordinate certificate is invalid: reason=$(cert.reason)",
    )
    moments = SDPX.value(result, w)
    bound = moments[2]
    hankel = _quartic_hankel(moments, order)
    spectrum = eigvals(Symmetric(hankel))
    return (
        bound=bound,
        moments=moments,
        hankel=hankel,
        spectrum=spectrum,
        certificate=cert,
        result=result,
    )
end

function _run_quartic(
    ::Type{T},
    bits,
    g_text::AbstractString,
    order::Int,
    bound_choice::AbstractString,
    threads::Int,
    blas_threads::Int,
    max_iterations::Int,
) where {T<:AbstractFloat}
    order >= 2 || error("--order must be at least 2")
    bound_choice in ("lower", "upper", "both") || error(
        "--bound must be lower, upper, or both",
    )
    return _precision_scope(T, bits) do
        g = _decimal(T, g_text)
        g > zero(T) || error("--g must be strictly positive")
        lower = bound_choice == "upper" ? nothing : _quartic_bound(
            T,
            bits,
            g_text,
            order,
            SDPX.Minimize(),
            threads,
            blas_threads,
            max_iterations,
        )
        upper = bound_choice == "lower" ? nothing : _quartic_bound(
            T,
            bits,
            g_text,
            order,
            SDPX.Maximize(),
            threads,
            blas_threads,
            max_iterations,
        )
        if lower !== nothing && upper !== nothing
            lower.bound < upper.bound || error(
                "quartic bounds are reversed: lower=$(lower.bound), upper=$(upper.bound)",
            )
            reference = _decimal(T, _QUARTIC_REFERENCE_W2)
            margin = _decimal(T, "2e-3")
            lower.bound <= reference + margin || error(
                "reference W2 is below the finite-order lower bound",
            )
            upper.bound >= reference - margin || error(
                "reference W2 is above the finite-order upper bound",
            )
        end
        return (lower=lower, upper=upper)
    end
end

function _option(args, key::AbstractString, default::AbstractString)
    flag = "--" * key
    for index in eachindex(args)
        argument = args[index]
        startswith(argument, flag * "=") && return argument[(length(flag) + 2):end]
        argument == flag && begin
            index < length(args) || error("missing value after $flag")
            return args[index + 1]
        end
    end
    return default
end

function _quartic_help()
    println("quartic_integral_sdp.jl [--g 1] [--order 8] [--bound both]")
    println("  --arithmetic f64|f64x2|f64x4|bf256|bf512")
    println("  --threads N --blas-threads N --max-iterations N")
end

function main(args=ARGS)
    any(argument -> argument == "--help", args) && return _quartic_help()
    arithmetic_name = _option(args, "arithmetic", "f64")
    type, bits = _arithmetic_type(arithmetic_name)
    threads = parse(Int, _option(args, "threads", "1"))
    blas_threads = parse(Int, _option(args, "blas-threads", "1"))
    max_iterations = parse(Int, _option(args, "max-iterations", "250"))
    threads >= 1 || error("--threads must be at least 1")
    blas_threads >= 1 || error("--blas-threads must be at least 1")
    max_iterations >= 1 || error("--max-iterations must be at least 1")
    threads <= Base.Threads.nthreads() || error(
        "requested $threads worker threads, but Julia was started with " *
        "only $(Base.Threads.nthreads()); launch with julia -t$threads",
    )
    g_text = _option(args, "g", "1")
    order = parse(Int, _option(args, "order", "8"))
    bound_choice = lowercase(_option(args, "bound", "both"))
    values = try
        _run_quartic(
            type,
            bits,
            g_text,
            order,
            bound_choice,
            threads,
            blas_threads,
            max_iterations,
        )
    catch error_value
        if lowercase(arithmetic_name) != "f64"
            error(
                "arithmetic/provider unavailable for $arithmetic_name: " *
                sprint(showerror, error_value),
            )
        end
        rethrow()
    end
    println("quartic integral: g=$g_text, order=$order, arithmetic=$arithmetic_name")
    if values.lower !== nothing
        println("  lower W2 = ", values.lower.bound)
        println("  lower Hankel min eigenvalue = ", minimum(values.lower.spectrum))
        println("  lower primal residual = ", values.lower.certificate.primal_residual)
    end
    if values.upper !== nothing
        println("  upper W2 = ", values.upper.bound)
        println("  upper Hankel min eigenvalue = ", minimum(values.upper.spectrum))
        println("  upper primal residual = ", values.upper.certificate.primal_residual)
    end
    return values
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
