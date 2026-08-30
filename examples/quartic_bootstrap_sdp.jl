using SDPX
using GenericLinearAlgebra
using LinearAlgebra
using MultiFloats

"""Finite-order moment SDP for the zero-dimensional quartic integral.

The unknowns are the even moments `w[n + 1] = W_(2n)`. Integration by parts
gives exact linear recurrences, while positivity of `p(x)^2` gives one
parity-aware Hankel matrix constraint. Minimizing and maximizing `W_2`
produces the finite-order bootstrap interval.
"""

const QUARTIC_REFERENCE_W2 = "0.467919916973665188637421298330615640"

function load_requested_provider(::Type{T}, provider::Symbol) where {T<:AbstractFloat}
    provider in (:auto, :standard, :multifloat, :bfla) || error(
        "provider must be auto, standard, multifloat, or bfla for the native SDP example",
    )
    if provider === :multifloat ||
       (provider === :auto && T <: MultiFloats.MultiFloat)
        T <: MultiFloats.MultiFloat || error(
            "provider=:multifloat requires a fixed-width MultiFloat model",
        )
        try
        @eval import MultiFloatLinearAlgebra
        catch error_value
            error(
                "provider=:multifloat requires MultiFloatLinearAlgebra in the active " *
                "environment: $(sprint(showerror, error_value))",
            )
        end
    end
    if provider === :bfla || (provider === :auto && T === BigFloat)
        T === BigFloat || error("provider=:bfla requires a BigFloat model")
        try
        @eval import BigFloatLinearAlgebra
        catch error_value
            error(
                "provider=:bfla requires BigFloatLinearAlgebra in the active " *
                "environment: $(sprint(showerror, error_value))",
            )
        end
    end
    return nothing
end

function with_precision(f::Function, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        f()
    end
end

with_precision(f::Function, ::Type{T}, bits) where {T<:AbstractFloat} = f()

decimal(::Type{BigFloat}, text::AbstractString) = BigFloat(text)
decimal(::Type{Float64}, text::AbstractString) = parse(Float64, text)

function decimal(::Type{T}, text::AbstractString) where {T<:AbstractFloat}
    return setprecision(BigFloat, 256) do
        T(BigFloat(text))
    end
end

function arithmetic_type(name::AbstractString)
    key = lowercase(name)
    key == "f64" && return (Float64, nothing)
    key == "bf256" && return (BigFloat, 256)
    key == "bf512" && return (BigFloat, 512)
    key in ("f64x2", "f64x4") || error(
        "unknown arithmetic '$name'; choose f64, f64x2, f64x4, bf256, or bf512",
    )
    type_name = key == "f64x2" ? :Float64x2 : :Float64x4
    isdefined(MultiFloats, type_name) || error("MultiFloats does not define $type_name")
    return (getfield(MultiFloats, type_name), nothing)
end

"""Build the mathematical SDP. This is the instructional core of the file."""
function quartic_bootstrap_model(
    ::Type{T},
    bits,
    g::T,
    order::Int,
    sense,
) where {T<:AbstractFloat}
    order >= 2 || error("order must be at least 2")
    model = T === BigFloat ?
        Model(T; precision_bits=bits, name="quartic_bootstrap_sdp") :
        Model(T; name="quartic_bootstrap_sdp")

    # w = [W_0, W_2, ..., W_(2 order)].
    w = variable!(model, :w, order + 1; domain=Reals())
    constraint!(model, :normalization, w[1] - one(T), ZeroCone())

    # (2n+1)W_(2n) - W_(2n+2) - g W_(2n+4) = 0.
    for n in 0:(order - 2)
        recurrence = (2n + 1) * w[n + 1] - w[n + 2] - g * w[n + 3]
        constraint!(model, Symbol("recurrence_", n), recurrence, ZeroCone())
    end

    # Odd moments vanish. In the natural monomial ordering, the even and odd
    # Stieltjes sectors appear as the two parity blocks of this matrix.
    hankel = [
        iseven(i + j) ? w[(i + j) ÷ 2 + 1] : zero(T)
        for i in 0:order, j in 0:order
    ]
    constraint!(model, :moment_matrix, hankel, PSDCone())
    objective!(model, sense, w[2])
    return model, w
end

function solve_quartic_bound(
    ::Type{T},
    bits,
    g::T,
    order::Int,
    sense;
    provider::Symbol=:auto,
    threads::Int=1,
    max_iterations::Int=250,
) where {T<:AbstractFloat}
    model, w = quartic_bootstrap_model(T, bits, g, order, sense)
    settings = Settings(
        model;
        # The public native route chooses the formulation and arithmetic
        # provider from the model. `provider` is only a dependency-loading
        # hint handled by `load_requested_provider` before this call.
        formulation=:auto,
        provider=:auto,
        sparse=:off,
        presolve=:off,
        limits=Limits(iterations=max_iterations, time=120.0, threads=threads),
        verbosity=0,
        timing=true,
        diagnostics=:summary,
        certification=true,
    )
    outputs = Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
    result = nothing
    optimize_seconds = @elapsed result = optimize!(model; settings=settings, outputs=outputs)
    status(result) === :optimal || error("quartic SDP ended with status $(status(result))")

    cert = certificate(result)
    cert.valid || error("invalid original-coordinate certificate: $(cert.reason)")
    moments = value(result, w)
    hankel = [
        iseven(i + j) ? moments[(i + j) ÷ 2 + 1] : zero(T)
        for i in 0:order, j in 0:order
    ]
    spectrum = eigvals(Symmetric(hankel))
    solve_diagnostics = diagnostics(result)
    return (
        bound=moments[2],
        moments=moments,
        hankel=hankel,
        spectrum=spectrum,
        certificate=cert,
        plan=execution_plan(result),
        diagnostics=solve_diagnostics,
        optimize_seconds=optimize_seconds,
        result=result,
    )
end

function run_quartic_bootstrap(
    ::Type{T},
    bits,
    g_text::AbstractString,
    order::Int,
    bound_choice::AbstractString;
    provider::Symbol=:auto,
    threads::Int=1,
    max_iterations::Int=250,
) where {T<:AbstractFloat}
    return with_precision(T, bits) do
        g = decimal(T, g_text)
        g > zero(T) || error("g must be strictly positive")
        bound_choice in ("lower", "upper", "both") ||
            error("bound must be lower, upper, or both")

        lower = bound_choice == "upper" ? nothing : solve_quartic_bound(
            T, bits, g, order, Minimize();
            provider, threads, max_iterations,
        )
        upper = bound_choice == "lower" ? nothing : solve_quartic_bound(
            T, bits, g, order, Maximize();
            provider, threads, max_iterations,
        )

        if lower !== nothing && upper !== nothing
            lower.bound <= upper.bound || error("computed bootstrap interval is reversed")
            if g_text == "1"
                reference = decimal(T, QUARTIC_REFERENCE_W2)
                tolerance = T === Float64 ? T(2e-6) : T(1e-12)
                lower.bound <= reference + tolerance ||
                    error("known g=1 value lies below the computed lower bound")
                reference <= upper.bound + tolerance ||
                    error("known g=1 value lies above the computed upper bound")
            end
        end
        return (lower=lower, upper=upper)
    end
end

function option(args, key::AbstractString, default::AbstractString)
    flag = "--" * key
    for index in eachindex(args)
        argument = args[index]
        startswith(argument, flag * "=") && return argument[(length(flag) + 2):end]
        if argument == flag
            index < length(args) || error("missing value after $flag")
            return args[index + 1]
        end
    end
    return default
end

function print_help()
    println("quartic_bootstrap_sdp.jl [--g 1] [--order 8] [--bound both]")
    println("  --arithmetic f64|f64x2|f64x4|bf256|bf512")
    println("  --provider auto|standard|multifloat|bfla|legacy")
    println("  --threads N --max-iterations N")
end

function main(args=ARGS)
    any(==("--help"), args) && return print_help()
    arithmetic_name = lowercase(option(args, "arithmetic", "f64"))
    T, bits = arithmetic_type(arithmetic_name)
    provider = Symbol(lowercase(option(args, "provider", "auto")))
    threads = parse(Int, option(args, "threads", "1"))
    max_iterations = parse(Int, option(args, "max-iterations", "250"))
    order = parse(Int, option(args, "order", "8"))
    g_text = option(args, "g", "1")
    bound_choice = lowercase(option(args, "bound", "both"))

    threads >= 1 || error("threads must be at least 1")
    threads <= Base.Threads.nthreads() || error(
        "requested $threads SDPX threads, but Julia owns only $(Base.Threads.nthreads())",
    )
    load_requested_provider(T, provider)

    # Optional provider extensions may have been loaded just above.  Enter the
    # latest method world before planning so their registered methods are
    # visible even when this file is executed directly as a script.
    values = Base.invokelatest(
        run_quartic_bootstrap,
        T,
        bits,
        g_text,
        order,
        bound_choice;
        provider,
        threads,
        max_iterations,
    )

    println("quartic bootstrap SDP: g=$g_text, order=$order, arithmetic=$arithmetic_name")
    for (label, record) in (("lower", values.lower), ("upper", values.upper))
        record === nothing && continue
        println("  $label W2 = ", record.bound)
        println("  $label min eigenvalue = ", minimum(record.spectrum))
        println("  $label executed provider = ",
            record.diagnostics.selected_algorithms.la_executed_provider)
        println("  $label optimize! seconds = ", record.optimize_seconds)
        println("  $label recorded core seconds = ", record.diagnostics.timings.core)
        println("  $label iterations = ", record.diagnostics.termination.iterations)
        println("  $label certificate residuals = (",
            record.certificate.primal_residual, ", ",
            record.certificate.dual_residual, ", ",
            record.certificate.relative_gap, ")")
    end
    return values
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
